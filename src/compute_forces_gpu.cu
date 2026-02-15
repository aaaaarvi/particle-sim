#include <cmath>
#include "compute_forces_gpu.cuh"
#include "quad_tree.h"
#include "quad_tree_flat.h"

__global__
void compute_forces_direct_(
    int n_particles,
    float* positions_x,
    float* positions_y,
    float* forces_x,
    float* forces_y,
    float epsilon,
    int extends) {
    /*
    threadIdx.x: The index of the thread within its block.
    blockIdx.x: The index of the block within the grid.
    blockDim.x: The total number of threads in the block.
    gridDim.x: The total number of blocks in the grid.
    */
    int index = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    for (int i = index; i < n_particles; i += stride) {
        float fx = 0.0f;
        float fy = 0.0f;
        for (int j = 0; j < n_particles; j++) {
            if (i == j) continue;
            for (float xx = -extends; xx <= extends; xx++) {
                for (float yy = -extends; yy <= extends; yy++) {
                    float dx = positions_x[j] - positions_x[i] + xx;
                    float dy = positions_y[j] - positions_y[i] + yy;
                    float dist = sqrt(dx*dx + dy*dy) + epsilon;
                    fx += dx / (dist * dist * dist);
                    fy += dy / (dist * dist * dist);
                }
            }
        }
        forces_x[i] = fx;
        forces_y[i] = fy;
    }
}

// Allocate device memory for quadtree recursively
// (Allocation happens root-to-leaf, and data is copied leaf-to-root)
void allocate_quadtree_on_device(const quad_tree::node_t* node, quad_tree::node_t** d_node) {
    if (node == nullptr) {
        *d_node = nullptr;
        return;
    }

    // Allocate device memory for this node
    cudaMalloc((void**)d_node, sizeof(quad_tree::node_t));

    // Recursively allocate children on device and get device pointers
    quad_tree::node_t* d_nw = nullptr;
    quad_tree::node_t* d_ne = nullptr;
    quad_tree::node_t* d_sw = nullptr;
    quad_tree::node_t* d_se = nullptr;
    allocate_quadtree_on_device(node->north_west, &d_nw);
    allocate_quadtree_on_device(node->north_east, &d_ne);
    allocate_quadtree_on_device(node->south_west, &d_sw);
    allocate_quadtree_on_device(node->south_east, &d_se);

    // Make a host copy, but replace child pointers with device pointers
    quad_tree::node_t h_node = *node;
    h_node.north_west = d_nw;
    h_node.north_east = d_ne;
    h_node.south_west = d_sw;
    h_node.south_east = d_se;

    // Copy the host copy (with device-child pointers) into device memory
    cudaMemcpy(*d_node, &h_node, sizeof(quad_tree::node_t), cudaMemcpyHostToDevice);
}

// Allocate flattened quadtree on device and return device pointer
quad_tree_flat::node_t* allocate_flattened_quadtree_on_device(const quad_tree_flat::tree_t* h_tree) {
    int tree_size = (int)h_tree->nodes.size();
    quad_tree_flat::node_t* d_nodes = nullptr;
    cudaMalloc((void**)&d_nodes, tree_size * sizeof(quad_tree_flat::node_t));
    cudaMemcpy(d_nodes, h_tree->nodes.data(), tree_size * sizeof(quad_tree_flat::node_t), cudaMemcpyHostToDevice);
    return d_nodes;
}

// Free device memory for quadtree recursively
void free_quadtree_on_device(quad_tree::node_t* d_node) {
    if (d_node == nullptr) return;

    // Copy the current node back to host to access child pointers
    quad_tree::node_t h_node;
    cudaMemcpy(&h_node, d_node, sizeof(quad_tree::node_t), cudaMemcpyDeviceToHost);

    // Recursively free memory for children
    free_quadtree_on_device(h_node.north_west);
    free_quadtree_on_device(h_node.north_east);
    free_quadtree_on_device(h_node.south_west);
    free_quadtree_on_device(h_node.south_east);

    // Free memory for the current node
    cudaFree(d_node);
}

// Free flattened quadtree on device
void free_flattened_quadtree_on_device(quad_tree_flat::node_t* d_nodes) {
    if (d_nodes != nullptr) {
        cudaFree(d_nodes);
    }
}

// Re-implementation of quad_tree::compute_force for device code
__device__
void compute_force_quadtree_(
    quad_tree::node_t* node,
    float* force_x,
    float* force_y,
    float x,
    float y,
    float mass,
    float theta_max,
    float epsilon) {

    if (node == nullptr) return;
    if (node->num_particles == 0) return;

    float dx = node->center_of_mass_x - x;
    float dy = node->center_of_mass_y - y;
    float dist = sqrtf(dx * dx + dy * dy) + epsilon;
    float theta = node->width / dist;

    // Check if we can approximate
    if (theta < theta_max || node->num_particles == 1) {
        // Compute force contribution
        float force_magnitude = (mass * node->mass) / (dist * dist);
        *force_x += force_magnitude * (dx / dist);
        *force_y += force_magnitude * (dy / dist);
    } else {
        // Recurse into children
        compute_force_quadtree_(node->north_west, force_x, force_y, x, y, mass, theta_max, epsilon);
        compute_force_quadtree_(node->north_east, force_x, force_y, x, y, mass, theta_max, epsilon);
        compute_force_quadtree_(node->south_west, force_x, force_y, x, y, mass, theta_max, epsilon);
        compute_force_quadtree_(node->south_east, force_x, force_y, x, y, mass, theta_max, epsilon);
    }
}

// Iterative traversal of flattened quadtree on device (index-based)
__device__
void compute_force_quadtree_flat_(
    quad_tree_flat::node_t* nodes,
    float* force_x,
    float* force_y,
    float x,
    float y,
    float mass,
    float theta_max,
    float epsilon) {

    if (nodes == nullptr) return;

    // Use a stack to traverse the tree iteratively (avoids recursion overhead)
    int stack[64];  // enough depth for typical quadtrees
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;  // start at root

    while (stack_ptr > 0) {
        int idx = stack[--stack_ptr];
        quad_tree_flat::node_t* node = &nodes[idx];

        if (node->num_particles == 0) continue;

        float dx = node->center_of_mass_x - x;
        float dy = node->center_of_mass_y - y;
        float dist = sqrtf(dx * dx + dy * dy) + epsilon;
        float theta = node->width / dist;

        // Check if we can approximate
        if (theta < theta_max || node->num_particles == 1) {
            // Compute force contribution
            float force_magnitude = (mass * node->mass) / (dist * dist);
            *force_x += force_magnitude * (dx / dist);
            *force_y += force_magnitude * (dy / dist);
        } else {
            // Push children onto stack (in reverse order for depth-first traversal)
            if (node->se >= 0) stack[stack_ptr++] = node->se;
            if (node->sw >= 0) stack[stack_ptr++] = node->sw;
            if (node->ne >= 0) stack[stack_ptr++] = node->ne;
            if (node->nw >= 0) stack[stack_ptr++] = node->nw;
        }
    }
}

__global__
void compute_forces_barnes_hut_v1_(
    int n_particles,
    float* positions_x,
    float* positions_y,
    float* forces_x,
    float* forces_y,
    float epsilon,
    int extends,
    float theta_max,
    quad_tree::node_t* root) {
    /*
    threadIdx.x: The index of the thread within its block.
    blockIdx.x: The index of the block within the grid.
    blockDim.x: The total number of threads in the block.
    gridDim.x: The total number of blocks in the grid.
    */
    int index = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    for (int i = index; i < n_particles; i += stride) {
        float fx = 0.0f;
        float fy = 0.0f;
        compute_force_quadtree_(
            root, &fx, &fy, positions_x[i], positions_y[i], 1.0f, theta_max, epsilon);
        forces_x[i] = fx;
        forces_y[i] = fy;
    }
}

__global__
void compute_forces_barnes_hut_v2_(
    int n_particles,
    float* positions_x,
    float* positions_y,
    float* forces_x,
    float* forces_y,
    float epsilon,
    int extends,
    float theta_max,
    quad_tree_flat::node_t* nodes) {
    /*
    threadIdx.x: The index of the thread within its block.
    blockIdx.x: The index of the block within the grid.
    blockDim.x: The total number of threads in the block.
    gridDim.x: The total number of blocks in the grid.
    */
    int index = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    for (int i = index; i < n_particles; i += stride) {
        float fx = 0.0f;
        float fy = 0.0f;
        compute_force_quadtree_flat_(
            nodes, &fx, &fy, positions_x[i], positions_y[i], 1.0f, theta_max, epsilon);
        forces_x[i] = fx;
        forces_y[i] = fy;
    }
}

void compute_forces(
    std::vector<Particle<float>>* particles,
    const float epsilon,
    const int extends) {

    int n_particles = particles->size();

    int num_threads = 256; // Number of threads per block
    int num_blocks = (n_particles + num_threads - 1) / num_threads; // Number of blocks needed

    // Copy positions to host pointers
    float h_positions_x[n_particles], h_positions_y[n_particles];
    #pragma omp parallel for
    for (int i = 0; i < n_particles; i++) {
        h_positions_x[i] = (*particles)[i].pos.x;
        h_positions_y[i] = (*particles)[i].pos.y;
    }

    // Initialize device pointers
    float *d_positions_x, *d_positions_y;
    float *d_forces_x, *d_forces_y;

    // Allocate device memory
    cudaMalloc((void**)&d_positions_x, n_particles * sizeof(float));
    cudaMalloc((void**)&d_positions_y, n_particles * sizeof(float));
    cudaMalloc((void**)&d_forces_x, n_particles * sizeof(float));
    cudaMalloc((void**)&d_forces_y, n_particles * sizeof(float));

    // Copy positions to device
    cudaMemcpy(d_positions_x, h_positions_x, n_particles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions_y, h_positions_y, n_particles * sizeof(float), cudaMemcpyHostToDevice);

    // Direct force computation, O(N^2)
    // (Assumes all particles have mass 1.)
    /** /

    // Compute forces
    compute_forces_direct_<<<num_blocks, num_threads>>>(
        n_particles,
        d_positions_x,
        d_positions_y,
        d_forces_x,
        d_forces_y,
        epsilon,
        extends);
    //*/

    // Barnes-Hut algorithm v1, O(N log N)
    /** /
    float theta_max = 0.5f;

    // Compute min/max x/y coordinates
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    for (const auto& p : *particles) {
        if (p.pos.x - (float)extends < min_x) min_x = p.pos.x - (float)extends;
        if (p.pos.x + (float)extends > max_x) max_x = p.pos.x + (float)extends;
        if (p.pos.y - (float)extends < min_y) min_y = p.pos.y - (float)extends;
        if (p.pos.y + (float)extends > max_y) max_y = p.pos.y + (float)extends;
    }
    float origo_x = (min_x + max_x) * 0.5f;
    float origo_y = (min_y + max_y) * 0.5f;
    float width = std::max(max_x - min_x, max_y - min_y);

    // Initialize quadtree
    quad_tree::node_t* h_root;
    quad_tree::init(&h_root, origo_x, origo_y, width);
    for (const auto& p : *particles) {
        for (int xx = -extends; xx <= extends; xx++) {
            for (int yy = -extends; yy <= extends; yy++) {
                quad_tree::insert(h_root, p.pos.x + (float)xx, p.pos.y + (float)yy, p.mass);
            }
        }
    }

    // Allocate device memory for quadtree
    quad_tree::node_t* d_root;
    allocate_quadtree_on_device(h_root, &d_root);

    // Compute forces
    compute_forces_barnes_hut_v1_<<<num_blocks, num_threads>>>(
        n_particles,
        d_positions_x,
        d_positions_y,
        d_forces_x,
        d_forces_y,
        epsilon,
        extends,
        theta_max,
        d_root);

    // Free device memory for quadtree
    free_quadtree_on_device(d_root);
    //*/

    // Barnes-Hut algorithm with flattened quadtree, O(N log N)
    /**/
    float theta_max = 0.5f;

    // Compute min/max x/y coordinates
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    for (const auto& p : *particles) {
        if (p.pos.x - (float)extends < min_x) min_x = p.pos.x - (float)extends;
        if (p.pos.x + (float)extends > max_x) max_x = p.pos.x + (float)extends;
        if (p.pos.y - (float)extends < min_y) min_y = p.pos.y - (float)extends;
        if (p.pos.y + (float)extends > max_y) max_y = p.pos.y + (float)extends;
    }
    float origo_x = (min_x + max_x) * 0.5f;
    float origo_y = (min_y + max_y) * 0.5f;
    float width = std::max(max_x - min_x, max_y - min_y);

    // Initialize flattened quadtree on host
    quad_tree_flat::tree_t h_tree;
    quad_tree_flat::init(&h_tree, origo_x, origo_y, width);
    for (const auto& p : *particles) {
        for (int xx = -extends; xx <= extends; xx++) {
            for (int yy = -extends; yy <= extends; yy++) {
                quad_tree_flat::insert(&h_tree, p.pos.x + (float)xx, p.pos.y + (float)yy, p.mass);
            }
        }
    }

    // Allocate device memory for flattened quadtree (single bulk allocation)
    quad_tree_flat::node_t* d_nodes = allocate_flattened_quadtree_on_device(&h_tree);

    // Compute forces
    compute_forces_barnes_hut_v2_<<<num_blocks, num_threads>>>(
        n_particles,
        d_positions_x,
        d_positions_y,
        d_forces_x,
        d_forces_y,
        epsilon,
        extends,
        theta_max,
        d_nodes);

    // Free device memory for flattened quadtree
    free_flattened_quadtree_on_device(d_nodes);
    //*/

    // Copy forces back to host
    float h_forces_x[n_particles], h_forces_y[n_particles];
    cudaMemcpy(h_forces_x, d_forces_x, n_particles * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_forces_y, d_forces_y, n_particles * sizeof(float), cudaMemcpyDeviceToHost);
    #pragma omp parallel for
    for (int i = 0; i < n_particles; i++) {
        (*particles)[i].acc.x = h_forces_x[i];
        (*particles)[i].acc.y = h_forces_y[i];
    }

    // Free device memory
    cudaFree(d_positions_x);
    cudaFree(d_positions_y);
    cudaFree(d_forces_x);
    cudaFree(d_forces_y);
}
