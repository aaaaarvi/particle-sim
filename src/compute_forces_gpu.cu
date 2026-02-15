#include <cmath>
#include "compute_forces_gpu.cuh"

__global__
void compute_forces_(
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
        float fx = 0.0;
        float fy = 0.0;
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

    // Launch kernel
    compute_forces_<<<num_blocks, num_threads>>>(
        n_particles,
        d_positions_x,
        d_positions_y,
        d_forces_x,
        d_forces_y,
        epsilon,
        extends);

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
