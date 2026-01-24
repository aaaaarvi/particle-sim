#include <cmath>
#include <iostream>
#include "quad_tree.h"

namespace quad_tree {

void init(node_t** root, double origo_x, double origo_y, double width) {
    *root = new node_t;
    (*root)->north_west = nullptr;
    (*root)->north_east = nullptr;
    (*root)->south_west = nullptr;
    (*root)->south_east = nullptr;
    (*root)->origo_x = origo_x;
    (*root)->origo_y = origo_y;
    (*root)->width = width;
    (*root)->center_of_mass_x = origo_x;
    (*root)->center_of_mass_y = origo_y;
    (*root)->mass = 0.0;
    (*root)->id = -1;
    (*root)->num_particles = 0;
}

void insert(node_t* node, double x, double y, double mass, int id) {
    double half_width = node->width / 2.0;
    double quarter_width = node->width / 4.0;

    // If node is empty, insert particle here
    if (node->num_particles == 0) {
        // If outside bounds, do nothing
        if (x <= node->origo_x - node->width / 2.0 || x >= node->origo_x + node->width / 2.0 ||
            y <= node->origo_y - node->width / 2.0 || y >= node->origo_y + node->width / 2.0) {
            std::cout << "Warning: Particle out of bounds (" << x << ", " << y << "), skipping\n";
            return;
        }
        node->center_of_mass_x = x;
        node->center_of_mass_y = y;
        node->mass = mass;
        node->id = id;
        node->num_particles = 1;
        return;
    }

    // If node is a leaf, subdivide and re-insert existing particle
    if (node->num_particles == 1) {
        bool insert_north = node->center_of_mass_y >= node->origo_y;
        bool insert_west = node->center_of_mass_x <= node->origo_x;
        if (insert_north && insert_west) {
            if (node->north_west == nullptr)
                init(&node->north_west, node->origo_x - quarter_width, node->origo_y + quarter_width, half_width);
            insert(node->north_west, node->center_of_mass_x, node->center_of_mass_y, node->mass, node->id);
        } else if (insert_north && !insert_west) {
            if (node->north_east == nullptr)
                init(&node->north_east, node->origo_x + quarter_width, node->origo_y + quarter_width, half_width);
            insert(node->north_east, node->center_of_mass_x, node->center_of_mass_y, node->mass, node->id);
        } else if (!insert_north && insert_west) {
            if (node->south_west == nullptr)
                init(&node->south_west, node->origo_x - quarter_width, node->origo_y - quarter_width, half_width);
            insert(node->south_west, node->center_of_mass_x, node->center_of_mass_y, node->mass, node->id);
        } else {
            if (node->south_east == nullptr)
                init(&node->south_east, node->origo_x + quarter_width, node->origo_y - quarter_width, half_width);
            insert(node->south_east, node->center_of_mass_x, node->center_of_mass_y, node->mass, node->id);
        }
        node->id = -1; // Reset id since it's no longer a leaf
    }

    // Insert new particle
    bool insert_north = y >= node->origo_y;
    bool insert_west = x <= node->origo_x;
    if (insert_north && insert_west) {
        if (node->north_west == nullptr)
            init(&node->north_west, node->origo_x - quarter_width, node->origo_y + quarter_width, half_width);
        insert(node->north_west, x, y, mass, id);
    } else if (insert_north && !insert_west) {
        if (node->north_east == nullptr)
            init(&node->north_east, node->origo_x + quarter_width, node->origo_y + quarter_width, half_width);
        insert(node->north_east, x, y, mass, id);
    } else if (!insert_north && insert_west) {
        if (node->south_west == nullptr)
            init(&node->south_west, node->origo_x - quarter_width, node->origo_y - quarter_width, half_width);
        insert(node->south_west, x, y, mass, id);
    } else {
        if (node->south_east == nullptr)
            init(&node->south_east, node->origo_x + quarter_width, node->origo_y - quarter_width, half_width);
        insert(node->south_east, x, y, mass, id);
    }

    // Update this nodes properties
    node->center_of_mass_x = (node->center_of_mass_x * node->mass + x * mass) / (node->mass + mass);
    node->center_of_mass_y = (node->center_of_mass_y * node->mass + y * mass) / (node->mass + mass);
    node->mass += mass;
    node->num_particles += 1;
}

// Compute force recursively
void compute_force(node_t* node, double* force_x, double* force_y, int id, double x, double y, double mass, double theta_max, double epsilon) {
    if (node == nullptr) return;
    if (node->num_particles == 0) return;
    if (node->id == id) return;

    double dx = x - node->center_of_mass_x;
    double dy = y - node->center_of_mass_y;
    double dist = sqrt(dx * dx + dy * dy) + epsilon;
    double theta = node->width / dist;

    // Check if we can approximate
    if (theta < theta_max || node->num_particles == 1) {
        // Compute force contribution
        double force_magnitude = (mass * node->mass) / (dist * dist);
        *force_x += force_magnitude * (dx / dist);
        *force_y += force_magnitude * (dy / dist);
    } else {
        // Recurse into children
        compute_force(node->north_west, force_x, force_y, id, x, y, mass, theta_max, epsilon);
        compute_force(node->north_east, force_x, force_y, id, x, y, mass, theta_max, epsilon);
        compute_force(node->south_west, force_x, force_y, id, x, y, mass, theta_max, epsilon);
        compute_force(node->south_east, force_x, force_y, id, x, y, mass, theta_max, epsilon);
    }
}

// Print the tree
void print_tree(node_t* node, int depth, bool is_root) {
    if (node == nullptr) return;

    if (is_root) {
        printf("Root: ");
        print_tree(node, depth, false);
        return;
    }

    if (node->num_particles == 0) {
        std::cout << "[empty]\n";
        return;
    }

    if (node->num_particles == 1) {
        printf("[leaf:%d] pos = (%.4f, %.4f), mass = %.1f\n",
               node->id, node->center_of_mass_x, node->center_of_mass_y, node->mass);
        return;
    }

    printf("[cluster] origo = (%.4f, %.4f), width = %.4f, mass = %.1f\n",
           node->origo_x, node->origo_y, node->width, node->mass);

    if (node->north_west != nullptr) {
        std::cout << std::string(depth + 1, ' ') << "NW: ";
        print_tree(node->north_west, depth + 1, false);
    }
    if (node->north_east != nullptr) {
        std::cout << std::string(depth + 1, ' ') << "NE: ";
        print_tree(node->north_east, depth + 1, false);
    }
    if (node->south_west != nullptr) {
        std::cout << std::string(depth + 1, ' ') << "SW: ";
        print_tree(node->south_west, depth + 1, false);
    }
    if (node->south_east != nullptr) {
        std::cout << std::string(depth + 1, ' ') << "SE: ";
        print_tree(node->south_east, depth + 1, false);
    }
}

// Recursive function to free memory
void free_tree(node_t* node) {
    if (node == nullptr) return;
    free_tree(node->north_west);
    free_tree(node->north_east);
    free_tree(node->south_west);
    free_tree(node->south_east);
    delete node;
}

} // namespace quadtree
