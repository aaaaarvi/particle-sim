#include <cmath>
#include "compute_forces_cpu.h"
#include "quad_tree.h"

void compute_forces(
    int n_particles,
    double* positions_x,
    double* positions_y,
    double* forces_x,
    double* forces_y,
    double extends,
    double epsilon) {

    // Direct force computation, O(N^2)
    /** /
    #pragma omp parallel for
    for (int i = 0; i < n_particles; i++) {
        forces_x[i] = 0;
        forces_y[i] = 0;
        for (int j = 0; j < n_particles; j++) {
            if (i == j) continue;
            for (double xx = -extends; xx <= extends; xx++) {
                for (double yy = -extends; yy <= extends; yy++) {
                    double dx = positions_x[i] - positions_x[j] + xx;
                    double dy = positions_y[i] - positions_y[j] + yy;
                    double dist = sqrt(dx*dx + dy*dy) + epsilon;
                    //dist = dist < epsilon ? epsilon : dist;
                    forces_x[i] += dx / (dist * dist * dist);
                    forces_y[i] += dy / (dist * dist * dist);
                }
            }
        }
    }
    //*/

    // Barnes-Hut algorithm, O(N log N)
    /**/
    double theta_max = 0.5;

    // Initialize quadtree
    quad_tree::node_t* root;
    quad_tree::init(&root);
    for (int i = 0; i < n_particles; i++) {
        quad_tree::insert(root, positions_x[i], positions_y[i], 1.0, i);
    }

    // Compute forces
    #pragma omp parallel for
    for (int i = 0; i < n_particles; i++) {
        forces_x[i] = 0;
        forces_y[i] = 0;
        quad_tree::compute_force(root, &forces_x[i], &forces_y[i], i,
                                 positions_x[i], positions_y[i], 1.0,
                                 theta_max, epsilon);
    }

    // Free quadtree
    quad_tree::free_tree(root);
    //*/
}
