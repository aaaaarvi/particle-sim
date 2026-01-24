#include <cmath>
#include "compute_forces_cpu.h"

void compute_forces(
    int n_particles,
    double* positions_x,
    double* positions_y,
    double* forces_x,
    double* forces_y,
    double extends,
    double epsilon) {

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
}
