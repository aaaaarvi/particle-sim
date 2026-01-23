#pragma once

void compute_forces_gpu(
    int n_particles,
    double* positions_x,
    double* positions_y,
    double* forces_x,
    double* forces_y,
    double extends,
    double epsilon);
