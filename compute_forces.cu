#include <cmath>
#include "compute_forces.cuh"

__global__
void compute_forces_gpu_(
    int n_particles,
    double* positions_x,
    double* positions_y,
    double* forces_x,
    double* forces_y,
    double extends,
    double epsilon) {
    /*
    threadIdx.x: The index of the thread within its block.
    blockIdx.x: The index of the block within the grid.
    blockDim.x: The total number of threads in the block.
    gridDim.x: The total number of blocks in the grid.
    */
    int index = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    for (int i = index; i < n_particles; i += stride) {
        double fx = 0.0;
        double fy = 0.0;
        for (int j = 0; j < n_particles; j++) {
            for (double xx = -extends; xx <= extends; xx++) {
                for (double yy = -extends; yy <= extends; yy++) {
                    double dx = positions_x[i] - positions_x[j] + xx;
                    double dy = positions_y[i] - positions_y[j] + yy;
                    double dist = sqrt(dx*dx + dy*dy) + epsilon;
                    fx += dx / (dist * dist * dist);
                    fy += dy / (dist * dist * dist);
                }
            }
        }
        forces_x[i] = fx;
        forces_y[i] = fy;
    }
}

void compute_forces_gpu(
    int n_particles,
    double* positions_x,
    double* positions_y,
    double* forces_x,
    double* forces_y,
    double extends,
    double epsilon) {

    int num_threads = 256; // Number of threads per block
    int num_blocks = (n_particles + num_threads - 1) / num_threads; // Number of blocks needed
    compute_forces_gpu_<<<num_blocks, num_threads>>>(
        n_particles,
        positions_x,
        positions_y,
        forces_x,
        forces_y,
        extends,
        epsilon);
}
