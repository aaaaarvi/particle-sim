#include <cmath>
#include "compute_forces_gpu.cuh"

__global__
void compute_forces_(
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

void compute_forces(
    int n_particles,
    double* positions_x,
    double* positions_y,
    double* forces_x,
    double* forces_y,
    double extends,
    double epsilon) {

    int num_threads = 256; // Number of threads per block
    int num_blocks = (n_particles + num_threads - 1) / num_threads; // Number of blocks needed

    // Initialize device pointers
    double *d_positions_x, *d_positions_y;
    double *d_forces_x, *d_forces_y;

    // Allocate device memory
    cudaMalloc((void**)&d_positions_x, n_particles * sizeof(double));
    cudaMalloc((void**)&d_positions_y, n_particles * sizeof(double));
    cudaMalloc((void**)&d_forces_x, n_particles * sizeof(double));
    cudaMalloc((void**)&d_forces_y, n_particles * sizeof(double));

    // Copy positions to device
    cudaMemcpy(d_positions_x, positions_x, n_particles * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions_y, positions_y, n_particles * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    compute_forces_<<<num_blocks, num_threads>>>(
        n_particles,
        d_positions_x,
        d_positions_y,
        d_forces_x,
        d_forces_y,
        extends,
        epsilon);

    // Copy forces back to host
    cudaMemcpy(forces_x, d_forces_x, n_particles * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(forces_y, d_forces_y, n_particles * sizeof(double), cudaMemcpyDeviceToHost);
}
