#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <sys/time.h>
#include <vector>

#include <cuda_runtime.h>

#include "Vector.h"
#include "compute_forces.cuh"

// Include the appropriate window header based on the operating system
#ifdef _WIN32
#include "Window.h"
#else
#include "WindowLinux.h"
#endif

struct timeval tv;
unsigned long long get_time_us() {
    gettimeofday(&tv, NULL);
    return (unsigned long long)(tv.tv_sec) * 1000000 +
           (unsigned long long)(tv.tv_usec);
}

void compute_forces_cpu(
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
            for (double xx = -extends; xx <= extends; xx++) {
                for (double yy = -extends; yy <= extends; yy++) {
                    double dx = positions_x[i] - positions_x[j] + xx;
                    double dy = positions_y[i] - positions_y[j] + yy;
                    double dist = sqrt(dx*dx + dy*dy) + epsilon;
                    forces_x[i] += dx / (dist * dist * dist);
                    forces_y[i] += dy / (dist * dist * dist);
                }
            }
        }
    }
}

void compute_forces(
    int n_particles,
    double* h_positions_x,
    double* h_positions_y,
    double* h_forces_x,
    double* h_forces_y,
    double* d_positions_x,
    double* d_positions_y,
    double* d_forces_x,
    double* d_forces_y,
    double extends,
    double epsilon,
    bool cuda) {

    if (cuda) {
        // Copy positions to device
        cudaMemcpy(d_positions_x, h_positions_x, n_particles * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_positions_y, h_positions_y, n_particles * sizeof(double), cudaMemcpyHostToDevice);

        // Launch kernel
        compute_forces_gpu(n_particles, d_positions_x, d_positions_y, d_forces_x, d_forces_y, extends, epsilon);

        // Copy forces back to host
        cudaMemcpy(h_forces_x, d_forces_x, n_particles * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_forces_y, d_forces_y, n_particles * sizeof(double), cudaMemcpyDeviceToHost);
    } else {
        compute_forces_cpu(n_particles, h_positions_x, h_positions_y, h_forces_x, h_forces_y, extends, epsilon);
    }
}

int main()
{
    srand(static_cast<unsigned>(time(0)));

    const int n_particles = 10000;
    const double g_const = 0.001 / (double)n_particles;
    const double epsilon = 0.01;
    const double delta_t = 0.01;
    const int width = 720;
    const int height = 720;
    const int offset_w = 100;
    const int offset_h = 100;
    const int extends = 0;
    const bool periodic = false;
    const bool timings = true;
    const bool cuda = true;

    double *h_positions_x, *h_positions_y;
    double *h_velocities_x, *h_velocities_y;
    double *h_forces_x, *h_forces_y;
    double *d_positions_x, *d_positions_y;
    double *d_forces_x, *d_forces_y;

    std::cout << "Allocating memory\n";

    if (cuda) {
        // Allocate host memory
        h_positions_x = new double[n_particles];
        h_positions_y = new double[n_particles];
        h_velocities_x = new double[n_particles];
        h_velocities_y = new double[n_particles];
        h_forces_x = new double[n_particles];
        h_forces_y = new double[n_particles];

        // Allocate device memory
        cudaMalloc((void**)&d_positions_x, n_particles * sizeof(double));
        cudaMalloc((void**)&d_positions_y, n_particles * sizeof(double));
        cudaMalloc((void**)&d_forces_x, n_particles * sizeof(double));
        cudaMalloc((void**)&d_forces_y, n_particles * sizeof(double));
    } else {
        // Allocate host memory directly
        h_positions_x = new double[n_particles];
        h_positions_y = new double[n_particles];
        h_velocities_x = new double[n_particles];
        h_velocities_y = new double[n_particles];
        h_forces_x = new double[n_particles];
        h_forces_y = new double[n_particles];

        d_positions_x = h_positions_x;
        d_positions_y = h_positions_y;
        d_forces_x = h_forces_x;
        d_forces_y = h_forces_y;
    }

    // Alias for simplicity
    double *positions_x = h_positions_x;
    double *positions_y = h_positions_y;
    double *velocities_x = h_velocities_x;
    double *velocities_y = h_velocities_y;
    double *forces_x = h_forces_x;
    double *forces_y = h_forces_y;

    std::cout << "Initializing particles\n";

    // Initialize "two galaxies"
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < n_particles / 2; i++) {
        positions_x[i] = 0.25 + 0.01*distribution(generator);
        positions_y[i] = 0.5 + 0.01*distribution(generator);
        velocities_x[i] = 0.0;
        velocities_y[i] = 0.1;
        forces_x[i] = 0.0;
        forces_y[i] = 0.0;
    }
    for (int i = n_particles / 2; i < n_particles; i++) {
        positions_x[i] = 0.75 + 0.01*distribution(generator);
        positions_y[i] = 0.5 + 0.01*distribution(generator);
        velocities_x[i] = 0.0;
        velocities_y[i] = -0.1;
        forces_x[i] = 0.0;
        forces_y[i] = 0.0;
    }

    std::cout << "Setting up pixels\n";

    // Initialize pixels
    std::vector<std::vector<int>> pixels(n_particles, std::vector<int>(2));

    // Create window
    std::cout << "Creating Window\n";
    MyWindow* pWindow = new MyWindow(width, height, offset_w, offset_h);

    std::cout << "Starting simulation\n";

    bool running = true;
    while (running) {

        unsigned long long t0 = get_time_us();

        if (!pWindow->ProcessMessages()) {
            std::cout << "Closing Window\n";
            running = false;
        }

        unsigned long long t1 = get_time_us();

        // Compute forces
        compute_forces(n_particles, positions_x, positions_y, forces_x, forces_y, d_positions_x, d_positions_y, d_forces_x, d_forces_y, extends, epsilon, cuda);

        unsigned long long t2 = get_time_us();

        // Apply forces
        #pragma omp parallel for
        for (int i = 0; i < n_particles; i++) {
            velocities_x[i] -= g_const * forces_x[i];
            velocities_y[i] -= g_const * forces_y[i];
        }

        unsigned long long t3 = get_time_us();

        // Update positions
        #pragma omp parallel for
        for (int i = 0; i < n_particles; i++) {
            positions_x[i] = positions_x[i] + delta_t * velocities_x[i];
            positions_y[i] = positions_y[i] + delta_t * velocities_y[i];
            if (periodic) {
                positions_x[i] = std::fmod(positions_x[i], 1.0);
                positions_y[i] = std::fmod(positions_y[i], 1.0);
                if (positions_x[i] < 0) {
                    positions_x[i] += 1 - (double)(int)positions_x[i];
                }
                if (positions_y[i] < 0) {
                    positions_y[i] += 1 - (double)(int)positions_y[i];
                }
            }
            pixels[i][0] = (int)(positions_x[i] * width);
            pixels[i][1] = (int)(positions_y[i] * height);
        }

        unsigned long long t4 = get_time_us();

        // Render
        pWindow->DrawPixels(pixels);
        //Sleep(1000);

        unsigned long long t5 = get_time_us();

        if (timings) {
            std::cout << (t1 - t0)/1000 << " "
                    << (t2 - t1)/1000 << " "
                    << (t3 - t2)/1000 << " "
                    << (t4 - t3)/1000 << " "
                    << (t5 - t4)/1000 << std::endl;

        }
    }

    delete pWindow;

    if (cuda) {
        cudaFree(d_positions_x);
        cudaFree(d_positions_y);
        cudaFree(d_forces_x);
        cudaFree(d_forces_y);
    }

    delete[] h_positions_x;
    delete[] h_positions_y;
    delete[] h_velocities_x;
    delete[] h_velocities_y;
    delete[] h_forces_x;
    delete[] h_forces_y;

    return 0;
}
