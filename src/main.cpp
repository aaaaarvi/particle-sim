#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <sys/time.h>
#include <vector>

#include "vector.h"

// CPU vs GPU implementation
#ifdef USE_CUDA
#include "compute_forces_gpu.cuh"
#else
#include "compute_forces_cpu.h"
#endif

// Include the appropriate window header based on the operating system
#ifdef _WIN32
#include "window_win.h"
#else
#include "window_linux.h"
#endif

struct timeval tv;
unsigned long long get_time_us() {
    gettimeofday(&tv, NULL);
    return (unsigned long long)(tv.tv_sec) * 1000000 +
           (unsigned long long)(tv.tv_usec);
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

    double positions_x[n_particles];
    double positions_y[n_particles];
    double velocities_x[n_particles];
    double velocities_y[n_particles];
    double forces_x[n_particles];
    double forces_y[n_particles];

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

    // Initialize pixels
    std::vector<std::vector<int>> pixels(n_particles, std::vector<int>(2));

    // Create window
    std::cout << "Creating Window\n";
    MyWindow* pWindow = new MyWindow(width, height, offset_w, offset_h);

    bool running = true;
    while (running) {

        unsigned long long t0 = get_time_us();

        if (!pWindow->ProcessMessages()) {
            std::cout << "Closing Window\n";
            running = false;
        }

        unsigned long long t1 = get_time_us();

        // Compute forces
        compute_forces(n_particles, positions_x, positions_y, forces_x, forces_y, extends, epsilon);

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

    return 0;
}
