#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <sys/time.h>
#include <vector>

#include "quad_tree.h"

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

long double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long double)(ts.tv_sec) +
           (long double)(ts.tv_nsec) / (long double)1000000000.0L;
}

int main()
{
    srand(static_cast<unsigned>(time(0)));

    // TODO: Move constants to config file
    // TODO: Investigate cuda memory transfer overhead
    // TODO: Implement true periodic boundary conditions
    // TODO: Preserve conserved quantities (momentum, energy)
    // TODO: Add physical particle interactions (collisions, mergers, slowdown)
    // TODO: Unit tests
    // TODO: Zooming and panning in the window
    // TODO: Optimize quadtree memory allocations

    // Parameters
    const int n_particles = 100000; // 1000
    const double g_const = 0.1 / (double)n_particles; // 100
    const double epsilon = 0.01; // 1e-3
    const double delta_t = 0.01; // 1e-5
    const int width = 720;
    const int height = 720;
    const int offset_w = 100;
    const int offset_h = 100;
    const int extends = 0;
    const bool periodic = false;
    const bool verbose = false;

    // Timing statistics
    long double delta_t_tot_1 = 0.0;
    long double delta_t_tot_2 = 0.0;
    long double delta_t_tot_3 = 0.0;
    long double delta_t_tot_4 = 0.0;
    long double delta_t_tot_5 = 0.0;
    int num_deltas = 0;

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
        velocities_y[i] = 0.1; // 5.0
    }
    for (int i = n_particles / 2; i < n_particles; i++) {
        positions_x[i] = 0.75 + 0.01*distribution(generator);
        positions_y[i] = 0.5 + 0.01*distribution(generator);
        velocities_x[i] = 0.0;
        velocities_y[i] = -0.1; // -5.0
    }

    // Initialize uniform distribution
    /** /
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    for (int i = 0; i < n_particles; i++) {
        positions_x[i] = uniform_dist(generator);
        positions_y[i] = uniform_dist(generator);
        velocities_x[i] = 0.0;
        velocities_y[i] = 0.0;
    }
    //*/

    // Initialize pixels
    std::vector<std::vector<int>> pixels(n_particles, std::vector<int>(2));

    // Print initial quadtree
    /** /
    unsigned long long time0 = get_time_us();
    quad_tree::node_t* root;
    quad_tree::init(&root);
    for (int i = 0; i < n_particles; i++) {
        quad_tree::insert(root, positions_x[i], positions_y[i], 1.0, i);
    }
    //quad_tree::print_tree(root);
    quad_tree::free_tree(root);
    unsigned long long time1 = get_time_us();
    std::cout << "Quadtree time: " << (time1 - time0) / 1000 << " ms\n";
    //*/

    // Create window
    std::cout << "Creating Window\n";
    MyWindow* pWindow = new MyWindow(width, height, offset_w, offset_h);

    bool running = true;
    while (running) {

        long double t0 = get_time();

        if (!pWindow->ProcessMessages()) {
            std::cout << "Closing Window\n";
            running = false;
        }

        long double t1 = get_time();

        // Compute forces
        compute_forces(n_particles, positions_x, positions_y, forces_x, forces_y, extends, epsilon);

        long double t2 = get_time();

        // Apply forces
        #pragma omp parallel for
        for (int i = 0; i < n_particles; i++) {
            velocities_x[i] += delta_t * g_const * forces_x[i];
            velocities_y[i] += delta_t * g_const * forces_y[i];
        }

        long double t3 = get_time();

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

        long double t4 = get_time();

        // Render
        pWindow->DrawPixels(pixels);
        //Sleep(1000);

        long double t5 = get_time();

        // Timings
        long double diff1 = (t1 - t0);
        long double diff2 = (t2 - t1);
        long double diff3 = (t3 - t2);
        long double diff4 = (t4 - t3);
        long double diff5 = (t5 - t4);
        delta_t_tot_1 = (diff1 + num_deltas * delta_t_tot_1) / (num_deltas + 1);
        delta_t_tot_2 = (diff2 + num_deltas * delta_t_tot_2) / (num_deltas + 1);
        delta_t_tot_3 = (diff3 + num_deltas * delta_t_tot_3) / (num_deltas + 1);
        delta_t_tot_4 = (diff4 + num_deltas * delta_t_tot_4) / (num_deltas + 1);
        delta_t_tot_5 = (diff5 + num_deltas * delta_t_tot_5) / (num_deltas + 1);
        num_deltas++;
        if (verbose) {
            std::cout << (int)(diff1 * 1000) << " "
                      << (int)(diff2 * 1000) << " "
                      << (int)(diff3 * 1000) << " "
                      << (int)(diff4 * 1000) << " "
                      << (int)(diff5 * 1000) << "    "
                      << (int)(delta_t_tot_1 * 1000) << " "
                      << (int)(delta_t_tot_2 * 1000) << " "
                      << (int)(delta_t_tot_3 * 1000) << " "
                      << (int)(delta_t_tot_4 * 1000) << " "
                      << (int)(delta_t_tot_5 * 1000) << " " << std::endl;
        }
    }

    std::cout << "Timing statistics" << std::endl;
    std::cout << "  Window messages:  " << (int)(delta_t_tot_1 * 1000) << " ms" << std::endl;
    std::cout << "  Compute forces:   " << (int)(delta_t_tot_2 * 1000) << " ms" << std::endl;
    std::cout << "  Apply forces:     " << (int)(delta_t_tot_3 * 1000) << " ms" << std::endl;
    std::cout << "  Update positions: " << (int)(delta_t_tot_4 * 1000) << " ms" << std::endl;
    std::cout << "  Render:           " << (int)(delta_t_tot_5 * 1000) << " ms" << std::endl;

    delete pWindow;

    return 0;
}
