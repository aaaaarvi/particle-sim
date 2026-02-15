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
    // TODO: Change DrawPixels to DrawParticles, which accepts a vector of particles
    // TODO: Draw in grayscale/colormap -> higher particle density gives higher brightness

    // Parameters
    const int n_particles = 100000; // 1000
    const float g_const = 0.1f / n_particles; // 100
    const float epsilon = 0.01f; // 1e-3
    const float delta_t = 0.01f; // 1e-5
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

    // Initialize "two galaxies"
    /**/
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    std::vector<Particle<float>> particles;
    for (int i = 0; i < n_particles / 2; i++) {
        particles.push_back(Particle<float>(
            Vector2<float>(
                0.25f + 0.01f * distribution(generator),
                0.5f + 0.01f * distribution(generator)),
            Vector2<float>(0.0f, 0.1f)));
    }
    for (int i = n_particles / 2; i < n_particles; i++) {
        particles.push_back(Particle<float>(
            Vector2<float>(
                0.75f + 0.01f * distribution(generator),
                0.5f + 0.01f * distribution(generator)),
            Vector2<float>(0.0f, -0.1f)));
    }
    //*/

    // Initialize uniform distribution
    /** /
    std::default_random_engine generator;
    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
    std::vector<Particle<float>> particles;
    for (int i = 0; i < n_particles; i++) {
        particles.push_back(Particle<float>(
            Vector2<float>(uniform_dist(generator), uniform_dist(generator)),
            Vector2<float>(0.0f, 0.0f)));
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
        compute_forces(&particles, epsilon, extends);

        long double t2 = get_time();

        // Apply forces
        #pragma omp parallel for
        for (auto& p : particles) {
            p.vel += p.acc * g_const * delta_t;
        }

        long double t3 = get_time();

        // Update positions
        #pragma omp parallel for
        for (auto& p : particles) {
            p.pos += p.vel * delta_t;
            if (periodic) {
                p.pos.x = std::fmod(p.pos.x, 1.0f);
                p.pos.y = std::fmod(p.pos.y, 1.0f);
                if (p.pos.x < 0) p.pos.x += 1 - (float)(int)p.pos.x;
                if (p.pos.y < 0) p.pos.y += 1 - (float)(int)p.pos.y;
            }
        }

        // Update pixels
        for (int i = 0; i < n_particles; i++) {
            pixels[i][0] = (int)(particles[i].pos.x * width);
            pixels[i][1] = (int)(particles[i].pos.y * height);
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
