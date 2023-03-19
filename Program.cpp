#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <sys/time.h>
#include <vector>

#include "Vector.h"
#include "Window.h"

struct timeval tv;
unsigned long long get_time_us() {
    gettimeofday(&tv, NULL);
    return (unsigned long long)(tv.tv_sec) * 1000000 +
           (unsigned long long)(tv.tv_usec);
}

int main()
{
    srand(static_cast<unsigned>(time(0)));

    const int n_particles = 4000;
    const double g_const = 0.001 / (double)n_particles;
    const double epsilon = 0.01;
    const double delta_t = 0.01;
    const int width = 720;
    const int height = 720;
    const int offset_w = 100;
    const int offset_h = 100;
    const int extends = 0;
    const bool periodic = false;
    const bool timings = false;

    // Create window
    std::cout << "Creating Window\n";
    Window* pWindow = new Window(width, height, offset_w, offset_h);

    // Initialize particles (uniform random positions)
    std::vector<Vector2D> positions(n_particles);
    std::vector<Vector2D> velocities(n_particles, Vector2D(0, 0));
    std::vector<Vector2D> forces(n_particles, Vector2D(0, 0));
    std::vector<std::vector<int>> pixels(n_particles, std::vector<int>(2));

    // "Two galaxies"
    /* */
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < n_particles / 2; i++) {
        positions[i].x = 0.25 + 0.01*distribution(generator);
        positions[i].y = 0.5 + 0.01*distribution(generator);
        velocities[i].y = 0.1;
    }
    for (int i = n_particles / 2; i < n_particles; i++) {
        positions[i].x = 0.75 + 0.01*distribution(generator);
        positions[i].y = 0.5 + 0.01*distribution(generator);
        velocities[i].y = -0.1;
    }
    // */

    bool running = true;
    while (running) {

        unsigned long long t0 = get_time_us();

        if (!pWindow->ProcessMessages()) {
            std::cout << "Closing Window\n";
            running = false;
        }

        unsigned long long t1 = get_time_us();

        // Compute forces
        #pragma omp parallel for
        for (int i = 0; i < n_particles; i++) {
            forces[i].x = 0;
            forces[i].y = 0;
            for (int j = 0; j < n_particles; j++) {
                for (double xx = -extends; xx <= extends; xx++) {
                    for (double yy = -extends; yy <= extends; yy++) {
                        double dx = positions[i].x - positions[j].x + xx;
                        double dy = positions[i].y - positions[j].y + yy;
                        double dist = sqrt(dx*dx + dy*dy) + epsilon;
                        forces[i].x += dx / (dist * dist * dist);
                        forces[i].y += dy / (dist * dist * dist);
                    }
                }
            }
        }

        unsigned long long t2 = get_time_us();

        // Apply forces
        #pragma omp parallel for
        for (int i = 0; i < n_particles; i++) {
            velocities[i].x -= g_const * forces[i].x;
            velocities[i].y -= g_const * forces[i].y;
        }

        unsigned long long t3 = get_time_us();

        // Update positions
        #pragma omp parallel for
        for (int i = 0; i < n_particles; i++) {
            positions[i].x = positions[i].x + delta_t * velocities[i].x;
            positions[i].y = positions[i].y + delta_t * velocities[i].y;
            if (periodic) {
                positions[i].x = std::fmod(positions[i].x, 1.0);
                positions[i].y = std::fmod(positions[i].y, 1.0);
                if (positions[i].x < 0) {
                    positions[i].x += 1 - (double)(int)positions[i].x;
                }
                if (positions[i].y < 0) {
                    positions[i].y += 1 - (double)(int)positions[i].y;
                }
            }
            pixels[i][0] = (int)(positions[i].x * width);
            pixels[i][1] = (int)(positions[i].y * height);
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
