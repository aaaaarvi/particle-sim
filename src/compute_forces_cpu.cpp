#include <cmath>
#include "compute_forces_cpu.h"
#include "quad_tree.h"

void compute_forces(
    std::vector<Particle<float>>* particles,
    const float epsilon,
    const int extends) {

    // Direct force computation, O(N^2)
    // (Assumes all particles have mass 1.)
    /** /
    #pragma omp parallel for
    for (auto& p : *particles) {
        p.acc.x = 0.0f;
        p.acc.y = 0.0f;
        for (const auto& other : *particles) {
            for (int xx = -extends; xx <= extends; xx++) {
                for (int yy = -extends; yy <= extends; yy++) {
                    float dx = other.pos.x - p.pos.x + (float)xx;
                    float dy = other.pos.y - p.pos.y + (float)yy;
                    float dist = sqrtf(dx * dx + dy * dy) + epsilon;
                    p.acc.x += dx / (dist * dist * dist);
                    p.acc.y += dy / (dist * dist * dist);
                }
            }
        }
    }
    //*/

    // Barnes-Hut algorithm v1, O(N log N)
    /**/
    float theta_max = 0.5f;

    // Compute min/max x/y coordinates
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    for (const auto& p : *particles) {
        if (p.pos.x - (float)extends < min_x) min_x = p.pos.x - (float)extends;
        if (p.pos.x + (float)extends > max_x) max_x = p.pos.x + (float)extends;
        if (p.pos.y - (float)extends < min_y) min_y = p.pos.y - (float)extends;
        if (p.pos.y + (float)extends > max_y) max_y = p.pos.y + (float)extends;
    }
    float origo_x = (min_x + max_x) * 0.5f;
    float origo_y = (min_y + max_y) * 0.5f;
    float width = std::max(max_x - min_x, max_y - min_y);

    // Initialize quadtree
    quad_tree::node_t* root;
    quad_tree::init(&root, origo_x, origo_y, width);
    for (const auto& p : *particles) {
        for (int xx = -extends; xx <= extends; xx++) {
            for (int yy = -extends; yy <= extends; yy++) {
                quad_tree::insert(root, p.pos.x + (float)xx, p.pos.y + (float)yy, p.mass);
            }
        }
    }

    // Compute forces
    #pragma omp parallel for
    for (auto& p : *particles) {
        p.acc.x = 0.0f;
        p.acc.y = 0.0f;
        quad_tree::compute_force(
            root, &p.acc.x, &p.acc.y, p.pos.x, p.pos.y, p.mass, theta_max, epsilon);
    }

    // Free quadtree
    quad_tree::free_tree(root);
    //*/

    // Barnes-Hut algorithm v2, O(N log N)
    // (Doesn't support 'extends'.)
    /** /
    float theta_max = 0.5f;

    quad_tree::Quad quad = quad_tree::Quad(*particles);
    quad_tree::Quadtree quadtree = quad_tree::Quadtree(theta_max, epsilon);
    quadtree.clear(quad);

    for (const auto& p : *particles) {
        quadtree.insert(p.pos, p.mass);
    }

    quadtree.propagate();

    #pragma omp parallel for
    for (auto& p : *particles) {
        p.acc = quadtree.acc(p.pos) * p.mass;
    }
    //*/
}
