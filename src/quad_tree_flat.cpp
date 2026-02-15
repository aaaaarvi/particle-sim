#include "quad_tree_flat.h"
#include <algorithm>
#include <cmath>

namespace quad_tree_flat {

void init(tree_t* tree, float origo_x, float origo_y, float width) {
    tree->nodes.clear();
    // Create root node (index 0)
    node_t root;
    root.nw = -1;
    root.ne = -1;
    root.sw = -1;
    root.se = -1;
    root.origo_x = origo_x;
    root.origo_y = origo_y;
    root.width = width;
    root.center_of_mass_x = 0.0f;
    root.center_of_mass_y = 0.0f;
    root.mass = 0.0f;
    root.num_particles = 0;
    tree->nodes.push_back(root);
    tree->root_idx = 0;
}

// Helper: determine which quadrant (0=NW, 1=NE, 2=SW, 3=SE)
static int get_quadrant(float x, float y, float origo_x, float origo_y) {
    int q = 0;
    if (x > origo_x) q |= 1;  // East
    if (y < origo_y) q |= 2;  // South
    return q;
}

// Helper: get child index from parent node and quadrant
static int& get_child(node_t* node, int quadrant) {
    int* children[4] = {&node->nw, &node->ne, &node->sw, &node->se};
    return *children[quadrant];
}

// Helper: compute new origo and width for a quadrant
static void get_quadrant_bounds(float origo_x, float origo_y, float width,
                                 int quadrant, float& new_x, float& new_y, float& new_width) {
    new_width = width * 0.5f;
    new_x = origo_x + ((quadrant & 1) ? new_width : -new_width);
    new_y = origo_y + ((quadrant & 2) ? -new_width : new_width);
}

void insert(tree_t* tree, float x, float y, float mass) {
    if (tree->nodes.empty()) return;

    int idx = tree->root_idx;
    node_t* node = &tree->nodes[idx];
    float origo_x = node->origo_x;
    float origo_y = node->origo_y;
    float width = node->width;

    // Traverse/build tree
    while (true) {
        node->mass += mass;
        node->center_of_mass_x = (node->center_of_mass_x * node->num_particles + x * mass) / (node->mass);
        node->center_of_mass_y = (node->center_of_mass_y * node->num_particles + y * mass) / (node->mass);
        node->num_particles++;

        // Determine quadrant
        int quadrant = get_quadrant(x, y, origo_x, origo_y);
        int& child_idx = get_child(node, quadrant);

        if (child_idx == -1) {
            // Create new child
            float new_x, new_y, new_width;
            get_quadrant_bounds(origo_x, origo_y, width, quadrant, new_x, new_y, new_width);

            node_t child;
            child.nw = -1;
            child.ne = -1;
            child.sw = -1;
            child.se = -1;
            child.origo_x = new_x;
            child.origo_y = new_y;
            child.width = new_width;
            child.center_of_mass_x = x;
            child.center_of_mass_y = y;
            child.mass = mass;
            child.num_particles = 1;

            child_idx = (int)tree->nodes.size();
            tree->nodes.push_back(child);
            break;
        } else {
            // Descend
            idx = child_idx;
            node = &tree->nodes[idx];
            origo_x = node->origo_x;
            origo_y = node->origo_y;
            width = node->width;
        }
    }
}

int tree_size(const tree_t* tree) {
    return (int)tree->nodes.size();
}

void free_tree(tree_t* tree) {
    tree->nodes.clear();
    tree->root_idx = -1;
}

} // namespace quad_tree_flat
