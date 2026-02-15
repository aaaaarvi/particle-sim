#pragma once

#include <vector>
#include <cstring>

namespace quad_tree_flat {

// Flattened quadtree node: uses integer indices instead of pointers
struct node_t {
    int nw;  // index of north_west child (-1 if none)
    int ne;  // index of north_east child (-1 if none)
    int sw;  // index of south_west child (-1 if none)
    int se;  // index of south_east child (-1 if none)
    float origo_x;
    float origo_y;
    float width;
    float center_of_mass_x;
    float center_of_mass_y;
    float mass;
    int num_particles;
};

// Container for a flattened quadtree on host
struct tree_t {
    std::vector<node_t> nodes;
    int root_idx;
};

// Initialize a flattened quadtree
void init(tree_t* tree, float origo_x, float origo_y, float width);

// Insert a particle into the flattened quadtree
void insert(tree_t* tree, float x, float y, float mass);

// Get tree size
int tree_size(const tree_t* tree);

// Free tree (just clears the vector)
void free_tree(tree_t* tree);

} // namespace quad_tree_flat
