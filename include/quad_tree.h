#pragma once

namespace quad_tree {

typedef struct quad_node {
    struct quad_node* north_west;
    struct quad_node* north_east;
    struct quad_node* south_west;
    struct quad_node* south_east;
    double origo_x;
    double origo_y;
    double width;
    double center_of_mass_x;
    double center_of_mass_y;
    double mass;
    int id;
    int num_particles;
} node_t;

void init(node_t** root, double origo_x = 0.5, double origo_y = 0.5, double width = 100.0);
void insert(node_t* node, double x, double y, double mass, int id);
void compute_force(node_t* node, double* force_x, double* force_y, int id, double x, double y, double mass, double theta_max, double epsilon);
void print_tree(node_t* node, int depth = 0, bool is_root = true);
void free_tree(node_t* node);

} // namespace quadtree
