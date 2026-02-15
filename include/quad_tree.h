#pragma once

#include <vector>

#include "vector.h"
#include "particle.h"

namespace quad_tree {

typedef struct quad_node {
    struct quad_node* north_west;
    struct quad_node* north_east;
    struct quad_node* south_west;
    struct quad_node* south_east;
    float origo_x;
    float origo_y;
    float width;
    float center_of_mass_x;
    float center_of_mass_y;
    float mass;
    int num_particles;
} node_t;

void init(node_t** root, float origo_x = 0.5f, float origo_y = 0.5f, float width = 100.0f);
void insert(node_t* node, float x, float y, float mass);
void compute_force(node_t* node, float* force_x, float* force_y, float x, float y, float mass, float theta_max, float epsilon);
void print_tree(node_t* node, int depth = 0, bool is_root = true);
int tree_size(node_t* node);
void free_tree(node_t* node);

class Quad {
public:
    Vector2<float> center;
    float size;

    Quad();
    Quad(std::vector<Particle<float>> particles);
    ~Quad() {}

    unsigned int find_quadrant(const Vector2<float>& point) const;
    Quad into_quadrant(unsigned int quadrant);
    std::array<Quad, 4> subdivide();
};

class Node {
public:
    unsigned int children;
    unsigned int next;
    Vector2<float> pos;
    float mass;
    Quad quad;

    Node(unsigned int Next, const Quad Quad)
        : children(0), next(Next), pos(0, 0), mass(0), quad(Quad) {}
    ~Node() {}

    bool is_leaf() const { return children == 0; }
    bool is_branch() const { return children != 0; }
    bool is_empty() const { return mass == 0; }
};

class Quadtree {
public:
    const unsigned int ROOT = 0;
    float theta_squared;
    float epsilon_squared;
    std::vector<Node> nodes;
    std::vector<unsigned int> parents;

    Quadtree(float theta, float epsilon)
        : theta_squared(theta * theta), epsilon_squared(epsilon * epsilon) {}
    ~Quadtree() {}

    void clear(Quad quad);
    unsigned int subdivide(unsigned int node);
    void insert(Vector2<float> pos, float mass);
    void propagate();
    Vector2<float> acc(Vector2<float> pos);
};

} // namespace quadtree
