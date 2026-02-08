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
