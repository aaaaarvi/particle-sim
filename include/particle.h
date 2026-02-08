#pragma once

#include "vector.h"

template<typename T>
class Particle {
public:
    Vector2<T> pos;
    Vector2<T> vel;
    Vector2<T> acc;
    T mass;
    Particle<T>()
        : pos(0, 0), vel(0, 0), acc(0, 0), mass(1) {}
    Particle<T>(Vector2<T> Pos, Vector2<T> Vel, T Mass = 1)
        : pos(Pos), vel(Vel), acc(0, 0), mass(Mass) {}
    ~Particle<T>() {}
    void update(float dt) {
        vel += acc * dt;
        pos += vel * dt;
    }
};

typedef Particle<float> Particle_f;
typedef Particle<double> Particle_d;
typedef Particle<long double> Particle_l;
