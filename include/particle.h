#pragma once

#include "vector.h"

template<typename T>
class Particle {
public:
    Vector2<T> pos;
    Vector2<T> vel;
    Vector2<T> acc;
    T mass;
    Particle<T>(Vector2<T> Pos, Vector2<T> Vel, T Mass)
        : pos(Pos), vel(Vel), acc(0, 0), mass(Mass) {}
    ~Particle<T>() {}
    void update(float dt) {
        vel += acc * dt;
        pos += vel * dt;
    }
};
