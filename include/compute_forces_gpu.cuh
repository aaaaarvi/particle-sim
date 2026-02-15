#pragma once

#include <vector>
#include "particle.h"

void compute_forces(
    std::vector<Particle<float>>* particles,
    const float epsilon,
    const int extends);
