#include "vector.h"

Vector2D::Vector2D()
    : x(static_cast<double>(rand()) / static_cast<double>(RAND_MAX)),
      y(static_cast<double>(rand()) / static_cast<double>(RAND_MAX))
{
}

Vector2D::Vector2D(double x, double y)
    : x(x), y(y)
{
}

Vector2D::~Vector2D()
{
}
