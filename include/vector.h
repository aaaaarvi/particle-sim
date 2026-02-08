#pragma once

#include<cmath>

template<typename T>
class Vector2 {
public:
    T x;
    T y;
    Vector2<T>(T X = 0, T Y = 0) : x(X), y(Y) {}
    ~Vector2<T>() {}
    Vector2<T> operator+(const Vector2<T>& other) const {
        return Vector2<T>(x + other.x, y + other.y);
    }
    Vector2<T> operator-(const Vector2<T>& other) const {
        return Vector2<T>(x - other.x, y - other.y);
    }
    Vector2<T> operator*(T scalar) const {
        return Vector2<T>(x * scalar, y * scalar);
    }
    Vector2<T> operator/(T scalar) const {
        return Vector2<T>(x / scalar, y / scalar);
    }
    Vector2<T>& operator+=(const Vector2<T>& other) {
        x += other.x;
        y += other.y;
        return *this;
    }
    Vector2<T>& operator-=(const Vector2<T>& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }
    Vector2<T>& operator*=(const T scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }
    Vector2<T>& operator/=(const T scalar) {
        x /= scalar;
        y /= scalar;
        return *this;
    }
    bool operator==(const Vector2<T>& other) {
        return (x == other.x) && (y == other.y);
    }
    T norm();
    T norm_squared() {
        return x * x + y * y;
    }
};

template <>
inline float Vector2<float>::norm() {
    return sqrtf(x * x + y * y);
}

template <>
inline double Vector2<double>::norm() {
    return sqrt(x * x + y * y);
}

template <>
inline long double Vector2<long double>::norm() {
    return sqrtl(x * x + y * y);
}

typedef Vector2<float> Vector2_f;
typedef Vector2<double> Vector2_d;
typedef Vector2<long double> Vector2_l;
