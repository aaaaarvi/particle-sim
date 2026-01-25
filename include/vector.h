#pragma once

template<typename T>
class Vector2 {
public:
    T x;
    T y;
    Vector2<T>(T X = 0, T Y = 0) : x(X), y(Y) {}
    ~Vector2() {}
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
    Vector2<T>& operator*=(const Vector2<T>& other) {
        x *= other.x;
        y *= other.y;
        return *this;
    }
    Vector2<T>& operator/=(const Vector2<T>& other) {
        x /= other.x;
        y /= other.y;
        return *this;
    }
};

typedef Vector2<float> Vector2f;
typedef Vector2<double> Vector2d;
typedef Vector2<int> Vector2i;
typedef Vector2<unsigned int> Vector2u;
