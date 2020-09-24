#pragma once

#include <cmath>
#include <cstdlib>
#include <limits>

__device__ const double Infinity = std::numeric_limits<double>::infinity();
__device__ const double Pi = 3.1415926535897932385;

__device__ inline float radians(float angle) {
    return (angle * Pi) / 180;
}

__device__ inline double ffmin(double a, double b) {
    return a <= b ? a : b;
}

__device__ inline float ffmax(double a, double b) {
    return a >= b ? a : b;
}

__device__ inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__device__ inline float schlick(float cosine, float refIdx) {
    float r0 = (1 - refIdx) / (1 + refIdx);
    r0 *= r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}