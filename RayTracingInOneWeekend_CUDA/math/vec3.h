#pragma once

#include "..\global.h"

class Vec3 {
public:
    float val[3];

public:
    __host__ __device__ Vec3() :val{ 0,0,0 } {}
    __host__ __device__ Vec3(float v1, float v2, float v3) : val{ v1, v2, v3 } {}
    __host__ __device__ Vec3(const Vec3& rhs) : val{ rhs.val[0], rhs.val[1], rhs.val[2] } {}

    __device__ float getX() const { return val[0]; }
    __device__ float getY() const { return val[1]; }
    __device__ float getZ() const { return val[2]; }

    __host__ __device__ float lengthSquared() {
        return val[0] * val[0] + val[1] * val[1] + val[2] * val[2];
    }

    __host__ __device__ float length() {
        return sqrt(lengthSquared());
    }

    __host__ __device__ Vec3 operator-() const { return Vec3(-val[0], -val[1], -val[2]); }
    __host__ __device__ float operator[](int i) const { return val[i]; }
    __host__ __device__ float& operator[](int i) { return val[i]; }
    __host__ __device__ Vec3& operator+=(const Vec3& rhs) {
        val[0] += rhs.val[0];
        val[1] += rhs.val[1];
        val[2] += rhs.val[2];
        return *this;
    }
    __host__ __device__ Vec3& operator*=(const float& rhs) {
        val[0] *= rhs;
        val[1] *= rhs;
        val[2] *= rhs;
        return *this;
    }
    __host__ __device__ friend Vec3 operator+(const Vec3& l, const Vec3& r) {
        return Vec3(l.val[0] + r.val[0], l.val[1] + r.val[1], l.val[2] + r.val[2]);
    }
    __host__ __device__ friend Vec3 operator-(const Vec3& l, const Vec3& r) {
        return Vec3(l.val[0] - r.val[0], l.val[1] - r.val[1], l.val[2] - r.val[2]);
    }
    __host__ __device__ friend Vec3 operator*(const Vec3& l, const float& r) {
        return Vec3(l.val[0] * r, l.val[1] * r, l.val[2] * r);
    }
    __host__ __device__ friend Vec3 operator*(const float& l, const Vec3& r) {
        return r * l;
    }
    __host__ __device__ friend Vec3 operator*(const Vec3& l, const Vec3& r) {
        return Vec3(l.val[0] * r.val[0], l.val[1] * r.val[1], l.val[2] * r.val[2]);
    }
    __host__ __device__ friend Vec3 operator/(const Vec3& l, const float& r) {
        return l * (1 / r);
    }
    __host__ __device__ friend float dot(const Vec3& l, const Vec3& r) {
        return l.val[0] * r.val[0] + l.val[1] * r.val[1] + l.val[2] * r.val[2];
    }
    __host__ __device__ friend Vec3 cross(const Vec3& l, const Vec3& r) {
        return Vec3(l.val[1] * r.val[2] - l.val[2] * r.val[1],
            l.val[2] * r.val[0] - l.val[0] * r.val[2],
            l.val[0] * r.val[1] - l.val[1] * r.val[0]);
    }
    __host__ __device__ friend Vec3 normalize(Vec3 vec) {
        return vec / vec.length();
    }

    __device__ inline static Vec3 random(curandState* localRandState) {
        return Vec3(curand_uniform(localRandState),
                    curand_uniform(localRandState), 
                    curand_uniform(localRandState));
    }
    __device__ inline static Vec3 random(float min, float max, curandState* localRandState) {
        return Vec3(curand_uniform(localRandState) * (max - min) + min,
                    curand_uniform(localRandState) * (max - min) + min,
                    curand_uniform(localRandState) * (max - min) + min);
    }
};

__device__ Vec3 randomInSphere(curandState* localRandState) {
    auto a = curand_uniform(localRandState) * 2.0 * Pi;
    auto z = curand_uniform(localRandState) * 2.0 - 1.0;
    auto r = sqrt(1 - z * z);
    return Vec3(r * cos(a), r * sin(a), z);
}

__device__ Vec3 randomInDisk(curandState* localRandState) {
    while (true) {
        auto p = 2.0 * Vec3(curand_uniform(localRandState), curand_uniform(localRandState), 0) - Vec3(1.0, 1.0, 0);
        if (dot(p, p) >= 1)continue;
        return p;
    }
}