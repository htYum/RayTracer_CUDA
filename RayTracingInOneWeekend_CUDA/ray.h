#pragma once

#include "Math/Vec3.h"

class Ray {
public:
    __device__ Ray() = default;
    __device__ Ray(const Vec3& position, const Vec3& direction, float t = 0) :
        pos(position),
        dir(normalize(direction)),
        time(t){}

    __device__ Vec3 getPos() const { return pos; }
    __device__ Vec3 getDir() const { return dir; }
    __device__ float getTime() const { return time; }

    __device__ Vec3 at(float t) const {
        return pos + t * dir;
    }

public:
    Vec3 pos;
    Vec3 dir;
    float time;
};