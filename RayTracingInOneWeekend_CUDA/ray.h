#pragma once

#include "Math/vec3.h"

class ray {
public:
    __device__ ray() = default;
    __device__ ray(const vec3& position, const vec3& direction, float t = 0) :
        pos(position),
        dir(direction),
        time(t){}

    __device__ vec3 getPos() const { return pos; }
    __device__ vec3 getDir() const { return dir; }
    __device__ float getTime() const { return time; }

    __device__ vec3 at(float t) const {
        return pos + t * dir;
    }

public:
    vec3 pos;
    vec3 dir;
    float time;
};