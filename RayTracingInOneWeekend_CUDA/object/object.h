#pragma once

#include "..\ray.h"
#include "..\material\material.h"

struct hitRecord {
    vec3 p;
    vec3 normal;
    material* mat;
    float t;
    bool frontFace;

    __device__ inline void setFaceNormal(const ray& r, const vec3& outwardNormal) {
        frontFace = dot(r.getDir(), outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

class object {
public:
    __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const = 0;
};