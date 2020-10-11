#pragma once

#include "..\Ray.h"
#include "..\Material\Material.h"
#include "AABB.h"

struct HitRecord {
    Vec3 p;
    Vec3 normal;
    Material* mat;
    float t;
    float u;
    float v;
    bool frontFace;

    __device__ inline void setFaceNormal(const Ray& r, const Vec3& outwardNormal) {
        frontFace = dot(r.getDir(), outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

class Object {
public:
    __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState) const = 0;
    __device__ virtual bool boundingBox(AABB& box, float t0, float t1) const = 0;
};