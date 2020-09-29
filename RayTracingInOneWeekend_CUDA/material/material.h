#pragma once

#include "..\global.h"
#include "..\Ray.h"
#include "..\Object\Object.h"

struct HitRecord;

class Material {
public:
    __device__ virtual bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState) const = 0;
};

__device__ Vec3 reflect(const Vec3& vec, const Vec3& normal) {
    return vec - 2 * dot(vec, normal) * normal;
}

// refractRate n/n'
__device__ Vec3 refract(const Vec3& vec, const Vec3& normal, float refractRate) {
    auto cosa = dot(-vec, normal);
    Vec3 rParallel = refractRate * (vec + cosa * normal);
    Vec3 rPerp = -sqrt(1 - dot(rParallel, rParallel)) * normal;
    return rParallel + rPerp;
}