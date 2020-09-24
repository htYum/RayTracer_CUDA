#pragma once

#include "..\global.h"
#include "..\ray.h"
#include "..\object\object.h"

struct hitRecord;

class material {
public:
    __device__ virtual bool scatter(const ray& rayIn, const hitRecord& rec, vec3& attenuation, ray& scattered, curandState* localRandState) const = 0;
};

__device__ vec3 reflect(const vec3& vec, const vec3& normal) {
    return vec - 2 * dot(vec, normal) * normal;
}

// refractRate n/n'
__device__ vec3 refract(const vec3& vec, const vec3& normal, float refractRate) {
    auto cosa = dot(-vec, normal);
    vec3 rParallel = refractRate * (vec + cosa * normal);
    vec3 rPerp = -sqrt(1 - dot(rParallel, rParallel)) * normal;
    return rParallel + rPerp;
}