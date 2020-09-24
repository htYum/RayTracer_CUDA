#pragma once

#include "material.h"

class dielectric : public material {
public:
    float refIdx;

public:
    __host__ __device__ dielectric(float ri) : refIdx(ri) {}

    __device__ virtual bool scatter(const ray& rayIn, const hitRecord& rec, vec3& attenuation, ray& scattered, curandState* localRandState)const;
};

bool dielectric::scatter(const ray& rayIn, const hitRecord& rec, vec3& attenuation, ray& scattered, curandState* localRandState)const {
    attenuation = vec3(1.0, 1.0, 1.0);
    float refractRate;
    refractRate = (rec.frontFace) ? 1.0 / refIdx : refIdx;
    vec3 dir = normalize(rayIn.getDir());
    float cosa = ffmin(dot(-dir, rec.normal), 1.0);
    float sina = sqrt(1.0 - cosa * cosa);
    float reflectProb = schlick(cosa, refractRate);
    if (sina * refractRate > 1.0 || curand_uniform(localRandState) < reflectProb) {
        vec3 reflected = reflect(dir, rec.normal);
        scattered = ray(rec.p, reflected);
        return true;
    }
    vec3 refracted = refract(dir, rec.normal, refractRate);
    scattered = ray(rec.p, refracted);
    return true;
}