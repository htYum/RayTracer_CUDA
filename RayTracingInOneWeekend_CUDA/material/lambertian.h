#pragma once

#include "material.h"

class lambertian : public material {
public:
    vec3 albedo;

public:
    __device__ lambertian(const vec3& a) : albedo(a) {}

    __device__ virtual bool scatter(const ray& rayIn, const hitRecord& rec, vec3& attenuation, ray& scattered, curandState* localRandState)const;
};

bool lambertian::scatter(const ray& rayIn, const hitRecord& rec, vec3& attenuation, ray& scattered, curandState* localRandState)const {
    vec3 scatterDir = rec.normal + randomInSphere(localRandState);
    scattered = ray(rec.p, scatterDir);
    attenuation = albedo;
    return true;
}