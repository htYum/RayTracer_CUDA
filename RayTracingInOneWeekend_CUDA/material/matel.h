#pragma once

#include "material.h"

class matel : public material {
public:
    vec3 albedo;
    float fuzz;

public:
    __host__ __device__ matel() {}
    __host__ __device__ matel(const vec3& a, float f = 0.0) :
        albedo(a),
        fuzz(f) {}
    __host__ __device__ ~matel() {}

    __device__ virtual bool scatter(const ray& rayIn, const hitRecord& rec, vec3& attenuation, ray& scattered, curandState* localRandState)const;
};

bool matel::scatter(const ray& rayIn, const hitRecord& rec, vec3& attenuation, ray& scattered, curandState* localRandState) const {
    vec3 reflected = reflect(normalize(rayIn.getDir()), rec.normal);
    scattered = ray(rec.p, reflected + fuzz * randomInSphere(localRandState), curand_uniform(localRandState));
    attenuation = albedo;
    return (dot(rec.normal, reflected) > 0);
}