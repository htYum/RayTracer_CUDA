#pragma once

#include "Material.h"

class Matel : public Material {
public:
    Vec3 albedo;
    float fuzz;

public:
    __device__ Matel() {}
    __device__ Matel(const Vec3& a, float f = 0.0) :
        albedo(a),
        fuzz(f) {}
    __device__ ~Matel() {}

    __device__ virtual bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState)const;
};

bool Matel::scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState) const {
    Vec3 reflected = reflect(normalize(rayIn.getDir()), rec.normal);
    scattered = Ray(rec.p, reflected + fuzz * randomInSphere(localRandState), rayIn.time);
    attenuation = albedo;
    return (dot(rec.normal, reflected) > 0);
}