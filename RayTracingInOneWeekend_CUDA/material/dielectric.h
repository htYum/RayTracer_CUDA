#pragma once

#include "Material.h"

class Dielectric : public Material {
public:
    float refIdx;

public:
    __device__ Dielectric(float ri) : refIdx(ri) {}

    __device__ virtual bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState)const;
};

bool Dielectric::scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState)const {
    attenuation = Vec3(1.0, 1.0, 1.0);
    float refractRate;
    refractRate = (rec.frontFace) ? 1.0 / refIdx : refIdx;
    Vec3 dir = normalize(rayIn.getDir());
    float cosa = ffmin(dot(-dir, rec.normal), 1.0);
    float sina = sqrt(1.0 - cosa * cosa);
    float reflectProb = schlick(cosa, refractRate);
    if (sina * refractRate > 1.0 || curand_uniform(localRandState) < reflectProb) {
        Vec3 reflected = reflect(dir, rec.normal);
        scattered = Ray(rec.p, reflected);
        return true;
    }
    Vec3 refracted = refract(dir, rec.normal, refractRate);
    scattered = Ray(rec.p, refracted, rayIn.time);
    return true;
}