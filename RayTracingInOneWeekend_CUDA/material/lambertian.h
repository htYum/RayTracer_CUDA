#pragma once

#include "Material.h"
#include "..\texture\texture.h"
#include "..\texture\RGBtexture.h"

class Lambertian : public Material {
public:
    Texture* albedo;

public:
    __device__ Lambertian(Texture* a) : albedo(a) {}

    __device__ virtual bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState)const;
};

bool Lambertian::scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState)const {
    Vec3 scatterDir = rec.normal + randomInSphere(localRandState);
    scattered = Ray(rec.p, scatterDir);
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    return true;
}