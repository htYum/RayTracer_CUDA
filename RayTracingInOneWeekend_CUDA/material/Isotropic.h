#pragma once

#include "material.h"
#include "..\texture\texture.h"

class Isotropic : public Material {
public:
	Texture* albedo;

public:
	__device__ Isotropic(){}
	__device__ Isotropic(Texture* a) : albedo(a){}
	__device__ ~Isotropic() {if(albedo != nullptr) delete albedo; }
	__device__ virtual bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState)const;
};

bool Isotropic::scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState)const {
	scattered = Ray(rec.p, randomInSphere(localRandState), rayIn.getTime());
	attenuation = albedo->value(rec.u, rec.v, rec.p);
	return true;
}