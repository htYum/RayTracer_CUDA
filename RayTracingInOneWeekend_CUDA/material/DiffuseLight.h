#pragma once

#include "material.h"
#include "..\texture\texture.h"

class DiffuseLight : public Material {
public:
	Texture* emit;

public:
	__device__ DiffuseLight(){}
	__device__ DiffuseLight(Texture* t) : emit(t){}
	__device__ ~DiffuseLight() {if(emit != nullptr) delete emit; }
	__device__ virtual bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState)const;
	__device__ virtual Vec3 emitted(float u, float v, const Vec3& p) const;
};

bool DiffuseLight::scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState)const {
	return false;
}

Vec3 DiffuseLight::emitted(float u, float v, const Vec3& p)const {
	return emit->value(u, v, p);
}