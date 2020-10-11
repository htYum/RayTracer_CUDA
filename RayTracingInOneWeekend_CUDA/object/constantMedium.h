#pragma once

#include "object.h"
#include "..\material\Isotropic.h"

class ConstantMedium : public Object {
public:
	Object* boundary;
	Material* phaseFunction;
	float negInvDensity;

public:
	__device__ ConstantMedium(){}
	__device__ ConstantMedium(Object* obj, float d, Texture* a) :boundary(obj), negInvDensity(-1.0 / d) { phaseFunction = new Isotropic(a); }
	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState)const;
	__device__ virtual bool boundingBox(AABB& box, float t0, float t1)const;
};

bool ConstantMedium::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState)const {
	HitRecord rec0, rec1;

	if (!boundary->hit(r, -INFINITY, INFINITY, rec0, localRandState)) {
		return false;
	}
	if (!boundary->hit(r, rec0.t + 0.0001, INFINITY, rec1, localRandState)) {
		return false;
	}
	if (rec0.t < tMin)rec0.t = tMin;
	if (rec1.t > tMax)rec1.t = tMax;
	if (rec1.t <= rec0.t)return false;
	if (rec0.t < 0)rec0.t = 0;
	const auto rayLen = r.getDir().length();
	const auto distanceInsideBoundary = (rec1.t - rec0.t) * rayLen;
	const auto hitDistance = negInvDensity * log(curand_uniform(localRandState));

	if (hitDistance > distanceInsideBoundary) {
		return false;
	}

	rec.t = rec0.t + hitDistance / rayLen;
	rec.p = r.at(rec.t);
	rec.normal = Vec3(1.0, 0, 0);
	rec.frontFace = true;
	rec.mat = phaseFunction;

	return true;
}

bool ConstantMedium::boundingBox(AABB& box, float t0, float t1)const {
	return boundary->boundingBox(box, t0, t1);
}