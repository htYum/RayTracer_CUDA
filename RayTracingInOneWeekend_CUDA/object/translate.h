#pragma once

#include "box.h"

class Translate : public Object {
public:
	Object* obj;
	Vec3 offset;

public:
	__device__ Translate(){}
	__device__ Translate(Object* o, const Vec3& move) : obj(o), offset(move){}

	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState)const;
	__device__ virtual bool boundingBox(AABB& box, float t0, float t1)const;
};

bool Translate::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState)const {
	Ray movedRay(r.getPos() - offset, r.getDir(), r.getTime());
	if (!obj->hit(movedRay, tMin, tMax, rec, localRandState)) {
		return false;
	}
	rec.p += offset;
	rec.setFaceNormal(movedRay, rec.normal);
	return true;
}

bool Translate::boundingBox(AABB& box, float t0, float t1)const {
	if (!obj->boundingBox(box, t0, t1)) {
		return false;
	}
	box = AABB(box.minp + offset, box.maxp + offset);
	return true;
}