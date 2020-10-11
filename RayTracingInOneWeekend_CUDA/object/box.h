#pragma once

#include "object.h"
#include "flat.h"
#include "aabb.h"

class Box : public Object {
public:
	Vec3 minp;
	Vec3 maxp;
	Object** sides;

public:
	__device__ Box(){}
	__device__ Box(const Vec3& min, const Vec3& max, Material* mat);
	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState)const;
	__device__ virtual bool boundingBox(AABB& box, float t0, float t1)const;
};

Box::Box(const Vec3& min, const Vec3& max, Material* mat) {
	minp = min;
	maxp = max;
	sides = new Object * [6];
	int i = 0;
	sides[i++] = new xyFlat(min.getX(), max.getX(), min.getY(), max.getY(), min.getZ(), mat);
	sides[i++] = new xyFlat(min.getX(), max.getX(), min.getY(), max.getY(), max.getZ(), mat);
	sides[i++] = new xzFlat(min.getX(), max.getX(), min.getZ(), max.getZ(), min.getY(), mat);
	sides[i++] = new xzFlat(min.getX(), max.getX(), min.getZ(), max.getZ(), max.getY(), mat);
	sides[i++] = new yzFlat(min.getY(), max.getY(), min.getZ(), max.getZ(), min.getX(), mat);
	sides[i++] = new yzFlat(min.getY(), max.getY(), min.getZ(), max.getZ(), max.getX(), mat);
}

bool Box::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState)const {
	HitRecord tempRec;
	bool hitOne = false;
	float far = tMax;

	for (int i = 0; i < 6; ++i) {
		if (sides[i]->hit(r, tMin, far, tempRec, localRandState)) {
			hitOne = true;
			far = tempRec.t;
			rec = tempRec;
		}
	}
	return hitOne;
}

bool Box::boundingBox(AABB& box, float t0, float t1)const {
	box = AABB(minp, maxp);
	return true;
}