#pragma once

#include "..\global.h"
#include "..\Math\Vec3.h"
#include "..\Ray.h"


class AABB {
public:
	__device__ AABB() {}
	__device__ AABB(const Vec3& a, const Vec3& b) : minp(a), maxp(b){}

	__device__ bool hit(const Ray& r, float tMin, float tMax) const;

public:
	Vec3 minp;
	Vec3 maxp;
};

bool AABB::hit(const Ray& r, float tMin, float tMax) const{
	for (int i = 0; i < 3; ++i) {
		float divDir = 1.0 / r.getDir()[i];
		float t0 = (minp[i] - r.getPos()[i]) * divDir;
		float t1 = (maxp[i] - r.getPos()[i]) * divDir;
		if (divDir < 0) {
			auto temp = t0;
			t0 = t1;
			t1 = temp;
		}
		tMin = (t0 > tMin) ? t0 : tMin;
		tMax = (t1 < tMax) ? t1 : tMax;
		if (tMin >= tMax) return false;
	}
	return true;
}

__device__ AABB surroundingBox(const AABB& box0, const AABB& box1) {
	Vec3 minp(ffmin(box0.minp[0], box1.minp[0]),
			  ffmin(box0.minp[1], box1.minp[1]),
			  ffmin(box0.minp[2], box1.minp[2]));
	Vec3 maxp(ffmax(box0.maxp[0], box1.maxp[0]),
			  ffmax(box0.maxp[1], box1.maxp[1]),
			  ffmax(box0.maxp[2], box1.maxp[2]));
	return AABB(minp, maxp);
}