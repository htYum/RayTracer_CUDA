#pragma once

#include "box.h"

class yRotate : public Object {
public:
	Object* obj;
	float sina;
	float cosa;
	bool hasBox;
	AABB bbox;

public:
	__device__ yRotate(){}
	__device__ yRotate(Object* o, float angle);
	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState)const;
	__device__ virtual bool boundingBox(AABB& box, float t0, float t1)const;
};

yRotate::yRotate(Object* o, float angle) :obj(o) {
	auto radians = angle * 3.1415926535 / 180;
	sina = sin(radians);
	cosa = cos(radians);
	hasBox = obj->boundingBox(bbox, 0, 1);

	Vec3 minv(INFINITY, INFINITY, INFINITY);
	Vec3 maxv(-INFINITY, -INFINITY, -INFINITY);

	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 2; ++k) {
				auto x = i * bbox.maxp.getX() + (1 - i) * bbox.minp.getX();
				auto y = j * bbox.maxp.getY() + (1 - j) * bbox.minp.getY();
				auto z = k * bbox.maxp.getZ() + (1 - k) * bbox.minp.getZ();
				auto newx = cosa * x + sina * z;
				auto newz = -sina * x + cosa * z;
				Vec3 tester(newx, y, newz);
				for (int c = 0; c < 3; ++c) {
					minv[c] = ffmin(minv[c], tester[c]);
					maxv[c] = ffmax(maxv[c], tester[c]);
				}
			}
		}
	}
	bbox = AABB(minv, maxv);
}

bool yRotate::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState)const {
	Vec3 pos = r.getPos();
	Vec3 dir = r.getDir();
	pos[0] = cosa * r.getPos()[0] - sina * r.getPos()[2];
	pos[2] = sina * r.getPos()[0] + cosa * r.getPos()[2];
	dir[0] = cosa * r.getDir()[0] - sina * r.getDir()[2];
	dir[2] = sina * r.getDir()[0] + cosa * r.getDir()[2];

	Ray rotetedRay(pos, dir, r.getTime());

	if (!obj->hit(rotetedRay, tMin, tMax, rec, localRandState)) {
		return false;
	}

	Vec3 p = rec.p;
	Vec3 normal = rec.normal;
	p[0] = cosa * rec.p[0] + sina * rec.p[2];
	p[2] = -sina * rec.p[0] + cosa * rec.p[2];
	normal[0] = cosa * rec.normal[0] + sina * rec.normal[2];
	normal[2] = -sina * rec.normal[0] + cosa * rec.normal[2];

	rec.p = p;
	rec.setFaceNormal(rotetedRay, normal);

	return true;
}

bool yRotate::boundingBox(AABB& box, float t0, float t1)const {
	box = bbox;
	return hasBox;
}