#pragma once

#include "object.h"

class xyFlat : public Object {
public:
	Material* mat;
	float x0, x1, y0, y1, k;

public:
	__device__ xyFlat(){}
	__device__ xyFlat(float _x0, float _x1, float _y0, float _y1, float _k, Material* m):	// z = k
		x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mat(m){}
	__device__ ~xyFlat() { delete mat; }

	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState)const {
		float t = (k - r.getPos().getZ()) / r.getDir().getZ();
		if (t < tMin || t> tMax)return false;
		float x = r.getPos().getX() + t * r.getDir().getX();
		float y = r.getPos().getY() + t * r.getDir().getY();
		if (x < x0 || x > x1 || y < y0 || y > y1)return false;
		rec.u = (x - x0) / (x1 - x0);
		rec.v = (y - y0) / (y1 - y0);
		rec.t = t;
		Vec3 outNormal = Vec3(0, 0, 1.0);
		rec.setFaceNormal(r, outNormal);
		rec.mat = mat;
		rec.p = r.at(t);
		return true;
	}

	__device__ virtual bool boundingBox(AABB& box, float t0, float t1)const {
		box = AABB(Vec3(x0, y0, k - 0.0001), Vec3(x1, y1, k + 0.0001));
		return true;
	}
};

class xzFlat : public Object {
public:
	Material* mat;
	float x0, x1, z0, z1, k;

public:
	__device__ xzFlat() {}
	__device__ xzFlat(float _x0, float _x1, float _z0, float _z1, float _k, Material* m) :	// y = k
		x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mat(m) {}
	__device__ ~xzFlat() { delete mat; }

	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState = nullptr)const {
		float t = (k - r.getPos().getY()) / r.getDir().getY();
		if (t < tMin || t> tMax)return false;
		float x = r.getPos().getX() + t * r.getDir().getX();
		float z = r.getPos().getZ() + t * r.getDir().getZ();
		if (x < x0 || x > x1 || z < z0 || z > z1)return false;
		rec.u = (x - x0) / (x1 - x0);
		rec.v = (z - z0) / (z1 - z0);
		rec.t = t;
		Vec3 outNormal = Vec3(0, 1.0, 0);
		rec.setFaceNormal(r, outNormal);
		rec.mat = mat;
		rec.p = r.at(t);
		return true;
	}

	__device__ virtual bool boundingBox(AABB& box, float t0, float t1)const {
		box = AABB(Vec3(x0, k - 0.0001, z0), Vec3(x1, k + 0.0001, z1));
		return true;
	}
};

class yzFlat : public Object {
public:
	Material* mat;
	float y0, y1, z0, z1, k;

public:
	__device__ yzFlat() {}
	__device__ yzFlat(float _y0, float _y1, float _z0, float _z1, float _k, Material* m) :	// x = k
		y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mat(m) {}
	__device__ ~yzFlat() { delete mat; }

	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState = nullptr)const {
		float t = (k - r.getPos().getX()) / r.getDir().getX();
		if (t < tMin || t> tMax)return false;
		float y = r.getPos().getY() + t * r.getDir().getY();
		float z = r.getPos().getZ() + t * r.getDir().getZ();
		if (y < y0 || y > y1 || z < z0 || z > z1)return false;
		rec.u = (y - y0) / (y1 - y0);
		rec.v = (z - z0) / (z1 - z0);
		rec.t = t;
		Vec3 outNormal = Vec3(1.0, 0, 0);
		rec.setFaceNormal(r, outNormal);
		rec.mat = mat;
		rec.p = r.at(t);
		return true;
	}

	__device__ virtual bool boundingBox(AABB& box, float t0, float t1)const {
		box = AABB(Vec3(k - 0.0001, y0, z0), Vec3(k + 0.0001, y1, z1));
		return true;
	}
};