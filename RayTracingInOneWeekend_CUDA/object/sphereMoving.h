#pragma once

#include "Object.h"

class SphereMoving : public Object {
public:
	__device__ SphereMoving(){}
    __device__ SphereMoving(const Vec3& cen0, const Vec3& cen1, float t0, float t1, float r, Material* m):
		center0(cen0),
        center1(cen1),
		time0(t0),
		time1(t1),
		radius(r),
		mat(m){}
    __device__ ~SphereMoving(){}
	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const;
    __device__ virtual bool boundingBox(AABB& box, float t0, float t1) const;
    __device__ Vec3 center(float time)const;

public:
    Vec3 center0, center1;
    float time0, time1;
    float radius;
    Material* mat;
};

bool SphereMoving::hit(const Ray& r, float tMin, float tMax, HitRecord& rec)const {
    Vec3 cen = center(r.time);
    Vec3 ac = r.getPos() - cen;
    float a = dot(r.getDir(), r.getDir());
    float halfb = dot(ac, r.getDir());
    float c = dot(ac, ac) - radius * radius;
    float result = halfb * halfb - a * c;

    if (result > 0) {
        float root = sqrt(result);
        float temp = (-halfb - root) / a;
        if (temp < tMax && temp > tMin) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            Vec3 outwardNormal = (rec.p - cen) / radius;
            // outwardNormal = normalize(outwardNormal);
            rec.setFaceNormal(r, outwardNormal);
            rec.mat = mat;
            return true;
        }
        temp = (-halfb + root) / a;
        if (temp < tMax && temp > tMin) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            Vec3 outwardNormal = (rec.p - cen) / radius;
            // outwardNormal = normalize(outwardNormal);
            rec.setFaceNormal(r, outwardNormal);
            rec.mat = mat;
            return true;
        }
    }
    return false;
}

bool SphereMoving::boundingBox(AABB& box, float t0, float t1)const {
    AABB box0 = AABB(center(t0) - Vec3(radius, radius, radius), center(t0) + Vec3(radius, radius, radius));
    AABB box1 = AABB(center(t1) - Vec3(radius, radius, radius), center(t1) + Vec3(radius, radius, radius));
    box = surroundingBox(box0, box1);
    return true;
}
Vec3 SphereMoving::center(float time)const {
    return center0 + (time - time0) / (time1 - time0) * (center1 - center0);
}