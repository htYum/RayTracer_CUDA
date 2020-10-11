#pragma once

#include "Object.h"

__device__ void getSphereUV(Vec3& p, float& u, float& v) {
    float x = p.getX();
    float y = p.getY();
    float z = p.getZ();
    float phi = atan2(z, x);
    float theta = asin(y);
    u = 0.5 - phi / (2 * Pi);
    v = theta / Pi + 0.5;
}

class Sphere : public Object {
public:
    __device__ Sphere() {}
    __device__ Sphere(const Vec3& cen, float r, Material* m) :
        center(cen),
        radius(r),
        mat(m) {}
    __device__ ~Sphere() {}
    __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState) const;
    __device__ virtual bool boundingBox(AABB& box, float t0, float t1) const;

public:
    Vec3 center;
    float radius;
    Material* mat;
};

bool Sphere::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState) const {
    Vec3 ac = r.getPos() - center;
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
            getSphereUV((rec.p - center) / radius, rec.u, rec.v);
            Vec3 outwardNormal = (rec.p - center) / radius;
            // outwardNormal = normalize(outwardNormal);
            rec.setFaceNormal(r, outwardNormal);
            rec.mat = mat;
            return true;
        }
        temp = (-halfb + root) / a;
        if (temp < tMax && temp > tMin) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            getSphereUV((rec.p - center) / radius, rec.u, rec.v);
            Vec3 outwardNormal = (rec.p - center) / radius;
            // outwardNormal = normalize(outwardNormal);
            rec.setFaceNormal(r, outwardNormal);
            rec.mat = mat;
            return true;
        }
    }
    return false;
}

bool Sphere::boundingBox(AABB& box, float t0, float t1)const {
    box = AABB(center - Vec3(radius, radius, radius), center + Vec3(radius, radius, radius));
    return true;
}