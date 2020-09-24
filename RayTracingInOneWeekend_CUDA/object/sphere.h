#pragma once

#include "object.h"

class sphere : public object {
public:
    __device__ sphere() {}
    __device__ sphere(const vec3& cen, float r, material* m) :
        center(cen),
        radius(r),
        mat(m) {}
    __device__ ~sphere() {}
    __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const;

public:
    vec3 center;
    float radius;
    material* mat;
};

bool sphere::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    vec3 ac = r.getPos() - center;
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
            vec3 outwardNormal = (rec.p - center) / radius;
            // outwardNormal = normalize(outwardNormal);
            rec.setFaceNormal(r, outwardNormal);
            rec.mat = mat;
            return true;
        }
        temp = (-halfb + root) / a;
        if (temp < tMax && temp > tMin) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            vec3 outwardNormal = (rec.p - center) / radius;
            // outwardNormal = normalize(outwardNormal);
            rec.setFaceNormal(r, outwardNormal);
            rec.mat = mat;
            return true;
        }
    }
    return false;
}