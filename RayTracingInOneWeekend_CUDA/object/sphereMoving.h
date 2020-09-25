#pragma once

#include "object.h"

/*
镜子反射的还是静止的小球
*/
class sphereMoving : public object {
public:
	__device__ sphereMoving(){}
	__device__ sphereMoving(const vec3& cen0, const vec3& cen1, float t0, float t1, float r, material* m):
		center0(cen0),
        center1(cen1),
		time0(t0),
		time1(t1),
		radius(r),
		mat(m){}
	__device__ ~sphereMoving(){}
	__device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const;
    __device__ vec3 center(float time)const;

public:
    vec3 center0, center1;
    float time0, time1;
    float radius;
    material* mat;
};

bool sphereMoving::hit(const ray& r, float tMin, float tMax, hitRecord& rec)const {
    vec3 cen = center(r.time);
    vec3 ac = r.getPos() - cen;
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
            vec3 outwardNormal = (rec.p - cen) / radius;
            // outwardNormal = normalize(outwardNormal);
            rec.setFaceNormal(r, outwardNormal);
            rec.mat = mat;
            return true;
        }
        temp = (-halfb + root) / a;
        if (temp < tMax && temp > tMin) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            vec3 outwardNormal = (rec.p - cen) / radius;
            // outwardNormal = normalize(outwardNormal);
            rec.setFaceNormal(r, outwardNormal);
            rec.mat = mat;
            return true;
        }
    }
    return false;
}

vec3 sphereMoving::center(float time)const {
    return center0 + (time - time0) / (time1 - time0) * (center1 - center0);
}