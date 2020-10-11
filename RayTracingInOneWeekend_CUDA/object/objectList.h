#pragma once

#include <thrust\device_vector.h>
#include "Object.h"
#include "..\global.h"

class ObjectList : public Object {
public:
    __device__ ObjectList() {}
    __device__ ObjectList(Object** list, int n) { objects = list; size = n; }
    __device__ ~ObjectList() {}

    __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState) const;
    __device__ virtual bool boundingBox(AABB& box, float t0, float t1) const;

public:
    Object** objects;
    int size;
};

bool ObjectList::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState) const {
    HitRecord tempRec;
    bool hitOne = false;
    float far = tMax;

    for (int i = 0; i < size; ++i) {
        if (objects[i]->hit(r, tMin, far, tempRec, localRandState)) {
            hitOne = true;
            far = tempRec.t;
            rec = tempRec;
        }
    }
    return hitOne;
}

bool ObjectList::boundingBox(AABB& box, float t0, float t1)const {
    if (size == 0) return false;

    AABB tempBox;
    bool firstBox;

    for (int i = 0; i < size; ++i) {
        if (objects[i]->boundingBox(tempBox, t0, t1)) {
            box = firstBox ? tempBox : surroundingBox(tempBox, box);
            firstBox = false;
        }
    }
    return true;
}