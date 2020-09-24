#pragma once

#include <vector>
#include "object.h"
#include "..\global.h"

class objectList : public object {
public:
    __device__ objectList() {}
    __device__ objectList(object** list, int n) { objects = list; size = n; }
    __device__ ~objectList() {}

    __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const;

public:
    object** objects;
    int size;
};

bool objectList::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    hitRecord tempRec;
    bool hitOne = false;
    float far = tMax;

    for (int i = 0; i < size; ++i) {
        if (objects[i]->hit(r, tMin, far, tempRec)) {
            hitOne = true;
            far = tempRec.t;
            rec = tempRec;
        }
    }
    return hitOne;
}