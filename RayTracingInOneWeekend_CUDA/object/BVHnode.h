#pragma once

#include <thrust\sort.h>
#include <thrust\device_vector.h>
#include <thrust\device_ptr.h>
#include <thrust\functional.h>
#include <iostream>
#include <algorithm>

#include "Object.h"
#include "ObjectList.h"
#include "AABB.h"

__device__ bool boxCompare(Object* a, Object* b, int axis) {
	AABB boxa;
	AABB boxb;
	if (!a->boundingBox(boxa, 0, 0) || !b->boundingBox(boxb, 0, 0)) {
		printf("No bounding box in BVHnode constructor\n");
	}
	return boxa.minp[axis] < boxb.minp[axis];
}

__device__ bool boxCompareX(Object* a, Object* b) {
	return boxCompare(a, b, 0);
}

__device__ bool boxCompareY(Object* a, Object* b) {
	return boxCompare(a, b, 1);
}

__device__ bool boxCompareZ(Object* a, Object* b) {
	return boxCompare(a, b, 2);
}

__device__ void objectSort(Object** objects, int start, int end, int axis) {
	Object** temp = new Object * [end - start];
	int lmin, lmax, rmin, rmax;
	int next;
	for (int i = 1; i < end - start; i *= 2) {
		for (lmin = start; lmin < end - i; lmin = rmax) {
			rmin = lmax = lmin + i;
			rmax = lmax + i;
			if (rmax > end)rmax = end;
			next = 0;
			while (lmin < lmax && rmin < rmax) {
				if (boxCompare(objects[lmin], objects[rmin], axis)) {
					temp[next++] = objects[lmin++];
				}
				else {
					temp[next++] = objects[rmin++];
				}
			}
			while (lmin < lmax) {
				objects[--rmin] = objects[--lmax];
			}
			while (next > 0) {
				objects[--rmin] = temp[--next];
			}
		}
	}
	free(temp);
}

class BVHnode : public Object {
public:
	Object* left;
	Object* right;
	AABB box;

public:
	__device__ BVHnode(){}
	//__device__ BVHnode(objectList** list, float time0, float time1, curandState* localRandState) : BVHnode((*list)->objects, 0, (*list)->size, time0, time1, localRandState){}
	__device__ BVHnode(Object** objects, int start, int end, float time0, float time1, curandState* localRandState);
	__device__ ~BVHnode(){}

	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState) const;
	__device__ virtual bool boundingBox(AABB& box, float t0, float t1) const;
};

bool BVHnode::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState)const {
	if (!box.hit(r, tMin, tMax))return false;

	bool hitLeft = left->hit(r, tMin, tMax, rec, localRandState);
	bool hitRight = right->hit(r, tMin, (hitLeft ? rec.t : tMax), rec, localRandState);

	return hitLeft || hitRight;
}

bool BVHnode::boundingBox(AABB& outBox, float t0, float t1)const {
	outBox = box;
	return true;
}

BVHnode::BVHnode(Object** objects, int start, int end, float time0, float time1, curandState* localRandState) {
	int axis = curand(localRandState) % 3;
	//int axis = 1;
	auto comparator = (axis == 0) ? boxCompareX : ((axis == 1) ? boxCompareY : boxCompareZ);
	int objectSpan = end - start;

	if (objectSpan == 1) {
		left = right = objects[start];
	}
	else if (objectSpan == 2) {
		if (comparator(objects[start], objects[start + 1])) {
			left = objects[start];
			right = objects[start + 1];
		}
		else {
			left = objects[start + 1];
			right = objects[start];
		}
	}
	else{
		objectSort(objects, start, end, axis);
		//thrust::stable_sort(thrust::seq, objects + start, objects + end, comparator);
		auto mid = start + objectSpan / 2;
		left = new BVHnode(objects, start, mid, time0, time1, localRandState);
		right = new BVHnode(objects, mid, end, time0, time1, localRandState);
	}

	AABB boxa, boxb;
	if (!left->boundingBox(boxa, time0, time1) || !right->boundingBox(boxb, time0, time1)) {
		printf("No bounding box in BVHnode constructor\n");
	}
	box = surroundingBox(boxa, boxb);
}
