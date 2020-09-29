#pragma once

#include "..\global.h"
#include "..\math\Vec3.h"

class Texture {
public:
	__device__ virtual Vec3 value(float u, float v, const Vec3& p) const = 0;
};