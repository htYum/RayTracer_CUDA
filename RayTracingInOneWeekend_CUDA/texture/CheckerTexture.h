#pragma once

#include "texture.h"

class CheckerTexture : public Texture{
public:
	Texture* odd;
	Texture* even;

public:
	__device__ CheckerTexture(){}
	__device__ CheckerTexture(Texture* t0, Texture* t1) : even(t0), odd(t1){}
	__device__ ~CheckerTexture(){}

	__device__ virtual Vec3 value(float u, float v, const Vec3& p)const;
};

Vec3 CheckerTexture::value(float u, float v, const Vec3& p)const {
	auto sines = sin(10 * p[0]) * sin(10 * p[1]) * sin(10 * p[2]);
	if (sines < 0) {
		return odd->value(u, v, p);
	}
	else return even->value(u, v, p);
}