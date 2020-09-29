#pragma once

#include "texture.h"

class RGBtexture : public Texture{
public:
	Vec3 color;

public:
	__device__ RGBtexture(){}
	__device__ RGBtexture(const Vec3& col) : color(col){}
	__device__ ~RGBtexture(){}

	__device__ virtual Vec3 value(float u, float v, const Vec3& p)const;
};

Vec3 RGBtexture::value(float u, float v, const Vec3& p)const {
	return color;
}