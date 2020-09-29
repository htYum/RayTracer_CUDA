#pragma once

#include "texture.h"

class ImageTexture : public Texture {
public:
	unsigned char* data;
	int width, height;
	int nChannal;

public:
	__device__ ImageTexture(){}
	__device__ ImageTexture(unsigned char* img, int w, int h, int n) : data(img), width(w), height(h), nChannal(n){}
	__device__ ~ImageTexture() { delete data; }

	__device__ virtual Vec3 value(float u, float v, const Vec3& p)const;
};

Vec3 ImageTexture::value(float u, float v, const Vec3& p)const {
	if (data == nullptr) {
		return Vec3(0, 1.0, 1.0);
	}

	int i = u * width;
	int j = v * height;

	if (i < 0)i = 0;
	if (j < 0)j = 0;
	if (i > width - 1)i = width - 1;
	if (j > height - 1)j = height - 1;

	float r = data[nChannal * i + nChannal * j * width + 0] / 255.0;
	float g = data[nChannal * i + nChannal * j * width + 1] / 255.0;
	float b = data[nChannal * i + nChannal * j * width + 2] / 255.0;

	return Vec3(r, g, b);
}