#pragma once

#include "global.h"
#include "Ray.h"

class Camera {
public:
    __device__ Camera(const Vec3& cameraPos, const Vec3& target, const Vec3& worldUp, float fov, double scale, float aperture, float focusDistance, float t0 = 0, float t1 = 0) {
        pos = cameraPos;
        lensRadius = aperture / 2;
        time0 = t0;
        time1 = t1;
        //float a = radians(fov);
        float a = fov * 3.1415926535 / 180;
        float halfHeight = tan(a / 2);
        float halfWidth = scale * halfHeight;
        z = normalize(cameraPos - target);
        x = normalize(cross(worldUp, z));
        y = cross(z, x);

        leftDown = pos - focusDistance * (halfWidth * x + halfHeight * y + z);
        horizontal = 2 * halfWidth * x * focusDistance;
        vertical = 2 * halfHeight * y * focusDistance;
    }
    ~Camera() {}

    __device__ Ray getRay(float u, float v, curandState* localRandState) {
        Vec3 rd = lensRadius * randomInDisk(localRandState);
        Vec3 offset = x * rd.getX() + y * rd.getY();
        float randTime = curand_uniform(localRandState) * (time1 - time0) + time0;
        return Ray(pos + offset, leftDown + u * horizontal + v * vertical - pos - offset, randTime);
    }

public:
    Vec3 pos;
    Vec3 leftDown;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 x, y, z;
    float lensRadius;
    float time0, time1;
};