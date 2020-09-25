#pragma once

#include "global.h"
#include "ray.h"

class camera {
public:
__device__ camera(const vec3& cameraPos, const vec3& target, const vec3& worldUp, float fov, double scale, float aperture, float focusDistance, float t0 = 0, float t1 = 0) {
        pos = cameraPos;
        lensRadius = aperture / 2;
        time0 = t0;
        time1 = t1;
        float a = radians(fov);
        float halfHeight = tan(a / 2);
        float halfWidth = scale * halfHeight;
        z = normalize(cameraPos - target);
        x = normalize(cross(worldUp, z));
        y = cross(z, x);

        leftDown = pos - focusDistance * (halfWidth * x + halfHeight * y + z);
        horizontal = 2 * halfWidth * x * focusDistance;
        vertical = 2 * halfHeight * y * focusDistance;
    }
    ~camera() {}

    __device__ ray getRay(float u, float v, curandState* localRandState) {
        vec3 rd = lensRadius * randomInDisk(localRandState);
        vec3 offset = x * rd.getX() + y * rd.getY();
        float randTime = curand_uniform(localRandState) * (time1 - time0) + time0;
        return ray(pos + offset, leftDown + u * horizontal + v * vertical - pos - offset, randTime);
    }

public:
    vec3 pos;
    vec3 leftDown;
    vec3 horizontal;
    vec3 vertical;
    vec3 x, y, z;
    float lensRadius;
    float time0, time1;
};