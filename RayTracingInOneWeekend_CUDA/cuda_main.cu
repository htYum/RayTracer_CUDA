#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <curand_kernel.h>
#include <thrust\device_vector.h>
#include <thrust\device_ptr.h>
#include <time.h>
#include <limits>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "global.h"
#include "ray.h"
#include "camera.h"

#include "object\objectList.h"
#include "object\sphere.h"
#include "object\sphereMoving.h"
#include "object\BVHnode.h"

#include "material\lambertian.h"
#include "material\matel.h"
#include "material\dielectric.h"

#include "texture\texture.h"
#include "texture\CheckerTexture.h"
#include "texture\ImageTexture.h"

#define CHECK_CUDA_ERRORS(val) checkCuda( (val), #val, __FILE__, __LINE__)

void checkCuda(cudaError_t result, char const *const func, const char* const file, int const line){
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << " :" << line << "'" << func << "'\n";
        cudaDeviceReset();
        system("pause");
        exit(99);
    }
}

__device__ Vec3 rayColor(const Ray& r, Object** world, int maxDepth, curandState* localRandState) {
    Ray currRay = r;
    Vec3 currAttenuation(1.0, 1.0, 1.0);
    for (int i = 0; i < maxDepth; ++i) {
        HitRecord rec;
        if ((*world)->hit(currRay, 0.001, Infinity, rec)) {
            Vec3 attenuation;
            Ray scattered;
            if (rec.mat->scatter(currRay, rec, attenuation, scattered, localRandState)) {
                currAttenuation = currAttenuation * attenuation;
                currRay = scattered;
            }
            else return Vec3(0, 0, 0);
        }
        else {
            Vec3 dir = normalize(r.getDir());
            float t = 0.5 * (dir.getY() + 1);
            return ((1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0)) * currAttenuation;
        }
    }
    return Vec3(0, 0, 0);
}

__global__ void cudaRandInit(curandState* randState) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1985, 0, 0, randState);
    }
}

__global__ void renderInit(int width, int height, curandState* randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;
    int index = j * width + i;
    curand_init(1985, index, 0, &randState[index]);
}

#define cuda_rand (curand_uniform(&localRandState))

__global__ void render(unsigned char* fb, int width, int height, Camera** cam, Object** world, int samplerPerPixel, int maxDepth, curandState* randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;

    int index = 4 * (j * width + i);
    Vec3 color(0.0, 0.0, 0.0);
    curandState localRandState = randState[index/4];
    for (int k = 0; k < samplerPerPixel; ++k) {
        float u = (i + cuda_rand) / width;
        float v = (j + cuda_rand) / height;
        Ray r = (*cam)->getRay(u, v, &localRandState);
        color += rayColor(r, world, maxDepth, &localRandState);
    }
    randState[index / 4] = localRandState;
    color = color / samplerPerPixel;
    int inverseIndex = 4 * ((height - 1 - j) * width + i);
    fb[inverseIndex + 0] = sqrt(color.getX()) * 255.999;
    fb[inverseIndex + 1] = sqrt(color.getY()) * 255.999;
    fb[inverseIndex + 2] = sqrt(color.getZ()) * 255.999;
    fb[inverseIndex + 3] = 255;
}

__global__ void createWorld(Object** list, Object** world, Camera** cam, int width, int height, curandState* randState, int* worldObjNum, unsigned char* earth, int w, int h, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState localRandState = *randState;
        auto checker = new CheckerTexture(new RGBtexture(Vec3(0.2, 0.3, 0.1)), new RGBtexture(Vec3(0.9, 0.9, 0.9)));

        int i = 0;
        list[i++] = new Sphere(Vec3(0.0, -1000.0, 0.0), 1000.0, new Lambertian(checker));
        for (int a = -10; a < 10; ++a) {
            for (int b = -10; b < 10; ++b) {
                float chooseMat = cuda_rand;
                Vec3 center(a + cuda_rand * 0.9, 0.2, b + cuda_rand * 0.9);
                if (chooseMat < 0.5) {
                    list[i++] = new SphereMoving(center, center + Vec3(0, cuda_rand * 0.5, 0), 0.0, 1.0,  0.2, new Lambertian(new RGBtexture(Vec3(cuda_rand * cuda_rand, cuda_rand * cuda_rand, cuda_rand * cuda_rand))));
                }
                else if (chooseMat < 0.85) {
                    list[i++] = new Sphere(center, 0.2, new Matel(Vec3(0.5 * (1.0 + cuda_rand), 0.5 * (1.0 + cuda_rand), 0.5 * (1.0 + cuda_rand)), 0.35 * cuda_rand));
                }
                else {
                    list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
                }
            }
        }
        list[i++] = new Sphere(Vec3(0, 1.0, 0), 1.0, new Dielectric(1.5));
        list[i++] = new Sphere(Vec3(4, 1.0, 0), 1.0, new Matel(Vec3(0.7, 0.6, 0.5), 0.0));
        list[i++] = new Sphere(Vec3(-4.0, 1.0, 0), 1.0, new Lambertian(new ImageTexture(earth, w, h, n)));
        *worldObjNum = i;

        *randState = localRandState;
        //*world = new BVHnode(list, 0, i, 0.0, 1.0, &localRandState);
        *world = new ObjectList(list, i);

        // camera initialize
        Vec3 cameraPos(17, 2, -1.5);
        Vec3 cameraTarget(0, 0, -0.5);
        float fov = 30.0;
        float focusDistance = 12.0;
        float aperture = 0;
        float beginTime = 0.0;
        float endTime = 1.0;

        *cam = new Camera(cameraPos, cameraTarget, Vec3(0, 1.0, 0), fov, float(width) / float(height), aperture, focusDistance, beginTime, endTime);

        //free(list);
    }
}

__global__ void freeWorld(Object** list, Object** world, Camera** cam, int* objectNum) {
    for (int i = 0; i < *objectNum; ++i) {
        delete ((Sphere*)list[i])->mat;
        delete list[i];
    }
    delete world;
    delete cam;
    delete objectNum;
}

int main()
{
	/*int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;*/

    const int maxDepth = 50;
    const int samplerPerPixel = 500;
    const int width = 1920;
    const int height = 1080;

    // image
    const int pixelsNum = width * height;

    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << width << " * " << height << " image ";
    std::cerr << " in " << tx << " * " << ty << " blocks.\n";

    // fb
    size_t fbSize = 4 * pixelsNum * sizeof(unsigned char);

    // allocate fb
    unsigned char* fb;
    CHECK_CUDA_ERRORS(cudaMallocManaged((void**)&fb, fbSize));

    // allocate random state
    curandState* totalRandState;
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&totalRandState, pixelsNum * sizeof(curandState)));
    curandState* randState;
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&randState, sizeof(curandState)));

    cudaRandInit<<<1, 1 >>> (randState);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    // world & camera data
    Object** list;
    CHECK_CUDA_ERRORS(cudaMallocManaged((void**)&list, (20 * 20 + 1 + 3) * sizeof(Object*)));
    Object** world;
    CHECK_CUDA_ERRORS(cudaMallocManaged((void**)&world, sizeof(Object*)));
    Camera** cam;
    CHECK_CUDA_ERRORS(cudaMallocManaged((void**)&cam, sizeof(Camera*)));
    int* worldObjNum;
    CHECK_CUDA_ERRORS(cudaMallocManaged((void**)&worldObjNum, sizeof(int)));

    int w, h, n;
    unsigned char* earthImg = stbi_load("earth.jpg", &w, &h, &n, 0);
    unsigned char* cudaEarth;
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&cudaEarth, sizeof(unsigned char) * w * h * n));
    cudaMemcpy(cudaEarth, earthImg, w * h * n * sizeof(unsigned char), cudaMemcpyHostToDevice);

    *worldObjNum = 0;
    createWorld<<<1, 1 >>> (list, world, cam, width, height, randState, worldObjNum, cudaEarth, w, h, n);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    clock_t begin, end;
    std::cerr << "\nStart rendering\n";
    begin = clock();
    // render buffer
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);
    renderInit<<<blocks, threads>>> (width, height, totalRandState);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, width, height, cam, world, samplerPerPixel, maxDepth, totalRandState);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    end = clock();
    double time = ((double)(end - begin)) / CLOCKS_PER_SEC;
    std::cerr << "End\ntook " << time << " s.\n";

    /*unsigned char* data[4 * pixelsNum];
    cudaMemcpy(fb, data, fbSize, cudaMemcpyDeviceToHost);*/

    // output image png
    stbi_write_png("main.png", width, height, 4, fb, width * 4);

    //freeWorld<<<1, 1 >>>(list, world, cam, worldObjNum);
    CHECK_CUDA_ERRORS(cudaFree(totalRandState));
    CHECK_CUDA_ERRORS(cudaFree(randState));
    CHECK_CUDA_ERRORS(cudaFree(worldObjNum));
    CHECK_CUDA_ERRORS(cudaFree(list));
    CHECK_CUDA_ERRORS(cudaFree(world));
    CHECK_CUDA_ERRORS(cudaFree(cam));
    CHECK_CUDA_ERRORS(cudaFree(fb));
    cudaDeviceReset();
    system("pause");
}