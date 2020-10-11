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
#include "object\flat.h"
#include "object\box.h"
#include "object\yRotate.h"
#include "object\translate.h"
#include "object\constantMedium.h"

#include "material\lambertian.h"
#include "material\matel.h"
#include "material\dielectric.h"
#include "material\DiffuseLight.h"
#include "material\Isotropic.h"

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

__device__ Vec3 rayColor(const Ray& r, Object** world, const Vec3& background, int maxDepth, curandState* localRandState) {
    Ray currRay = r;
    Vec3 currAttenuation(1.0, 1.0, 1.0);
    Vec3 lastAttenuation = currAttenuation;
    Vec3 currEmitted(0, 0, 0);
    for (int i = 0; i < maxDepth; ++i) {
        HitRecord rec;
        if ((*world)->hit(currRay, 0.001, Infinity, rec, localRandState)) {
            Vec3 attenuation;
            Vec3 emitted = rec.mat->emitted(rec.u, rec.v, rec.p);
            Ray scattered;
            if (rec.mat->scatter(currRay, rec, attenuation, scattered, localRandState)) {
                currAttenuation = currAttenuation * attenuation;
                currEmitted += emitted * lastAttenuation;
                lastAttenuation = attenuation;
                currRay = scattered;
            }
            else return emitted * currAttenuation + currEmitted;
        }
        else {
            return background * currAttenuation + currEmitted;
        }
    }
    return Vec3(0, 0, 0);
}

//__device__ Vec3 rayColor(const Ray& r, Object** world, int maxDepth, curandState* localRandState) {
//    Ray currRay = r;
//    Vec3 currAttenuation(1.0, 1.0, 1.0);
//    for (int i = 0; i < maxDepth; ++i) {
//        HitRecord rec;
//        if ((*world)->hit(currRay, 0.001, Infinity, rec)) {
//            Vec3 attenuation;
//            Ray scattered;
//            if (rec.mat->scatter(currRay, rec, attenuation, scattered, localRandState)) {
//                currAttenuation = currAttenuation * attenuation;
//                currRay = scattered;
//            }
//            else return Vec3(0, 0, 0);
//        }
//        else {
//            Vec3 dir = normalize(r.getDir());
//            float t = 0.5 * (dir.getY() + 1);
//            return ((1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0)) * currAttenuation;
//        }
//    }
//    return Vec3(0, 0, 0);
//}

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
    Vec3 background(0, 0, 0);
    curandState localRandState = randState[index/4];
    for (int k = 0; k < samplerPerPixel; ++k) {
        float u = (i + cuda_rand) / width;
        float v = (j + cuda_rand) / height;
        Ray r = (*cam)->getRay(u, v, &localRandState);
        color += rayColor(r, world, background, maxDepth, &localRandState);
        //color += rayColor(r, world, maxDepth, &localRandState);
    }
    randState[index / 4] = localRandState;
    color = color / samplerPerPixel;
    int inverseIndex = 4 * ((height - 1 - j) * width + i);
    int r = sqrt(color.getX()) * 255.999;
    int g = sqrt(color.getY()) * 255.999;
    int b = sqrt(color.getZ()) * 255.999;
    fb[inverseIndex + 0] = r > 255 ? 255 : r;
    fb[inverseIndex + 1] = g > 255 ? 255 : g;
    fb[inverseIndex + 2] = b > 255 ? 255 : b;
    fb[inverseIndex + 3] = 255;
}

//__global__ void createWorld(Object** list, Object** world, Camera** cam, int width, int height, curandState* randState, int* worldObjNum, unsigned char* earth, int w, int h, int n) {
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//        curandState localRandState = *randState;
//        auto checker = new CheckerTexture(new RGBtexture(Vec3(0.2, 0.3, 0.1)), new RGBtexture(Vec3(0.9, 0.9, 0.9)));
//
//        int i = 0;
//        list[i++] = new Sphere(Vec3(0.0, -1000.0, 0.0), 1000.0, new Lambertian(checker));
//        for (int a = -2; a < 2; ++a) {
//            for (int b = -2; b < 2; ++b) {
//                float chooseMat = cuda_rand;
//                Vec3 center(a + cuda_rand * 0.9, 0.2, b + cuda_rand * 0.9);
//                if (chooseMat < 0.5) {
//                    list[i++] = new SphereMoving(center, center + Vec3(0, cuda_rand * 0.5, 0), 0.0, 1.0,  0.2, new Lambertian(new RGBtexture(Vec3(cuda_rand * cuda_rand, cuda_rand * cuda_rand, cuda_rand * cuda_rand))));
//                }
//                else if (chooseMat < 0.85) {
//                    list[i++] = new Sphere(center, 0.2, new Matel(Vec3(0.5 * (1.0 + cuda_rand), 0.5 * (1.0 + cuda_rand), 0.5 * (1.0 + cuda_rand)), 0.35 * cuda_rand));
//                }
//                else {
//                    list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
//                }
//            }
//        }
//        list[i++] = new Sphere(Vec3(0, 1.0, 0), 1.0, new Dielectric(1.5));
//        list[i++] = new Sphere(Vec3(4, 1.0, 0), 1.0, new Matel(Vec3(0.7, 0.6, 0.5), 0.0));
//        list[i++] = new Sphere(Vec3(-4.0, 1.0, 0), 1.0, new Lambertian(new ImageTexture(earth, w, h, n)));
//        //list[i++] = new Sphere(Vec3(0, 3, -3), 1.5, new DiffuseLight(new RGBtexture(Vec3(4.0, 4.0, 4.0))));
//        list[i++] = new xyFlat(3, 5, 1, 3, -2, new DiffuseLight(new RGBtexture(Vec3(4.0, 4.0, 4.0))));
//        *worldObjNum = i;
//
//        *randState = localRandState;
//        //*world = new BVHnode(list, 0, i, 0.0, 1.0, &localRandState);
//        *world = new ObjectList(list, i);
//
//        // camera initialize
//        Vec3 cameraPos(17, 2, -1.5);
//        Vec3 cameraTarget(0, 0, -0.5);
//        float fov = 30.0;
//        float focusDistance = 12.0;
//        float aperture = 0;
//        float beginTime = 0.0;
//        float endTime = 1.0;
//
//        *cam = new Camera(cameraPos, cameraTarget, Vec3(0, 1.0, 0), fov, float(width) / float(height), aperture, focusDistance, beginTime, endTime);
//    }
//}

__global__ void createWorld(Object** list, Object** world, Camera** cam, int width, int height, curandState* randState, int* worldObjNum, unsigned char* earth, int w, int h, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState localRandState = *randState;

        int i = 0;

        auto ground = new Lambertian(new RGBtexture(Vec3(0.48, 0.83, 0.53)));
        int boxesPerSide = 5;
        Object** floor = new Object *[boxesPerSide * boxesPerSide];
        int f = 0;
        for (int m = 0; m < boxesPerSide; ++m) {
            for (int n = 0; n < boxesPerSide; ++n) {
                auto w0 = 200.0;
                auto x0 = -400.0 + m * w0;
                auto z0 = n * w0;
                auto y0 = 0.0;
                auto x1 = x0 + w0;
                auto y1 = curand_uniform(&localRandState) * 100 + 1;
                auto z1 = z0 + w0;
                floor[f++] = new Box(Vec3(x0, y0, z0), Vec3(x1, y1, z1), ground);
                //list[i++] = new Box(Vec3(x0, y0, z0), Vec3(x1, y1, z1), ground);
            }
        }
        list[i++] = new BVHnode(floor, 0, f, 0.0, 1.0, &localRandState);

        auto light = new DiffuseLight(new RGBtexture(Vec3(7.0, 7.0, 7.0)));
        list[i++] = new xzFlat(123, 423, 147, 412, 554, light);

        auto center0 = Vec3(400.0, 400.0, 200.0);
        auto center1 = center0 + Vec3(30.0, 0, 0);
        auto movingSphereMaterial = new Lambertian(new RGBtexture(Vec3(0.7, 0.15, 0.02)));
        list[i++] = new SphereMoving(center0, center1, 0, 1.0, 50, movingSphereMaterial);

        list[i++] = new Sphere(Vec3(260, 150, 45), 50, new Dielectric(1.5));
        list[i++] = new Sphere(Vec3(0, 150, 145), 50, new Matel(Vec3(0.8, 0.8, 0.9), 10.0));

        auto boundary = new Sphere(Vec3(360, 150, 145), 70, new Dielectric(1.5));
        list[i++] = boundary;
        list[i++] = new ConstantMedium(boundary, 0.2, new RGBtexture(Vec3(0.2, 0.4, 0.9)));
        boundary = new Sphere(Vec3(0, 0, 0), 5000, new Dielectric(1.5));
        list[i++] = new ConstantMedium(boundary, 0.0001, new RGBtexture(Vec3(1, 1, 1)));

        auto earthTexture = new Lambertian(new ImageTexture(earth, w, h, n));
        list[i++] = new Sphere(Vec3(220, 280, 300), 80, earthTexture);

        //int ns = 1000;
        //Object** sphereBox = new Object * [ns];
        auto white = new Lambertian(new RGBtexture(Vec3(0.73, 0.73, 0.73)));
        auto matelWhite = new Matel(Vec3(0.8, 0.8, 0.9), 0);
        list[i++] = new Sphere(Vec3(-100, 320, 395), 80, matelWhite);
        /*for (int k = 0; k < ns; ++k) {
            sphereBox[k] = new Sphere(Vec3::random(&localRandState) * 165, 10, white);
        }
        list[i++] = new Translate(new yRotate(new BVHnode(sphereBox, 0, ns, 0.0, 1.0, &localRandState), 15.0), Vec3(-100, 270, 395));*/
        /*auto white = new Lambertian(new RGBtexture(Vec3(0.73, 0.73, 0.73)));
        auto red = new Lambertian(new RGBtexture(Vec3(0.65, 0.05, 0.05)));
        auto green = new Lambertian(new RGBtexture(Vec3(0.15, 0.45, 0.15)));
        auto blue = new Lambertian(new RGBtexture(Vec3(0.12, 0.12, 0.45)));
        auto light = new DiffuseLight(new RGBtexture(Vec3(7.0, 7.0, 7.0)));
        list[i++] = new yzFlat(0, 555, 0, 555, 555, blue);
        list[i++] = new yzFlat(0, 555, 0, 555, 0, red);
        list[i++] = new xzFlat(113, 443, 127, 432, 554, light);
        list[i++] = new xzFlat(0, 555, 0, 555, 0, white);
        list[i++] = new xzFlat(0, 555, 0, 555, 555, green);
        list[i++] = new xyFlat(0, 555, 0, 555, 555, white);
        Box* box0 = new Box(Vec3(0, 0, 0), Vec3(165, 330, 165), white);
        Box* box1 = new Box(Vec3(0, 0, 0), Vec3(165, 165, 165), white);
        list[i++] = new ConstantMedium(new Translate(new yRotate(box0, 15.0), Vec3(265, 0, 295)), 0.01, new RGBtexture(Vec3(0, 0, 0)));
        list[i++] = new ConstantMedium(new Translate(new yRotate(box1, -18.0), Vec3(130, 0, 65)), 0.01, new RGBtexture(Vec3(1.0, 1.0, 1.0)));*/

        *worldObjNum = i;

        *randState = localRandState;
        //*world = new BVHnode(list, 0, i, 0.0, 1.0, &localRandState);
        *world = new ObjectList(list, i);

        // camera initialize
        /*Vec3 cameraPos(17, 2, -1.5);
        Vec3 cameraTarget(0, 0, -0.5);
        float fov = 30.0;
        float focusDistance = 12.0;
        float aperture = 0;
        float beginTime = 0.0;
        float endTime = 1.0;*/
        Vec3 cameraPos(478, 278, -600);
        Vec3 cameraTarget(278, 278, 0);
        float fov = 40.0;
        float focusDistance = 10.0;
        float aperture = 0;
        float beginTime = 0.0;
        float endTime = 1.0;

        *cam = new Camera(cameraPos, cameraTarget, Vec3(0, 1.0, 0), fov, float(width) / float(height), aperture, focusDistance, beginTime, endTime);
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
    const int samplerPerPixel = 5000;
    const int width = 800;
    const int height = 800;

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
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&list, (10 + 25) * sizeof(Object*)));
    Object** world;
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&world, sizeof(Object*)));
    Camera** cam;
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&cam, sizeof(Camera*)));
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
    int hours = time / 3600;
    if (hours > 0)time = time - 3600 * hours;
    int minute = time / 60;
    if (minute > 0)time = time - 60 * minute;
    std::cerr << "End\ntook ";
    if (hours > 0)std::cerr << hours << " h ";
    if (minute > 0)std::cerr << minute << " m ";
    std::cerr << time << " s \n";

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