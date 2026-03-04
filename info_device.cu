#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "time.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include <iostream>
#include <fstream>   
#include "device_launch_parameters.h"
#include <direct.h>

#include "src/info_device.cuh"

void getdevice()
{
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    }
    else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Device %d: \"%s\"\n", dev, deviceProp.name);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
            deviceProp.major, deviceProp.minor);
        printf("  Total amount of global memory:                 %.2f GB\n",
            (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
        printf("  Multiprocessors:                               %d\n",
            deviceProp.multiProcessorCount);
        printf("  CUDA Cores/MP:                                 %d\n",
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
        printf("  Total CUDA Cores:                              %d\n",
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n",
            deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
        printf("  Memory Clock rate:                             %.0f MHz\n",
            deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",
            deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n",
                deviceProp.l2CacheSize);
        }

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
            deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
            deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
            deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
            deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
            deviceProp.maxTexture2DLayered[2]);

        printf("  Total amount of constant memory:               %lu bytes\n",
            deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n",
            deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n",
            deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",
            deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n",
            deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n",
            deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
            deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
            deviceProp.maxGridSize[0],
            deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n",
            deviceProp.memPitch);
        printf("  Texture alignment:                             %lu bytes\n",
            deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n",
            (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n",
            deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n",
            deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n",
            deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n",
            deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n",
            deviceProp.ECCEnabled ? "Enabled" : "Disabled");
        printf("  Unified Addressing (UVA):                      %s\n",
            deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  PCI Bus ID / PCI location ID:                  %d / %d\n",
            deviceProp.pciBusID, deviceProp.pciDeviceID);

        const char* sComputeMode[] = {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }
}

int _ConvertSMVer2Cores(int major, int minor) {
    typedef struct {
        int SM;
        int Cores;
    } SMtoCores;

    SMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192}, // Kepler
        {0x32, 192}, // Kepler
        {0x35, 192}, // Kepler
        {0x37, 192}, // Kepler
        {0x50, 128}, // Maxwell
        {0x52, 128}, // Maxwell
        {0x53, 128}, // Maxwell
        {0x60, 64},  // Pascal
        {0x61, 128}, // Pascal
        {0x62, 128}, // Pascal
        {0x70, 64},  // Volta
        {0x72, 64},  // Xavier
        {0x75, 64},  // Turing
        {0x80, 64},  // Ampere
        {0x86, 128}, // Ampere
        {-1, -1}
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    printf("MapSMtoCores undefined SM version %d.%d!\n", major, minor);
    return -1;
}
