#include <iostream>
#include "src/info_device.cuh"// ����cudaͷ�ļ�
#include "src/parameters.h"
#include "src/init.cuh"
#include "src/rhs.cuh"


#include "cufft.h"
#include <cuda_runtime.h>
#include "cusparse.h"
#include "cusparse_v2.h"
#include <chrono>

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

using namespace std;



#define CHECK_CUSPARSE(func) {                                          \
    cusparseStatus_t status = (func);                                   \
    if (status != CUSPARSE_STATUS_SUCCESS) {                            \
        std::cerr << "CUSPARSE API failed at line " << __LINE__         \
                  << " with error: " << cusparseGetErrorString(status)  \
                  << std::endl;                                         \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

#define CHECK_CUDA(func) {                                              \
    cudaError_t status = (func);                                        \
    if (status != cudaSuccess) {                                        \
        std::cerr << "CUDA API failed at line " << __LINE__             \
                  << " with error: " << cudaGetErrorString(status)      \
                  << std::endl;                                         \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}



int main()
{
    /**/
    //int minGridSize; // ��С������
    //int blockSize;   // �Ƽ����߳̿��С
    ////int gridSize;    // ���յ������С
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, rhsx, 0, 0);    //// ��ȡ�Ƽ��Ŀ��С
    //int maxActiveBlocks;
    //size_t dynamicSMemSize = 0;
    //// ���ú�������ȡÿ�� SM ���������
    //cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //    &maxActiveBlocks,
    //    rhsx,             // �ں˺���
    //    blockSize,            // ÿ������߳���
    //    dynamicSMemSize       // ��̬�����ڴ��С
    //);
    //int numSMs;
    //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    //printf("blockSize = %d   minGridSize = %d  \n maxActiveBlocks =  %d  maxConcurrentBlocks = %d", blockSize, minGridSize, maxActiveBlocks, maxActiveBlocks*numSMs);
    /**/
    cudaSetDevice(1);

    //sprintf(output_path, "C:\\Users\\customer\\Desktop\\DNS_0603\\");
    sprintf(output_path, "E:\\lch\\");

    CHECK_CUDA(cudaMalloc((void**)&flu, nxp * (nyp + 1) * nzp * sizeof(fluid)));
    CHECK_CUDA(cudaMalloc((void**)&var, nxp * (nyp + 1) * nzp * sizeof(process_variables)));

    CHECK_CUDA(cudaMalloc((void**)&prsrc, nxp * (nyp + 1) * nzp * sizeof(cufftDoubleComplex)));

    dim3 blockDim(8, 8, 8); // ÿ���߳̿��е��߳���
    dim3 gridDim((nxp + blockDim.x - 1) / blockDim.x,
        (nyp + 1 + blockDim.y - 1) / blockDim.y,
        (nzp + blockDim.z - 1) / blockDim.z);// �����еĿ���

    init_gpu_var << < gridDim, blockDim >> > (flu, var);

    init_mesh_para();

#ifdef Restart
    init_restart(flu, flu_host, var);
#else 
    init_fluid(flu_host);
#endif
 
    CHECK_CUDA(cudaMemcpy(flu, flu_host, nxp * (nyp + 1) * nzp * sizeof(fluid), cudaMemcpyHostToDevice));


    calcuate();
#ifdef Output_Restart
    output_restart(flu_host, var);   
#endif 
 

Error:
    cudaFree(flu);
    cudaFree(var);
    free(flu_host);

    return 0;
}

