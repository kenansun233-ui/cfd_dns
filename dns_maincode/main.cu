#include <iostream>
#include "src/info_device.cuh"// ����cudaͷ�ļ�
#include "src/parameters.h"
#include "src/init.cuh"
#include "src/rhs.cuh"


#include "cufft.h"
#include <cuda_runtime.h>
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
    CHECK_CUDA(cudaSetDevice(1));

    //sprintf(output_path, "C:\\Users\\customer\\Desktop\\DNS_0603\\");
    sprintf(output_path, "E:\\lch\\");
    _mkdir(output_path);

    char run_info_name[200];
    sprintf_s(run_info_name, "%srun_info.dat", output_path);
    FILE* run_info = nullptr;
    fopen_s(&run_info, run_info_name, "w");
    if (run_info == nullptr)
    {
        fprintf(stderr, "Failed to open file: %s\n", run_info_name);
        exit(EXIT_FAILURE);
    }

    fprintf(run_info, "Re_omega %.15e\n", Re_omega);
    fprintf(run_info, "Re_delta %.15e\n", Re_delta);
    fprintf(run_info, "U_osc %.15e\n", U_osc);
    fprintf(run_info, "omega %.15e\n", omega);
    fprintf(run_info, "nu %.15e\n", nu);
    fprintf(run_info, "stokes_delta %.15e\n", stokes_delta);
    fprintf(run_info, "simulation_cycles %.15e\n", simulation_cycles);
    fprintf(run_info, "dt %.15e\n", dt);
    fprintf(run_info, "timemax %d\n", timemax);
    fprintf(run_info, "nxp %d\n", nxp);
    fprintf(run_info, "nyp %d\n", nyp);
    fprintf(run_info, "nzp %d\n", nzp);
    fprintf(run_info, "xlength %.15e\n", xlength);
    fprintf(run_info, "ylength %.15e\n", ylength);
    fprintf(run_info, "zlength %.15e\n", zlength);
    fprintf(run_info, "enable_bulk_pressure_feedback %d\n", enable_bulk_pressure_feedback ? 1 : 0);
    fprintf(run_info, "bypass_perturbation_amp %.15e\n", bypass_perturbation_amp);
    fprintf(run_info, "stat_output_interval %d\n", stat_output_interval);
    fprintf(run_info, "stat_output_dt %.15e\n", stat_output_dt);
    fprintf(run_info, "tau_wall_map_interval %d\n", tau_wall_map_interval);
    fprintf(run_info, "tau_wall_map_output_dt %.15e\n", tau_wall_map_output_dt);
    fprintf(run_info, "restart_output_interval %d\n", restart_output_interval);
    fprintf(run_info, "restart_input_step %d\n", restart_input_step);
#ifdef Restart
    fprintf(run_info, "Restart 1\n");
#else
    fprintf(run_info, "Restart 0\n");
#endif
    fclose(run_info);

    CHECK_CUDA(cudaMalloc((void**)&flu, nxp * (nyp + 1) * nzp * sizeof(fluid)));
    CHECK_CUDA(cudaMalloc((void**)&var, nxp * (nyp + 1) * nzp * sizeof(process_variables)));

    CHECK_CUDA(cudaMalloc((void**)&prsrc, nxp * (nyp + 1) * nzp * sizeof(cufftDoubleComplex)));

    dim3 blockDim(8, 8, 8); // ÿ���߳̿��е��߳���
    dim3 gridDim((nxp + blockDim.x - 1) / blockDim.x,
        (nyp + 1 + blockDim.y - 1) / blockDim.y,
        (nzp + blockDim.z - 1) / blockDim.z);// �����еĿ���

    init_gpu_var << < gridDim, blockDim >> > (flu, var);
    CHECK_CUDA(cudaDeviceSynchronize());

    init_mesh_para();

#ifdef Restart
    init_restart(flu, flu_host, var);
#else 
    init_fluid(flu_host);
#endif

    fopen_s(&run_info, run_info_name, "a+");
    if (run_info == nullptr)
    {
        fprintf(stderr, "Failed to open file: %s\n", run_info_name);
        exit(EXIT_FAILURE);
    }
    fprintf(run_info, "restart_start_step %d\n", restart_start_step);
    fclose(run_info);
 
    CHECK_CUDA(cudaMemcpy(flu, flu_host, nxp * (nyp + 1) * nzp * sizeof(fluid), cudaMemcpyHostToDevice));


    calcuate();

    CHECK_CUDA(cudaFree(flu));
    CHECK_CUDA(cudaFree(var));
    CHECK_CUDA(cudaFree(prsrc));
    CHECK_CUDA(cudaFree(ypara_device));
    CHECK_CUDA(cudaFree(dydir_device));
    CHECK_CUDA(cudaFree(rk_device));
    free(flu_host);
    free(ypara_host);
    free(dydir);
    free(rk);

    return 0;
}

