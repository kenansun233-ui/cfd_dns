#include <iostream>
#include "info_device.cuh"// ����cudaͷ�ļ�
#include "parameters.h"
#include "init.cuh"
#include "rhs.cuh"
#include "platform_compat.h"


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
#include "device_launch_parameters.h"
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

static void make_dir_if_needed(const char* path)
{
    if (path == nullptr || path[0] == '\0') {
        return;
    }

    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "Failed to create directory: %s\n", path);
        exit(EXIT_FAILURE);
    }
}

static void make_dirs_recursive(const char* path)
{
    char current[512];
    const int written = snprintf(current, sizeof(current), "%s", path);
    if (written < 0 || static_cast<size_t>(written) >= sizeof(current)) {
        fprintf(stderr, "Output path is too long: %s\n", path);
        exit(EXIT_FAILURE);
    }

    const size_t len = strlen(current);
    if (len == 0) {
        return;
    }
    if (current[len - 1] == '/') {
        current[len - 1] = '\0';
    }

    for (char* p = current + 1; *p != '\0'; ++p) {
        if (*p == '/') {
            *p = '\0';
            make_dir_if_needed(current);
            *p = '/';
        }
    }
    make_dir_if_needed(current);
}

static void set_output_path(int argc, char** argv)
{
    const char* run_dir = (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') ? argv[1] : "default";
    if (run_dir[0] == '/' || strstr(run_dir, "..") != nullptr) {
        fprintf(stderr, "Invalid run directory: %s\n", run_dir);
        exit(EXIT_FAILURE);
    }

    const int written = snprintf(output_path, sizeof(output_path), "./OutputFile/%s/", run_dir);
    if (written < 0 || static_cast<size_t>(written) >= sizeof(output_path)) {
        fprintf(stderr, "Output path is too long for run directory: %s\n", run_dir);
        exit(EXIT_FAILURE);
    }
    make_dirs_recursive(output_path);
}

int main(int argc, char** argv)
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
    CHECK_CUDA(cudaSetDevice(0));

    set_output_path(argc, argv);

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

    char run_info_name[512];
    sprintf_s(run_info_name, "%srun_info.dat", output_path);
    FILE* run_info = nullptr;
    fopen_s(&run_info, run_info_name, "w");
    if (run_info == nullptr)
    {
        fprintf(stderr, "Failed to open file: %s\n", run_info_name);
        exit(EXIT_FAILURE);
    }

    fprintf(run_info, "Re_omega %.15e\n", Re_omega);
    fprintf(run_info, "U_osc %.15e\n", U_osc);
    fprintf(run_info, "omega %.15e\n", omega);
    fprintf(run_info, "nu %.15e\n", nu);
    fprintf(run_info, "dt %.15e\n", dt);
    fprintf(run_info, "timemax %d\n", timemax);
    fprintf(run_info, "stat_output_interval %d\n", stat_output_interval);
    fprintf(run_info, "nxp %d\n", nxp);
    fprintf(run_info, "nyp %d\n", nyp);
    fprintf(run_info, "nzp %d\n", nzp);
    fprintf(run_info, "xlength %.15e\n", xlength);
    fprintf(run_info, "ylength %.15e\n", ylength);
    fprintf(run_info, "zlength %.15e\n", zlength);
    fprintf(run_info, "enable_wall_oscillation %d\n", enable_wall_oscillation ? 1 : 0);
    fprintf(run_info, "enable_bulk_pressure_feedback %d\n", enable_bulk_pressure_feedback ? 1 : 0);
    fprintf(run_info, "restart_start_step %d\n", restart_start_step);
    fprintf(run_info, "oscillation_start_step %d\n", restart_continues_oscillation_time ? restart_start_step : 0);
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

