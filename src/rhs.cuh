#ifndef rhs
#define rhs

#include "cusparse.h"

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

#define CHECK_CUFFT(call) \
    { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            std::cerr << "CUFFT Error at " << __FILE__ << ":" << __LINE__ << " - " << err << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

__device__ int d_Ord3(int x, int y, int z, int nzp, int nxp);
__device__ void d_Ord3r(int id, int* x, int* y, int* z, int nzp, int nxp);


__global__ void Update_VelInterp_uvw(fluid* flu, double* uxhx, double* uyhx, double* uzhx, double* uxhz, double* uyhz, double* uzhz);
__global__ void rhsx(fluid* flu, process_variables* var, dyDir* dydir, Ypara* ypara, double* uxhx, double* uyhx, double* uzhx, double* uxhz, double* uyhz, double* uzhz, int ns, double* s3tot,RK* rk);
__global__ void rhsz(fluid* flu, process_variables* var, dyDir* dydir, Ypara* ypara, double* uxhx, double* uyhx, double* uzhx, double* uxhz, double* uyhz, double* uzhz, int ns, RK* rk);
__global__ void rhsy(fluid* flu, process_variables* var, dyDir* dydir, Ypara* ypara, double* uxhx, double* uyhx, double* uzhx, double* uxhz, double* uyhz, double* uzhz, int ns, RK* rk);

__global__ void correct_rhsx(fluid* flu, process_variables* var, dyDir* dydir, Ypara* ypara, double* s3tot, int ns, RK* rk);

// __global__ void uhat_coe(process_variables* var, Ypara* ypara, int ns, double* a, double* b, double* c, double* d, int k, RK* rk);
// void uhat_clc(fluid* flu, process_variables* var, Ypara* ypara, double* a, double* b, double* c, double* d, int ns, cusparseHandle_t handle, RK* rk);

__global__ void uhat_coe(process_variables* var, Ypara* ypara, int ns, double* a, double* b, double* c, double* d, int k, RK* rk, double u_wall_next);

void uhat_clc(fluid* flu, process_variables* var, Ypara* ypara, double* a, double* b, double* c, double* d, int ns, cusparseHandle_t handle, RK* rk, double u_wall_next);

__global__ void uhat_update(fluid* flu, double* d, int k);

__global__ void vhat_coe(process_variables* var, Ypara* ypara, int ns, double* a, double* b, double* c, double* d, int k, RK* rk);
void vhat_clc(fluid* flu, process_variables* var, Ypara* ypara, double* a, double* b, double* c, double* d, int ns, cusparseHandle_t handle, RK* rk);
__global__ void vhat_update(fluid* flu, double* d, int k);

__global__ void what_coe(process_variables* var, Ypara* ypara, int ns, double* a, double* b, double* c, double* d, int k, RK* rk);
void what_clc(fluid* flu, process_variables* var, Ypara* ypara, double* a, double* b, double* c, double* d, int ns, cusparseHandle_t handle, RK* rk);
__global__ void what_update(fluid* flu, double* d, int k);

//__global__ void bc_velocity(fluid* flu);
__global__ void bc_velocity(fluid* flu, double current_time);

__global__ void bc_prsrc(cufftDoubleComplex* prsrc);

__global__ void clcprsrc(fluid* flu, process_variables* var, dyDir* dydir, int ns, double* divmax, RK* rk, cufftDoubleComplex* prsrc);
void initPPE(double* ak1, double* ak3);
__global__ void clcPPE(fluid* flu, process_variables* var, Ypara* ypara, double* ak1, double* ak3, cufftDoubleComplex* prsrc);
__global__ void bc_presure(fluid* flu);

__global__ void update_velocity(fluid* flu, dyDir* dydir, RK* rk, int ns, cufftDoubleComplex* prsrc);

__global__ void update_v_dir_velocity(fluid* flu, dyDir* dydir, RK* rk, int ns, cufftDoubleComplex* prsrc);

__global__ void update_pressure(fluid* flu, Ypara* ypara, RK* rk, int ns, cufftDoubleComplex* prsrc);

__global__ void test(double* a, double* b, double* c, double* d);
void calcuate();


__global__ void clcPPE_cof(Ypara* ypara, double* ak1, double* ak3, cufftDoubleComplex* prsrc, double* a, double* b, double* c, double* d, double* e, int k);
void clcPPE_1025(Ypara* ypara, double* ak1, double* ak3, cufftDoubleComplex* prsrc, double* a, double* b, double* c, double* d, double* e, cusparseHandle_t handle);
__global__ void clcPPE_update(cufftDoubleComplex* prsrc, double* d, double* e, int k);

__global__ void ForwardElimination(int m, int n, double* aj, double* bj, double* cj, double* fj, double* vecm, double* arrmn);
__global__ void BackwardSubstitution(int m, int n, double* fj, double* arrmn);
void InverseTridiagonalDevice(int m, int n, double* aj, double* bj, double* cj, double* fj);

__global__ void average_xz(fluid* flu, dyDir* dydir, Stat* stat, int nx, int ny, int nz);
#endif