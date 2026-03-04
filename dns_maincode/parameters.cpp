#include "src/parameters.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include "cufft.h"

fluid* flu;

fluid* flu_host = (fluid*)malloc(nxp * (nyp + 1) * nzp * sizeof(fluid));

process_variables* var;

cufftDoubleComplex* prsrc;

Ypara* ypara_host = (Ypara*)malloc((nyp + 1) * sizeof(Ypara));// 为了方便 将所有系数数组大小设置为一致  在实际运行过程中 会有不使用  且在主机上可以定义
dyDir* dydir = (dyDir*)malloc((nyp + 1) * sizeof(dyDir));
RK* rk = (RK*)malloc(3 * sizeof(RK));

Ypara* ypara_device;
dyDir* dydir_device;
RK* rk_device;

char output_path[100];

