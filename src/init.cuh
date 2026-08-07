#ifndef init
#define init

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "time.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include <iostream>
#include "device_launch_parameters.h"
#include "parameters.h"

extern int restart_start_step;

void init_mesh_para();
void init_fluid(fluid* flu);
void output_velocity(fluid* flu, fluid* flu_host, int time_step);
int Ord3(int x, int y, int z, int nzp, int nxp);
__global__ void init_gpu_var(fluid* flu, process_variables* var);
void output_prgrad(double prgradaver);

void output_restart(fluid* flu, fluid* flu_host, process_variables* var, int time_step);/*for restart*/
void init_restart(fluid* flu, fluid* flu_host, process_variables* var);/*for restart*/
//extern __device__ int d_Ord3(int x, int y, int z, int nzp, int nxp);
void clcstat(fluid* flu, double current_time, int time_step);


#endif
