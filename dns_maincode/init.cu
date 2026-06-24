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

#include <random>

#include "src/parameters.h"
#include "src/init.cuh"
#include "src/rhs.cuh"

void init_mesh_para()
{
	memset(ypara_host, 0, (static_cast<size_t>(nyp) + 1) * sizeof(Ypara));
	memset(dydir, 0, (static_cast<size_t>(nyp) + 1) * sizeof(dyDir));
	memset(rk, 0, 3 * sizeof(RK));

	/* 单侧双曲正切(tanh)非均匀网格划分：仅在底部(y=0)极度加密 */
	// double gamma_mesh = 3.5; 
	double gamma_mesh = 4.5; 
	for (int j = 1; j <= nyp; j++)
	{
		double xi = (double(j) - 1.0) / nyc; 
		dydir[j].yp = ylength * (1.0 + tanh(gamma_mesh * (xi - 1.0)) / tanh(gamma_mesh));
	}
	dydir[1].yp = 0.0;
	dydir[nyp].yp = ylength;

	/*define para related to y-dir (保留原始计算逻辑，自适应不对称)*/
	for (int j = 1; j <= nyc; j++)
	{
		dydir[j].yc = 0.5 * (dydir[j].yp + dydir[j + 1].yp);
		dydir[j].dyp = dydir[j + 1].yp - dydir[j].yp;
	}
	dydir[0].yc = 2 * dydir[1].yp - dydir[1].yc;
	dydir[nyp].yc = 2 * dydir[nyp].yp - dydir[nyc].yc;

	dydir[0].dyp = dydir[1].dyp;
	dydir[nyp].dyp = dydir[nyc].dyp;

	for (int j = 0; j <= nyp; j++)
	{
		dydir[j].volcell = dydir[j].dyp * dx * dz;
		dydir[j].deltacell = pow(dydir[j].dyp * dx * dz, 1.0 / 3.0);
		dydir[j].rdyp = 1.0 / dydir[j].dyp;
	}

	for (int j = 2; j <= nyc; j++)
	{
		dydir[j].dyc = dydir[j].yc - dydir[j - 1].yc;
	}
	dydir[1].dyc = dydir[1].dyp;
	dydir[nyp].dyc = dydir[nyc].dyp;

	for (int j = 1; j <= nyp; j++)
	{
		dydir[j].rdyc = 1.0 / dydir[j].dyc;
	}

	for (int j = 1; j <= nyc; j++)
	{
		ypara_host[j].am2ph = dydir[j].rdyp * dydir[j].rdyc;
		ypara_host[j].ap2ph = dydir[j].rdyp * dydir[j + 1].rdyc;
	}
	ypara_host[1].am2ph = 0.0;
	ypara_host[1].ap2ph = dydir[1].rdyp * dydir[2].rdyc;
	ypara_host[nyc].am2ph = dydir[nyc].rdyp * dydir[nyc].rdyc;
	ypara_host[nyc].ap2ph = 0.0;

	for (int j = 1; j <= nyc; j++)
	{
		ypara_host[j].ac2ph = -(ypara_host[j].am2ph + ypara_host[j].ap2ph);
	}

	for (int j = 1; j <= nyc; j++)
	{
		ypara_host[j].am2c = dydir[j].rdyp * dydir[j].rdyc;
		ypara_host[j].ap2c = dydir[j].rdyp * dydir[j + 1].rdyc;
	}
	ypara_host[1].am2c = 4.0 * dydir[1].rdyc / (dydir[1].dyc + 2.0 * dydir[2].dyc);
	ypara_host[1].ap2c = 4.0 * dydir[2].rdyc / (dydir[1].dyc + 2.0 * dydir[2].dyc);
	ypara_host[nyc].am2c = 4.0 * dydir[nyc].rdyc / (dydir[nyp].dyc + 2.0 * dydir[nyc].dyc);
	ypara_host[nyc].ap2c = 4.0 * dydir[nyp].rdyc / (dydir[nyp].dyc + 2.0 * dydir[nyc].dyc);

	for (int j = 1; j <= nyc; j++)
	{
		ypara_host[j].ac2c = -(ypara_host[j].am2c + ypara_host[j].ap2c);
		ypara_host[j].ap2cForCN = ypara_host[j].ap2c;
		ypara_host[j].ac2cForCN = ypara_host[j].ac2c;
		ypara_host[j].am2cForCN = ypara_host[j].am2c;
	}
	ypara_host[1].ap2cForCN = ypara_host[1].ap2c;
	ypara_host[1].ac2cForCN = ypara_host[1].ac2c - ypara_host[1].am2c;
	ypara_host[1].am2cForCN = 2 * ypara_host[1].am2c;

	ypara_host[nyc].am2cForCN = ypara_host[nyc].am2c;
	ypara_host[nyc].ac2cForCN = ypara_host[nyc].ac2c - ypara_host[nyc].ap2c;
	ypara_host[nyc].ap2cForCN = 2 * ypara_host[nyc].ap2c;

	for (int j = 1; j <= nyc; j++)
	{
		ypara_host[j].am2p = dydir[j].rdyc * dydir[j - 1].rdyp;
		ypara_host[j].ap2p = dydir[j].rdyc * dydir[j].rdyp;
		ypara_host[j].ac2p = -(ypara_host[j].am2p + ypara_host[j].ap2p);
	}

	for (int j = 1; j <= nyp; j++)
	{
		ypara_host[j].yinterpCoe = dydir[j].dyp / (dydir[j].dyp + dydir[j - 1].dyp);
	}

	rk[0].alpha = 8.0 / 15.0 * dt;
	rk[1].alpha = 2.0 / 15.0 * dt;
	rk[2].alpha = 1.0 / 3.0 * dt;

	rk[0].gamma = 8.0 / 15.0 * dt;
	rk[1].gamma = 5.0 / 12.0 * dt;
	rk[2].gamma = 3.0 / 4.0 * dt;

	rk[0].theta = 0.0 * dt;
	rk[1].theta = -17.0 / 60.0 * dt;
	rk[2].theta = -5.0 / 12.0 * dt;

	CHECK_CUDA(cudaMalloc((void**)&ypara_device, (nyp + 1) * sizeof(Ypara)));
	CHECK_CUDA(cudaMalloc((void**)&dydir_device, (nyp + 1) * sizeof(dyDir)));
	CHECK_CUDA(cudaMalloc((void**)&rk_device, 3 * sizeof(RK)));
	cudaMemcpy(ypara_device, ypara_host, (nyp + 1) * sizeof(Ypara), cudaMemcpyHostToDevice);
	cudaMemcpy(dydir_device, dydir, (nyp + 1) * sizeof(dyDir), cudaMemcpyHostToDevice);
	cudaMemcpy(rk_device, rk, 3 * sizeof(RK), cudaMemcpyHostToDevice);
}

void init_fluid(fluid* flu)
{
	int ic, jc, kc;
	// double height = 0.5 * ylength;//全槽道
	double height = ylength;//半槽道
	// 保留你原本严格验证过的雷诺数计算逻辑，但当 ubulk = 0 时规避 NaN 运算
	double Re = 1.0; 
	if (fabs(ubulk) > 1e-12) {
		Re = ubulk * height / nu;
	}
	double Retau = 0.1538 * pow(Re, 0.887741);
	double utau = Retau * nu / height;

	double uzmean[nyp] = { 0.0 };
	double uxmean = 0.0;
	
	double wx = 2.0 * PI / 500.0;
	double wz = 2.0 * PI / 200.0;

	if (fabs(ubulk) > 1e-12) {
		double xlength_plus = xlength * utau / nu;
		double zlength_plus = zlength * utau / nu;

		int m1 = (int)(xlength_plus * wx / (2.0 * PI)) + 1;
		int m2 = (int)(zlength_plus * wz / (2.0 * PI)) + 1;

		wx = m1 * 2.0 * PI / xlength_plus;
		wz = m2 * 2.0 * PI / zlength_plus;
	}

	std::random_device rd;  
	std::mt19937 gen(rd());  
	std::uniform_real_distribution<double> dis(0.9, 1.1);  

	for (jc = 1; jc < nyp; jc++)
	{
		uzmean[jc] = 0.0;
		double yct = dydir[jc].yc;
		double ybar = yct / height;
		// if (ybar > 1.0) ybar = 2.0 - ybar;//全槽道

		for (kc = 0; kc < nzp; kc++)
		{
			double zp = kc * dz + dz * 0.5;
			double zplus = utau * zp / nu;
			for (ic = 0; ic < nxp; ic++)
			{
				double random_number = dis(gen);
				double xp = ic * dx + dx * 0.5;
				double xplus = utau * xp / nu;

				flu[Ord3(ic, jc, kc, nzp, nxp)].u = ubulk * ybar * exp(-4.5 * ybar * ybar) * cos(wz * zplus) * random_number + 3.0 * ubulk * (ybar - 0.5 * ybar * ybar);
				flu[Ord3(ic, jc, kc, nzp, nxp)].v = 0.0;
				flu[Ord3(ic, jc, kc, nzp, nxp)].w = ubulk * ybar * exp(-4.5 * ybar * ybar) * sin(wx * xplus) * random_number;

				uzmean[jc] = uzmean[jc] + flu[Ord3(ic, jc, kc, nzp, nxp)].w;
				uxmean = uxmean + flu[Ord3(ic, jc, kc, nzp, nxp)].u * dydir[jc].dyp;
			}
		}
		uzmean[jc] = uzmean[jc] / (nxzc);
	}
	
  if (fabs(ubulk) > 1e-12) {
		uxmean = uxmean / (nxzc * ylength);
	}

	std::uniform_real_distribution<double> noise(-1.0, 1.0);


	double amp = bypass_perturbation_amp; 

	double Lx = nxp * dx;
	double Lz = nzp * dz;
	double kx1 = 2.0 * PI / Lx;
	double kz1 = 2.0 * PI / Lz;
	double kx2 = 2.0 * kx1;
	double kz2 = kz1;
	double kx3 = kx1;
	double kz3 = 2.0 * kz1;


	double local_delta = sqrt(2.0 * nu / omega);

	for (jc = 1; jc < nyp; jc++)
	{

		double y = dydir[jc].yc;
		double y_star = y / local_delta;

		double f_y = 1.85 * (y_star * y_star) * exp(-y_star);
		double near_wall_window = 1.0 - exp(-pow(y_star / 0.20, 4.0));
		f_y *= near_wall_window;

		for (kc = 0; kc < nzp; kc++)
		{
			double zp = kc * dz + 0.5 * dz;
			for (ic = 0; ic < nxp; ic++)
			{
				double xp = ic * dx + 0.5 * dx;

				if (fabs(ubulk) < 1e-12) {
					double mode11 = sin(kz1 * zp) * cos(kx1 * xp);
					double mode21 = sin(kz2 * zp + 0.35) * cos(kx2 * xp + 0.20);
					double mode12 = sin(kz3 * zp + 0.70) * cos(kx3 * xp - 0.15);

					double vort11 = cos(kz1 * zp) * sin(kx1 * xp);
					double vort21 = cos(kz2 * zp + 0.35) * sin(kx2 * xp + 0.20);
					double vort12 = cos(kz3 * zp + 0.70) * sin(kx3 * xp - 0.15);

					double wall_normal = sin(kx1 * xp + 0.40) * sin(kz1 * zp - 0.25)
						+ 0.50 * sin(kx2 * xp - 0.10) * sin(kz2 * zp + 0.45)
						+ 0.35 * sin(kx3 * xp + 0.25) * sin(kz3 * zp + 0.10);

					double seed_noise = noise(gen);
					double noise_envelope = exp(-2.0 * y_star);
					flu[Ord3(ic, jc, kc, nzp, nxp)].u = amp * noise_envelope * seed_noise
						+ amp * f_y * (mode11 + 0.45 * mode21 + 0.30 * mode12);
					flu[Ord3(ic, jc, kc, nzp, nxp)].v = 0.25 * amp * f_y * wall_normal;
					flu[Ord3(ic, jc, kc, nzp, nxp)].w = 0.50 * amp * noise_envelope * noise(gen)
						- amp * f_y * (
						(kx1 / kz1) * vort11
						+ 0.45 * (kx2 / kz2) * vort21
						+ 0.30 * (kx3 / kz3) * vort12);
				}
				else {
					flu[Ord3(ic, jc, kc, nzp, nxp)].u = flu[Ord3(ic, jc, kc, nzp, nxp)].u * ubulk / uxmean - uCRF;
					flu[Ord3(ic, jc, kc, nzp, nxp)].w = flu[Ord3(ic, jc, kc, nzp, nxp)].w - uzmean[jc];
				}
			}
		}
	}

	/* 边界初始化：顶部滑移，底部 Stokes */
	for (kc = 0; kc < nzp; kc++)
		for (ic = 0; ic < nxp; ic++)
		{
			flu[Ord3(ic, nyp, kc, nzp, nxp)].u = flu[Ord3(ic, nyc, kc, nzp, nxp)].u;
			flu[Ord3(ic, nyp, kc, nzp, nxp)].w = flu[Ord3(ic, nyc, kc, nzp, nxp)].w;
			flu[Ord3(ic, nyp, kc, nzp, nxp)].v = 0.0;  

			double initial_u_wall = U_osc * cos(0.0) - uCRF;
			flu[Ord3(ic, 0, kc, nzp, nxp)].u = 2.0 * initial_u_wall - flu[Ord3(ic, 1, kc, nzp, nxp)].u;
			flu[Ord3(ic, 0, kc, nzp, nxp)].w = -flu[Ord3(ic, 1, kc, nzp, nxp)].w;
			flu[Ord3(ic, 1, kc, nzp, nxp)].v = 0.0;   
		}

	for (jc = 0; jc <= nyp; jc++)
		for (kc = 0; kc < nzp; kc++)
			for (ic = 0; ic < nxp; ic++)
			{
				flu[Ord3(ic, jc, kc, nzp, nxp)].p = 0.0;
			}

	FILE* fp = NULL;
	char filename[100];
	char flu_name[100];

	sprintf_s(flu_name, "mesh.dat");
	sprintf_s(filename, output_path);
	strcat_s(filename, flu_name);
	fopen_s(&fp, filename, "w+");

	for (ic = 0; ic < nxp; ic = ic + xskip) { fprintf(fp, "%e\n", ic * dx); }
	for (jc = 0; jc <= nyp; jc = jc + yskip) { fprintf(fp, "%e\n", dydir[jc].yc); }
	for (kc = 0; kc < nzp; kc = kc + zskip) { fprintf(fp, "%e\n", kc * dz); }

	fclose(fp);
}

int Ord3(int x, int y, int z, int nzp, int nxp)
{
	return x + z * nxp + (nxp * nzp) * y;
}

void output_velocity(fluid* flu, fluid* flu_host, int t)
{
	int ic, jc, kc;
	int xyzsize = nxp * (nyp + 1) * nzp;

	//fluid* flu_host = (fluid*)malloc(xyzsize * sizeof(fluid));
	cudaMemcpy(flu_host, flu, xyzsize * sizeof(fluid), cudaMemcpyDeviceToHost);

	FILE* fp = NULL;//�ļ�ָ��
	char filename[100];//�ļ���
	char flu_name[100];

#ifdef Restart
	sprintf_s(flu_name, "restart%d.dat", t);
#else
	sprintf_s(flu_name, "dns_data%d.dat", t);
#endif // Restart

	sprintf_s(filename, output_path);
	strcat_s(filename, flu_name);
	fopen_s(&fp, filename, "w+");

	for (jc = 0; jc <= nyp; jc = jc + yskip)
		for (kc = 0; kc < nzp; kc = kc + zskip)
			for (ic = 0; ic < nxp; ic = ic + xskip)
			{
				//fprintf(fp, "%d %d %e %e %d\n", flu[i].x, flu[i].y, flu[i].ux, flu[i].uy, flu[i].type);
				fprintf(fp, "%e %e %e %e\n", flu_host[Ord3(ic, jc, kc, nzp, nxp)].u, flu_host[Ord3(ic, jc, kc, nzp, nxp)].v, flu_host[Ord3(ic, jc, kc, nzp, nxp)].w, flu_host[Ord3(ic, jc, kc, nzp, nxp)].p);
				//fprintf(fp, "%e %e %e\n", ic * dx, dydir[jc].yc, kc * dz);
			}
	fclose(fp);
}

__global__ void init_gpu_var(fluid* flu, process_variables* var)
{
	int ic = blockIdx.x * blockDim.x + threadIdx.x;
	int jc = blockIdx.y * blockDim.y + threadIdx.y;
	int kc = blockIdx.z * blockDim.z + threadIdx.z;

	if (ic < nxp && jc <= nyp && kc < nzp) {
		flu[ic + kc * nxp + (nxp * nzp) * jc].p = 0.0;
		flu[ic + kc * nxp + (nxp * nzp) * jc].u = 0.0;
		flu[ic + kc * nxp + (nxp * nzp) * jc].v = 0.0;
		flu[ic + kc * nxp + (nxp * nzp) * jc].w = 0.0;

		var[ic + kc * nxp + (nxp * nzp) * jc].rhsx = 0.0;
		var[ic + kc * nxp + (nxp * nzp) * jc].rhsy = 0.0;
		var[ic + kc * nxp + (nxp * nzp) * jc].rhsz = 0.0;
		var[ic + kc * nxp + (nxp * nzp) * jc].uold = 0.0;
		var[ic + kc * nxp + (nxp * nzp) * jc].vold = 0.0;
		var[ic + kc * nxp + (nxp * nzp) * jc].wold = 0.0;
	}
}

void output_prgrad(double prgradaver)
{
	FILE* fp = NULL;//�ļ�ָ��
	char filename[100];//�ļ���
	char flu_name[100];
	sprintf_s(flu_name, "prgrad.dat");
	sprintf_s(filename, output_path);
	strcat_s(filename, flu_name);
	fopen_s(&fp, filename, "a+");

	fprintf(fp, "%e\n", prgradaver);

	fclose(fp);
}

void output_restart(fluid* flu, fluid* flu_host, process_variables* var)
{
	int ic, jc, kc;
	int xyzsize = nxp * (nyp + 1) * nzp;

	cudaMemcpy(flu_host, flu, xyzsize * sizeof(fluid), cudaMemcpyDeviceToHost);
	process_variables* var_for_output = (process_variables*)malloc(xyzsize * sizeof(process_variables));
	cudaMemcpy(var_for_output, var, xyzsize * sizeof(process_variables), cudaMemcpyDeviceToHost);


	FILE* fp = NULL;//�ļ�ָ��
	char filename[100];//�ļ���
	char flu_name[100];

	sprintf_s(flu_name, "restart.dat");
	sprintf_s(filename, output_path);
	strcat_s(filename, flu_name);
	fopen_s(&fp, filename, "w+");

	for (jc = 0; jc <= nyp; jc = jc + yskip)
		for (kc = 0; kc < nzp; kc = kc + zskip)
			for (ic = 0; ic < nxp; ic = ic + xskip)
			{
				fprintf(fp, "%e %e %e %e %e %e %e\n", flu_host[Ord3(ic, jc, kc, nzp, nxp)].u, flu_host[Ord3(ic, jc, kc, nzp, nxp)].v, flu_host[Ord3(ic, jc, kc, nzp, nxp)].w, flu_host[Ord3(ic, jc, kc, nzp, nxp)].p, var_for_output[Ord3(ic, jc, kc, nzp, nxp)].uold, var_for_output[Ord3(ic, jc, kc, nzp, nxp)].vold, var_for_output[Ord3(ic, jc, kc, nzp, nxp)].wold);
			}
	fclose(fp);

	free(var_for_output);
}

void init_restart(fluid* flu, fluid* flu_host, process_variables* var)
{
	std::ifstream file("E:\\lch\\restart.dat");  // ���ļ�
	if (!file) {
		std::cerr << "Unable to open file: " << std::endl;
		return;
	}

	int ic, jc, kc;
	int xyzsize = nxp * (nyp + 1) * nzp;
	process_variables* var_for_input = (process_variables*)malloc(xyzsize * sizeof(process_variables));

	for (jc = 0; jc <= nyp; jc = jc + yskip)
		for (kc = 0; kc < nzp; kc = kc + zskip)
			for (ic = 0; ic < nxp; ic = ic + xskip)
			{
				if (!(file
					>> flu_host[Ord3(ic, jc, kc, nzp, nxp)].u
					>> flu_host[Ord3(ic, jc, kc, nzp, nxp)].v
					>> flu_host[Ord3(ic, jc, kc, nzp, nxp)].w
					>> flu_host[Ord3(ic, jc, kc, nzp, nxp)].p
					>> var_for_input[Ord3(ic, jc, kc, nzp, nxp)].uold
					>> var_for_input[Ord3(ic, jc, kc, nzp, nxp)].vold
					>> var_for_input[Ord3(ic, jc, kc, nzp, nxp)].wold)
					)
				{
					std::cerr << "Error reading data" << std::endl;
					break;  // �ļ������ݲ���ʱ��ǰ�˳�
				}
			}

	int iu, ku;

	/* Interpolate the skipped restart points for xskip = 2, yskip = 1, zskip = 2. */
	for (jc = 0; jc <= nyp; jc = jc + yskip)
		for (kc = 0; kc < nzp; kc = kc + zskip)
			for (ic = 0; ic < nxp; ic = ic + xskip)
			{
				iu = ic + xskip;
				ku = kc + zskip;
				if (ic == nxp - 2) { iu = 0; }
				if (kc == nzp - 2) { ku = 0; }

				flu_host[Ord3(ic + 1, jc, kc, nzp, nxp)].u = 0.5 * (flu_host[Ord3(ic, jc, kc, nzp, nxp)].u + flu_host[Ord3(iu, jc, kc, nzp, nxp)].u);
				flu_host[Ord3(ic, jc, kc + 1, nzp, nxp)].u = 0.5 * (flu_host[Ord3(ic, jc, kc, nzp, nxp)].u + flu_host[Ord3(ic, jc, ku, nzp, nxp)].u);
				flu_host[Ord3(ic + 1, jc, kc + 1, nzp, nxp)].u = 0.25 * (
					flu_host[Ord3(ic, jc, kc, nzp, nxp)].u +
					flu_host[Ord3(ic, jc, ku, nzp, nxp)].u +
					flu_host[Ord3(iu, jc, kc, nzp, nxp)].u +
					flu_host[Ord3(iu, jc, ku, nzp, nxp)].u);

				flu_host[Ord3(ic + 1, jc, kc, nzp, nxp)].v = 0.5 * (flu_host[Ord3(ic, jc, kc, nzp, nxp)].v + flu_host[Ord3(iu, jc, kc, nzp, nxp)].v);
				flu_host[Ord3(ic, jc, kc + 1, nzp, nxp)].v = 0.5 * (flu_host[Ord3(ic, jc, kc, nzp, nxp)].v + flu_host[Ord3(ic, jc, ku, nzp, nxp)].v);
				flu_host[Ord3(ic + 1, jc, kc + 1, nzp, nxp)].v = 0.25 * (
					flu_host[Ord3(ic, jc, kc, nzp, nxp)].v +
					flu_host[Ord3(ic, jc, ku, nzp, nxp)].v +
					flu_host[Ord3(iu, jc, kc, nzp, nxp)].v +
					flu_host[Ord3(iu, jc, ku, nzp, nxp)].v);

				flu_host[Ord3(ic + 1, jc, kc, nzp, nxp)].w = 0.5 * (flu_host[Ord3(ic, jc, kc, nzp, nxp)].w + flu_host[Ord3(iu, jc, kc, nzp, nxp)].w);
				flu_host[Ord3(ic, jc, kc + 1, nzp, nxp)].w = 0.5 * (flu_host[Ord3(ic, jc, kc, nzp, nxp)].w + flu_host[Ord3(ic, jc, ku, nzp, nxp)].w);
				flu_host[Ord3(ic + 1, jc, kc + 1, nzp, nxp)].w = 0.25 * (
					flu_host[Ord3(ic, jc, kc, nzp, nxp)].w +
					flu_host[Ord3(ic, jc, ku, nzp, nxp)].w +
					flu_host[Ord3(iu, jc, kc, nzp, nxp)].w +
					flu_host[Ord3(iu, jc, ku, nzp, nxp)].w);

				flu_host[Ord3(ic + 1, jc, kc, nzp, nxp)].p = 0.5 * (flu_host[Ord3(ic, jc, kc, nzp, nxp)].p + flu_host[Ord3(iu, jc, kc, nzp, nxp)].p);
				flu_host[Ord3(ic, jc, kc + 1, nzp, nxp)].p = 0.5 * (flu_host[Ord3(ic, jc, kc, nzp, nxp)].p + flu_host[Ord3(ic, jc, ku, nzp, nxp)].p);
				flu_host[Ord3(ic + 1, jc, kc + 1, nzp, nxp)].p = 0.25 * (
					flu_host[Ord3(ic, jc, kc, nzp, nxp)].p +
					flu_host[Ord3(ic, jc, ku, nzp, nxp)].p +
					flu_host[Ord3(iu, jc, kc, nzp, nxp)].p +
					flu_host[Ord3(iu, jc, ku, nzp, nxp)].p);

				var_for_input[Ord3(ic + 1, jc, kc, nzp, nxp)].uold = 0.5 * (var_for_input[Ord3(ic, jc, kc, nzp, nxp)].uold + var_for_input[Ord3(iu, jc, kc, nzp, nxp)].uold);
				var_for_input[Ord3(ic, jc, kc + 1, nzp, nxp)].uold = 0.5 * (var_for_input[Ord3(ic, jc, kc, nzp, nxp)].uold + var_for_input[Ord3(ic, jc, ku, nzp, nxp)].uold);
				var_for_input[Ord3(ic + 1, jc, kc + 1, nzp, nxp)].uold = 0.25 * (
					var_for_input[Ord3(ic, jc, kc, nzp, nxp)].uold +
					var_for_input[Ord3(ic, jc, ku, nzp, nxp)].uold +
					var_for_input[Ord3(iu, jc, kc, nzp, nxp)].uold +
					var_for_input[Ord3(iu, jc, ku, nzp, nxp)].uold);

				var_for_input[Ord3(ic + 1, jc, kc, nzp, nxp)].vold = 0.5 * (var_for_input[Ord3(ic, jc, kc, nzp, nxp)].vold + var_for_input[Ord3(iu, jc, kc, nzp, nxp)].vold);
				var_for_input[Ord3(ic, jc, kc + 1, nzp, nxp)].vold = 0.5 * (var_for_input[Ord3(ic, jc, kc, nzp, nxp)].vold + var_for_input[Ord3(ic, jc, ku, nzp, nxp)].vold);
				var_for_input[Ord3(ic + 1, jc, kc + 1, nzp, nxp)].vold = 0.25 * (
					var_for_input[Ord3(ic, jc, kc, nzp, nxp)].vold +
					var_for_input[Ord3(ic, jc, ku, nzp, nxp)].vold +
					var_for_input[Ord3(iu, jc, kc, nzp, nxp)].vold +
					var_for_input[Ord3(iu, jc, ku, nzp, nxp)].vold);

				var_for_input[Ord3(ic + 1, jc, kc, nzp, nxp)].wold = 0.5 * (var_for_input[Ord3(ic, jc, kc, nzp, nxp)].wold + var_for_input[Ord3(iu, jc, kc, nzp, nxp)].wold);
				var_for_input[Ord3(ic, jc, kc + 1, nzp, nxp)].wold = 0.5 * (var_for_input[Ord3(ic, jc, kc, nzp, nxp)].wold + var_for_input[Ord3(ic, jc, ku, nzp, nxp)].wold);
				var_for_input[Ord3(ic + 1, jc, kc + 1, nzp, nxp)].wold = 0.25 * (
					var_for_input[Ord3(ic, jc, kc, nzp, nxp)].wold +
					var_for_input[Ord3(ic, jc, ku, nzp, nxp)].wold +
					var_for_input[Ord3(iu, jc, kc, nzp, nxp)].wold +
					var_for_input[Ord3(iu, jc, ku, nzp, nxp)].wold);


			}


	CHECK_CUDA(cudaMemcpy(var, var_for_input, xyzsize * sizeof(process_variables), cudaMemcpyHostToDevice));

	free(var_for_input);
}

void clcstat(fluid* flu, double current_time, int time_step)
{
	int block = 256;
	int grid = (nyp + 1 + block - 1) / block;
	Stat* stat = (Stat*)malloc((nyp + 1) * sizeof(Stat));
	Stat* stat_dev;
	CHECK_CUDA(cudaMalloc((void**)&stat_dev, (nyp + 1) * sizeof(Stat)));

	average_xz << < grid, block >> > (flu, dydir_device,stat_dev, nxp, nyp + 1, nzp);
	CHECK_CUDA(cudaMemcpy(stat, stat_dev, (nyp + 1) * sizeof(Stat), cudaMemcpyDeviceToHost));

	int jc;
 
	FILE* fp = NULL;//�ļ�ָ��
	char filename[100];//�ļ���
	char flu_name[100];

	sprintf_s(flu_name, "stat.dat");

	sprintf_s(filename, output_path);
	strcat_s(filename, flu_name);
	fopen_s(&fp, filename, "a+");// “a+”追加读写

	for (int ii = 0; ii < 11; ii++) {
		for (jc = 1; jc < nyp; jc++)
		{
			switch (ii)
			{
			case 0:
				fprintf(fp, "%.15e ", stat[jc].wx); break;
			case 1:
				fprintf(fp, "%.15e ", stat[jc].wy); break;
			case 2:
				fprintf(fp, "%.15e ", stat[jc].wz); break;
			case 3:
				fprintf(fp, "%.15e ", stat[jc].um); break;
			case 4:
				fprintf(fp, "%.15e ", stat[jc].vm); break;
			case 5:
				fprintf(fp, "%.15e ", stat[jc].wm); break;
			case 6:
				fprintf(fp, "%.15e ", stat[jc].pm); break;
			case 7:
				fprintf(fp, "%.15e ", stat[jc].um2); break;
			case 8:
				fprintf(fp, "%.15e ", stat[jc].vm2); break;
			case 9:
				fprintf(fp, "%.15e ", stat[jc].wm2); break;
			case 10:
				fprintf(fp, "%.15e ", stat[jc].pm2); break;
			}
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf_s(flu_name, "tau_wall.dat");
	sprintf_s(filename, output_path);
	strcat_s(filename, flu_name);
	fopen_s(&fp, filename, "a+");

	double y1 = dydir[1].yc;
	double y2 = dydir[2].yc;
	double c0 = (y1 + y2) / (y1 * y2);
	double c1 = -y2 / (y1 * (y2 - y1));
	double c2 = y1 / (y2 * (y2 - y1));
	double u_wall = U_osc * cos(omega * current_time) - uCRF;
	double tau_fd2 = nu * (c0 * u_wall + c1 * stat[1].um + c2 * stat[2].um);

	fprintf(fp, "%.15e %.15e %.15e %.15e %.15e\n",
		current_time, u_wall, tau_fd2, stat[1].um, stat[2].um);
	fclose(fp);

	if (output_tau_wall_maps && time_step % tau_wall_map_interval == 0)
	{
		int xyzsize = nxp * (nyp + 1) * nzp;
		CHECK_CUDA(cudaMemcpy(flu_host, flu, xyzsize * sizeof(fluid), cudaMemcpyDeviceToHost));

		sprintf_s(flu_name, "tau_wall_map_%08d.dat", time_step);
		sprintf_s(filename, output_path);
		strcat_s(filename, flu_name);
		fopen_s(&fp, filename, "w+");

		fprintf(fp, "%.15e %.15e %d %d\n", current_time, u_wall, nxp, nzp);
		for (int kc = 0; kc < nzp; kc++)
		{
			for (int ic = 0; ic < nxp; ic++)
			{
				double u1_local = flu_host[Ord3(ic, 1, kc, nzp, nxp)].u;
				double u2_local = flu_host[Ord3(ic, 2, kc, nzp, nxp)].u;
				double tau_local = nu * (c0 * u_wall + c1 * u1_local + c2 * u2_local);
				fprintf(fp, "%.15e ", tau_local);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	free(stat);
	cudaFree(stat_dev);
}
