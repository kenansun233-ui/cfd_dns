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

#include "src/parameters.h"
#include "src/init.cuh"
#include "src/rhs.cuh"

#include "cufft.h"


__device__ int d_Ord3(int x, int y, int z, int nzp, int nxp) {
	return x + z * nxp + (nxp * nzp) * y;
}

__device__ void d_Ord3r(int id, int* x, int* y, int* z, int nzp, int nxp) {
	*y = id / (nxp * nzp);
	*x = id % nxp;
	*z = (id / nxp) % nzp;
	return;
}

__global__ void Update_VelInterp_uvw(fluid* flu, double* uxhx, double* uyhx, double* uzhx, double* uxhz, double* uyhz, double* uzhz)
{

	int ic = blockIdx.x * blockDim.x + threadIdx.x;
	int jc = blockIdx.y * blockDim.y + threadIdx.y;
	int kc = blockIdx.z * blockDim.z + threadIdx.z;

	int ks = kc - 2, km = kc - 1, kp = kc + 1, ku = kc + 2;
	int is = ic - 2, im = ic - 1, ip = ic + 1, iu = ic + 2;

	if (ic == 0) { im = nxp - 1; is = nxp - 2; }
	if (kc == 0) { km = nzp - 1; ks = nzp - 2; }
	if (ic == nxp - 1) { ip = 0; iu = 1; }
	if (kc == nzp - 1) { kp = 0; ku = 1; }

	if (ic == 1) { im = 0; is = nxp - 1; }
	if (kc == 1) { km = 0; ks = nzp - 1; }
	if (ic == nxp - 2) { ip = nxp - 1; iu = 0; }
	if (kc == nzp - 2) { kp = nzp - 1; ku = 0; }

	int id = d_Ord3(ic, jc, kc, nzp, nxp);

	if (ic < nxp && jc <= nyp && kc < nzp)
	{
		uxhx[id] = interpcoe1 * flu[d_Ord3(im, jc, kc, nzp, nxp)].u + interpcoe2 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].u + interpcoe3 * flu[d_Ord3(ip, jc, kc, nzp, nxp)].u + interpcoe4 * flu[d_Ord3(iu, jc, kc, nzp, nxp)].u;
		uyhx[id] = interpcoe1 * flu[d_Ord3(is, jc, kc, nzp, nxp)].v + interpcoe2 * flu[d_Ord3(im, jc, kc, nzp, nxp)].v + interpcoe3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].v + interpcoe4 * flu[d_Ord3(ip, jc, kc, nzp, nxp)].v;
		uzhx[id] = interpcoe1 * flu[d_Ord3(is, jc, kc, nzp, nxp)].w + interpcoe2 * flu[d_Ord3(im, jc, kc, nzp, nxp)].w + interpcoe3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].w + interpcoe4 * flu[d_Ord3(ip, jc, kc, nzp, nxp)].w;

		uxhz[id] = interpcoe1 * flu[d_Ord3(ic, jc, ks, nzp, nxp)].u + interpcoe2 * flu[d_Ord3(ic, jc, km, nzp, nxp)].u + interpcoe3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].u + interpcoe4 * flu[d_Ord3(ic, jc, kp, nzp, nxp)].u;
		uyhz[id] = interpcoe1 * flu[d_Ord3(ic, jc, ks, nzp, nxp)].v + interpcoe2 * flu[d_Ord3(ic, jc, km, nzp, nxp)].v + interpcoe3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].v + interpcoe4 * flu[d_Ord3(ic, jc, kp, nzp, nxp)].v;
		uzhz[id] = interpcoe1 * flu[d_Ord3(ic, jc, km, nzp, nxp)].w + interpcoe2 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].w + interpcoe3 * flu[d_Ord3(ic, jc, kp, nzp, nxp)].w + interpcoe4 * flu[d_Ord3(ic, jc, ku, nzp, nxp)].w;

		//if (kc == 100 && ns==1) {
		//	printf("Thread %d: data[%d] = %f, %f\n", threadIdx.x, kc, uzhx[id], uyhx[id]);
		//}

	}

	// if (jc == nyp)
	// {
	// 	uyhx[d_Ord3(ic, jc, kc, nzp, nxp)] = 0.0;
	// 	uyhz[d_Ord3(ic, jc, kc, nzp, nxp)] = 0.0;
	// }

}

__global__ void rhsx(fluid* flu, process_variables* var, dyDir* dydir, Ypara* ypara, double* uxhx, double* uyhx, double* uzhx, double* uxhz, double* uyhz, double* uzhz, int ns, double* s3tot, RK* rk)
{
	int ic = blockIdx.x * blockDim.x + threadIdx.x;
	int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int kc = blockIdx.z * blockDim.z + threadIdx.z;


	int ks = kc - 2, km = kc - 1, kp = kc + 1, ku = kc + 2;
	int is = ic - 2, im = ic - 1, ip = ic + 1, iu = ic + 2;
	int jm = jc - 1, jp = jc + 1;

	if (ic == 0) { im = nxp - 1; is = nxp - 2; }
	if (kc == 0) { km = nzp - 1; ks = nzp - 2; }
	if (ic == nxp - 1) { ip = 0; iu = 1; }
	if (kc == nzp - 1) { kp = 0; ku = 1; }

	if (ic == 1) { im = 0; is = nxp - 1; }
	if (kc == 1) { km = 0; ks = nzp - 1; }
	if (kc == nzp - 2) { kp = nzp - 1; ku = 0; }
	if (ic == nxp - 2) { ip = nxp - 1; iu = 0; }


	if (kc < nzp && jc <= (nyp - 1) && ic < nxp) {

		double sucaj = dydir[jc].rdyp;

		double InterpY1 = ypara[jc].yinterpCoe, InterpY2 = 1.0 - InterpY1;
		double InterpY3 = ypara[jp].yinterpCoe, InterpY4 = 1.0 - InterpY3;

		double uxhyc = InterpY1 * flu[d_Ord3(ic, jm, kc, nzp, nxp)].u + InterpY2 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].u;
		double uxhyp = InterpY3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].u + InterpY4 * flu[d_Ord3(ic, jp, kc, nzp, nxp)].u;

		double h11 = dxcoe1 * uxhx[d_Ord3(is, jc, kc, nzp, nxp)] * (uxhx[d_Ord3(is, jc, kc, nzp, nxp)] + uCRF) +
			dxcoe2 * uxhx[d_Ord3(im, jc, kc, nzp, nxp)] * (uxhx[d_Ord3(im, jc, kc, nzp, nxp)] + uCRF) +
			dxcoe3 * uxhx[d_Ord3(ic, jc, kc, nzp, nxp)] * (uxhx[d_Ord3(ic, jc, kc, nzp, nxp)] + uCRF) +
			dxcoe4 * uxhx[d_Ord3(ip, jc, kc, nzp, nxp)] * (uxhx[d_Ord3(ip, jc, kc, nzp, nxp)] + uCRF);

		double h12 = (uyhx[d_Ord3(ic, jp, kc, nzp, nxp)] * uxhyp - uyhx[d_Ord3(ic, jc, kc, nzp, nxp)] * uxhyc) * sucaj;

		double h13 = dzcoe1 * uxhz[d_Ord3(ic, jc, km, nzp, nxp)] * uzhx[d_Ord3(ic, jc, km, nzp, nxp)] +
			dzcoe2 * uxhz[d_Ord3(ic, jc, kc, nzp, nxp)] * uzhx[d_Ord3(ic, jc, kc, nzp, nxp)] +
			dzcoe3 * uxhz[d_Ord3(ic, jc, kp, nzp, nxp)] * uzhx[d_Ord3(ic, jc, kp, nzp, nxp)] +
			dzcoe4 * uxhz[d_Ord3(ic, jc, ku, nzp, nxp)] * uzhx[d_Ord3(ic, jc, ku, nzp, nxp)];

		double d11q1 = dxxcoe1 * flu[d_Ord3(is, jc, kc, nzp, nxp)].u +
			dxxcoe2 * flu[d_Ord3(im, jc, kc, nzp, nxp)].u +
			dxxcoe3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].u +
			dxxcoe4 * flu[d_Ord3(ip, jc, kc, nzp, nxp)].u +
			dxxcoe5 * flu[d_Ord3(iu, jc, kc, nzp, nxp)].u;

		double d22q1 = ypara[jc].ap2c * flu[d_Ord3(ic, jp, kc, nzp, nxp)].u +
			ypara[jc].ac2c * flu[d_Ord3(ic, jc, kc, nzp, nxp)].u +
			ypara[jc].am2c * flu[d_Ord3(ic, jm, kc, nzp, nxp)].u;

		double d33q1 = dzzcoe1 * flu[d_Ord3(ic, jc, ks, nzp, nxp)].u +
			dzzcoe2 * flu[d_Ord3(ic, jc, km, nzp, nxp)].u +
			dzzcoe3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].u +
			dzzcoe4 * flu[d_Ord3(ic, jc, kp, nzp, nxp)].u +
			dzzcoe5 * flu[d_Ord3(ic, jc, ku, nzp, nxp)].u;

		double dcq13 = d11q1 + d33q1;
		atomicAdd(s3tot, (dcq13 + d22q1) * dydir[jc].dyp);
		//atomicAdd(s3tot, dcq13);

		double convEd1 = -h11 - h12 - h13 + nu * dcq13;
		double gradp1 = dxcoe1 * flu[d_Ord3(is, jc, kc, nzp, nxp)].p +
			dxcoe2 * flu[d_Ord3(im, jc, kc, nzp, nxp)].p +
			dxcoe3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].p +
			dxcoe4 * flu[d_Ord3(ip, jc, kc, nzp, nxp)].p;

		var[d_Ord3(ic, jc, kc, nzp, nxp)].rhsx = rk[ns].gamma * convEd1 + rk[ns].theta * var[d_Ord3(ic, jc, kc, nzp, nxp)].uold - rk[ns].alpha * gradp1 + 2.0 * (0.5 * rk[ns].alpha * nu) * d22q1;
		var[d_Ord3(ic, jc, kc, nzp, nxp)].uold = convEd1;

		//if (kc == 100 && jc == nyp - 1) {
		//	printf("i %d: k[%d] j[%d] ns[%d]=     %f,         %f,          %f\n", ic, kc, jc, ns, (dcq13 + d22q1) * dydir[jc].dyp, dcq13,        *s3tot);//dpdx, dpdy, dpdz);
		//}

	}
}

__global__ void rhsz(fluid* flu, process_variables* var, dyDir* dydir, Ypara* ypara, double* uxhx, double* uyhx, double* uzhx, double* uxhz, double* uyhz, double* uzhz, int ns, RK* rk)
{
	int ic = blockIdx.x * blockDim.x + threadIdx.x;
	int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int kc = blockIdx.z * blockDim.z + threadIdx.z;


	int ks = kc - 2, km = kc - 1, kp = kc + 1, ku = kc + 2;
	int is = ic - 2, im = ic - 1, ip = ic + 1, iu = ic + 2;
	int jm = jc - 1, jp = jc + 1;

	if (ic == 0) { im = nxp - 1; is = nxp - 2; }
	if (kc == 0) { km = nzp - 1; ks = nzp - 2; }
	if (ic == nxp - 1) { ip = 0; iu = 1; }
	if (kc == nzp - 1) { kp = 0; ku = 1; }

	if (ic == 1) { im = 0; is = nxp - 1; }
	if (kc == 1) { km = 0; ks = nzp - 1; }
	if (ic == nxp - 2) { ip = nxp - 1; iu = 0; }
	if (kc == nzp - 2) { kp = nzp - 1; ku = 0; }

	if (kc < nzp && jc <= (nyp - 1) && ic < nxp) {

		double sucaj = dydir[jc].rdyp;

		double InterpY1 = ypara[jc].yinterpCoe, InterpY2 = 1.0 - InterpY1;
		double InterpY3 = ypara[jp].yinterpCoe, InterpY4 = 1.0 - InterpY3;

		double uzhyc = InterpY1 * flu[d_Ord3(ic, jm, kc, nzp, nxp)].w + InterpY2 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].w;
		double uzhyp = InterpY3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].w + InterpY4 * flu[d_Ord3(ic, jp, kc, nzp, nxp)].w;

		double h31 = dxcoe1 * uzhx[d_Ord3(im, jc, kc, nzp, nxp)] * (uxhz[d_Ord3(im, jc, kc, nzp, nxp)] + uCRF) +
			dxcoe2 * uzhx[d_Ord3(ic, jc, kc, nzp, nxp)] * (uxhz[d_Ord3(ic, jc, kc, nzp, nxp)] + uCRF) +
			dxcoe3 * uzhx[d_Ord3(ip, jc, kc, nzp, nxp)] * (uxhz[d_Ord3(ip, jc, kc, nzp, nxp)] + uCRF) +
			dxcoe4 * uzhx[d_Ord3(iu, jc, kc, nzp, nxp)] * (uxhz[d_Ord3(iu, jc, kc, nzp, nxp)] + uCRF);

		double h32 = (uyhz[d_Ord3(ic, jp, kc, nzp, nxp)] * uzhyp - uyhz[d_Ord3(ic, jc, kc, nzp, nxp)] * uzhyc) * sucaj;

		double h33 = dzcoe1 * uzhz[d_Ord3(ic, jc, ks, nzp, nxp)] * uzhz[d_Ord3(ic, jc, ks, nzp, nxp)] +
			dzcoe2 * uzhz[d_Ord3(ic, jc, km, nzp, nxp)] * uzhz[d_Ord3(ic, jc, km, nzp, nxp)] +
			dzcoe3 * uzhz[d_Ord3(ic, jc, kc, nzp, nxp)] * uzhz[d_Ord3(ic, jc, kc, nzp, nxp)] +
			dzcoe4 * uzhz[d_Ord3(ic, jc, kp, nzp, nxp)] * uzhz[d_Ord3(ic, jc, kp, nzp, nxp)];

		double d11q3 = dxxcoe1 * flu[d_Ord3(is, jc, kc, nzp, nxp)].w +
			dxxcoe2 * flu[d_Ord3(im, jc, kc, nzp, nxp)].w +
			dxxcoe3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].w +
			dxxcoe4 * flu[d_Ord3(ip, jc, kc, nzp, nxp)].w +
			dxxcoe5 * flu[d_Ord3(iu, jc, kc, nzp, nxp)].w;

		double d22q3 = ypara[jc].ap2c * flu[d_Ord3(ic, jp, kc, nzp, nxp)].w +
			ypara[jc].ac2c * flu[d_Ord3(ic, jc, kc, nzp, nxp)].w +
			ypara[jc].am2c * flu[d_Ord3(ic, jm, kc, nzp, nxp)].w;

		double d33q3 = dzzcoe1 * flu[d_Ord3(ic, jc, ks, nzp, nxp)].w +
			dzzcoe2 * flu[d_Ord3(ic, jc, km, nzp, nxp)].w +
			dzzcoe3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].w +
			dzzcoe4 * flu[d_Ord3(ic, jc, kp, nzp, nxp)].w +
			dzzcoe5 * flu[d_Ord3(ic, jc, ku, nzp, nxp)].w;

		double dcq13 = d11q3 + d33q3;

		double convEd3 = -h31 - h32 - h33 + nu * dcq13;
		double gradp3 = dzcoe1 * flu[d_Ord3(ic, jc, ks, nzp, nxp)].p +
			dzcoe2 * flu[d_Ord3(ic, jc, km, nzp, nxp)].p +
			dzcoe3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].p +
			dzcoe4 * flu[d_Ord3(ic, jc, kp, nzp, nxp)].p;

		var[d_Ord3(ic, jc, kc, nzp, nxp)].rhsz = rk[ns].gamma * convEd3 + rk[ns].theta * var[d_Ord3(ic, jc, kc, nzp, nxp)].wold - rk[ns].alpha * gradp3 + 2.0 * (0.5 * rk[ns].alpha * nu) * d22q3;
		var[d_Ord3(ic, jc, kc, nzp, nxp)].wold = convEd3;

		//if (kc == 100 && ic == 0) {
		//	printf("i %d: k[%d] j[%d] ns[%d]= %f, %f, %f\n", ic, kc, jc, ns, var[d_Ord3(ic, jc, kc, nzp, nxp)].uold, var[d_Ord3(ic, 0, kc, nzp, nxp)].vold, var[d_Ord3(ic, 0, kc, nzp, nxp)].wold);//dpdx, dpdy, dpdz);
		//}

	}
}

__global__ void rhsy(fluid* flu, process_variables* var, dyDir* dydir, Ypara* ypara, double* uxhx, double* uyhx, double* uzhx, double* uxhz, double* uyhz, double* uzhz, int ns, RK* rk)
{
	int ic = blockIdx.x * blockDim.x + threadIdx.x;
	int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int kc = blockIdx.z * blockDim.z + threadIdx.z;


	int ks = kc - 2, km = kc - 1, kp = kc + 1, ku = kc + 2;
	int is = ic - 2, im = ic - 1, ip = ic + 1, iu = ic + 2;
	int jm = jc - 1, jp = jc + 1;

	if (ic == 0) { im = nxp - 1; is = nxp - 2; }
	if (kc == 0) { km = nzp - 1; ks = nzp - 2; }
	if (ic == nxp - 1) { ip = 0; iu = 1; }
	if (kc == nzp - 1) { kp = 0; ku = 1; }

	if (ic == 1) { im = 0; is = nxp - 1; }
	if (kc == 1) { km = 0; ks = nzp - 1; }
	if (ic == nxp - 2) { ip = nxp - 1; iu = 0; }
	if (kc == nzp - 2) { kp = nzp - 1; ku = 0; }

	if (kc < nzp && jc <= (nyp - 1) && ic < nxp) {

		double sucac = dydir[jc].rdyp;
		double qsucac = 0.25 * sucac;
		double InterpY1 = ypara[jc].yinterpCoe, InterpY2 = 1.0 - InterpY1;

		double uxhym = InterpY1 * flu[d_Ord3(im, jm, kc, nzp, nxp)].u + InterpY2 * flu[d_Ord3(im, jc, kc, nzp, nxp)].u + uCRF;
		double uxhyc = InterpY1 * flu[d_Ord3(ic, jm, kc, nzp, nxp)].u + InterpY2 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].u + uCRF;
		double uxhyp = InterpY1 * flu[d_Ord3(ip, jm, kc, nzp, nxp)].u + InterpY2 * flu[d_Ord3(ip, jc, kc, nzp, nxp)].u + uCRF;
		double uxhyu = InterpY1 * flu[d_Ord3(iu, jm, kc, nzp, nxp)].u + InterpY2 * flu[d_Ord3(iu, jc, kc, nzp, nxp)].u + uCRF;

		double uzhym = InterpY1 * flu[d_Ord3(ic, jm, km, nzp, nxp)].w + InterpY2 * flu[d_Ord3(ic, jc, km, nzp, nxp)].w;
		double uzhyc = InterpY1 * flu[d_Ord3(ic, jm, kc, nzp, nxp)].w + InterpY2 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].w;
		double uzhyp = InterpY1 * flu[d_Ord3(ic, jm, kp, nzp, nxp)].w + InterpY2 * flu[d_Ord3(ic, jc, kp, nzp, nxp)].w;
		double uzhyu = InterpY1 * flu[d_Ord3(ic, jm, ku, nzp, nxp)].w + InterpY2 * flu[d_Ord3(ic, jc, ku, nzp, nxp)].w;

		double h21 = dxcoe1 * uyhx[d_Ord3(im, jc, kc, nzp, nxp)] * uxhym +
			dxcoe2 * uyhx[d_Ord3(ic, jc, kc, nzp, nxp)] * uxhyc +
			dxcoe3 * uyhx[d_Ord3(ip, jc, kc, nzp, nxp)] * uxhyp +
			dxcoe4 * uyhx[d_Ord3(iu, jc, kc, nzp, nxp)] * uxhyu;

		double h22 = ((flu[d_Ord3(ic, jp, kc, nzp, nxp)].v + flu[d_Ord3(ic, jc, kc, nzp, nxp)].v) *
			(flu[d_Ord3(ic, jp, kc, nzp, nxp)].v + flu[d_Ord3(ic, jc, kc, nzp, nxp)].v) -
			(flu[d_Ord3(ic, jc, kc, nzp, nxp)].v + flu[d_Ord3(ic, jm, kc, nzp, nxp)].v) *
			(flu[d_Ord3(ic, jc, kc, nzp, nxp)].v + flu[d_Ord3(ic, jm, kc, nzp, nxp)].v)) * qsucac;

		double h23 = dzcoe1 * uyhz[d_Ord3(ic, jc, km, nzp, nxp)] * uzhym +
			dzcoe2 * uyhz[d_Ord3(ic, jc, kc, nzp, nxp)] * uzhyc +
			dzcoe3 * uyhz[d_Ord3(ic, jc, kp, nzp, nxp)] * uzhyp +
			dzcoe4 * uyhz[d_Ord3(ic, jc, ku, nzp, nxp)] * uzhyu;

		double d11q2 = dxxcoe1 * flu[d_Ord3(is, jc, kc, nzp, nxp)].v +
			dxxcoe2 * flu[d_Ord3(im, jc, kc, nzp, nxp)].v +
			dxxcoe3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].v +
			dxxcoe4 * flu[d_Ord3(ip, jc, kc, nzp, nxp)].v +
			dxxcoe5 * flu[d_Ord3(iu, jc, kc, nzp, nxp)].v;

		double d22q2 = ypara[jc].ap2p * flu[d_Ord3(ic, jp, kc, nzp, nxp)].v +
			ypara[jc].ac2p * flu[d_Ord3(ic, jc, kc, nzp, nxp)].v +
			ypara[jc].am2p * flu[d_Ord3(ic, jm, kc, nzp, nxp)].v;

		double d33q2 = dzzcoe1 * flu[d_Ord3(ic, jc, ks, nzp, nxp)].v +
			dzzcoe2 * flu[d_Ord3(ic, jc, km, nzp, nxp)].v +
			dzzcoe3 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].v +
			dzzcoe4 * flu[d_Ord3(ic, jc, kp, nzp, nxp)].v +
			dzzcoe5 * flu[d_Ord3(ic, jc, ku, nzp, nxp)].v;

		double dcq13 = d11q2 + d33q2;

		double convEd2 = -h21 - h22 - h23 + nu * dcq13;
		double gradp2 = (flu[d_Ord3(ic, jc, kc, nzp, nxp)].p - flu[d_Ord3(ic, jm, kc, nzp, nxp)].p) * sucac;
		var[d_Ord3(ic, jc, kc, nzp, nxp)].rhsy = rk[ns].gamma * convEd2 + rk[ns].theta * var[d_Ord3(ic, jc, kc, nzp, nxp)].vold - rk[ns].alpha * gradp2 + 2.0 * (0.5 * rk[ns].alpha * nu) * d22q2;
		var[d_Ord3(ic, jc, kc, nzp, nxp)].vold = convEd2;
	}
}

__global__ void correct_rhsx(fluid* flu, process_variables* var, dyDir* dydir, Ypara* ypara, double* s3tot, int ns, RK* rk)  
{
	double dp1ns = nu * (*s3tot) / nxzc / ylength * rk[ns].alpha;

	if (!enable_bulk_pressure_feedback) {
		dp1ns = 0.0;
	}

	int ic = blockIdx.x * blockDim.x + threadIdx.x;
	int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int kc = blockIdx.z * blockDim.z + threadIdx.z;

	if (kc < nzp && jc < nyp && ic < nxp)
	{
		var[d_Ord3(ic, jc, kc, nzp, nxp)].rhsx = var[d_Ord3(ic, jc, kc, nzp, nxp)].rhsx - dp1ns;

	}
}


/*uhat*/
__global__ void uhat_coe(process_variables* var, Ypara* ypara, int ns, double* a, double* b, double* c, double* d, int k, RK* rk, double dU_wall)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if (i < nxp && j < nyc) {
		a[i * nyc + j] = -(0.5 * rk[ns].alpha * nu) * ypara[(j + 1)].am2cForCN;
		b[i * nyc + j] = 1.0 - (0.5 * rk[ns].alpha * nu) * ypara[(j + 1)].ac2cForCN;
		c[i * nyc + j] = -(0.5 * rk[ns].alpha * nu) * ypara[(j + 1)].ap2cForCN;
		d[i * nyc + j] = var[d_Ord3(i, (j + 1), k, nzp, nxp)].rhsx;

        if (j == 0) {
			d[i * nyc + j] += ( 0.5 * rk[ns].alpha * nu) * ypara[1].am2cForCN * dU_wall;
		}
		
	}
}

void uhat_clc(fluid* flu, process_variables* var, Ypara* ypara, double* a, double* b, double* c, double* d, int ns, RK* rk, double dU_wall, double* tri_vecm, double* tri_arrmn)
{
	//cusparseHandle_t handle;
	//CHECK_CUSPARSE(cusparseCreate(&handle));



	for (int k = 0; k < nzp; k++)
	{
		//size_t bufferSize;
		//void* buffer = nullptr;
		uhat_coe << <nyc, nxp >> > (var, ypara, ns, a, b, c, d, k, rk, dU_wall);

		//if (ns == 1)
		//{
		//	double* temp = (double*)malloc((nyc * nxp) * sizeof(double));
		//	cudaMemcpy(temp, d, (nyc * nxp) * sizeof(double), cudaMemcpyDeviceToHost);
		//	for (int i = 0; i < nyc; i++)
		//	{
		//		std::cout << temp[i + 2*nxp] << std::endl;
		//	}		
		//}
		//CHECK_CUSPARSE(cusparseDgtsv2StridedBatch_bufferSizeExt(handle, nyc, a, b, c, d, nxp, nyc, &bufferSize));
		//CHECK_CUDA(cudaMalloc(&buffer, bufferSize));
		//CHECK_CUSPARSE(cusparseDgtsv2StridedBatch(handle, nyc, a, b, c, d, nxp, nyc, buffer));
		//uhat_update << <nyc, nxp >> > (flu, d, k);
		//CHECK_CUDA(cudaFree(buffer));

		InverseTridiagonalDevice(nxp, nyc, a, b, c, d, tri_vecm, tri_arrmn);
		uhat_update << <nyc, nxp >> > (flu, d, k);



		CHECK_CUDA(cudaDeviceSynchronize());
	}
	//CHECK_CUSPARSE(cusparseDestroy(handle));
}

__global__ void uhat_update(fluid* flu, double* d, int k)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if (i < nxp && j < nyc) {
		flu[d_Ord3(i, j + 1, k, nzp, nxp)].u += d[i * nyc + j];
	}
}
/*uhat*/

/*vhat*/
__global__ void vhat_coe(process_variables* var, Ypara* ypara, int ns, double* a, double* b, double* c, double* d, int k, RK* rk)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if (i < nxp && j < nyc && j > 0)
	{
		a[i * nyc + j] = -(0.5 * rk[ns].alpha * nu) * ypara[(j + 1)].am2p;
		b[i * nyc + j] = 1.0 - (0.5 * rk[ns].alpha * nu) * ypara[(j + 1)].ac2p;
		c[i * nyc + j] = -(0.5 * rk[ns].alpha * nu) * ypara[(j + 1)].ap2p;
		d[i * nyc + j] = var[d_Ord3(i, (j + 1), k, nzp, nxp)].rhsy;
	}
	else if (j == 0)
	{
		a[i * nyc + j] = 0.0;
		b[i * nyc + j] = 1.0;
		c[i * nyc + j] = 0.0;
		d[i * nyc + j] = 0.0;
	}
}

void vhat_clc(fluid* flu, process_variables* var, Ypara* ypara, double* a, double* b, double* c, double* d, int ns, RK* rk, double* tri_vecm, double* tri_arrmn)
{
	//cusparseHandle_t handle;
	//CHECK_CUSPARSE(cusparseCreate(&handle));
	for (int k = 0; k < nzp; k++)
	{
		vhat_coe << <nyc, nxp >> > (var, ypara, ns, a, b, c, d, k, rk);

		//if (ns == 1)
		//{
		//	double* temp = (double*)malloc((nyc * nxp) * sizeof(double));
		//	cudaMemcpy(temp, d, (nyc * nxp) * sizeof(double), cudaMemcpyDeviceToHost);
		//	for (int i = 0; i < nyc; i++)
		//	{
		//		std::cout << temp[i + 2 * nxp] << std::endl;
		//	}
		//}

		//size_t bufferSize;
		//void* buffer;
		//CHECK_CUSPARSE(cusparseDgtsv2StridedBatch_bufferSizeExt(handle, nyc, a, b, c, d, nxp, nyc, &bufferSize));
		//CHECK_CUDA(cudaMalloc(&buffer, bufferSize));
		//CHECK_CUSPARSE(cusparseDgtsv2StridedBatch(handle, nyc, a, b, c, d, nxp, nyc, buffer));
		//vhat_update << <nyc, nxp >> > (flu, d, k);
		//CHECK_CUDA(cudaFree(buffer));


		InverseTridiagonalDevice(nxp, nyc, a, b, c, d, tri_vecm, tri_arrmn);
		vhat_update << <nyc, nxp >> > (flu, d, k);


		CHECK_CUDA(cudaDeviceSynchronize());


	}
	//CHECK_CUSPARSE(cusparseDestroy(handle));
}

__global__ void vhat_update(fluid* flu, double* d, int k)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if (i < nxp && j < nyc) {
		flu[d_Ord3(i, j + 1, k, nzp, nxp)].v = flu[d_Ord3(i, j + 1, k, nzp, nxp)].v + d[i * nyc + j];
	}
}
/*vhat*/

/*what*/
__global__ void what_coe(process_variables* var, Ypara* ypara, int ns, double* a, double* b, double* c, double* d, int k, RK* rk)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if (i < nxp && j < nyc) {
		a[i * nyc + j] = -(0.5 * rk[ns].alpha * nu) * ypara[(j + 1)].am2cForCN;
		b[i * nyc + j] = 1.0 - (0.5 * rk[ns].alpha * nu) * ypara[(j + 1)].ac2cForCN;
		c[i * nyc + j] = -(0.5 * rk[ns].alpha * nu) * ypara[(j + 1)].ap2cForCN;
		d[i * nyc + j] = var[d_Ord3(i, (j + 1), k, nzp, nxp)].rhsz;
	}
}

void what_clc(fluid* flu, process_variables* var, Ypara* ypara, double* a, double* b, double* c, double* d, int ns, RK* rk, double* tri_vecm, double* tri_arrmn)
{
	//cusparseHandle_t handle;
	//CHECK_CUSPARSE(cusparseCreate(&handle));
	for (int k = 0; k < nzp; k++)
	{
		what_coe << <nyc, nxp >> > (var, ypara, ns, a, b, c, d, k, rk);

		//size_t bufferSize;
		//void* buffer;
		//CHECK_CUSPARSE(cusparseDgtsv2StridedBatch_bufferSizeExt(handle, nyc, a, b, c, d, nxp, nyc, &bufferSize));
		//CHECK_CUDA(cudaMalloc(&buffer, bufferSize));
		//CHECK_CUSPARSE(cusparseDgtsv2StridedBatch(handle, nyc, a, b, c, d, nxp, nyc, buffer));
		//what_update << <nyc, nxp >> > (flu, d, k);
		//CHECK_CUDA(cudaFree(buffer));

		InverseTridiagonalDevice(nxp, nyc, a, b, c, d, tri_vecm, tri_arrmn);
		what_update << <nyc, nxp >> > (flu, d, k);


		CHECK_CUDA(cudaDeviceSynchronize());
	}
	//CHECK_CUSPARSE(cusparseDestroy(handle));
}

__global__ void what_update(fluid* flu, double* d, int k)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if (i < nxp && j < nyc) {
		flu[d_Ord3(i, j + 1, k, nzp, nxp)].w = flu[d_Ord3(i, j + 1, k, nzp, nxp)].w + d[i * nyc + j];
	}
}
/*what*/


__global__ void clcprsrc(fluid* flu, process_variables* var, dyDir* dydir, int ns, double* divmax, RK* rk, cufftDoubleComplex* prsrc)
{
	int ic = blockIdx.x * blockDim.x + threadIdx.x;
	int jc = blockIdx.y * blockDim.y + threadIdx.y;
	int kc = blockIdx.z * blockDim.z + threadIdx.z;

	int ks = kc - 2, km = kc - 1, kp = kc + 1, ku = kc + 2;
	int is = ic - 2, im = ic - 1, ip = ic + 1, iu = ic + 2;
	int jm = jc - 1, jp = jc + 1;

	if (ic == 0) { im = nxp - 1; is = nxp - 2; }
	if (kc == 0) { km = nzp - 1; ks = nzp - 2; }
	if (ic == nxp - 1) { ip = 0; iu = 1; }
	if (kc == nzp - 1) { kp = 0; ku = 1; }

	if (ic == 1) { im = 0; is = nxp - 1; }
	if (kc == 1) { km = 0; ks = nzp - 1; }
	if (ic == nxp - 2) { ip = nxp - 1; iu = 0; }
	if (kc == nzp - 2) { kp = nzp - 1; ku = 0; }

	if (ic >= nxp || jc > nyp || kc >= nzp) {
		return;
	}

	prsrc[d_Ord3(ic, jc, kc, nzp, nxp)].y = 0.0;

	if (jc >= 1 && jc < nyp) {
		double dudx = dxcoe1 * flu[d_Ord3(im, jc, kc, nzp, nxp)].u +
			dxcoe2 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].u +
			dxcoe3 * flu[d_Ord3(ip, jc, kc, nzp, nxp)].u +
			dxcoe4 * flu[d_Ord3(iu, jc, kc, nzp, nxp)].u;

		double sucaj = dydir[jc].rdyp;
		double dvdy = (flu[d_Ord3(ic, jp, kc, nzp, nxp)].v - flu[d_Ord3(ic, jc, kc, nzp, nxp)].v) * sucaj;

		double dwdz = dzcoe1 * flu[d_Ord3(ic, jc, km, nzp, nxp)].w +
			dzcoe2 * flu[d_Ord3(ic, jc, kc, nzp, nxp)].w +
			dzcoe3 * flu[d_Ord3(ic, jc, kp, nzp, nxp)].w +
			dzcoe4 * flu[d_Ord3(ic, jc, ku, nzp, nxp)].w;

		double rdiv = dudx + dvdy + dwdz;
		double sudtal = 1.0 / rk[ns].alpha;

		////atomicMax(divmax, abs(rdiv));// check the max div for 
		//var[d_Ord3(ic, jc, kc, nzp, nxp)].prsrc->x= sudtal * rdiv;
		//var[d_Ord3(ic, jc, kc, nzp, nxp)].prsrc->y = 0.0; 
		prsrc[d_Ord3(ic, jc, kc, nzp, nxp)].x = sudtal * rdiv;

		//if (kc == 0 && ic == 0) {
		//	printf("i [%d] k[%d] j[%d] ns[%d]= %f, %f,    bbb      %f\n", ic, kc, jc, ns, dvdy, prsrc[d_Ord3(ic, jm, kc, nzp, nxp)].x, flu[d_Ord3(ic, jm, kc, nzp, nxp)].v);//dpdx, dpdy, dpdz);
		//}
	}
}


void initPPE(double* ak1, double* ak3)
{
	double* ak1_host = new double[nxp];
	double* ak3_host = new double[nzp];
	for (int i = 0; i < nxp; i++)
	{
		ak1_host[i] = 1.0 / (dx2 * 288.0) * (cos(3.0 * 2.0 * PI * i / nxp) - 54.0 * cos(2.0 * 2.0 * PI * i / nxp) + 783.0 * cos(1.0 * 2.0 * PI * i / nxp) - 730.0);
	}
	for (int k = 0; k < nzp; k++)
	{
		ak3_host[k] = 1.0 / (dz2 * 288.0) * (cos(3.0 * 2.0 * PI * k / nzp) - 54.0 * cos(2.0 * 2.0 * PI * k / nzp) + 783.0 * cos(1.0 * 2.0 * PI * k / nzp) - 730.0);
	}
	CHECK_CUDA(cudaMemcpy(ak1, ak1_host, nxp * sizeof(double), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(ak3, ak3_host, nzp * sizeof(double), cudaMemcpyHostToDevice));

	delete[] ak1_host;
	delete[] ak3_host;
}

/*to be check*/
__global__ void clcPPE(fluid* flu, process_variables* var, Ypara* ypara, double* ak1, double* ak3, cufftDoubleComplex* prsrc)
{
	int i = threadIdx.x;
	int k = blockIdx.x;


	//float a[256 + 1], b[256 + 1], c[256 + 1], 
	double a[nyp], b[nyp], c[nyp];
	double b1;
	double bn;
	double d[nyp], x[nyp];
	double cc[nyp];
	double dd[nyp];
	int max = nyp - 1, min = 1;

	int j;

	for (j = 1; j < nyp; j++) {
		a[j] = ypara[j].am2ph;
		b[j] = ypara[j].ac2ph + ak1[i] + ak3[k];
		c[j] = ypara[j].ap2ph;
		//d[j] = var[d_Ord3(i, j, k, nzp, nxp)].prsrc->x;
		d[j] = prsrc[d_Ord3(i, j, k, nzp, nxp)].x;

		//if (i == 0 && k == nzp - 1) {
		//	printf("i = %d j = %d k = %d           %f, %f, %f, %f\n", i,j, k, a[j], b[j], c[j], d[j]);
		//}
	}

	b1 = ypara[1].ac2ph + ak1[i] + ak3[k] + ypara[1].am2ph;
	bn = ypara[nyc].ac2ph + ak1[i] + ak3[k] + ypara[nyc].ap2ph;


	for (j = 1; j <= max; j++) {

		cc[j] = 0.0;
		dd[j] = 0.0;
	}

	if (i == 0 && k == 0) {
		min = 2;
		max = nyp - 1;
		cc[min] = c[min] / b[min];
		dd[min] = d[min] / b[min];

	}
	else {
		min = 1;
		max = nyp - 1;
		cc[min] = c[min] / b1;
		dd[min] = d[min] / b1;

	}

	// Forward
	for (j = min + 1; j < max; j++) {
		cc[j] = c[j] / (b[j] - a[j] * cc[j - 1]);
	}
	cc[max] = c[max] / (bn - a[max] * cc[max - 1]);


	for (j = min + 1; j < max; j++) {
		dd[j] = (d[j] - a[j] * dd[j - 1]) / (b[j] - a[j] * cc[j - 1]);
	}
	dd[max] = (d[max] - a[max] * dd[max - 1]) / (bn - a[max] * cc[max - 1]);

	x[max] = dd[max];

	for (j = max - 1; j >= min; j--) {
		x[j] = dd[j] - cc[j] * x[j + 1];
	}

	for (j = 1; j < nyc + 1; j++) {
		//P_device[j * nx_d * nz_d + k * nx_d + i].x = x[j];
		//var[d_Ord3(i, j, k, nzp, nxp)].prsrc->x = x[j];
		prsrc[d_Ord3(i, j, k, nzp, nxp)].x = x[j];

		if (isnan(prsrc[d_Ord3(i, j, k, nzp, nxp)].x)) {
			printf("Thread %d: data[%d] = %f, %f\n", threadIdx.x, k, prsrc[d_Ord3(i, j, k, nzp, nxp)].x, prsrc[d_Ord3(i, j, k, nzp, nxp)].y);
		}


		//if (i == 0 && k == 0) {
		//	printf("i [%d] k[%d] j[%d] ns[%d]= %f, %f,          %f\n", i, k, j, j, flu[d_Ord3(i, j, k, nzp, nxp)].u, prsrc[d_Ord3(i, j, k, nzp, nxp)].x, prsrc[d_Ord3(i, j, k, nzp, nxp)].y);//dpdx, dpdy, dpdz);
		//}

	}




	if (i == 0 && k == 0) {
		//P_device[0 * nx_d * nz_d + k * nx_d + i].x = 0.0f;
		//var[d_Ord3(i, 1, k, nzp, nxp)].prsrc->x = 0.0;

		prsrc[d_Ord3(i, 1, k, nzp, nxp)].x = 0.0;
		//prsrc[d_Ord3(i, nyc, k, nzp, nxp)].x = 0.0;
	}





	for (j = 1; j < nyp; j++) {
		//d[j] = var[d_Ord3(i, j, k, nzp, nxp)].prsrc->y;

		d[j] = prsrc[d_Ord3(i, j, k, nzp, nxp)].y;
	}

	for (j = 0; j <= max; j++) {
		cc[j] = 0.0;
		dd[j] = 0.0;
	}
	if (i == 0 && k == 0) {
		min = 2;
		max = nyc;
		cc[min] = c[min] / b[min];
		dd[min] = d[min] / b[min];

	}
	else {
		min = 1;
		max = nyc;
		cc[min] = c[min] / b1;
		dd[min] = d[min] / b1;

	}

	// Forward
	for (j = min + 1; j < max; j++) {
		cc[j] = c[j] / (b[j] - a[j] * cc[j - 1]);
	}
	cc[max] = c[max] / (bn - a[max] * cc[max - 1]);

	for (j = min + 1; j < max; j++) {
		dd[j] = (d[j] - a[j] * dd[j - 1]) / (b[j] - a[j] * cc[j - 1]);
	}
	dd[max] = (d[max] - a[max] * dd[max - 1]) / (bn - a[max] * cc[max - 1]);

	x[max] = dd[max];
	for (j = max - 1; j >= min; j--) {
		x[j] = dd[j] - cc[j] * x[j + 1];
	}

	for (j = 1; j < nyp; j++) {
		//P_device[j * nx_d * nz_d + k * nx_d + i].y = x[j];
		//var[d_Ord3(i, 1, k, nzp, nxp)].prsrc->y = x[j];

		prsrc[d_Ord3(i, j, k, nzp, nxp)].y = x[j];

		if (isnan(prsrc[d_Ord3(i, j, k, nzp, nxp)].x)) {
			printf("Thread %d: data[%d] = %f, %f\n", threadIdx.x, k, prsrc[d_Ord3(i, j, k, nzp, nxp)].x, prsrc[d_Ord3(i, j, k, nzp, nxp)].y);
		}

	}
	if (i == 0 && k == 0) {
		//P_device[0 * nx_d * nz_d + k * nx_d + i].y = 0.0f;
		//var[d_Ord3(i, 1, k, nzp, nxp)].prsrc->y = 0.0;
		prsrc[d_Ord3(i, 1, k, nzp, nxp)].y = 0.0;
		//prsrc[d_Ord3(i, nyc, k, nzp, nxp)].y = 0.0;
	}

}

void clcPPE_1025(Ypara* ypara, double* ak1, double* ak3, cufftDoubleComplex* prsrc, double* a, double* b, double* c, double* d, double* e, double* tri_vecm, double* tri_arrmn)
{
	for (int k = 0; k < nzp; k++)
	{
		clcPPE_cof << <nyc, nxp >> > (ypara, ak1, ak3, prsrc, a, b, c, d, e, k);

		//size_t bufferSize;
		//void* buffer1;
		//CHECK_CUSPARSE(cusparseDgtsv2StridedBatch_bufferSizeExt(handle, nyc, a, b, c, d, nxp, nyc, &bufferSize));
		//CHECK_CUDA(cudaMalloc(&buffer1, bufferSize));
		//CHECK_CUSPARSE(cusparseDgtsv2StridedBatch(handle, nyc, a, b, c, d, nxp, nyc, buffer1));
		//CHECK_CUDA(cudaFree(buffer1));
		//
		//void* buffer2;
		//CHECK_CUSPARSE(cusparseDgtsv2StridedBatch_bufferSizeExt(handle, nyc, a, b, c, e, nxp, nyc, &bufferSize));
		//CHECK_CUDA(cudaMalloc(&buffer2, bufferSize));
		//CHECK_CUSPARSE(cusparseDgtsv2StridedBatch(handle, nyc, a, b, c, e, nxp, nyc, buffer2));
		//CHECK_CUDA(cudaFree(buffer2));
		InverseTridiagonalDevice(nxp, nyc, a, b, c, d, tri_vecm, tri_arrmn);
		InverseTridiagonalDevice(nxp, nyc, a, b, c, e, tri_vecm, tri_arrmn);

		clcPPE_update << <nyc, nxp >> > (prsrc, d, e, k);
		CHECK_CUDA(cudaDeviceSynchronize());
	}
}

__global__ void clcPPE_cof(Ypara* ypara, double* ak1, double* ak3, cufftDoubleComplex* prsrc, double* a, double* b, double* c, double* d, double* e, int k)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if (k == 0)
	{
		if (i == 0 && j == 0)
		{
			a[i * nyc + j] = 0.0;
			b[i * nyc + j] = 1.0;
			c[i * nyc + j] = 0.0;

			d[i * nyc + j] = 0.0;
			e[i * nyc + j] = 0.0;
		}
		else
		{
			if (i < nxp && j < nyc) {
				a[i * nyc + j] = ypara[j + 1].am2ph;
				c[i * nyc + j] = ypara[j + 1].ap2ph;

				b[i * nyc + j] = ak1[i] + ak3[k] + ypara[j + 1].ac2ph;

				d[i * nyc + j] = prsrc[d_Ord3(i, (j + 1), k, nzp, nxp)].x;
				e[i * nyc + j] = prsrc[d_Ord3(i, (j + 1), k, nzp, nxp)].y;
			}
		}
	}


	else
	{
		if (i < nxp && j < nyc) {
			a[i * nyc + j] = ypara[j + 1].am2ph;
			c[i * nyc + j] = ypara[j + 1].ap2ph;

			b[i * nyc + j] = ak1[i] + ak3[k] + ypara[j + 1].ac2ph;

			d[i * nyc + j] = prsrc[d_Ord3(i, (j + 1), k, nzp, nxp)].x;
			e[i * nyc + j] = prsrc[d_Ord3(i, (j + 1), k, nzp, nxp)].y;
		}
	}
}

__global__ void clcPPE_update(cufftDoubleComplex* prsrc, double* d, double* e, int k)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if (i < nxp && j < nyc) {
		prsrc[d_Ord3(i, j + 1, k, nzp, nxp)].x = d[i * nyc + j] / nxzc;
		prsrc[d_Ord3(i, j + 1, k, nzp, nxp)].y = e[i * nyc + j] / nxzc;
		//prsrc[d_Ord3(i, j + 1, k, nzp, nxp)].x = d[i * nyc + j];
		//prsrc[d_Ord3(i, j + 1, k, nzp, nxp)].y = e[i * nyc + j];
	}
}


__global__ void update_velocity(fluid* flu, dyDir* dydir, RK* rk, int ns, cufftDoubleComplex* prsrc)
{
	int ic = blockIdx.x * blockDim.x + threadIdx.x;
	int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int kc = blockIdx.z * blockDim.z + threadIdx.z;


	int ks = kc - 2, km = kc - 1, kp = kc + 1, ku = kc + 2;
	int is = ic - 2, im = ic - 1, ip = ic + 1, iu = ic + 2;
	int jm = jc - 1, jp = jc + 1;

	if (ic == 0) { im = nxp - 1; is = nxp - 2; }
	if (kc == 0) { km = nzp - 1; ks = nzp - 2; }
	if (ic == nxp - 1) { ip = 0; iu = 1; }
	if (kc == nzp - 1) { kp = 0; ku = 1; }

	if (ic == 1) { im = 0; is = nxp - 1; }
	if (kc == 1) { km = 0; ks = nzp - 1; }
	if (ic == nxp - 2) { ip = nxp - 1; iu = 0; }
	if (kc == nzp - 2) { kp = nzp - 1; ku = 0; }

	if (kc < nzp && jc <= nyc && ic < nxp) {
		double sucac = dydir[jc].rdyc;
		double dpdy = 0.0;
		double dpdx = dxcoe1 * prsrc[d_Ord3(is, jc, kc, nzp, nxp)].x +
			dxcoe2 * prsrc[d_Ord3(im, jc, kc, nzp, nxp)].x +
			dxcoe3 * prsrc[d_Ord3(ic, jc, kc, nzp, nxp)].x +
			dxcoe4 * prsrc[d_Ord3(ip, jc, kc, nzp, nxp)].x;

		//if (jc == nyc) {
		//	double dpdy = (prsrc[d_Ord3(ic, jp, kc, nzp, nxp)].x - prsrc[d_Ord3(ic, jc, kc, nzp, nxp)].x) * sucac;
		//}
		//else {
		dpdy = (prsrc[d_Ord3(ic, jc, kc, nzp, nxp)].x - prsrc[d_Ord3(ic, jm, kc, nzp, nxp)].x) * sucac;
		//}
		double dpdz = dzcoe1 * prsrc[d_Ord3(ic, jc, ks, nzp, nxp)].x +
			dzcoe2 * prsrc[d_Ord3(ic, jc, km, nzp, nxp)].x +
			dzcoe3 * prsrc[d_Ord3(ic, jc, kc, nzp, nxp)].x +
			dzcoe4 * prsrc[d_Ord3(ic, jc, kp, nzp, nxp)].x;

		flu[d_Ord3(ic, jc, kc, nzp, nxp)].u = flu[d_Ord3(ic, jc, kc, nzp, nxp)].u - rk[ns].alpha * dpdx;
		flu[d_Ord3(ic, jc, kc, nzp, nxp)].v = flu[d_Ord3(ic, jc, kc, nzp, nxp)].v - rk[ns].alpha * dpdy;
		flu[d_Ord3(ic, jc, kc, nzp, nxp)].w = flu[d_Ord3(ic, jc, kc, nzp, nxp)].w - rk[ns].alpha * dpdz;

		//if (kc == 0 && ic == 0) {
		//	printf("i [%d] k[%d] j[%d] ns[%d]= %f, %f,  aaa   %f\n", ic, kc, jc, ns, flu[d_Ord3(im, jc, kc, nzp, nxp)].v, prsrc[d_Ord3(ic, jm, kc, nzp, nxp)].x, dpdy);//dpdx, dpdy, dpdz);
		//}
	}
}

__global__ void update_v_dir_velocity(fluid* flu, dyDir* dydir, RK* rk, int ns, cufftDoubleComplex* prsrc)
{
	int ic = blockIdx.x * blockDim.x + threadIdx.x;
	int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int kc = blockIdx.z * blockDim.z + threadIdx.z;


	int jm = jc - 1, jp = jc + 1;
	//int ks = kc - 2, km = kc - 1, kp = kc + 1, ku = kc + 2;
	//int is = ic - 2, im = ic - 1, ip = ic + 1, iu = ic + 2;

	//if (ic == 0) { im = nxp - 1; is = nxp - 2; }
	//if (kc == 0) { km = nzp - 1; ks = nzp - 2; }
	//if (ic == nxp - 1) { ip = 0; iu = 1; }
	//if (kc == nzp - 1) { kp = 0; ku = 1; }

	//if (ic == 1) { im = 0; is = nxp - 1; }
	//if (kc == 1) { km = 0; ks = nzp - 1; }
	//if (ic == nxp - 2) { ip = nxp - 1; iu = 0; }
	//if (kc == nzp - 2) { kp = nzp - 1; ku = 0; }

	if (kc < nzp && jc <= nyc && ic < nxp) {
		double sucac = dydir[jc].rdyc;
		double dpdy = 0.0;

		dpdy = (prsrc[d_Ord3(ic, jc, kc, nzp, nxp)].x - prsrc[d_Ord3(ic, jm, kc, nzp, nxp)].x) * sucac;

		flu[d_Ord3(ic, jc, kc, nzp, nxp)].v = flu[d_Ord3(ic, jc, kc, nzp, nxp)].v - rk[ns].alpha * dpdy;

	}
}

__global__ void update_pressure(fluid* flu, Ypara* ypara, RK* rk, int ns, cufftDoubleComplex* prsrc)
{
	int ic = blockIdx.x * blockDim.x + threadIdx.x;
	int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int kc = blockIdx.z * blockDim.z + threadIdx.z;


	int ks = kc - 2, km = kc - 1, kp = kc + 1, ku = kc + 2;
	int is = ic - 2, im = ic - 1, ip = ic + 1, iu = ic + 2;
	int jm = jc - 1, jp = jc + 1;

	if (ic == 0) { im = nxp - 1; is = nxp - 2; }
	if (kc == 0) { km = nzp - 1; ks = nzp - 2; }
	if (ic == nxp - 1) { ip = 0; iu = 1; }
	if (kc == nzp - 1) { kp = 0; ku = 1; }

	if (ic == 1) { im = 0; is = nxp - 1; }
	if (kc == 1) { km = 0; ks = nzp - 1; }
	if (kc == nzp - 2) { kp = nzp - 1; ku = 0; }
	if (ic == nxp - 2) { ip = nxp - 1; iu = 0; }

	//if (kc == 100 && ic == 100) {
	//	printf("i %d: k[%d] = %f, %f\n", ic, kc, prsrc[d_Ord3(ic, jc, kc, nzp, nxp)].x, prsrc[d_Ord3(ic, jc, kc, nzp, nxp)].y);
	//}

	if (kc < nzp && jc <= (nyp - 1) && ic < nxp) {
		double betap = -0.5 * rk[ns].alpha * nu * ypara[jc].ap2ph;
		double betac = -0.5 * rk[ns].alpha * nu * ypara[jc].ac2ph + 1.0;
		double betam = -0.5 * rk[ns].alpha * nu * ypara[jc].am2ph;
		flu[d_Ord3(ic, jc, kc, nzp, nxp)].p = flu[d_Ord3(ic, jc, kc, nzp, nxp)].p + (betap * prsrc[d_Ord3(ic, jp, kc, nzp, nxp)].x + betac * prsrc[d_Ord3(ic, jc, kc, nzp, nxp)].x + betam * prsrc[d_Ord3(ic, jm, kc, nzp, nxp)].x);
	}

}

__global__ void bc_velocity(fluid* flu, double current_time)
{
	int ic = threadIdx.x;
	int kc = blockIdx.x;

	if (kc < nzp && ic < nxp) {

		// 顶部：自由滑移/零梯度
		flu[d_Ord3(ic, nyp, kc, nzp, nxp)].v = 0.0; 
		flu[d_Ord3(ic, nyp, kc, nzp, nxp)].u = flu[d_Ord3(ic, nyc, kc, nzp, nxp)].u;
		flu[d_Ord3(ic, nyp, kc, nzp, nxp)].w = flu[d_Ord3(ic, nyc, kc, nzp, nxp)].w;

		// 底部：Stokes震荡壁面
		flu[d_Ord3(ic, 1, kc, nzp, nxp)].v = 0.0; 
		double u_wall = U_osc * sin(omega * current_time) - uCRF;
		flu[d_Ord3(ic, 0, kc, nzp, nxp)].u = 2.0 * u_wall - flu[d_Ord3(ic, 1, kc, nzp, nxp)].u;
		flu[d_Ord3(ic, 0, kc, nzp, nxp)].w = -flu[d_Ord3(ic, 1, kc, nzp, nxp)].w;
	}
}

__global__ void bc_presure(fluid* flu)
{
	int ic = threadIdx.x;
	int kc = blockIdx.x;

	if (kc < nzp && ic < nxp) {

		flu[d_Ord3(ic, nyp, kc, nzp, nxp)].p = flu[d_Ord3(ic, nyc, kc, nzp, nxp)].p;
		flu[d_Ord3(ic, 0, kc, nzp, nxp)].p = flu[d_Ord3(ic, 1, kc, nzp, nxp)].p;

	}
}

__global__ void bc_prsrc(cufftDoubleComplex* prsrc)
{
	int ic = threadIdx.x;
	int kc = blockIdx.x;

	if (kc < nzp && ic < nxp) {

		prsrc[d_Ord3(ic, nyp, kc, nzp, nxp)].x = prsrc[d_Ord3(ic, nyc, kc, nzp, nxp)].x;
		prsrc[d_Ord3(ic, 0, kc, nzp, nxp)].x = prsrc[d_Ord3(ic, 1, kc, nzp, nxp)].x;

	}
}




__global__ void ForwardElimination(int m, int n, double* aj, double* bj, double* cj, double* fj, double* vecm, double* arrmn) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < m) {
		// ������һ��
		vecm[i] = bj[i * n];           // vecm(i) = bj(i,1)
		fj[i * n] /= vecm[i];          // fj(i,1) = fj(i,1) / vecm(i)

		// �ӵڶ��п�ʼ
		for (int j = 1; j < n; j++) {
			int idx = i * n + j;
			arrmn[idx] = cj[idx - 1] / vecm[i];    // arrmn(i,j) = cj(i,j-1) / vecm(i)
			vecm[i] = bj[idx] - aj[idx] * arrmn[idx]; // vecm(i) = bj(i,j) - aj(i,j) * arrmn(i,j)
			fj[idx] = (fj[idx] - aj[idx] * fj[idx - 1]) / vecm[i]; // fj(i,j) = (fj(i,j) - aj(i,j) * fj(i,j-1)) / vecm(i)
		}
	}
}

__global__ void BackwardSubstitution(int m, int n, double* fj, double* arrmn) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < m) {
		// �ӵ����ڶ��л���
		for (int j = n - 2; j >= 0; j--) {
			int idx = i * n + j;
			fj[idx] -= arrmn[idx + 1] * fj[idx + 1];  // fj(i,j) = fj(i,j) - arrmn(i,j+1) * fj(i,j+1)
		}
	}
}

// ��������ֻ���� GPU ����������
void InverseTridiagonalDevice(int m, int n, double* aj, double* bj, double* cj, double* fj, double* vecm, double* arrmn) {
	int threadsPerBlock = 1024;
	int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;

	// ǰ����ȥ
	ForwardElimination << <blocksPerGrid, threadsPerBlock >> > (m, n, aj, bj, cj, fj, vecm, arrmn);
	CHECK_CUDA(cudaDeviceSynchronize());

	// ����ش�
	BackwardSubstitution << <blocksPerGrid, threadsPerBlock >> > (m, n, fj, arrmn);
	CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void average_xz(fluid* flu, dyDir* dydir, Stat* stat, int nx, int ny, int nz) {

	int jc = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (jc < nyp)  // ������� y ����Ĵ�С��������
	{
		int jm = jc - 1, jp = jc + 1;

		double cac = dydir[jc].rdyc;
		double cacU = dydir[jp].rdyc;
		double caj = dydir[jc].rdyp;


		double sum[11] = { 0.0 };

		for (int ic = 0; ic < nx; ic++) {
			for (int kc = 0; kc < nz; kc++) {

				int ks = kc - 2, km = kc - 1, kp = kc + 1, ku = kc + 2;
				int is = ic - 2, im = ic - 1, ip = ic + 1, iu = ic + 2;
				int jm = jc - 1, jp = jc + 1;

				if (ic == 0) { im = nxp - 1; is = nxp - 2; }
				if (kc == 0) { km = nzp - 1; ks = nzp - 2; }
				if (ic == nxp - 1) { ip = 0; iu = 1; }
				if (kc == nzp - 1) { kp = 0; ku = 1; }

				if (ic == 1) { im = 0; is = nxp - 1; }
				if (kc == 1) { km = 0; ks = nzp - 1; }
				if (kc == nzp - 2) { kp = nzp - 1; ku = 0; }
				if (ic == nxp - 2) { ip = nxp - 1; iu = 0; }

				double dwdy = (flu[d_Ord3(ic, jc, kc, nz, nx)].w - flu[d_Ord3(ic, jm, kc, nz, nx)].w) * cac;
				double dvdz =
					dzcoe1 * flu[d_Ord3(ic, jc, ks, nz, nx)].v +
					dzcoe2 * flu[d_Ord3(ic, jc, km, nz, nx)].v +
					dzcoe3 * flu[d_Ord3(ic, jc, kc, nz, nx)].v +
					dzcoe4 * flu[d_Ord3(ic, jc, kp, nz, nx)].v;
				double dudz =
					dzcoe1 * flu[d_Ord3(ic, jc, ks, nz, nx)].u +
					dzcoe2 * flu[d_Ord3(ic, jc, km, nz, nx)].u +
					dzcoe3 * flu[d_Ord3(ic, jc, kc, nz, nx)].u +
					dzcoe4 * flu[d_Ord3(ic, jc, kp, nz, nx)].u;
				double dwdx =
					dxcoe1 * flu[d_Ord3(is, jc, kc, nz, nx)].w +
					dxcoe2 * flu[d_Ord3(im, jc, kc, nz, nx)].w +
					dxcoe3 * flu[d_Ord3(ic, jc, kc, nz, nx)].w +
					dxcoe4 * flu[d_Ord3(ip, jc, kc, nz, nx)].w;
				double dvdx =
					dxcoe1 * flu[d_Ord3(is, jc, kc, nz, nx)].v +
					dxcoe2 * flu[d_Ord3(im, jc, kc, nz, nx)].v +
					dxcoe3 * flu[d_Ord3(ic, jc, kc, nz, nx)].v +
					dxcoe4 * flu[d_Ord3(ip, jc, kc, nz, nx)].v;
				double dudy = (flu[d_Ord3(ic, jc, kc, nz, nx)].u - flu[d_Ord3(ic, jm, kc, nz, nx)].u) * cac;

				sum[0]+= (dwdy - dvdz) * (dwdy - dvdz);
				sum[1] += (dudz - dwdx) * (dudz - dwdx);
				sum[2] += (dvdx - dudy) * (dvdx - dudy);

				sum[3] += flu[d_Ord3(ic, jc, kc, nz, nx)].u + uCRF;
				sum[4] += flu[d_Ord3(ic, jc, kc, nz, nx)].v;
				sum[5] += flu[d_Ord3(ic, jc, kc, nz, nx)].w;
				sum[6] += flu[d_Ord3(ic, jc, kc, nz, nx)].p;

				sum[7] += (flu[d_Ord3(ic, jc, kc, nz, nx)].u + uCRF) * (flu[d_Ord3(ic, jc, kc, nz, nx)].u + uCRF);
				sum[8] += flu[d_Ord3(ic, jc, kc, nz, nx)].v * flu[d_Ord3(ic, jc, kc, nz, nx)].v;
				sum[9] += flu[d_Ord3(ic, jc, kc, nz, nx)].w * flu[d_Ord3(ic, jc, kc, nz, nx)].w;
				sum[10] += flu[d_Ord3(ic, jc, kc, nz, nx)].p * flu[d_Ord3(ic, jc, kc, nz, nx)].p;


			}
		}

		// �� x �� z ������ƽ��
		stat[jc].wx = sum[0] / (nx * nz);
		stat[jc].wy = sum[1] / (nx * nz);
		stat[jc].wz = sum[2] / (nx * nz);

		stat[jc].um = sum[3] / (nx * nz);
		stat[jc].vm = sum[4] / (nx * nz);
		stat[jc].wm = sum[5] / (nx * nz);
		stat[jc].pm = sum[6] / (nx * nz);

		stat[jc].um2 = sum[7] / (nx * nz);
		stat[jc].vm2 = sum[8] / (nx * nz);
		stat[jc].wm2 = sum[9] / (nx * nz);
		stat[jc].pm2 = sum[10] / (nx * nz);
	}
}



__global__ void test(double* a, double* b, double* c, double* d)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if (i < nxp && j < nyc) {
		a[i * nyc + j] = 1.0;
		b[i * nyc + j] = 1.0;
		c[i * nyc + j] = 1.0;
		d[i * nyc + j] = 1.0;
		//a[i * nyc + j - 1] = -(0.5 * rk[ns].alpha * nu) * ypara[j].am2cForCN;
		//b[i * nyc + j - 1] = 1.0 - (0.5 * rk[ns].alpha * nu) * ypara[j].ac2cForCN;
		//c[i * nyc + j - 1] = -(0.5 * rk[ns].alpha * nu) * ypara[j].ap2cForCN;
		//d[i * nyc + j - 1] = var[i+ k * nxp + (nxp * nzp) * j].rhsx;
	}

}

void calcuate()
{
	double* uxhx;
	double* uyhx;
	double* uzhx;
	double* uxhz;
	double* uyhz;
	double* uzhz;

	double* s3tot;

	double* aa, * bb, * cc, * dd;
	double* tri_vecm, * tri_arrmn;

	double* pp; //for the image of prsrc

	double* ak1, * ak3;

	//cudaMalloc((void**)&s3tot, sizeof(double));
	//cudaMemcpy(s3tot, &zero, sizeof(double), cudaMemcpyHostToDevice);

	CHECK_CUDA(cudaMalloc((void**)&s3tot, sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&uxhx, static_cast<size_t>(nxp) * static_cast<size_t>(nyp + 1) * static_cast<size_t>(nzp) * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&uyhx, static_cast<size_t>(nxp) * static_cast<size_t>(nyp + 1) * static_cast<size_t>(nzp) * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&uzhx, static_cast<size_t>(nxp) * static_cast<size_t>(nyp + 1) * static_cast<size_t>(nzp) * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&uxhz, static_cast<size_t>(nxp) * static_cast<size_t>(nyp + 1) * static_cast<size_t>(nzp) * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&uyhz, static_cast<size_t>(nxp) * static_cast<size_t>(nyp + 1) * static_cast<size_t>(nzp) * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&uzhz, static_cast<size_t>(nxp) * static_cast<size_t>(nyp + 1) * static_cast<size_t>(nzp) * sizeof(double)));

	CHECK_CUDA(cudaMalloc((void**)&aa, static_cast<size_t>(nxp) * static_cast<size_t>(nyc) * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&bb, static_cast<size_t>(nxp) * static_cast<size_t>(nyc) * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&cc, static_cast<size_t>(nxp) * static_cast<size_t>(nyc) * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&dd, static_cast<size_t>(nxp) * static_cast<size_t>(nyc) * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&pp, static_cast<size_t>(nxp) * static_cast<size_t>(nyc) * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&tri_vecm, static_cast<size_t>(nxp) * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&tri_arrmn, static_cast<size_t>(nxp) * static_cast<size_t>(nyc) * sizeof(double)));

	double* divmax;
	CHECK_CUDA(cudaMalloc((void**)&divmax, sizeof(double)));

	CHECK_CUDA(cudaMalloc((void**)&ak1, static_cast<size_t>(nxp) * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&ak3, static_cast<size_t>(nzp) * sizeof(double)));
	initPPE(ak1, ak3);//�޵�����ȥ

	cufftHandle plan;
	int nn[2];
	nn[0] = nzp;
	nn[1] = nxp;
	CHECK_CUFFT(cufftPlanMany(&plan, 2, nn, NULL, 1, nxp * nzp, NULL, 1, nxp * nzp, CUFFT_Z2Z, nyp + 1));

	dim3 blockDim(8, 8, 8); // ÿ���߳̿��е��߳���
	dim3 gridDim((nxp + blockDim.x - 1) / blockDim.x,
		(nyp + 1 + blockDim.y - 1) / blockDim.y,
		(nzp + blockDim.z - 1) / blockDim.z);// �����еĿ���


	//size_t bufferSize;
	//void* buffer = nullptr;

	double prgradaver = 0.0, s3tot_host = 0.0;

	clock_t start, finish;

	start = clock();
	bc_velocity << < nzp, nxp >> > (flu, restart_start_step * dt);
	CHECK_CUDA(cudaDeviceSynchronize());

	for (int t = 0; t < timemax; t++)
	{
        // 计算当前物理时间，用于传递给含时边界
		const int global_step_start = restart_start_step + t;
		double t_stage_start = global_step_start * dt;

		for (int ns = 0; ns < 3; ns++)
		{
			CHECK_CUDA(cudaMemcpy(s3tot, &zero, sizeof(double), cudaMemcpyHostToDevice));
            /*修改时间补偿*/
            double t_current = t_stage_start;
            double t_next    = t_current + rk[ns].alpha;

            // 分别计算当前时刻和下一时刻的物理壁面速度
            double u_wall_current = U_osc * sin(omega * t_current) - uCRF;
            double u_wall_next = U_osc * sin(omega * t_next) - uCRF;
            
            // 求出在这一个极小的 RK 子步内的速度增量
            double dU_wall = u_wall_next - u_wall_current;
			/*修改时间补偿*/

			Update_VelInterp_uvw << < gridDim, blockDim >> > (flu, uxhx, uyhx, uzhx, uxhz, uyhz, uzhz);  //flu ��Ҫ��gpu�϶���
			rhsx << < gridDim, blockDim >> > (flu, var, dydir_device, ypara_device, uxhx, uyhx, uzhx, uxhz, uyhz, uzhz, ns, s3tot, rk_device);
			rhsy << < gridDim, blockDim >> > (flu, var, dydir_device, ypara_device, uxhx, uyhx, uzhx, uxhz, uyhz, uzhz, ns, rk_device);
			rhsz << < gridDim, blockDim >> > (flu, var, dydir_device, ypara_device, uxhx, uyhx, uzhx, uxhz, uyhz, uzhz, ns, rk_device);
			CHECK_CUDA(cudaDeviceSynchronize());

			correct_rhsx << < gridDim, blockDim >> > (flu, var, dydir_device, ypara_device, s3tot, ns, rk_device);

			CHECK_CUDA(cudaDeviceSynchronize());
			//for (int k = 0; k < nzp; k++)
			//{
			//	uhat_coe << <nyc, nxp >> > (var, ypara_device, ns, aa, bb, cc, dd, k, rk_device); cudaDeviceSynchronize();
			//	//test << <nyc, nxp >> > (aa, bb, cc, dd);
			//	CHECK_CUDA(cudaMemcpy(h_d, cc, nxp * nyc * sizeof(double), cudaMemcpyDeviceToHost));
			//	for (int i = 0; i < nyc; i++)
			//	{
			//		std::cout << h_d[i] << std::endl;
			//	}
			//	CHECK_CUSPARSE(cusparseDgtsv2StridedBatch_bufferSizeExt(handle, nyc, aa, bb, cc, dd, nxp, nyc, &bufferSize));
			//	CHECK_CUDA(cudaMalloc(&buffer, bufferSize));
			//	CHECK_CUSPARSE(cusparseDgtsv2StridedBatch(handle, nyc, aa, bb, cc, dd, nxp, nyc, buffer));
			//	uhat_update << <nxp, nyp >> > (flu, dd, k);
			//	cudaDeviceSynchronize();
			//	CHECK_CUDA(cudaFree(buffer));
			//}

            /*修改真实速度*/
			uhat_clc(flu, var, ypara_device, aa, bb, cc, dd, ns, rk_device, dU_wall, tri_vecm, tri_arrmn);
			/*修改真实速度*/
			what_clc(flu, var, ypara_device, aa, bb, cc, dd, ns, rk_device, tri_vecm, tri_arrmn);
			vhat_clc(flu, var, ypara_device, aa, bb, cc, dd, ns, rk_device, tri_vecm, tri_arrmn);

			bc_velocity << < nzp, nxp >> > (flu, t_next);
			CHECK_CUDA(cudaDeviceSynchronize());

			/*吹吸边界条件*/
            /*
			double current_time = t * dt;
            bc_velocity << < nzp, nxp >> > (flu, current_time);
			*/
			

			/*PPE*/
			CHECK_CUDA(cudaMemcpy(divmax, &zero, sizeof(double), cudaMemcpyHostToDevice));


			clcprsrc << < gridDim, blockDim >> > (flu, var, dydir_device, ns, divmax, rk_device, prsrc);

			CHECK_CUFFT(cufftExecZ2Z(plan, (cufftDoubleComplex*)prsrc, (cufftDoubleComplex*)prsrc, CUFFT_FORWARD)); CHECK_CUDA(cudaDeviceSynchronize());

			clcPPE_1025(ypara_device, ak1, ak3, prsrc, aa, bb, cc, dd, pp, tri_vecm, tri_arrmn);

			//clcPPE << < nzp, nxp >> > (flu, var, ypara_device, ak1, ak3, prsrc); cudaDeviceSynchronize();

			CHECK_CUFFT(cufftExecZ2Z(plan, (cufftDoubleComplex*)prsrc, (cufftDoubleComplex*)prsrc, CUFFT_INVERSE)); CHECK_CUDA(cudaDeviceSynchronize());
			bc_prsrc << < nzp, nxp >> > (prsrc);
			/*PPE*/


			//bc_velocity << < nzp, nxp >> > (flu);
			/*update*/
			update_velocity << < gridDim, blockDim >> > (flu, dydir_device, rk_device, ns, prsrc);
			//update_v_dir_velocity << < gridDim, blockDim >> > (flu, dydir_device, rk_device, ns, prsrc);
			update_pressure << < gridDim, blockDim >> > (flu, ypara_device, rk_device, ns, prsrc); CHECK_CUDA(cudaDeviceSynchronize());

			bc_presure << < nzp, nxp >> > (flu); CHECK_CUDA(cudaDeviceSynchronize());

			bc_velocity << < nzp, nxp >> > (flu, t_next);
			CHECK_CUDA(cudaDeviceSynchronize());

			/*update*/

			CHECK_CUDA(cudaMemcpy(&s3tot_host, s3tot, sizeof(double), cudaMemcpyDeviceToHost));
			//prgradaver += nu * (s3tot_host) / nxzc / ylength * rk[ns].alpha / dt;

			t_stage_start = t_next;

		}


		const int completed_step = restart_start_step + t + 1;
		const double completed_time = completed_step * dt;

		if (completed_step % stat_output_interval == 0) {
			printf("%d     prgrad= %f \n", completed_step, prgradaver);
		}
#ifdef  Restart
		if (t % 1 == 0)
		{
			output_prgrad(prgradaver);
			prgradaver = 0.0;
		}
		if (completed_step % stat_output_interval == 0)
		{
			//output_velocity(flu, flu_host, t);
			clcstat(flu, completed_time, completed_step);
		}
		if (completed_step % restart_output_interval == 0 || t + 1 == timemax)
		{
			//output_velocity(flu, flu_host, t);
			output_restart(flu, flu_host, var, completed_step);
		}
#else
		if (t % 1 == 0)
		{
			output_prgrad(prgradaver);
			prgradaver = 0.0;
		}
		//if (t % 500 == 0 && t >= 5000 && t < timemax)
		//{
		//	output_velocity(flu, flu_host, t);
		//}
		if (completed_step % stat_output_interval == 0)
		{
			clcstat(flu, completed_time, completed_step);
		}
		if (completed_step % restart_output_interval == 0 || t + 1 == timemax)
		{
			//output_velocity(flu, flu_host, t);
			output_restart(flu, flu_host, var, completed_step);
		}

#endif
	}

	finish = clock();
	printf("Time Take : %7.2lf h\n", (double)(finish - start)/3600 / CLOCKS_PER_SEC);

	//output for restart


	CHECK_CUDA(cudaFree(uxhx));
	CHECK_CUDA(cudaFree(uyhx));
	CHECK_CUDA(cudaFree(uzhx));
	CHECK_CUDA(cudaFree(uxhz));
	CHECK_CUDA(cudaFree(uyhz));
	CHECK_CUDA(cudaFree(uzhz));
	CHECK_CUDA(cudaFree(s3tot));
	CHECK_CUDA(cudaFree(aa));
	CHECK_CUDA(cudaFree(bb));
	CHECK_CUDA(cudaFree(cc));
	CHECK_CUDA(cudaFree(dd));
	CHECK_CUDA(cudaFree(pp));
	CHECK_CUDA(cudaFree(tri_vecm));
	CHECK_CUDA(cudaFree(tri_arrmn));
	CHECK_CUDA(cudaFree(divmax));
	CHECK_CUDA(cudaFree(ak1));
	CHECK_CUDA(cudaFree(ak3));
	CHECK_CUFFT(cufftDestroy(plan));
}



