#include <cufft.h>
#ifndef parameters
#define parameters

struct fluid
{
	/*all variables are on the center mesh
	and only define on the gpu
	u v w p need to define on the host
	*/

	double u;
	double v;
	double w;
	double p;
};

struct process_variables
{
	double uold;
	double vold;
	double wold;

	double rhsx;
	double rhsy;
	double rhsz;

	// Cached interpolation fields were moved to separate work arrays.
	// double uxhx;
	// double uyhx;
	// double uzhx;
	// double uxhz;
	// double uyhz;
	// double uzhz;
};

struct Ypara //y-dir interpolation para and the number =1:nyc
{
	//Pressure Laplacian metries in y-dir
	double ap2ph;
	double ac2ph;
	double am2ph;

	//ux/uz Laplacian metries in y-dir (STAGGERED VARIABLE)
	double ap2c;
	double ac2c;
	double am2c;

	//ux/uz Laplacian metries in y-dir (for Crank-Nicolson scheme purpose)
	double ap2cForCN;
	double ac2cForCN;
	double am2cForCN;

	//uy Laplacian metries in y-dir (CENTERED VARIABLE)
	double ap2p;
	double ac2p;
	double am2p;

	//
	double yinterpCoe;
};

struct dyDir
{
	double yp; //point coordinate in y-dir. Suffix 'v' means 'vector'   
	double dyp; //point coordinate interval in y-dir
	double yc;
	double dyc;
	double rdyp;
	double rdyc;

	double volcell;
	double deltacell;
};


extern fluid* flu;
extern fluid* flu_host;
extern Ypara* ypara_host;
extern dyDir* dydir;
extern process_variables* var;

extern cufftDoubleComplex* prsrc;


//#define Restart
#define Output_Restart

constexpr double PI = 3.14159265358979323846264338327950288;

// ==============================================================================
// 物理模拟控制开关 (通过注释切换)
// ==============================================================================
#define ZERO_CROSS_FLOW 

// --- Stokes 震荡底壁控制参数 ---
// 当前算例: Re_omega = 1.0e5
// 定义:
// Re_omega = U_osc^2 / (omega * nu)
// Re_delta = U_osc * sqrt(2.0 * nu / omega) / nu = sqrt(2.0 * Re_omega)
// stokes_delta = sqrt(2.0 * nu / omega)
// 若切换 Re_omega，请同步修改 U_osc、Re_delta、dt、simulation_cycles、timemax 和输出间隔。
constexpr double Re_omega = 100000.0;  // Re_omega = U_osc^2 / (omega * nu)
constexpr double omega = 2.0 * PI;     // 固定震荡圆频率 (周期 T = 1.0)
constexpr double nu_fixed = 8.841941282883074e-7; // 固定运动粘度
constexpr double U_osc = 7.453559924999298e-1;
constexpr double Re_delta = 4.472135954999579e2;
constexpr double stokes_delta = 5.305164769729844e-4;
constexpr bool enable_bulk_pressure_feedback = false;

// --- Bypass transition controls ---
// Set bypass_perturbation_amp = 0.0 for a clean laminar Stokes validation.
constexpr double bypass_perturbation_amp = 1.0e-6 * U_osc;

constexpr bool output_tau_wall_maps = true;
constexpr double stat_output_dt = 0.005;         // 记录/核对用；实际步数见 stat_output_interval
constexpr double tau_wall_map_output_dt = 0.02;  // 记录/核对用；实际步数见 tau_wall_map_interval

#ifdef ZERO_CROSS_FLOW
	// 阶段一：纯震荡层流验证
	constexpr double uCRF = 0.0;
	constexpr double ubulk = 0.0;
	constexpr double nu = nu_fixed;
#else
	// 阶段二：定来流研究 (保留你原有的设置)
	constexpr double uCRF = 0.0; 
	constexpr double ubulk = 0.6666666666666666666666667;
	constexpr double nu = nu_fixed;  
#endif

// Use a pi-fraction domain for the oscillating boundary-layer calculation.
constexpr double xlength = PI / 16.0;
// constexpr double ylength = 2.0;//全槽道
constexpr double ylength = 1.0;//半槽道
constexpr double zlength = PI / 64.0;

// constexpr int nxp = 640;
// constexpr int nyp = 511;
// constexpr int nzp = 640;

// constexpr int nxp = 364;
// // constexpr int nyp = 191;
// constexpr int nyp = 257;
// constexpr int nzp = 256;

constexpr int nxp = 768;
constexpr int nyp = 257;
constexpr int nzp = 192;

constexpr int nxc = nxp - 1;
constexpr int nyc = nyp - 1;
constexpr int nzc = nzp - 1;

constexpr double dx = xlength / nxp;
constexpr double dy = ylength / nyc;
constexpr double dz = zlength / nzp;

constexpr double nxzc = (double)nxp * (double)nzp;

constexpr double dx2 = dx * dx;
constexpr double dy2 = dy * dy;
constexpr double dz2 = dz * dz;

constexpr double dxcoe1 = 1.0 / 24.0 / dx;
constexpr double dxcoe2 = -1.125 / dx;
constexpr double dxcoe3 = 1.125 / dx;
constexpr double dxcoe4 = -1.0 / 24.0 / dx;

constexpr double dzcoe1 = 1.0 / 24.0 / dz;
constexpr double dzcoe2 = -1.125 / dz;
constexpr double dzcoe3 = 1.125 / dz;
constexpr double dzcoe4 = -1.0 / 24.0 / dz;

constexpr double dxxcoe1 = -1.0 / 12.0 / dx2;
constexpr double dxxcoe2 = 4.0 / 3.0 / dx2;
constexpr double dxxcoe3 = -2.5 / dx2;
constexpr double dxxcoe4 = 4.0 / 3.0 / dx2;
constexpr double dxxcoe5 = -1.0 / 12.0 / dx2;

constexpr double dzzcoe1 = -1.0 / 12.0 / dz2;
constexpr double dzzcoe2 = 4.0 / 3.0 / dz2;
constexpr double dzzcoe3 = -2.5 / dz2;
constexpr double dzzcoe4 = 4.0 / 3.0 / dz2;
constexpr double dzzcoe5 = -1.0 / 12.0 / dz2;

constexpr double interpcoe1 = -1.0 / 16.0;
constexpr double interpcoe2 = 9.0 / 16.0;
constexpr double interpcoe3 = 9.0 / 16.0;
constexpr double interpcoe4 = -1.0 / 16.0;

constexpr double zero = 0.0;

// --- 当前算例时间参数 ---
constexpr double dt = 5.0e-5;
constexpr double simulation_cycles = 15.0;
constexpr int timemax = 300000;              // simulation_cycles / dt
constexpr int stat_output_interval = 100;    // 0.005 / dt
constexpr int tau_wall_map_interval = 400;   // 0.02 / dt


struct RK
{
	double gamma;
	double theta;
	double alpha;
	//double beta;
};
extern RK* rk;

extern Ypara* ypara_device;
extern dyDir* dydir_device;
extern RK* rk_device;


extern char output_path[100];

/*output set*/
constexpr int xskip = 2;
constexpr int yskip = 1;
constexpr int zskip = 2;

struct Stat
{
	double wx;
	double wy;
	double wz;

	double um;
	double vm;
	double wm;
	double pm;

	double um2;
	double vm2;
	double wm2;
	double pm2;
};


#endif
