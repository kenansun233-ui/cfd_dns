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

	//cufftDoubleComplex* prsrc;
	//double prsrc;

	/*想办法优化*/
	//double uxhx;
	//double uyhx;
	//double uzhx;
	//double uxhz;
	//double uyhz;
	//double uzhz;
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

/* Retau = 1000 */
//constexpr double ubulk = 1.000000000;
//constexpr double nu = 5.00000E-05;
//constexpr double PI = 3.14159265358979323846264338327950288;
//constexpr double uCRF = 1.000000000;  //参考系速度
// 
//constexpr double xlength = 6.283185307;
//constexpr double ylength = 2.0;
//constexpr double zlength = 3.141592654;
//
//constexpr int nxp = 576;
//constexpr int nyp = 383;
//constexpr int nzp = 576;
/* Retau = 1000 */

/* Retau = 180 */
//constexpr double ubulk = 0.6666666666666666666666667;
////constexpr double ubulk = 0.333333333333333333333333333333;
//constexpr double nu = 2.3310E-4;
//constexpr double PI = 3.14159265358979323846264338327950288;
//constexpr double uCRF = 0.6666666666666666666666667;  //参考系速度,若壁面静止则为0；流体相对静止则为ubulk
//
//constexpr double xlength = 12.56637061;
//constexpr double ylength = 2.0;
//constexpr double zlength = 6.283185307;
//
//constexpr int nxp = 384;
//constexpr int nyp = 191;
//constexpr int nzp = 256;
/* Retau = 180 */


constexpr double PI = 3.14159265358979323846264338327950288;


// ==============================================================================
// 物理模拟控制开关 (通过注释切换)
// ==============================================================================
// #define ZERO_CROSS_FLOW  // 仅保留作零来流 Stokes 验证的历史配置。
constexpr bool enable_wall_oscillation = true;
// 当前算例: Re_tau = 180, h = 0.1, Re_omega = 100
// 定义:
// Re_omega = U_osc^2 / (omega * nu)
// Re_delta = U_osc * sqrt(2.0 * nu / omega) / nu = sqrt(2.0 * Re_omega)
// stokes_delta = sqrt(2.0 * nu / omega)
// 若切换 Re_omega，请同步修改 U_osc、Re_delta 和输出设置。
constexpr double Re_omega = 100.0;  // Re_omega = U_osc^2 / (omega * nu)
constexpr double omega = 2.0 * PI / 5.0; // Patil & Fringer (2022): Tw = 5, omega = 2*pi/5
constexpr double nu_fixed = 8.841941282883074e-7; // 固定运动粘度
constexpr double U_osc = 1.054092553389460e-2;
constexpr double Re_delta = 1.414213562373095e1;
constexpr double stokes_delta = 1.186270905695295e-3;
constexpr bool enable_bulk_pressure_feedback = true;
constexpr bool restart_continues_oscillation_time = false; // false: restart from developed channel and start wall phase at t=0

// --- 零来流 Stokes 扰动历史参数（当前槽道流初始化不使用） ---
// constexpr double bypass_perturbation_amp = 1.0e-4 * U_osc;

constexpr bool output_tau_wall_maps = true;
constexpr double stat_output_dt = 0.05;          // 记录/核对用；实际步数见 stat_output_interval
constexpr double tau_wall_map_output_dt = 0.5;   // 记录/核对用；实际步数见 tau_wall_map_interval

// 零来流 Stokes 验证配置：
// constexpr double uCRF = 0.0;
// constexpr double ubulk = 0.0;
// constexpr double nu = nu_fixed;

// 当前配置：Re_tau = 180 的半槽道定流量算例。
constexpr double uCRF = 0.0;
constexpr double ubulk = 2.528521911424e-2;
constexpr double nu = nu_fixed;

// 半槽高 H = 0.1；网格/流域按 Patil & Fringer (2022) bumpy-wall case:
// Lx = 2*pi*H, Lz = pi*H；grid = 512(streamwise) x 256(spanwise) x 128(wall-normal).

// --- 当前算例时间参数 ---
constexpr double dt = 5.0e-3;
constexpr int timemax = 10000;  // 10 个震荡周期，dt = 0.005 且 T = 5
constexpr int stat_output_interval = 10;
constexpr int tau_wall_map_interval = 100;
constexpr bool output_field_files = false;
constexpr int field_output_interval = 400;
constexpr int restart_output_interval = 1000;
constexpr int restart_input_step = -1;       // -1 reads the latest restart_*.dat; otherwise reads restart_%08d.dat
constexpr double simulation_cycles = 10.0;

constexpr double ylength = 0.1;
constexpr double xlength = 2.0 * PI * ylength;
constexpr double zlength = PI * ylength;

constexpr int nyp = 128;
constexpr int nxp = 512;
constexpr int nzp = 256;


/* Retau=150 */
//constexpr double ubulk = 1.000000000;
//constexpr double uCRF = 1.000000000;  //参考系速度,若壁面静止则为0；流体相对静止则为ubulk
//constexpr double nu = 4.5903E-4;
//constexpr double PI = 3.14159265358979323846264338327950288;
//
//
//constexpr double V_wall_blowing = 0.0034 * ubulk;//  (Uniform Blowing/Suction)0.5%
//
//constexpr double xlength = 15.70796327;
//constexpr double ylength = 2.0;
//constexpr double zlength = 6.283185307;
//
//constexpr int nxp = 128;
//constexpr int nyp = 191;
//constexpr int nzp = 128;
/* Retau=150 */

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
