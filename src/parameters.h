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

	/*��취�Ż�*/
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

// ==============================================================================
// 计算选择 (阶段切换)#define Restart
// 取消注释下方宏定义进行继续计算；注释掉则从头开始
// ==============================================================================
//#define Restart
#define Output_Restart

// ==============================================================================
// 物理模拟控制面板 (阶段切换)#define ZERO_CROSS_FLOW 
// 取消注释下方宏定义进行“零横流验证”；注释掉则进行“高雷诺数横流研究”
// ==============================================================================
#define ZERO_CROSS_FLOW 

// --- Stokes 震荡底壁控制参数 ---
constexpr double U_osc = 1.0;          // 震荡速度振幅
constexpr double omega = 2.0 * PI;     // 震荡圆频率 (周期 T = 1.0)
constexpr double Re_delta = 100.0;     // 目标震荡雷诺数 (控制穿透深度和转捩)

#ifdef ZERO_CROSS_FLOW
	// [阶段一] 零横流纯震荡 (实验室绝对静止参考系)
	constexpr double uCRF = 0.0;
	constexpr double ubulk = 0.0;
	// 核心：由 Re_delta 反推运动粘度，绝对保证物理相似性
	constexpr double nu = (2.0 * U_osc * U_osc) / (omega * Re_delta * Re_delta);
#else
	// [阶段二] 恒定横流 + 震荡底壁 (移动参考系)
	constexpr double Re_tau = 180.0;   // 目标摩擦雷诺数 (横流)
	constexpr double uCRF = 0.6666666666666666666666667; // 参考系移动速度
	constexpr double ubulk = 0.6666666666666666666666667;
	constexpr double nu = 3.37668E-5;  // 横流基准粘度
#endif

// 辅助计算：Stokes 穿透深度 (运行前请比对第一层网格高度 Delta_y1)
constexpr double stokes_delta = sqrt(2.0 * nu / omega);

// 几何与网格参数保持不变
constexpr double PI = 3.14159265358979323846264338327950288;
constexpr double xlength = 6.283185307;
constexpr double ylength = 2.0;
constexpr double zlength = 3.141592654;
constexpr int nxp = 640;
constexpr int nyp = 511;
constexpr int nzp = 640;


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

constexpr double dt = 0.0002;

#ifdef Restart
constexpr int timemax = 20000;
#else
constexpr int timemax = 20000;
#endif // Restart


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



/* 吹吸控制参数*/

/*

// 定义吹吸产生的垂直速度 V_wall
// 建议设置为 0.005 (即千分之五的参考速度)，这是文献中的典型值。
// 如果你想效果更剧烈，可以改成 0.01；想微弱一点，改成 0.002。
constexpr double V_wall_blowing = 0.005; 


*/

#endif