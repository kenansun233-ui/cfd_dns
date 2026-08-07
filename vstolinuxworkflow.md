# CFD_DNS：Visual Studio / Windows 工程迁移到 CentOS 7 + CUDA 11.4 + Makefile

## 1. 迁移目标

本流程用于把 Windows / Visual Studio CUDA 工程迁移为 Linux 可编译、可运行、可清理输出的工程。

迁移链路：

```text
Windows / Visual Studio 工程
        ↓
Linux 兼容修改
        ↓
CentOS 7 + CUDA 11.4 + RTX 3090
        ↓
nvcc + g++
        ↓
Makefile
        ↓
bin/dns_solver
```

编译成功时应看到：

```text
Linking: bin/dns_solver
Build complete: ./bin/dns_solver
```

推荐目录结构：

```text
~/cfd_dns/
├── Makefile
├── maincode/
│   ├── main.cu
│   ├── init.cu
│   ├── rhs.cu
│   ├── info_device.cu
│   └── parameters.cpp
├── src/
│   ├── parameters.h
│   ├── init.cuh
│   ├── rhs.cuh
│   ├── info_device.cuh
│   └── platform_compat.h
├── obj/
├── bin/
└── OutputFile/
```

核心原则：

- 只迁移构建系统、文件 API、路径和 Windows CRT 兼容层。
- 不修改 Navier-Stokes 方程、RHS 离散、时间推进、Poisson 求解、cuFFT 或 CUDA kernel 主体。

## 2. 输出目录逻辑

### 2.1 目标规则

输出文件不再固定写死到单一目录：

```cpp
sprintf(output_path, "./OutputFile/");
```

推荐改成：

```text
OutputFile/<用户指定的任意子目录>/
```

例如：

```text
OutputFile/Re180/omega_2pi/
OutputFile/Re180/omega_pi/
OutputFile/test/debug_case_01/
```

这样可以把不同算例的结果隔离保存，避免多个工况互相覆盖。

### 2.2 推荐运行方式

建议程序支持第一个命令行参数作为输出子目录：

```bash
./bin/dns_solver Re180/omega_2pi
```

对应输出目录：

```text
./OutputFile/Re180/omega_2pi/
```

如果不传参数，默认写入：

```text
./OutputFile/default/
```

### 2.3 推荐 C/C++ 实现

在 `maincode/main.cu` 中使用：

```cpp
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
```

推荐输出路径构造逻辑：

```cpp
static void make_dir_if_needed(const char* path)
{
    if (mkdir(path, 0755) != 0 && errno != EEXIST)
    {
        fprintf(stderr, "Failed to create directory: %s\n", path);
        exit(EXIT_FAILURE);
    }
}

static void set_output_path(int argc,
                            char** argv,
                            char* output_path,
                            size_t output_path_size)
{
    const char* output_subdir = "default";

    if (argc >= 2 && argv[1] != nullptr && argv[1][0] != '\0')
        output_subdir = argv[1];

    if (strstr(output_subdir, "..") != nullptr || output_subdir[0] == '/')
    {
        fprintf(stderr, "Invalid output subdirectory: %s\n", output_subdir);
        exit(EXIT_FAILURE);
    }

    make_dir_if_needed("./OutputFile");

    snprintf(output_path,
             output_path_size,
             "./OutputFile/%s/",
             output_subdir);

    make_dir_if_needed(output_path);
}
```

在 `main()` 中调用：

```cpp
char output_path[512];

set_output_path(argc,
                argv,
                output_path,
                sizeof(output_path));
```

注意：

- 输出路径只允许位于 `./OutputFile/` 下。
- 禁止绝对路径和 `..`，避免误写到工程外部目录。
- 若需要多级目录，例如 `Re180/omega_2pi`，需要递归创建目录；最简单做法是提前用 `mkdir -p` 创建，或在程序里实现递归 mkdir。

### 2.4 用 Makefile 传递输出目录

Makefile 中增加：

```makefile
RUN_DIR ?= default
```

运行规则：

```makefile
run: $(TARGET)
	./$(TARGET) $(RUN_DIR)
```

使用示例：

```bash
make run RUN_DIR=Re180/omega_2pi
```

等价于：

```bash
./bin/dns_solver Re180/omega_2pi
```

## 3. 文件修改清单

### 3.1 `maincode/main.cu`

#### 头文件路径

原 Visual Studio 写法：

```cpp
#include "src/info_device.cuh"
#include "src/parameters.h"
#include "src/init.cuh"
#include "src/rhs.cuh"
```

Makefile 已使用：

```makefile
-I./src
```

因此改为：

```cpp
#include "info_device.cuh"
#include "parameters.h"
#include "init.cuh"
#include "rhs.cuh"
```

否则编译器可能查找：

```text
./src/src/parameters.h
```

#### 删除 Windows 专属头文件

删除：

```cpp
#include <direct.h>
```

Linux 下目录创建需要：

```cpp
#include <sys/stat.h>
#include <sys/types.h>
```

#### 输出路径

不要再写死 Windows 路径：

```cpp
sprintf(output_path, "E:\\lch\\");
```

也不建议写死用户目录：

```cpp
sprintf(output_path, "/home/dell/cfd_dns/OutputFile/");
```

推荐使用第 2 节的逻辑，让结果写入：

```text
./OutputFile/<指定子目录>/
```

#### 文件打开

推荐写法：

```cpp
FILE* run_info = fopen(run_info_name, "w");

if (run_info == nullptr)
{
    fprintf(stderr, "Failed to open file: %s\n", run_info_name);
    exit(EXIT_FAILURE);
}
```

追加模式：

```cpp
FILE* run_info = fopen(run_info_name, "a+");
```

#### GPU 选择

服务器有两张 RTX 3090：

```text
GPU 0
GPU 1
```

使用第二张 GPU：

```cpp
CHECK_CUDA(cudaSetDevice(1));
```

运行前检查：

```bash
nvidia-smi
nvidia-smi -i 1
```

### 3.2 `maincode/init.cu`

删除：

```cpp
#include <direct.h>
#include <io.h>
#include <fstream>
```

加入：

```cpp
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
```

保留：

```cpp
#include <random>
#include <iostream>
```

include 路径统一为：

```cpp
#include "parameters.h"
#include "init.cuh"
#include "rhs.cuh"
#include "platform_compat.h"
```

推荐文件打开函数：

```cpp
static FILE* open_file_checked(const char* filename,
                               const char* mode)
{
    FILE* fp = fopen(filename, mode);

    if (fp == nullptr)
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    return fp;
}
```

#### Restart 文件搜索

Windows 原 API：

```cpp
_findfirst
_findnext
_findclose
_finddata_t
```

Linux 改为：

```cpp
DIR* dir = opendir(output_path);
struct dirent* entry;

while ((entry = readdir(dir)) != nullptr)
{
    ...
}

closedir(dir);
```

逻辑保持：

```text
扫描当前 output_path
        ↓
寻找 restart_XXXXXXXX.dat
        ↓
提取 timestep
        ↓
找到最大的 timestep
        ↓
作为最新 restart
```

#### 将 `std::ifstream` 改为 C 文件读取

原：

```cpp
std::ifstream file(filename);
```

改：

```cpp
FILE* file = fopen(filename, "r");

if (file == nullptr)
{
    fprintf(stderr, "Unable to open restart file: %s\n", filename);
    exit(EXIT_FAILURE);
}
```

读取：

```cpp
int ret = fscanf(file,
                 "%lf %lf %lf %lf %lf %lf %lf",
                 &flu_host[id].u,
                 &flu_host[id].v,
                 &flu_host[id].w,
                 &flu_host[id].p,
                 &var_for_input[id].uold,
                 &var_for_input[id].vold,
                 &var_for_input[id].wold);

if (ret != 7)
{
    fprintf(stderr, "Error reading restart data: %s\n", filename);
    exit(EXIT_FAILURE);
}

fclose(file);
```

### 3.3 `maincode/rhs.cu`

迁移原则：

```text
不修改 DNS 数值算法，只修改平台相关内容。
```

删除：

```cpp
#include <direct.h>
#include <fstream>
```

保留：

```cpp
#include <iostream>
```

include 路径统一：

```cpp
#include "parameters.h"
#include "init.cuh"
#include "rhs.cuh"
```

当前 warning 不影响编译：

```text
warning: variable "ks" was set but never used
warning: variable "iu" was set but never used
warning: variable "jp" was declared but never referenced
```

### 3.4 `maincode/info_device.cu`

删除：

```cpp
#include <direct.h>
#include <fstream>
```

原：

```cpp
#include "src/info_device.cuh"
```

改：

```cpp
#include "info_device.cuh"
```

GPU 查询逻辑保持：

```cpp
cudaGetDeviceCount()
cudaGetDeviceProperties()
```

### 3.5 `maincode/parameters.cpp`

原：

```cpp
#include "src/parameters.h"
```

改：

```cpp
#include "parameters.h"
```

因为 `parameters.h` 使用：

```cpp
#include <cufft.h>
```

所以 Makefile 中编译 `.cpp` 时，`g++` 必须包含：

```makefile
-I/usr/local/cuda/include
```

否则会报：

```text
fatal error: cufft.h: No such file or directory
```

### 3.6 头文件

`src/init.cuh`：

- 删除 `<direct.h>` 和 `<fstream>`。
- 保留真正需要的 CUDA 和项目头文件。

`src/rhs.cuh`：

- 保留 CUDA kernel 和辅助函数声明。
- 通常不需要平台相关修改。

`src/info_device.cuh`：

```cpp
#ifndef info_device
#define info_device

void getdevice();
int _ConvertSMVer2Cores(int major, int minor);

#endif
```

`src/parameters.h`：

- 保持物理参数、网格、结构体不变。
- 不因操作系统迁移修改 DNS 数值模型。

## 4. 新增 `src/platform_compat.h`

用于兼容原 Visual Studio 中的：

```cpp
fopen_s
sprintf_s
strcat_s
sscanf_s
```

参考内容：

```cpp
#ifndef PLATFORM_COMPAT_H
#define PLATFORM_COMPAT_H

#ifndef _WIN32

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <cstddef>

inline int fopen_s(FILE** fp,
                   const char* filename,
                   const char* mode)
{
    if (fp == nullptr)
        return EINVAL;

    *fp = fopen(filename, mode);

    return (*fp != nullptr) ? 0 : errno;
}

template <size_t N, typename... Args>
inline int sprintf_s(char (&buffer)[N],
                     const char* format,
                     Args... args)
{
    return snprintf(buffer, N, format, args...);
}

template <typename... Args>
inline int sprintf_s(char* buffer,
                     size_t size,
                     const char* format,
                     Args... args)
{
    return snprintf(buffer, size, format, args...);
}

template <size_t N>
inline int strcat_s(char (&dest)[N],
                    const char* src)
{
    size_t len = strlen(dest);

    if (len >= N)
        return ERANGE;

    strncat(dest, src, N - len - 1);

    return 0;
}

#define sscanf_s sscanf

#endif
#endif
```

长期建议逐步把：

```cpp
sprintf_s
strcat_s
```

直接改为标准接口：

```cpp
snprintf
```

然后删除兼容层。

## 5. Makefile

推荐 Makefile：

```makefile
# ============================================================================
# CFD_DNS CUDA Makefile
#
# GPU  : NVIDIA RTX 3090
# CUDA : 11.4
# Arch : sm_86
# ============================================================================

NVCC := nvcc
CXX  := g++

CUDA_PATH := /usr/local/cuda

CUDA_INCLUDE := -I$(CUDA_PATH)/include
CUDA_LIB     := -L$(CUDA_PATH)/lib64

SRC_DIR    := ./maincode
INC_DIR    := ./src
OBJ_DIR    := ./obj
BIN_DIR    := ./bin
OUTPUT_DIR := ./OutputFile
RUN_DIR    ?= default

TARGET := $(BIN_DIR)/dns_solver

INCLUDES := -I$(INC_DIR)
ARCH_FLAGS := -arch=sm_86

NVCC_FLAGS := -O3 \
              -std=c++11 \
              $(ARCH_FLAGS) \
              $(CUDA_INCLUDE) \
              $(INCLUDES) \
              -Xcompiler -fopenmp

CXX_FLAGS := -O3 \
             -std=c++11 \
             -fopenmp \
             $(CUDA_INCLUDE) \
             $(INCLUDES)

LDFLAGS := $(CUDA_LIB) \
           -lcudart \
           -lcufft \
           -lm \
           -lpthread \
           -Xcompiler -fopenmp

SOURCES := \
    $(SRC_DIR)/info_device.cu \
    $(SRC_DIR)/init.cu \
    $(SRC_DIR)/main.cu \
    $(SRC_DIR)/parameters.cpp \
    $(SRC_DIR)/rhs.cu

OBJECTS := \
    $(OBJ_DIR)/info_device.o \
    $(OBJ_DIR)/init.o \
    $(OBJ_DIR)/main.o \
    $(OBJ_DIR)/parameters.o \
    $(OBJ_DIR)/rhs.o

.PHONY: all clean clean-output clean-case cleanall run debug gpuinfo

all: $(TARGET)

$(OBJ_DIR) $(BIN_DIR) $(OUTPUT_DIR):
	@mkdir -p $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling C++: $<"
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(TARGET): $(OBJECTS) | $(BIN_DIR) $(OUTPUT_DIR)
	@echo "Linking: $@"
	$(NVCC) $(ARCH_FLAGS) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Build complete: ./$(TARGET)"

clean:
	@echo "Cleaning object and binary files..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)

clean-output:
	@echo "Cleaning all generated output files..."
	rm -rf $(OUTPUT_DIR)
	@mkdir -p $(OUTPUT_DIR)

clean-case:
	@echo "Cleaning generated output case: $(OUTPUT_DIR)/$(RUN_DIR)"
	rm -rf $(OUTPUT_DIR)/$(RUN_DIR)

cleanall: clean clean-output

run: $(TARGET) | $(OUTPUT_DIR)
	./$(TARGET) $(RUN_DIR)

debug: NVCC_FLAGS := -O0 -g -G -std=c++11 \
                     $(ARCH_FLAGS) \
                     $(CUDA_INCLUDE) \
                     $(INCLUDES)

debug: clean all

gpuinfo:
	nvidia-smi
```

关键点：

- RTX 3090 使用 `ARCH_FLAGS := -arch=sm_86`。
- `parameters.cpp` 需要 `$(CUDA_INCLUDE)`。
- cuFFT 必须链接 `-lcufft`。
- Makefile 命令前必须是真正的 Tab。
- `RUN_DIR` 控制 `OutputFile` 下的算例子目录。

## 6. 生成文件删除功能

### 6.1 删除编译生成文件

删除 `obj/` 和 `bin/`：

```bash
make clean
```

不会删除计算结果。

### 6.2 删除全部计算输出

删除整个 `OutputFile/` 并重建空目录：

```bash
make clean-output
```

或：

```bash
make cleanall
```

注意：这会删除所有算例结果、restart 文件和日志。

### 6.3 删除指定算例输出

删除 `OutputFile` 下某个子目录：

```bash
make clean-case RUN_DIR=Re180/omega_2pi
```

实际删除：

```text
OutputFile/Re180/omega_2pi/
```

这适合只清理某个工况，不影响其他输出目录。

### 6.4 手动删除输出文件

查看输出目录：

```bash
find OutputFile -maxdepth 2 -type d
```

删除指定目录：

```bash
rm -rf OutputFile/Re180/omega_2pi
```

删除前建议确认：

```bash
ls -lh OutputFile/Re180/omega_2pi
```

## 7. 常用命令

### 7.1 Windows 本地 SSH 登录

完整地址：

```bash
ssh dell@10.249.181.204
```

如果已配置 SSH Host：

```bash
ssh dell
```

登录成功后类似：

```text
(base) [dell@node01 ~]$
```

### 7.2 上传工程

上传整个工程：

```bash
scp -r D:\cfd_dns dell@10.249.181.204:/home/dell/
```

上传单个文件：

```bash
scp D:\xxx\main.cu dell@10.249.181.204:/home/dell/cfd_dns/maincode/
```

### 7.3 进入工程

```bash
cd ~/cfd_dns
pwd
```

预期：

```text
/home/dell/cfd_dns
```

### 7.4 查看文件

```bash
ls
ls -lh
ls -lh maincode
ls -lh src
```

### 7.5 查看身份与服务器

```bash
whoami
hostname
```

### 7.6 GPU 检查

```bash
nvidia-smi
watch -n 1 nvidia-smi
nvidia-smi -i 1
```

### 7.7 CUDA / GCC 检查

```bash
nvcc --version
which nvcc
gcc --version
g++ --version
```

## 8. 编译、运行与调试

### 8.1 普通编译

```bash
make -j4
```

如果只修改 `.cu` 或 `.cpp`，一般直接使用：

```bash
make -j4
```

如果修改了 `.h`、`.cuh` 或 `Makefile`，建议：

```bash
make clean
make -j4
```

因为当前 Makefile 尚未自动生成完整头文件依赖。

### 8.2 运行默认算例

```bash
./bin/dns_solver
```

或：

```bash
make run
```

默认输出：

```text
OutputFile/default/
```

### 8.3 运行指定输出目录

```bash
./bin/dns_solver Re180/omega_2pi
```

或：

```bash
make run RUN_DIR=Re180/omega_2pi
```

输出：

```text
OutputFile/Re180/omega_2pi/
```

### 8.4 后台运行正式算例

```bash
nohup ./bin/dns_solver Re180/omega_2pi > run_Re180_omega_2pi.log 2>&1 &
```

查看进程：

```bash
ps -ef | grep '[d]ns_solver'
```

查看日志：

```bash
tail -f run_Re180_omega_2pi.log
```

退出日志实时查看：

```text
Ctrl + C
```

这只会停止 `tail`，不会停止后台的 `dns_solver`。

## 9. 检查程序和输出

### 9.1 检查生成的程序

```bash
ls -lh bin
file bin/dns_solver
ldd bin/dns_solver
```

如果 `ldd` 出现：

```text
not found
```

说明运行库路径存在问题。

### 9.2 查看输出文件

默认输出：

```bash
ls -lh OutputFile/default
head OutputFile/default/run_info.dat
tail OutputFile/default/stat.dat
tail -f OutputFile/default/prgrad.dat
```

指定算例输出：

```bash
ls -lh OutputFile/Re180/omega_2pi
head OutputFile/Re180/omega_2pi/run_info.dat
tail OutputFile/Re180/omega_2pi/stat.dat
tail -f OutputFile/Re180/omega_2pi/prgrad.dat
```

## 10. 终止运算

如果程序正在前台运行，按：

```text
Ctrl + C
```

不要输入：

```text
SIGINT
```

`Ctrl+C` 会给程序发送 SIGINT 信号。

检查程序是否仍在运行：

```bash
ps -ef | grep '[d]ns_solver'
```

没有输出，表示已经停止。

终止后台程序：

```bash
ps -ef | grep '[d]ns_solver'
kill 12345
```

如果仍无法结束：

```bash
kill -9 12345
```

一般优先使用普通 `kill`。

## 11. 搜索源码

检查 Windows 残留：

```bash
grep -rn "direct.h" maincode src
grep -rn "io.h" maincode src
grep -rn "fstream" maincode src
```

查 GPU 选择：

```bash
grep -rn "cudaSetDevice" maincode src
```

查 `timemax`：

```bash
grep -rn "timemax" maincode src
```

查 Restart：

```bash
grep -rn "Restart" maincode src
```

## 12. 查看指定源码行

例如编译报：

```text
main.cu(70): error
```

查看第 65 到 75 行：

```bash
nl -ba maincode/main.cu | sed -n '65,75p'
```

通用格式：

```bash
nl -ba 文件名 | sed -n '起始行,结束行p'
```

## 13. Vim 常用命令

打开文件：

```bash
vim maincode/main.cu
```

进入编辑：

```text
i
```

退出编辑模式：

```text
Esc
```

保存退出：

```text
:wq
```

不保存退出：

```text
:q!
```

跳到第一行：

```text
gg
```

删除整个文件内容：

```text
gg
dG
```

## 14. 推荐日常开发流程

```text
1. SSH 登录
        ↓
2. cd ~/cfd_dns
        ↓
3. 修改 .cu / .cpp / .h / .cuh
        ↓
4. make clean && make -j4
        ↓
5. make run RUN_DIR=<算例目录>
        ↓
6. watch -n 1 nvidia-smi
        ↓
7. 检查 OutputFile/<算例目录>
        ↓
8. 检查数值结果
```

正式 DNS 算例：

```text
修改参数
   ↓
make clean
make -j4
   ↓
确认 run_info
   ↓
nohup 后台运行
   ↓
监控 GPU
   ↓
检查 stat / tau_wall / restart
```
