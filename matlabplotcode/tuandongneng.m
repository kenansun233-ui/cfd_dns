clear; clc; close all;
%% ========================================================================
%% 1. 基础配置 (物理与路径参数)
%% ========================================================================
% --- 路径 ---
dns_dir = 'E:\file\zd\zero_flow\Re_delta=500';          % 修改为你实际存放数据的目录
fname_stat   = 'stat.dat';
fname_mesh   = 'mesh.dat';
% --- 网格输出参数 (对应你 C++ 里的 xskip, yskip, zskip) ---
% 如果 xskip=2, 那么 x_length = nxp/2 = 384/2 = 192
x_length = 384 / 2;      
% y_length = 191 + 1;    
y_length = 257 + 1; 
% y_length = 383 + 1; 
% y_length = 513 + 1; 
z_length = 256 / 2;   
fprintf('正在读取 %s...\n', fullfile(dns_dir, fname_mesh));
xyz = importdata(fullfile(dns_dir, fname_mesh));
if length(xyz) ~= (x_length + y_length + z_length)
     error('mesh.dat 长度不匹配，请检查 x/y/z_length 设置');
end
% 提取 y 坐标 (包含虚拟网格)
y = xyz(x_length+1 : x_length+y_length);
% 定义内部点坐标 yc
yc = y(2:end-1); 
nyc = length(yc);
U_osc = 1.0;
omega = 2.0 * pi;

Re_delta = 500.0;                 % 你当前跑的目标 Re_delta

nu = (2.0 * U_osc^2) / (omega * Re_delta^2);
delta = sqrt(2.0 * nu / omega); % Stokes 穿透深度
%% ========================================================================
%% 3. 读取 stat.dat (按你的矩阵逻辑)
%% ========================================================================
    stat = importdata(fullfile(dns_dir, fname_stat));
    data_num = 11; 
    num_steps = floor(size(stat, 1) / data_num);
    
    if num_steps < 1, error('stat.dat 数据不足'); end

    % 累加平均
    stat_sum = zeros(data_num, nyc); 
    
    % 检查 stat 列数是否匹配
    if size(stat, 2) ~= nyc
        warning('stat.dat 列数 (%d) 与 nyc (%d) 不一致，后续索引可能出错', size(stat, 2), nyc);
    end

    for i = 1:num_steps
        base_idx = data_num * (i-1);
        stat_sum = stat_sum + stat(base_idx+1:base_idx+data_num, :);
    end
    stat_avg = stat_sum / num_steps;
%% 1. 提取数据与计算脉动量
% 提取平均速度与速度平方的均值
um = stat_avg(4, :); vm = stat_avg(5, :); wm = stat_avg(6, :); 
um2= stat_avg(8, :); vm2= stat_avg(9, :); wm2= stat_avg(10,:); 

% 计算三个方向的脉动速度均方值 (u'^2, v'^2, w'^2)
% 加入 max(..., 0) 是为了防止由于浮点数截断误差在死水区产生极小的负数
uu = max(um2 - um.^2, 0);
vv = max(vm2 - vm.^2, 0);
ww = max(wm2 - wm.^2, 0);

% 计算湍流动能 TKE = 0.5 * (u'^2 + v'^2 + w'^2)
tke = 0.5 * (uu + vv + ww);

% 注意：请在这里准备好你的法向坐标 yc (从 mesh.dat 读取) 或无量纲坐标 (yc/delta 或 y^+)
% 这里暂时用网格索引 1:length(tke) 占位，实际绘图时请将其替换为真实的 Y 坐标
y_axis = yc/delta; 

%% 2. 绘制湍流动能 (TKE) 剖面图
figure('Position', [100, 150, 500, 650], 'Color', 'w');
hold on; box on; grid on;
uu_tke = max(sqrt(tke), 1e-8);
plot(uu_tke, y_axis, '-', 'LineWidth', 2.5, 'Color', [0.8500 0.3250 0.0980]);

% 图表美化
set(gca, 'XScale', 'log'); % ⚡️ 核心魔法：开启 X 轴对数显示 ⚡️
xlim([1e-4, max(uu_tke)*1.5]); % 限制合适的显示范围
xlabel('湍动能 {\it k}', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('法向高度 {\it y}', 'FontSize', 16, 'FontWeight', 'bold');
title('湍动能法向分布', 'FontSize', 18);
set(gca, 'FontSize', 14, 'LineWidth', 1.5, 'GridAlpha', 0.3);

%% 3. [进阶] 揭开视觉错觉：三维脉动强度的真实对比
figure('Position', [200, 200, 950, 500], 'Color', 'w');
uu_safe = max(sqrt(uu), 1e-8);
vv_safe = max(sqrt(vv), 1e-8);
ww_safe = max(sqrt(ww), 1e-8);

% ==============================================================
% 左图：你现在看到的“视觉错觉” (线性坐标)
% ==============================================================
subplot(1, 2, 1);
hold on; box on; grid on;

plot(uu_safe, y_axis, '-', 'LineWidth', 2.5, 'Color', [0 0.4470 0.7410], 'DisplayName', 'u_{rms} (流向脉动)');
plot(vv_safe, y_axis, '-', 'LineWidth', 2.5, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'v_{rms} (法向脉动)');
plot(ww_safe, y_axis, '-', 'LineWidth', 2.5, 'Color', [0.9290 0.6940 0.1250], 'DisplayName', 'w_{rms} (展向脉动)');

xlabel('脉动速度方根 (线性)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('法向高度', 'FontSize', 14, 'FontWeight', 'bold');
title('线性坐标 (v_{rms} 显得极其微小)', 'FontSize', 15);
legend('Location', 'best', 'FontSize', 12);
set(gca, 'FontSize', 13, 'LineWidth', 1.5, 'GridAlpha', 0.3);

% ==============================================================
% 右图：物理真相 (对数坐标 XScale = 'log')
% ==============================================================
subplot(1, 2, 2);
hold on; box on; grid on;

% 过滤掉 0 值以防止对数坐标报错

plot(uu_safe, y_axis, '-', 'LineWidth', 2.5, 'Color', [0 0.4470 0.7410]);
plot(vv_safe, y_axis, '-', 'LineWidth', 2.5, 'Color', [0.8500 0.3250 0.0980]);
plot(ww_safe, y_axis, '-', 'LineWidth', 2.5, 'Color', [0.9290 0.6940 0.1250]);

set(gca, 'XScale', 'log'); % ⚡️ 核心魔法：开启 X 轴对数显示 ⚡️
xlim([1e-4, max(uu_safe)*1.5]); % 限制合适的显示范围

xlabel('脉动速度方根 (对数 Log Scale)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('法向高度', 'FontSize', 14, 'FontWeight', 'bold');
title('对数坐标 (真相：v_{rms} 存在明显的峰值)', 'FontSize', 15);
set(gca, 'FontSize', 13, 'LineWidth', 1.5, 'GridAlpha', 0.3);