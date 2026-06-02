clear; clc; close all;
%% ========================================================================
%% 1. 基础配置 (物理与路径参数)
%% ========================================================================
% --- 路径 ---
dns_dir = 'E:\file\zd\zero_flow\Re_delta=1000';          % 修改为你实际存放数据的目录
fname_stat   = 'stat.dat';
fname_mesh   = 'mesh.dat';
% --- 网格输出参数 (对应你 C++ 里的 xskip, yskip, zskip) ---
% 如果 xskip=2, 那么 x_length = nxp/2 = 384/2 = 192
x_length = 384 / 2;      
% y_length = 191 + 1;    
% y_length = 257 + 1; 
% y_length = 383 + 1; 
y_length = 513 + 1; 
z_length = 256 / 2;   
% --- 物理与震荡参数 (务必与 parameters.h 保持一致) ---
U_osc = 1.0;
omega = 2.0 * pi;

Re_delta = 1000.0;                 % 你当前跑的目标 Re_delta

nu = (2.0 * U_osc^2) / (omega * Re_delta^2);
delta = sqrt(2.0 * nu / omega); % Stokes 穿透深度
dt = 0.0002;            % 时间步长
output_interval = 10;  % clcstat 的输出间隔 (t % 10 == 0)
%% ========================================================================
%% 2. 读取 mesh.dat (按你的逻辑精确提取 yc)
%% ========================================================================
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
%% ========================================================================
%% 3. 读取 stat.dat (按你的矩阵逻辑)
%% ========================================================================
fprintf('正在读取 %s...\n', fullfile(dns_dir, fname_stat));
stat = importdata(fullfile(dns_dir, fname_stat));
data_num = 11; 
num_steps = floor(size(stat, 1) / data_num);
if num_steps < 1
    error('stat.dat 数据不足'); 
end
if size(stat, 2) ~= nyc
    warning('stat.dat 列数 (%d) 与 nyc (%d) 不一致，请检查！', size(stat, 2), nyc);
end
% --- 构建瞬态速度矩阵 u_mean_transient ---
% 维度: [nyc 行(空间高度), num_steps 列(时间步)]
u_mean_transient = zeros(nyc, num_steps);
for i = 1:num_steps
    base_idx = data_num * (i - 1);
    
    % 取出第 4 个变量 (um)，注意转置成列向量以存入矩阵
    u_mean_transient(:, i) = stat(base_idx + 4, :)';
end
fprintf('成功提取 %d 个快照的瞬态数据！\n', num_steps);

% 统计落在 1 个 delta 物理厚度内的网格点数量
points_in_delta = sum(yc <= delta);
fprintf('>>> 当前 Re_delta = %g, 穿透层 (1 delta) 内分配了 %d 个网格。\n', Re_delta, points_in_delta);

% 检查第一层网格高度比
fprintf('>>> 第一层网格高度 y1 = %.4f delta (安全阈值 < 0.05)。\n', yc(1)/delta);
%% ========================================================================
%% 4. 画图对比：双色系演化与解析解比对
%% ========================================================================
figure('Position', [150, 150, 800, 600], 'Color', 'w');
hold on; box on; grid on;
% 计算一个震荡周期包含多少个输出快照
snaps_per_period = round((2*pi/omega) / (output_interval * dt));
start_snap = max(1, num_steps - snaps_per_period + 1);

% 智能抽样画图

max_plot_lines = 8; 

plot_step = max(1, round(snaps_per_period / max_plot_lines));
% plot_indices = start_snap : plot_step : num_steps;
plot_indices = 201 : plot_step : 700;

% === 核心修改：生成两组不同的渐变色 ===
% DNS 用 winter 冷色调 (蓝到绿)，解析解用 autumn 暖色调 (红到黄)
colors_dns = winter(length(plot_indices) + 2); 
colors_ana = autumn(length(plot_indices) + 2); 
plot_idx = 1;
for k = plot_indices
    t_current = k * output_interval * dt;
    
    % 1. 画数值解 (DNS 用冷色调散点)
    plot(u_mean_transient(:, k) / U_osc, yc / delta,'o', 'Color', colors_dns(plot_idx,:), ...
        'MarkerFaceColor', colors_dns(plot_idx,:), 'MarkerSize', 4, ...
        'DisplayName', sprintf('DNS (t=%.3fs)', t_current));
    
    % 2. 画解析解 (理论解用暖色调实线)
    u_analytical = U_osc * exp(-yc ./ delta) .* cos(omega * t_current - yc ./ delta);
    
    plot(u_analytical / U_osc, yc / delta,'-', 'Color', colors_ana(plot_idx,:), 'LineWidth', 1.5, ...
        'HandleVisibility', 'off'); % 关闭实线在循环里的重复图例
    
    plot_idx = plot_idx + 1;
end
% 单独画一条隐藏的红线，用于在图例中专门标示“解析解”
plot(NaN, NaN, '-', 'Color', colors_ana(round(end/2), :), 'LineWidth', 2, ...
    'DisplayName', 'Theory (理论解析解)');
% ================= 图表美化 =================
xlabel('无量纲速度 {\it u} / {\it U}_{osc}', 'FontSize', 14);
ylabel('无量纲高度 {\it y} / \delta', 'FontSize', 14);
title(sprintf('Stokes 第二类流动相位验证 (Re_\\delta = %g)', Re_delta), 'FontSize', 16);
yline(1, 'k--', 'LineWidth', 1.5, 'Label', '\delta (Stokes 穿透深度)', ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 12);
ylim([0, 3]); 
xlim([- 1.2, 1.2]);
legend('Location', 'bestoutside', 'NumColumns', 1);
set(gca, 'FontSize', 12, 'LineWidth', 1.2);
disp('绘图完成！蓝绿色代表 DNS 计算值，红黄色代表理论解析解。');