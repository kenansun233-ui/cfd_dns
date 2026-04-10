% =========================================================
% 从 51 个 restart 文件批量读取并绘制 Figure 5 
% (终极版：包含动参考系相位修正与分箱平均)
% =========================================================
clear; clc; close all;

%% 1. 物理参数与网格设定
nu = 3.5298e-4;         
% uCRF = 1.0;                 % 【修改】参考系速度设为零
nxp = 192;              
nyp = 191;  
nyc = nyp-1;
nzp = 160;              
xlength = 12.56637061;  
dx = xlength / nxp;    
% 吹吸与时间参数 (Case S)
alpha = 5.0;            
A_quad = 0.02986;       
% A_quad = 0.036216;
lambda_t = 2 * pi / alpha; 

% ==== 动参考系关键时间参数 ====
% dt = 0.001;
% start_step = 40000;      % 您的起始步数
% step_interval = 100;     % 输出间隔

% 输出 skip 设置
xskip = 2;
zskip = 2;
yskip = 1;
nxp_out = nxp / xskip;       
nzp_out = nzp / zskip;       
nyp_out = (nyp + 1) / yskip; % 192

%% 2. 批量读取与绝对相位映射
data_folder = 'H:\dns_2026\Quadrio2007\f5_0.47\'; 
dns_dir = 'H:\dns_2026\Quadrio2007\f5_0.47';  
% file_pattern = fullfile(data_folder, 'restart*.dat');
file_pattern = fullfile(data_folder, 'dns_data*.dat');
files = dir(file_pattern);
num_files = length(files);

fname_stat   = 'stat.dat';
fname_prgrad = 'prgrad.dat';
fname_mesh   = 'mesh.dat';
% plot(prgrad, 'r-', 'LineWidth', 1.5);
% 【修改】读取 mesh.dat 以获取网格坐标，修复 xyz 未定义错误
xyz = importdata(fullfile(dns_dir, fname_mesh));

prgrad = importdata(fullfile(dns_dir, fname_prgrad));
rear_ratio = 0.5; % 明确：取后30%数据，p=0.3（可修改为0.2、0.8等，0<rear_ratio≤1）
Utau = sqrt(abs(mean(prgrad(floor(length(prgrad)*(1-rear_ratio))+1 : end))));
vormag = Utau*Utau/nu;
fprintf('Utau = %.6f, vormag = %.6f\n', Utau, vormag);   

if length(xyz) ~= (nxp_out + nyp_out + nzp_out)
     error('mesh.dat 长度不匹配，请检查 x/y/z_length 设置');
end
% 提取 yc (包含虚拟网格)
y = xyz(nxp_out+1 : nxp_out+nyp_out);
    
% 定义内部点坐标 yc
yc = y(2:end-1); 
y1 = yc(1);

if num_files == 0
    error('未找到任何 restart 文件，请检查文件夹路径。');
end
fprintf('共发现 %d 个瞬时流场文件。\n开始提取 %d 个数据点的绝对相位...\n', 96*num_files);

% 初始化大数组，用于收集所有时刻、所有流向位置的数据点
all_phases = [];
all_tau_w = [];

% 生成流向网格的相对计算坐标 (Computational Coordinate)
x_comp = (0:xskip:nxp-1)' * dx; 

for f = 1:num_files
    filename = fullfile(files(f).folder, files(f).name);
    
    if mod(f, 10) == 0 || f == 1 || f == num_files
        fprintf('正在处理文件: %d / %d ...\n', f, num_files);
    end
    
    % 加载数据，重构，展向平均
    data = load(filename);
    u_col = data(:, 1); 
    u_3d = reshape(u_col, [nxp_out, nzp_out, nyp_out]);
    
    % 【极其重要】提取第2层作为真实的底层流体速度！
    u_wall_layer = u_3d(:, :, 2); 
    u_1d = mean(u_wall_layer, 2); 
    
    % 计算当前截面的局部切应力
%     u_abs = u_1d + uCRF;
%     tau_w = nu * (u_abs / y1);
     u_abs = u_1d ;
     tau_w = nu * (u_abs  / y1);
    % ==== 核心：动坐标系相位追踪 ====
    % 计算当前文件的实际物理时间
%     t_current = (start_step + f * step_interval) * dt;
    % 将计算坐标转换回固定在壁面上的绝对物理坐标
%     x_phys = x_comp + uCRF * t_current;
    % 计算绝对相位
%     phase = mod(x_phys, lambda_t) / lambda_t;
    phase = mod( x_comp, lambda_t) / lambda_t;
    % 将当前时刻的 96 个点追加到大数组中
    all_phases = [all_phases; phase];
    all_tau_w = [all_tau_w; tau_w];
end

%% 3. 全局统计与相位分箱平均 (Phase Binning)
% 计算所有时空样本的全局平均切应力
cf_mean = mean(all_tau_w);          
all_cf_ratio = all_tau_w / cf_mean;     

% 分箱设置：将 0~1 的相位周期划分为 50 个极其精细的箱子
N_bins = 49; 
edges = linspace(0, 1, N_bins + 1);
bin_centers = (edges(1:end-1) + edges(2:end))' / 2;
binned_cf = zeros(N_bins, 1);

fprintf('正在进行相位分箱平均降噪...\n');
% for i = 1:N_bins
%     % 找到落在当前相位区间内的所有散点
%     idx = (all_phases >= edges(i)) & (all_phases < edges(i+1));
%     % 在箱子内部求平均，彻底抹平湍流噪声
%     binned_cf(i) = mean(all_cf_ratio(idx));
%     
% end

bin_counts = zeros(N_bins, 1);
for i = 1:N_bins
    idx = (all_phases >= edges(i)) & (all_phases < edges(i+1));
    bin_counts(i) = sum(idx);
    binned_cf(i) = mean(all_cf_ratio(idx));
end
disp('各箱子样本数：');
disp(bin_counts');
% 计算对应的理论吹吸速度用于画图
% 【修改】使用计算得到的 Utau 代替原先的硬编码 utau
v_wall_plus = (A_quad * cos(alpha .* bin_centers * lambda_t)) / Utau; 




%% 4. 绘图展示
figure('Position', [150, 150, 750, 500], 'Color', 'w');

% 左轴：局部摩擦系数（黑色实线，无数据点）
yyaxis left
plot(bin_centers, binned_cf, '-b', 'LineWidth', 1.5);
ylabel('$\widetilde{C}_f / \overline{C}_f$', 'Interpreter', 'latex', 'FontSize', 18);
ylim([0 2.5]);   % 适当放宽以容纳参考线峰值
ax = gca;
ax.YColor = 'k';

% 右轴：吹吸速度分布（红色虚线）
yyaxis right
plot(bin_centers, v_wall_plus, '--r', 'LineWidth', 2);
ylabel('$V_w^+$', 'Interpreter', 'latex', 'FontSize', 18);
ylim([-0.5 0.5]);
ax = gca;
ax.YColor = 'r';

% ==== 用户指定参考线文件路径 ====
ref_file_path = 'H:\dns_2026\Quadrio2007\dragc.txt';   % 请修改为您的实际路径

if exist(ref_file_path, 'file')
    % 使用 importdata 读取逗号分隔文件
    ref_data = importdata(ref_file_path);
    if isstruct(ref_data)
        % importdata 可能返回结构体，数据在 .data 字段
        ref_array = ref_data.data;
    else
        % 直接返回数值矩阵
        ref_array = ref_data;
    end
    ref_x = ref_array(:, 1);
    ref_y = ref_array(:, 2);
    
    yyaxis left
    hold on;
    plot(ref_x, ref_y, '-k', 'LineWidth', 1.5, 'DisplayName', 'Reference (dragc)');
    hold off;
else
    warning('参考线文件未找到：%s', ref_file_path);
end

xlabel('$x / \lambda_x$', 'Interpreter', 'latex', 'FontSize', 18);
xlim([0 1]);
xticks(0:0.2:1);
grid on;
title(sprintf('Phase-averaged Local Friction (Case S: \\alpha=5.0, Samples: %d)', num_files), 'FontSize', 14);
legend('Local Friction (Cf) - Phase Binned', 'Transpiration Velocity (V_w)', 'Reference (dragc)', 'Location', 'northeast');
fprintf('\n处理完成！\n');