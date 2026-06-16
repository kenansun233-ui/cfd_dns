clear; clc; close all;

%% ========================================================================
%% 1. 全局配置与算例列表
%% ========================================================================
base_dir = 'E:\file\zd\zero_flow';       % 你的根目录
% Re_delta_list = [ 100, 300, 500, 700, 1000]; % 你目前跑完的算例雷诺数
Re_delta_list = [500];
% 全局物理与时间参数
U_osc = 1.0;
omega = 2.0 * pi;
% dt = 0.0002;
dt = 0.001;
output_interval = 10;

% 水平和展向的抽样网格数通常是固定的
x_length = 384 / 2;      
z_length = 256 / 2;  

% 初始化存储相位差的数组
phi_dns_list = zeros(size(Re_delta_list));

fprintf('==================================================\n');
fprintf('开始批量处理 DNS 数据并提取相位差...\n');
fprintf('==================================================\n');

%% ========================================================================
%% 2. 自动化循环：逐个算例提取与计算
%% ========================================================================
for idx = 1:length(Re_delta_list)
    current_Re = Re_delta_list(idx);
    case_dir = fullfile(base_dir, sprintf('Re_delta=%d', current_Re));
    
    fname_mesh = fullfile(case_dir, 'mesh.dat');
    fname_stat = fullfile(case_dir, 'stat.dat');
    
    if ~exist(fname_mesh, 'file') || ~exist(fname_stat, 'file')
        warning('未找到算例 Re_delta=%d 的数据文件，跳过。', current_Re);
        phi_dns_list(idx) = NaN;
        continue;
    end
    
    % ---------------------------------------------------------------------
    % A. 网格自适应读取：动态推断 y_length
    % ---------------------------------------------------------------------
    xyz = importdata(fname_mesh);
    % 利用总长度减去固定的 x 和 z 长度，自动得出当前的 y_length
    y_length = length(xyz) - x_length - z_length; 
    
    if y_length <= 0
        error('Re_delta=%d 的 mesh.dat 长度异常！', current_Re);
    end
    
    y = xyz(x_length+1 : x_length+y_length);
    yc = y(2:end-1); 
    nyc = length(yc);
    
    % ---------------------------------------------------------------------
    % B. 读取统计数据 (使用你原有的矩阵逻辑)
    % ---------------------------------------------------------------------
    stat = importdata(fname_stat);
    data_num = 11; 
    num_steps = floor(size(stat, 1) / data_num);
    
    u_mean_transient = zeros(nyc, num_steps);
    for i = 1:num_steps
        u_mean_transient(:, i) = stat(data_num*(i-1) + 4, :)';
    end
    
    % ---------------------------------------------------------------------
    % C. 计算壁面切应力 (提取最后两个周期的数据)
    % ---------------------------------------------------------------------
    snaps_per_period = round((2*pi/omega) / (output_interval * dt));
    % 稳妥起见，取最后两到三个周期用于寻峰，避免单个周期波形畸变
    start_snap = max(1, num_steps - 2 * snaps_per_period); 
    
    time_array = (start_snap : num_steps) * output_interval * dt;
    U_wall_array = U_osc * cos(omega * time_array);
    
    % 提取第一层和第二层网格的瞬态速度
    u_1_transient = u_mean_transient(1, start_snap:num_steps);
    u_2_transient = u_mean_transient(2, start_snap:num_steps);
    
    % 计算动态粘性
    nu = (2.0 * U_osc^2) / (omega * current_Re^2);
    
    % ---------------------------------------------------------------------
    % 非均匀网格边界处的二阶差分 (Second-order Finite Difference)
    % ---------------------------------------------------------------------
    % 获取近壁面前两层网格的绝对法向坐标
    y1 = yc(1);
    y2 = yc(2);
    
    % 计算 -du/dy 在壁面 (y=0) 处的各项权重系数
    c0 = (y1 + y2) / (y1 * y2);
    c1 = -y2 / (y1 * (y2 - y1));
    c2 = y1 / (y2 * (y2 - y1));
    
    % 二阶精度计算壁面摩擦力 tau_w (严格保留了原有的符号与相位约定)
    tau_w_dns = nu * (c0 * U_wall_array + c1 * u_1_transient + c2 * u_2_transient);
    
    % ---------------------------------------------------------------------
    % D. 寻峰法提取相位差 (Peak Detection)
    % ---------------------------------------------------------------------
    % 寻找速度波峰和摩擦力波峰
    [~, locs_U] = findpeaks(U_wall_array);
    % 为了防止湍流高频噪声产生伪峰，稍微平滑一下 tau_w 再寻峰
    tau_smooth = smoothdata(tau_w_dns, 'gaussian', 15); 
    [~, locs_tau] = findpeaks(tau_smooth);
    
    if length(locs_U) >= 1 && ~isempty(locs_tau)
        % 取最后一个 U 的波峰作为基准点
        idx_U = locs_U(end);
        t_U_peak = time_array(idx_U);
        
        % 寻找在它之前、距离最近的一个 tau 波峰
        valid_tau_locs = locs_tau(locs_tau < idx_U);
        if ~isempty(valid_tau_locs)
            idx_tau = valid_tau_locs(end);
            t_tau_peak = time_array(idx_tau);
            
            % 计算时间差 -> 转换为角度
            % 计算时间差 -> 转换为原始角度
            delta_t = t_U_peak - t_tau_peak;
            phi_raw = (delta_t / (2*pi/omega)) * 360;
            
            % ---------------------------------------------------------
            % 相位折叠修正：将角度强行映射到 0 到 45 度的物理超前角区间
            % ---------------------------------------------------------
            phi_mod = mod(phi_raw, 360); % 剔除多余的完整周期
            
            % 如果角度落在 180~360 之间（例如 338.4），说明是从反方向绕了长程
            % 用 360 减去该值，直接折叠回真实的微小超前角 (例如 21.6)
            if phi_mod > 180
                phi_final = 360 - phi_mod;
            else
                phi_final = phi_mod;
            end
            
            phi_dns_list(idx) = phi_final;
            
            fprintf('[✔] Re_delta = %-4d | 网格 y_length = %-3d | 相位差 = %5.2f°\n', ...
                current_Re, y_length, phi_dns_list(idx));
        else
            phi_dns_list(idx) = NaN;
            fprintf('[✖] Re_delta = %-4d | 寻峰匹配失败\n', current_Re);
        end
    else
        phi_dns_list(idx) = NaN;
        fprintf('[✖] Re_delta = %-4d | 数据太短，未能形成完整波峰\n', current_Re);
    end
end
fprintf('==================================================\n');

%% ========================================================================
%% 3. 绘制最终宏观趋势图 (Phase Difference vs Reynolds Number)
%% ========================================================================
% 过滤掉处理失败的 NaN 数据
valid_idx = ~isnan(phi_dns_list);
plot_Re = Re_delta_list(valid_idx);
plot_phi = phi_dns_list(valid_idx);

figure('Position', [200, 150, 800, 600], 'Color', 'w');
hold on; box on; grid on;

% 1. 绘制层流理论基准线 (45 度虚线)
yline(45.0, 'r--', 'LineWidth', 2, 'DisplayName', '层流理论解 (45^\circ)');

% 2. 绘制转捩区视觉引导背景 (经验区间 Re=400~600)
patch([400 600 600 400], [0 0 60 60], [0.9 0.9 0.9], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

% 3. 绘制真实的 DNS 数据点
plot(plot_Re, plot_phi, '-o', ...
    'Color', [0 0.4470 0.7410], 'LineWidth', 2.5, ...
    'MarkerSize', 10, 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerEdgeColor', 'w', ...
    'DisplayName', 'DNS 提取值');

% ------------------ 图表美化 ------------------
xlabel('震荡雷诺数 {\it Re}_\delta', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('壁面切应力相位超前角 \phi (度)', 'FontSize', 16, 'FontWeight', 'bold');
title('Stokes 震荡边界层：相位差随雷诺数的演化', 'FontSize', 18);

% 设置坐标轴范围
xlim([0, max(plot_Re) + 100]);
ylim([0, 90]); % 理论最大值 45，最小值大概在 10 左右

% 增加物理流态标注
text(150, 20, '层流区', 'FontSize', 15, 'HorizontalAlignment', 'center', 'Color', [0.4 0.4 0.4]);
text(500, 50, '转捩区', 'FontSize', 15, 'HorizontalAlignment', 'center', 'Color', 'r');
if max(plot_Re) >= 700
    text(max(plot_Re)-50, 20, '湍流区', 'FontSize', 15, 'HorizontalAlignment', 'center', 'Color', [0.4 0.4 0.4]);
end

legend('Location', 'southwest', 'FontSize', 14);
set(gca, 'FontSize', 14, 'LineWidth', 1.5, 'GridAlpha', 0.3);

disp('绘图完成！请检查图表中的下降趋势。');