function plot_velocity_profiles
    clc; close all;
    % =========================================================================
    % 1. 基础配置
    % =========================================================================
    dns_dir = 'H:\dns_2026\work\work\cfd_matlab_code\Re_150_UB_ag60000_newbc';   
    fname_stat = 'stat.dat';
    fname_mesh = 'mesh.dat';
    % plot(prgrad, 'r-', 'LineWidth', 1.5);
    Up_file_path = 'H:\dns_2026\work\work\cfd_matlab_code\Up.txt';   % 请修改为实际路径
    % --- 物理参数 ---
    nu = 4.5903E-4;  
    
    % --- 网格参数 ---
    x_length = 128/2;      
    % y_length = 127+1;  
    y_length = 191+1;
    z_length = 128/2;    
    
    % =========================================================================
    % 2. 数据读取
    % =========================================================================
    
    % --- 读取网格 ---
    xyz = importdata(fullfile(dns_dir, fname_mesh));
    if length(xyz) ~= (x_length + y_length + z_length)
         error('mesh.dat 长度不匹配');
    end
    y = xyz(x_length+1 : x_length+y_length);
    yc = y(2:end-1); 
    nyc = length(yc);
    
    % --- 读取统计数据 ---
    stat = importdata(fullfile(dns_dir, fname_stat));
    data_num = 11; 
    num_steps = floor(size(stat, 1) / data_num);
    
    if size(stat, 2) ~= nyc
        warning('stat.dat 列数与 nyp 不一致');
    end
    
    % 统计平均
    stat_sum = zeros(data_num, nyc); 
    for i = 1:num_steps
        base_idx = data_num * (i-1);
        stat_sum = stat_sum + stat(base_idx+1:base_idx+data_num, :);
    end
    stat_avg = stat_sum / num_steps;
    
    % 提取平均速度 U
    um= stat_avg(4, :); 
    
    % --- 读取 Up.txt (文献数据) ---
    % 假设文件在当前目录下
    if exist(Up_file_path, 'file')
        ref_Up = importdata(Up_file_path);
        ref_Up(:,1) = ref_Up(:,1) + 1.0; % 将文献数据的 x 轴 (第一列) + 1，从 [-1, 1] 平移到 [0, 2]
    else
        ref_Up = [];
        warning('未找到 Up.txt 文件: %s', Up_file_path);
    end

    % =========================================================================
    % 3. 物理计算
    % =========================================================================
    % (此处保留计算逻辑，虽然后续绘图只用了 um_raw)
    
    % --- 计算下壁面 (Injection) 摩擦速度 ---
    y_bot = [yc(1), yc(2), yc(3), yc(4)];  % 下壁面插值节点
    u_bot = [um(1), um(2), um(3), um(4)];  % 对应速度值
    % 拉格朗日基函数在 y_bot(1)（第一个点）处的导数计算
    y1_bot = y_bot(1);  % 下壁面目标点（yc(1)）
    % 基函数 L1(y) 在 y1_bot 处的导数
    dL1_bot = ((y1_bot-y_bot(2))*(y1_bot-y_bot(3)) + (y1_bot-y_bot(2))*(y1_bot-y_bot(4)) + (y1_bot-y_bot(3))*(y1_bot-y_bot(4))) / ...
        ((y1_bot-y_bot(2))*(y1_bot-y_bot(3))*(y1_bot-y_bot(4)));
    % 基函数 L2(y) 在 y1_bot 处的导数
    dL2_bot = ((y1_bot-y_bot(1))*(y1_bot-y_bot(3)) + (y1_bot-y_bot(1))*(y1_bot-y_bot(4)) + (y1_bot-y_bot(3))*(y1_bot-y_bot(4))) / ...
        ((y_bot(2)-y_bot(1))*(y_bot(2)-y_bot(3))*(y_bot(2)-y_bot(4)));
    % 基函数 L3(y) 在 y1_bot 处的导数
    dL3_bot = ((y1_bot-y_bot(1))*(y1_bot-y_bot(2)) + (y1_bot-y_bot(1))*(y1_bot-y_bot(4)) + (y1_bot-y_bot(2))*(y1_bot-y_bot(4))) / ...
        ((y_bot(3)-y_bot(1))*(y_bot(3)-y_bot(2))*(y_bot(3)-y_bot(4)));
    % 基函数 L4(y) 在 y1_bot 处的导数
    dL4_bot = ((y1_bot-y_bot(1))*(y1_bot-y_bot(2)) + (y1_bot-y_bot(1))*(y1_bot-y_bot(3)) + (y1_bot-y_bot(2))*(y1_bot-y_bot(3))) / ...
        ((y_bot(4)-y_bot(1))*(y_bot(4)-y_bot(2))*(y_bot(4)-y_bot(3)));

    % 合并导数得到下壁面梯度
    grad_bot = u_bot(1)*dL1_bot + u_bot(2)*dL2_bot + u_bot(3)*dL3_bot + u_bot(4)*dL4_bot;
    tau_w_bot = nu * grad_bot;
    Utau_bot = sqrt(abs(tau_w_bot));
    fprintf('Injection Side (Bottom) Utau = %.6f\n', Utau_bot);
    
    % A. 全局数据
    y_global = yc;
    U_global = um;

    % =========================================================================
    % 4. 绘图 (Global Velocity Profile Only)
    % =========================================================================
    
    figure('Units', 'pixels', 'Position', [100, 100, 600, 500], 'Color', 'w', 'Name', 'Global Velocity Profile');
    hold on; box on;
    
    % 1. 绘制 DNS 全局速度 (黑色实线)
    p1 = plot(y_global, U_global, 'k-', 'LineWidth', 1.5);
    
    % 2. 绘制 Up.txt 数据 (绿色实线)
    if ~isempty(ref_Up)
        % 逗号前为横坐标(y)，逗号后为纵坐标(U)
        p2 = plot(ref_Up(:,1), ref_Up(:,2), 'g-', 'LineWidth', 1.5);
    end
    
    % 辅助线（兼容旧版 MATLAB，替换 xline 为 line）
    yl = ylim;
    line([0, 0], yl, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
    line([2.0, 2.0], yl, 'Color', 'b', 'LineStyle', '--', 'LineWidth', 1.5);
    line([1.0, 1.0], yl, 'Color', 'g', 'LineStyle', ':', 'LineWidth', 1.5);
    
    xlabel('$y/\delta$', 'Interpreter', 'latex');
    ylabel('$\overline{U}$', 'Interpreter', 'latex');
    title('Global Mean Velocity Profile');
    
    % 设置坐标轴范围
    xlim([0 2.0]); 
    if ~isempty(ref_Up)
        ylim([0 max([max(U_global), max(ref_Up(:,2))])*1.1]);
        legend([p1, p2], {'Present DNS', 'Ref (Up.txt)'}, 'Location', 'Best');
    else
        ylim([0 max(U_global)*1.1]);
        legend([p1], {'Present DNS'}, 'Location', 'Best');
    end
    
    grid on;
    set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
end