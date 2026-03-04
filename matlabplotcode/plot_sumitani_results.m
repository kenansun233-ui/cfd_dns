function plot_sumitani_results
    clc; close all;
    % --- 路径 ---
    dns_dir = 'E:\sce_workfile\cfd_matlab_code\Re_150_UB_ag40000';   
    fname_stat   = 'stat.dat';
    fname_prgrad = 'prgrad.dat';
    fname_mesh   = 'mesh.dat';
   
    % --- 物理参数 ---
    nu = 4.5903E-4;  
    
    % --- 网格参数 (保留原代码设置) ---
    x_length = 128/2;      
    y_length = 191+1;
    % y_length = 127+1;      
    z_length = 128/2;    
    xyz = importdata(fullfile(dns_dir, fname_mesh));
    if length(xyz) ~= (x_length + y_length + z_length)
         error('mesh.dat 长度不匹配，请检查 x/y/z_length 设置');
    end
    x = xyz(1 : x_length);
    z = xyz(x_length+y_length+1 : x_length + y_length + z_length);
    y = xyz(x_length+1 : x_length+y_length);
    yc = y(2:end-1);
    nyc = length(yc);
    % --- 2.3 读取统计数据 (stat.dat) ---
    stat = importdata(fullfile(dns_dir, fname_stat));
    data_num = 11; 
    num_steps = floor(size(stat, 1) / data_num);
    
    if num_steps < 1, error('stat.dat 数据不足'); end

    stat_sum = zeros(data_num, nyc); 

    if size(stat, 2) ~= nyc
        warning('stat.dat 列数 (%d) 与 nyp (%d) 不一致，后续索引可能出错', size(stat, 2), nyc);
    end

    
    for i = 1:num_steps
        base_idx = data_num * (i-1);
        stat_sum = stat_sum + stat(base_idx+1:base_idx+data_num, :);
    end
    stat_avg = stat_sum / num_steps;
    % 提取变量
    ox = stat_avg(1, :); oy = stat_avg(2, :); oz = stat_avg(3, :);
    um = stat_avg(4, :); vm = stat_avg(5, :); wm = stat_avg(6, :); pm = stat_avg(7, :);
    um2= stat_avg(8, :); vm2= stat_avg(9, :); wm2= stat_avg(10,:); pm2= stat_avg(11,:);
    % --- 梯度计算 ---
    dudyp = gradient(um, yc);
    dwdyp = gradient(wm, yc);

    prgrad = importdata(fullfile(dns_dir, fname_prgrad));
    rear_ratio = 0.3;             % 取后30%数据，p=0.3（可修改为0.2、0.8等，0<rear_ratio≤1）
    Utau = sqrt(abs(mean(prgrad(floor(length(prgrad)*(1-rear_ratio))+1 : end))));
    vormag = Utau*Utau/nu;
    fprintf('Utau = %.6f, vormag = %.6f\n', Utau, vormag);
    
    % 分别计算下壁面(Bot)和上壁面(Top)的局部摩擦速度
    % Injection Side (Bottom, Index 1)
    % 精度改进：使用三点法 (Three-point scheme) 计算壁面梯度
    % 公式：du/dy = (y2^2 * u1 - y1^2 * u2) / (y1 * y2 * (y2 - y1))
    
    % % --- 下壁面 (Bottom/Injection) ---
    % % 使用 yp(1), yp(2) 和壁面(0)
    % y1 = yc(1);  u1 = um(1);
    % y2 = yc(2);  u2 = um(2);
    % y3 = yc(3); 
    % grad_bot = (y2^2 * u1 - y1^2 * u2) / (y1 * y2 * (y2 - y1));
    % tau_w_bot = nu * grad_bot;
    % Utau_bot = sqrt(abs(tau_w_bot));
    % 
    % % --- 上壁面 (Top/Suction) ---
    % % 使用 yp(nyc), yp(nyc-1) 和壁面(2.0)
    % u_top1 = um(nyc);
    % u_top2 = um(nyc-1);
    % u_top3 = um(nyc-2);
    % 
    % denominator = (y1 - y2) * (y1 - y3);
    % grad_top = ((2*y1 - y2 - y3)*u_top1 + (y1 - y3)*u_top2 + (y1 - y2)*u_top3) /denominator;
    % tau_w_top = nu * abs(grad_top);
    % Utau_top = sqrt(abs(tau_w_top));
  

%%

y_bot = [yc(1), yc(2), yc(3), yc(4)];  % 下壁面插值节点
u_bot = [um(1), um(2), um(3), um(4)];  % 对应速度值

% 2. 检查下壁面节点是否重复（避免分母为0）
if length(unique(y_bot)) < 4
    error('下壁面插值节点yc(1-4)存在重复，无法进行三阶拉格朗日插值');
end

% 3. 拉格朗日基函数在y_bot(1)（第一个点）处的导数计算
y1_bot = y_bot(1);  % 下壁面目标点（yc(1)）
% 基函数L1(y)在y1_bot处的导数
dL1_bot = ((y1_bot-y_bot(2))*(y1_bot-y_bot(3)) + (y1_bot-y_bot(2))*(y1_bot-y_bot(4)) + (y1_bot-y_bot(3))*(y1_bot-y_bot(4))) / ...
    ((y1_bot-y_bot(2))*(y1_bot-y_bot(3))*(y1_bot-y_bot(4)));
% 基函数L2(y)在y1_bot处的导数
dL2_bot = ((y1_bot-y_bot(1))*(y1_bot-y_bot(3)) + (y1_bot-y_bot(1))*(y1_bot-y_bot(4)) + (y1_bot-y_bot(3))*(y1_bot-y_bot(4))) / ...
    ((y_bot(2)-y_bot(1))*(y_bot(2)-y_bot(3))*(y_bot(2)-y_bot(4)));
% 基函数L3(y)在y1_bot处的导数
dL3_bot = ((y1_bot-y_bot(1))*(y1_bot-y_bot(2)) + (y1_bot-y_bot(1))*(y1_bot-y_bot(4)) + (y1_bot-y_bot(2))*(y1_bot-y_bot(4))) / ...
    ((y_bot(3)-y_bot(1))*(y_bot(3)-y_bot(2))*(y_bot(3)-y_bot(4)));
% 基函数L4(y)在y1_bot处的导数
dL4_bot = ((y1_bot-y_bot(1))*(y1_bot-y_bot(2)) + (y1_bot-y_bot(1))*(y1_bot-y_bot(3)) + (y1_bot-y_bot(2))*(y1_bot-y_bot(3))) / ...
    ((y_bot(4)-y_bot(1))*(y_bot(4)-y_bot(2))*(y_bot(4)-y_bot(3)));

% 4. 合并导数得到下壁面梯度
grad_bot = u_bot(1)*dL1_bot + u_bot(2)*dL2_bot + u_bot(3)*dL3_bot + u_bot(4)*dL4_bot;

% ===================== 上壁面 (Top) 三阶拉格朗日插值 =====================
% 1. 取后4个点作为上壁面插值的四点（计算yc(nyc)处的导数）
y_top = [yc(nyc-3), yc(nyc-2), yc(nyc-1), yc(nyc)];  % 上壁面插值节点
u_top = [um(nyc-3), um(nyc-2), um(nyc-1), um(nyc)];  % 对应速度值

% 2. 检查上壁面节点是否重复（避免分母为0）
if length(unique(y_top)) < 4
    error('上壁面插值节点yc(%d:%d)存在重复，无法进行三阶拉格朗日插值', nyc-3, nyc);
end

% 3. 拉格朗日基函数在y_top(4)（最后一个点）处的导数计算
y4_top = y_top(4);  % 上壁面目标点（yc(nyc)）
% 基函数L1(y)在y4_top处的导数
dL1_top = ((y4_top-y_top(2))*(y4_top-y_top(3)) + (y4_top-y_top(2))*(y4_top-y_top(4)) + (y4_top-y_top(3))*(y4_top-y_top(4))) / ...
    ((y_top(1)-y_top(2))*(y_top(1)-y_top(3))*(y_top(1)-y_top(4)));
% 基函数L2(y)在y4_top处的导数
dL2_top = ((y4_top-y_top(1))*(y4_top-y_top(3)) + (y4_top-y_top(1))*(y4_top-y_top(4)) + (y4_top-y_top(3))*(y4_top-y_top(4))) / ...
    ((y_top(2)-y_top(1))*(y_top(2)-y_top(3))*(y_top(2)-y_top(4)));
% 基函数L3(y)在y4_top处的导数
dL3_top = ((y4_top-y_top(1))*(y4_top-y_top(2)) + (y4_top-y_top(1))*(y4_top-y_top(4)) + (y4_top-y_top(2))*(y4_top-y_top(4))) / ...
    ((y_top(3)-y_top(1))*(y_top(3)-y_top(2))*(y_top(3)-y_top(4)));
% 基函数L4(y)在y4_top处的导数
dL4_top = ((y4_top-y_top(1))*(y4_top-y_top(2)) + (y4_top-y_top(1))*(y4_top-y_top(3)) + (y4_top-y_top(2))*(y4_top-y_top(3))) / ...
    ((y_top(4)-y_top(1))*(y_top(4)-y_top(2))*(y_top(4)-y_top(3)));

% 4. 合并导数得到上壁面梯度
grad_top = u_top(1)*dL1_top + u_top(2)*dL2_top + u_top(3)*dL3_top + u_top(4)*dL4_top;

% ===================== 原有逻辑：计算剪切应力和摩擦速度 =====================
% 下壁面
tau_w_bot = nu * grad_bot;
Utau_bot = sqrt(abs(tau_w_bot));

% 上壁面（保持原逻辑的abs，若需要物理意义的符号可根据实际场景调整）
tau_w_top = nu * abs(grad_top);
Utau_top = sqrt(abs(tau_w_top));


%%
    fprintf('Injection (Bot) Utau (3-Point) = %.6f\n', Utau_bot);
    fprintf('Suction   (Top) Utau (3-Point) = %.6f\n', Utau_top);
    % --- 循环计算 turbh (分离上下壁面) ---
    % 使用 floor(nyp/2) 确保覆盖半宽
    limit_loop = floor(nyc/2);
    
    turbh_bot = zeros(limit_loop, 11);
    turbh_top = zeros(limit_loop, 11);
    % 关键修改：建立专门的 Y+ 坐标数组
    yh_bot = zeros(limit_loop, 1);
    yh_top = zeros(limit_loop, 1);

    for i = 1:limit_loop
        % --- 索引定义 (修正为严格对称) ---
        idx_bot = i;              % 下壁面: 第 i 个网格
        idx_top = nyc - i + 1;    % 上壁面: 倒数第 i 个网格
        
        % --- 计算 Y+ 坐标 ---
        % 下壁面 (Injection): 使用插值后的中心坐标 yc
        dist_bot_val = yc(idx_bot);
        yh_bot(i) = dist_bot_val * Utau_bot / nu;
        
        % 上壁面 (Suction): 距离 = 2.0 - yc
        dist_top_val = 2.0 - yc(idx_top);
        yh_top(i) = dist_top_val * Utau_top / nu;
        
        % --- 下壁面 (Injection) 数据填充 ---
        turbh_bot(i,1) = um(idx_bot) / Utau_bot;             % U+
        turbh_bot(i,5) = sqrt(um2(idx_bot) - um(idx_bot)^2) / Utau_bot; % u_rms+
        turbh_bot(i,6) = sqrt( vm2(idx_bot) - vm(idx_bot)^2) / Utau_bot; % v_rms+
        turbh_bot(i,7) = sqrt( wm2(idx_bot) - wm(idx_bot)^2) / Utau_bot; % w_rms+
        
        % --- 上壁面 (Suction) 数据填充 ---
        turbh_top(i,1) = um(idx_top) / Utau_top;             % U+
        turbh_top(i,5) = sqrt( um2(idx_top) - um(idx_top)^2) / Utau_top; % u_rms+
        turbh_top(i,6) = sqrt( vm2(idx_top) - vm(idx_top)^2) / Utau_top; % v_rms+
        turbh_top(i,7) = sqrt( wm2(idx_top) - wm(idx_top)^2) / Utau_top; % w_rms+
    end
    % =========================================================================
    % 读取文献数据
    % =========================================================================
    % 假设文件在当前路径，如果不存在则赋空
    if exist('1.txt', 'file'), ref_mean = importdata('1.txt'); else, ref_mean = []; end
    if exist('u+.txt', 'file'), ref_u = importdata('u+.txt'); else, ref_u = []; end
    if exist('v+.txt', 'file'), ref_v = importdata('v+.txt'); else, ref_v = []; end
    if exist('w+.txt', 'file'), ref_w = importdata('w+.txt'); else, ref_w = []; end

    fprintf('\n--- 下壁面 (Injection) 前 10 层 y+ ---\n');
    for i = 1:10
    fprintf('第%2d层: y+ = %.4f\n', i, yh_bot(i));
    end
    fprintf('------------------------------------\n');
    % =========================================================================
    % 4. 绘图
    % =========================================================================
    draw_plots_bottom(yh_bot, turbh_bot, yh_top, turbh_top, ref_mean, ref_u, ref_v, ref_w);
end

function draw_plots_bottom(yh_bot, turbh_bot, yh_top, turbh_top, ref_mean, ref_u, ref_v, ref_w)
    style_bot = {'r-', 'LineWidth', 1.5};
    
    figure('Units', 'pixels', 'Position', [100, 100, 1000, 500], 'Color', 'w', 'Name', 'Bot and Top');
    % Fig 3: Mean Velocity (左图)
    subplot(1, 2, 1); hold on; box on;
    p1 = plot(yh_bot, turbh_bot(:,1), 'r-', 'LineWidth', 1.5);
    p3 = plot(yh_top, turbh_top(:,1), 'b-', 'LineWidth', 1.5);
    
    % --- 绘制文献平均速度 (绿色) ---
    if ~isempty(ref_mean)
        p_ref = plot(ref_mean(:,1), ref_mean(:,2), 'g-', 'LineWidth', 1.5);
    end
    
    % 对数律参考线
    xref = linspace(1, 200, 100);
    p2 = semilogx(xref, 2.5*log(xref)+5.5, 'k--', 'LineWidth', 1.2);
    set(gca, 'XScale', 'log');
    % 绘制粘性底层 u+ = y+ 
    xref_vis = linspace(0.1, 10, 20);
    semilogx(xref_vis, xref_vis, 'k--', 'LineWidth', 1.2);
    xlabel('$y^+$', 'Interpreter', 'latex');
    ylabel('$\overline{u}^+$', 'Interpreter', 'latex');
    title('Fig 3: Mean Velocity (Injection)');
    
    % 更新图例
    if ~isempty(ref_mean)
        legend([p1, p3, p_ref, p2], {'Bot', 'Top', 'Ref', 'Log Law'}, 'Location', 'best');
    else
        legend([p1, p3, p2], {'Bot', 'Top', 'Log Law'}, 'Location', 'best');
    end
    
    xlim([1 100]); 
    ylim([0 25]); grid on;
    set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
    
    % Fig 4: RMS Fluctuations (右图)
    subplot(1, 2, 2); hold on; box on;
    plot(yh_bot, turbh_bot(:,5), 'r-', 'LineWidth', 1.5, 'DisplayName', 'u''');
    plot(yh_bot, turbh_bot(:,6), 'g-', 'LineWidth', 1.5, 'DisplayName', 'v''');
    plot(yh_bot, turbh_bot(:,7), 'b-', 'LineWidth', 1.5, 'DisplayName', 'w''');
    
    % --- 绘制文献 RMS 数据 ---
    if ~isempty(ref_u)
        plot(ref_u(:,1), ref_u(:,2), 'k--', 'LineWidth', 1.2, 'DisplayName', 'u'' Ref');
    end
    if ~isempty(ref_v)
        plot(ref_v(:,1), ref_v(:,2), 'k:', 'LineWidth', 1.2, 'DisplayName', 'v'' Ref');
    end
    if ~isempty(ref_w)
        plot(ref_w(:,1), ref_w(:,2), 'k-.', 'LineWidth', 1.2, 'DisplayName', 'w'' Ref');
    end
    
    xlabel('$y^+$', 'Interpreter', 'latex');
    ylabel('$RMS / u_{\tau}$', 'Interpreter', 'latex');
    title('Fig 4: Turbulence Intensities (Injection)');
    legend('Location', 'best');
    xlim([0 60]); 
    ylim([0 4.0]); grid on;
    set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
    sgtitle('Sumitani & Kasagi (1995) - Injection Side Only');
end

function [y,u,du,w,p,ur,vr,wr,pr,wx,wy,wz] = load_lm(d, f1, f2, f3)
    y=[];u=[];du=[];w=[];p=[];ur=[];vr=[];wr=[];pr=[];wx=[];wy=[];wz=[];
    d1 = rd(fullfile(d,f1)); if ~isempty(d1), y=d1(:,2); u=d1(:,3); du=d1(:,4); w=d1(:,5); p=d1(:,6); end
    d2 = rd(fullfile(d,f2)); if ~isempty(d2), ur=sqrt(d2(:,3)); vr=sqrt(d2(:,4)); wr=sqrt(d2(:,5)); end
    d3 = rd(fullfile(d,f3)); if ~isempty(d3), wx=sqrt(d3(:,3)); wy=sqrt(d3(:,4)); wz=sqrt(d3(:,5)); pr=sqrt(d3(:,9)); end
end
function d = rd(f), if exist(f,'file'), d=importdata(f); else, d=[]; end, end