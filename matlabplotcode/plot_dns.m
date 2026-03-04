function plot_dns_strict

    clc; close all;

    %% ========================================================================
    %% 1. 基础配置
    %% ========================================================================
    % --- 路径 ---
    dns_dir = 'E:\sce_workfile\cfd_matlab_code\Re180_40000step';   
    ref_dir = 'E:\sce_workfile\cfd_matlab_code\LM_Channel_0180';
    
    fname_stat   = 'stat.dat';
    fname_prgrad = 'prgrad.dat';
    fname_mesh   = 'mesh.dat';
    
    % --- 物理参数 ---
    % nu = 4.5903E-4;  
    nu = 2.3310E-4; 
    % nu = 5.00000E-05;

    % --- 网格参数 (需准确对应 mesh.dat) ---
    % x_length = 128/2;      
    % y_length = 127+1;      
    % z_length = 128/2;  
    % x_length = 576/2;      
    % y_length = 383+1;      
    % z_length = 576/2;      
    x_length = 384/2;      
    y_length = 191+1;      
    z_length = 256/2;    
    %% ========================================================================
    %% 2. 数据读取与预处理
    %% ========================================================================
    
    % --- 2.1 读取压力梯度与计算 Utau ---
    % Logic: strictly match "Utau=sqrt(abs(mean(prgrad)));"
    prgrad = importdata(fullfile(dns_dir, fname_prgrad));
    rear_ratio = 0.5; % 明确：取后30%数据，p=0.3（可修改为0.2、0.8等，0<rear_ratio≤1）
    Utau = sqrt(abs(mean(prgrad(floor(length(prgrad)*(1-rear_ratio))+1 : end))));
    vormag = Utau*Utau/nu;
    fprintf('Utau = %.6f, vormag = %.6f\n', Utau, vormag);

    % --- 2.2 读取网格并定义关键尺寸 ---
    xyz = importdata(fullfile(dns_dir, fname_mesh));
    if length(xyz) ~= (x_length + y_length + z_length)
         error('mesh.dat 长度不匹配，请检查 x/y/z_length 设置');
    end
    
    % 提取 yc (包含虚拟网格)
    y = xyz(x_length+1 : x_length+y_length);
    
    % 定义内部点坐标 yc
    yc = y(2:end-1); 
    
    nyc = length(yc);

    % --- 2.3 读取统计数据 (stat.dat) ---
    % Logic: import -> reshape -> mean
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

    ox = stat_avg(1, :); oy = stat_avg(2, :); oz = stat_avg(3, :);
    um = stat_avg(4, :); vm = stat_avg(5, :); wm = stat_avg(6, :); pm = stat_avg(7, :);
    um2= stat_avg(8, :); vm2= stat_avg(9, :); wm2= stat_avg(10,:); pm2= stat_avg(11,:);

    %% ========================================================================
    %% 3. 统计结果计算 (Strict Logic Replication)
    %% ========================================================================
    
    % --- 梯度计算 ---
    dudyp = gradient(um, yc);
    dwdyp = gradient(wm, yc);

    % --- 循环计算 turbh ---
    limit_loop = fix(nyc/2);
    turbh = zeros(limit_loop, 11);

    for i = 1:limit_loop

        idx_1 = i;              % 底部点
        idx_2 = nyc +1 - i;    % 顶部点1 (对应 2点平均)
        idx_3 = i + 1;          % 底部相邻点 (对应 4点平均)
        idx_4 = nyc- i;    % 顶部点2 (对应 4点平均)

        % 1. U+ (Mean U) - 使用 2点 (i, nyc+1-i)
        turbh(i,1) = 0.5 * (um(idx_1) + um(idx_2)) / Utau;

        % 2. dU+/dy (Mean Shear) - 使用 2点
        turbh(i,2) = 0.5 * (dudyp(idx_1) - dudyp(idx_2)) / vormag;

        % 3. W+ (Mean W) - 使用 2点
        turbh(i,3) = 0.5 * (wm(idx_1) + wm(idx_2)) / Utau;

        % 4. P+ (Mean P) - 使用 2点
        turbh(i,4) = 0.5 * (pm(idx_1) + pm(idx_2)) / Utau^2;

        % 5. u_rms+ - 使用 2点
        val_u1 = sqrt( um2(idx_1) - um(idx_1)^2);
        val_u2 = sqrt( um2(idx_2) - um(idx_2)^2);
        turbh(i,5) = 0.5 * (val_u1 + val_u2) / Utau;

        % 6. v_rms+ - [特殊] 使用 4点 
        % DNS.m: 0.25*(sqrt(v...)+sqrt(v...)+sqrt(v...)+sqrt(v...))
        v1 = sqrt(vm2(idx_1) - vm(idx_1)^2);
        v2 = sqrt(vm2(idx_2) - vm(idx_2)^2);
        v3 = sqrt(vm2(idx_3) - vm(idx_3)^2);
        v4 = sqrt(vm2(idx_4) - vm(idx_4)^2);
        turbh(i,6) = 0.25 * (v1 + v2 + v3 + v4) / Utau;

        % 7. w_rms+ - 使用 2点
        w1 = sqrt( wm2(idx_1) - wm(idx_1)^2);
        w2 = sqrt( wm2(idx_2) - wm(idx_2)^2);
        turbh(i,7) = 0.5 * (w1 + w2) / Utau;

        % 8. p_rms+ - 使用 2点
        p1 = sqrt( pm2(idx_1) - pm(idx_1)^2);
        p2 = sqrt( pm2(idx_2) - pm(idx_2)^2);
        turbh(i,8) = 0.5 * (p1 + p2) / Utau^2;

        % 9. wx_rms+ - [特殊] 使用 4点，且需减去 (dw/dy)^2
        wx1 = sqrt( ox(idx_1) - dwdyp(idx_1)^2);
        wx2 = sqrt( ox(idx_2) - dwdyp(idx_2)^2);
        wx3 = sqrt( ox(idx_3) - dwdyp(idx_3)^2);
        wx4 = sqrt( ox(idx_4) - dwdyp(idx_4)^2);
        turbh(i,9) = 0.25 / vormag * (wx1 + wx2 + wx3 + wx4);

        % 10. wy_rms+ - 使用 2点 (DNS.m line 290)
        wy1 = sqrt(oy(idx_1));
        wy2 = sqrt(oy(idx_2));
        turbh(i,10) = 0.5 / vormag * (wy1 + wy2);

        % 11. wz_rms+ - [特殊] 使用 4点，且需减去 (du/dy)^2
        wz1 = sqrt(oz(idx_1) - dudyp(idx_1)^2);
        wz2 = sqrt(oz(idx_2) - dudyp(idx_2)^2);
        wz3 = sqrt(oz(idx_3) - dudyp(idx_3)^2);
        wz4 = sqrt(oz(idx_4) - dudyp(idx_4)^2);
        turbh(i,11) = 0.25 / vormag * (wz1 + wz2 + wz3 + wz4);
    end

    % --- 生成 Y+ 坐标 ---
    yh = yc(1:limit_loop) * Utau / nu;

    draw_plots(yh, turbh, ref_dir);
end

function draw_plots(yh, turbh, ref_dir)
    % 基准数据文件名
    % f_mean = 'LM_Channel_1000_mean_prof.dat';
    % f_vel  = 'LM_Channel_1000_vel_fluc_prof.dat';
    % f_vor  = 'LM_Channel_1000_vor_pres_fluc_prof.dat';
    f_mean = 'LM_Channel_0180_mean_prof.dat';
    f_vel  = 'LM_Channel_0180_vel_fluc_prof.dat';
    f_vor  = 'LM_Channel_0180_vor_pres_fluc_prof.dat';
    
    [y_ref, U_ref, dUdy_ref, W_ref, p_mean_ref, ...
     urms_ref, vrms_ref, wrms_ref, prms_ref, ...
     wx_rms_ref, wy_rms_ref, wz_rms_ref] = load_lm(ref_dir, f_mean, f_vel, f_vor);

    style_ref = {'r-', 'LineWidth', 1};
    style_cur = {'k-', 'LineWidth', 1.2};
    
    figure('Units', 'pixels', 'Position', [100, 100, 1400, 800], 'Color', 'w');
    
    % Subplot helper
    psub = @(id, x, y, xr, yr, ylab) ...
        sub_plot_draw(id, x, y, xr, yr, ylab, style_cur, style_ref);

    subplot(3,4,1); axis off; hold on;
    L1=plot(nan,nan,style_ref{:}); L2=plot(nan,nan,style_cur{:});
    legend([L1,L2], {'Lee \& Moser', 'Present (DNS.m Logic)'}, 'Interpreter','latex','Location','best','FontSize',12);

    % Plotting (Columns map to turbh indices)
    psub(2,  yh, turbh(:,1),  y_ref, U_ref,      '$\overline{u}^+$');
    psub(3,  yh, turbh(:,2),  y_ref, dUdy_ref,   '$d\overline{u}/dy$');
    psub(4,  yh, turbh(:,3),  y_ref, W_ref,      '$\overline{w}$');
    psub(5,  yh, turbh(:,4),  y_ref, p_mean_ref, '$\overline{p}$');
    psub(6,  yh, turbh(:,5),  y_ref, urms_ref,   '$u_{rms}^+$');
    psub(7,  yh, turbh(:,6),  y_ref, vrms_ref,   '$v_{rms}^+$');
    psub(8,  yh, turbh(:,7),  y_ref, wrms_ref,   '$w_{rms}^+$');
    psub(9,  yh, turbh(:,8),  y_ref, prms_ref,   '$p_{rms}^+$');
    psub(10, yh, turbh(:,9),  y_ref, wx_rms_ref, '$\omega_{x,rms}^+$');
    psub(11, yh, turbh(:,10), y_ref, wy_rms_ref, '$\omega_{y,rms}^+$');
    psub(12, yh, turbh(:,11), y_ref, wz_rms_ref, '$\omega_{z,rms}^+$');
end

function sub_plot_draw(idx, x, y, xr, yr, ylab, sc, sr)
    subplot(3,4,idx); hold on; box on;
    if ~isempty(xr), plot(xr, yr, sr{:}); end
    plot(x, y, sc{:});
    xlabel('$y^+$','Interpreter','latex'); ylabel(ylab,'Interpreter','latex');
    % set(gca, 'XScale', 'log'); %对数横坐标
    % xlim([0.1 1000]); grid on;
    xlim([0 200]); grid on;
    % xlim([0 1000]); grid on;
    set(gca, 'FontSize', 10, 'FontName', 'Times New Roman');
end

function [y,u,du,w,p,ur,vr,wr,pr,wx,wy,wz] = load_lm(d, f1, f2, f3)
    % Initialize
    y=[];u=[];du=[];w=[];p=[];ur=[];vr=[];wr=[];pr=[];wx=[];wy=[];wz=[];
    % Load helpers
    d1 = rd(fullfile(d,f1)); if ~isempty(d1), y=d1(:,2); u=d1(:,3); du=d1(:,4); w=d1(:,5); p=d1(:,6); end
    d2 = rd(fullfile(d,f2)); if ~isempty(d2), ur=sqrt(d2(:,3)); vr=sqrt(d2(:,4)); wr=sqrt(d2(:,5)); end
    d3 = rd(fullfile(d,f3)); if ~isempty(d3), wx=sqrt(d3(:,3)); wy=sqrt(d3(:,4)); wz=sqrt(d3(:,5)); pr=sqrt(d3(:,9)); end
end
function d = rd(f), if exist(f,'file'), d=importdata(f); else, d=[]; end, end