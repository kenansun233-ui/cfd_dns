clear; clc; close all;

base_dir = 'E:\file\zd\stokes2';
Re_delta_list = [600, 700];

opts = struct();
opts.U_osc = 1.0;
opts.omega = 2.0 * pi;
opts.dt = 0.0005;
opts.output_interval = 10;
opts.x_length = 192;
opts.z_length = 128;
opts.fit_points = 12;
opts.start_periods = 5;

phi_fd2_list = nan(size(Re_delta_list));
phi_fit_list = nan(size(Re_delta_list));
tau_fd2_amp = nan(size(Re_delta_list));
tau_fit_amp = nan(size(Re_delta_list));

fprintf('==================================================\n');
fprintf('开始批量处理 Stokes DNS 数据：相位、剪切和扰动诊断\n');
fprintf('==================================================\n');

for idx = 1:length(Re_delta_list)
    current_Re = Re_delta_list(idx);
    case_dir = fullfile(base_dir, sprintf('Re_delta=%d', current_Re));

    if ~exist(fullfile(case_dir, 'mesh.dat'), 'file') || ~exist(fullfile(case_dir, 'stat.dat'), 'file')
        warning('未找到算例 Re_delta=%d 的数据文件，跳过。', current_Re);
        continue;
    end

    data = stokes_load_case(case_dir, current_Re, opts);
    range = data.start_snap:length(data.time);

    [phi_fd2_list(idx), tau_fd2_amp(idx)] = harmonic_phase(data.tau_fd2(range), data.u_wall(range), data.snaps_per_period);
    [phi_fit_list(idx), tau_fit_amp(idx)] = harmonic_phase(data.tau_fit(range), data.u_wall(range), data.snaps_per_period);

    d = data.diagnostics;
    if isempty(d.growth_flags)
        transition_text = '未见持续增长';
    else
        transition_text = sprintf('第 %s 周期增长', mat2str(d.growth_flags));
    end

    fprintf(['Re_delta=%-4d | y1/delta=%7.4f | tau_fd2=%7.3f deg | ', ...
        'tau_fit=%7.3f deg | max(k) first/last=%9.3e/%9.3e | %s\n'], ...
        current_Re, data.yc(1) / data.delta, phi_fd2_list(idx), phi_fit_list(idx), ...
        d.max_k(1), d.max_k(end), transition_text);
end

figure('Position', [160, 120, 980, 720], 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on; box on; grid on;
yline(45.0, 'k--', 'LineWidth', 1.4, 'DisplayName', '层流基准 45 deg');
plot(Re_delta_list, abs(phi_fd2_list), 'o-', 'LineWidth', 2.0, 'MarkerSize', 8, ...
    'DisplayName', 'tau\_fd2');
plot(Re_delta_list, abs(phi_fit_list), 's-', 'LineWidth', 2.0, 'MarkerSize', 8, ...
    'DisplayName', 'tau\_fit');
xlabel('Re_\delta');
ylabel('|壁面剪切相位| (deg)');
title('壁面剪切相位');
legend('Location', 'best');

nexttile;
hold on; box on; grid on;
plot(Re_delta_list, tau_fd2_amp, 'o-', 'LineWidth', 2.0, 'DisplayName', 'tau\_fd2');
plot(Re_delta_list, tau_fit_amp, 's-', 'LineWidth', 2.0, 'DisplayName', 'tau\_fit');
xlabel('Re_\delta');
ylabel('一阶谐波幅值');
title('壁面剪切幅值');
legend('Location', 'best');

nexttile([1 2]);
hold on; box on; grid on;
for idx = 1:length(Re_delta_list)
    current_Re = Re_delta_list(idx);
    case_dir = fullfile(base_dir, sprintf('Re_delta=%d', current_Re));
    if ~exist(fullfile(case_dir, 'stat.dat'), 'file')
        continue;
    end
    data = stokes_load_case(case_dir, current_Re, opts);
    plot(data.diagnostics.period_index, data.diagnostics.mean_k, '-o', ...
        'LineWidth', 1.8, 'DisplayName', sprintf('Re=%d mean(k)', current_Re));
end
set(gca, 'YScale', 'log');
xlabel('周期编号');
ylabel('近壁扰动动能均值');
title('旁路转捩诊断：扰动能量是否增长');
legend('Location', 'best');

fprintf('==================================================\n');
fprintf('处理完成。若 mean(k) 单调衰减且 vrms/wrms 很小，则当前数据未转捩。\n');

function [phase_deg, amp] = harmonic_phase(signal, reference, snaps_per_period)
valid_len = floor(length(signal) / snaps_per_period) * snaps_per_period;
signal = signal(1:valid_len);
reference = reference(1:valid_len);

signal_phase = mean(reshape(signal, snaps_per_period, []), 2);
ref_phase = mean(reshape(reference, snaps_per_period, []), 2);
n = (0:snaps_per_period-1)';

sig_a = 2 / snaps_per_period * sum(signal_phase .* cos(2*pi*n/snaps_per_period));
sig_b = 2 / snaps_per_period * sum(signal_phase .* sin(2*pi*n/snaps_per_period));
ref_a = 2 / snaps_per_period * sum(ref_phase .* cos(2*pi*n/snaps_per_period));
ref_b = 2 / snaps_per_period * sum(ref_phase .* sin(2*pi*n/snaps_per_period));

phase_deg = rad2deg(atan2(sig_b, sig_a) - atan2(ref_b, ref_a));
phase_deg = mod(phase_deg + 180, 360) - 180;
amp = hypot(sig_a, sig_b);
end
