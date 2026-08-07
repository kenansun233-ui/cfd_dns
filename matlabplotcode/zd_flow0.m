clear; clc; close all;

%% Single-case Stokes profile check.
dns_dir = 'E:\file\zd\stokes2\Re_delta=700';
Re_delta = 700.0;

opts = struct();
opts.U_osc = 1.0;
opts.omega = 2.0 * pi;
opts.dt = 0.0005;
opts.output_interval = 10;
opts.x_length = 192;
opts.z_length = 128;
opts.fit_points = 12;
opts.start_periods = 5;

data = stokes_load_case(dns_dir, Re_delta, opts);

fprintf('读取 %s\n', dns_dir);
fprintf('y方向网格点数: %d, y1/delta = %.5f, 1 delta 内点数 = %d\n', ...
    data.nyc + 2, data.yc(1) / data.delta, sum(data.yc <= data.delta));
fprintf('快照数: %d, 每周期快照数: %d\n', length(data.time), data.snaps_per_period);

figure('Position', [120, 80, 1100, 760], 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

%% Profile comparison over the last period.
nexttile([1 2]);
hold on; box on; grid on;

start_snap = max(1, length(data.time) - data.snaps_per_period + 1);
max_plot_lines = 10;
plot_step = max(1, round(data.snaps_per_period / max_plot_lines));
plot_indices = start_snap:plot_step:length(data.time);
colors_dns = winter(length(plot_indices) + 2);
colors_ana = autumn(length(plot_indices) + 2);

for p = 1:length(plot_indices)
    k = plot_indices(p);
    t_current = data.time(k);
    u_analytical = opts.U_osc * exp(-data.yc ./ data.delta) .* ...
        cos(opts.omega * t_current - data.yc ./ data.delta);

    plot(data.um(:, k) / opts.U_osc, data.yc / data.delta, 'o', ...
        'Color', colors_dns(p, :), 'MarkerFaceColor', colors_dns(p, :), ...
        'MarkerSize', 4, 'DisplayName', sprintf('DNS t=%.4f', t_current));
    plot(u_analytical / opts.U_osc, data.yc / data.delta, '-', ...
        'Color', colors_ana(p, :), 'LineWidth', 1.4, 'HandleVisibility', 'off');
end

plot(NaN, NaN, '-', 'Color', colors_ana(round(end/2), :), 'LineWidth', 2, ...
    'DisplayName', 'Stokes theory');
yline(1, 'k--', 'LineWidth', 1.2, 'Label', '\delta');
xlabel('u / U_{osc}');
ylabel('y / \delta');
title(sprintf('Stokes velocity profile check, Re_\\delta=%g', Re_delta));
ylim([0, 3]);
xlim([-1.2, 1.2]);
legend('Location', 'bestoutside');

%% Wall shear phase.
range = data.start_snap:length(data.time);
[phi_fd2, amp_fd2] = harmonic_phase(data.tau_fd2(range), data.u_wall(range), data.snaps_per_period);
[phi_fit, amp_fit] = harmonic_phase(data.tau_fit(range), data.u_wall(range), data.snaps_per_period);

nexttile;
hold on; box on; grid on;
plot(data.time, data.tau_fd2, '-', 'LineWidth', 1.2, 'DisplayName', 'tau\_fd2');
plot(data.time, data.tau_fit, '-', 'LineWidth', 1.2, 'DisplayName', 'tau\_fit');
xlabel('t');
ylabel('\tau_w');
title(sprintf('Wall shear: fd2 %.2f deg, fit %.2f deg', phi_fd2, phi_fit));
legend('Location', 'best');

%% Transition diagnostics.
nexttile;
hold on; box on; grid on;
plot(data.diagnostics.period_index, data.diagnostics.mean_k, '-o', ...
    'LineWidth', 1.8, 'DisplayName', 'mean(k)');
plot(data.diagnostics.period_index, data.diagnostics.max_vrms.^2, '-s', ...
    'LineWidth', 1.8, 'DisplayName', 'max(v_{rms})^2');
set(gca, 'YScale', 'log');
xlabel('Period index');
ylabel('Near-wall disturbance metric');
title('Transition diagnostic');
legend('Location', 'best');

fprintf('tau_fd2 phase = %.3f deg, amp = %.6e\n', phi_fd2, amp_fd2);
fprintf('tau_fit phase = %.3f deg, amp = %.6e\n', phi_fit, amp_fit);
if isempty(data.diagnostics.growth_flags)
    fprintf('扰动能量未出现连续周期增长：当前数据未显示转捩。\n');
else
    fprintf('检测到扰动能量增长周期: %s\n', mat2str(data.diagnostics.growth_flags));
end

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
