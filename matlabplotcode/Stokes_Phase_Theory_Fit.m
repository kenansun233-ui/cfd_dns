clear; clc; close all;

%% DNS phase extraction with robust wall-shear diagnostics.
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
u_star_fd2_list = nan(size(Re_delta_list));
u_star_fit_list = nan(size(Re_delta_list));
transition_note = strings(size(Re_delta_list));

fprintf('==================================================\n');
fprintf('Stokes DNS phase extraction with transition diagnostics\n');
fprintf('==================================================\n');

for idx = 1:length(Re_delta_list)
    current_Re = Re_delta_list(idx);
    case_dir = fullfile(base_dir, sprintf('Re_delta=%d', current_Re));

    if ~exist(fullfile(case_dir, 'mesh.dat'), 'file') || ~exist(fullfile(case_dir, 'stat.dat'), 'file')
        warning('Missing data for Re_delta=%d', current_Re);
        continue;
    end

    data = stokes_load_case(case_dir, current_Re, opts);
    range = data.start_snap:length(data.time);

    [phi_fd2, amp_fd2] = harmonic_phase(data.tau_fd2(range), data.u_wall(range), data.snaps_per_period);
    [phi_fit, amp_fit] = harmonic_phase(data.tau_fit(range), data.u_wall(range), data.snaps_per_period);

    phi_fd2_list(idx) = abs(phi_fd2);
    phi_fit_list(idx) = abs(phi_fit);
    u_star_fd2_list(idx) = sqrt(abs(amp_fd2)) / opts.U_osc;
    u_star_fit_list(idx) = sqrt(abs(amp_fit)) / opts.U_osc;

    if isempty(data.diagnostics.growth_flags)
        transition_note(idx) = "no sustained growth";
    else
        transition_note(idx) = "growth flags " + mat2str(data.diagnostics.growth_flags);
    end

    fprintf(['Re=%-4d | phi_fd2=%7.3f deg | phi_fit=%7.3f deg | ', ...
        'u*_fit/U0=%7.4f | mean(k) first/last=%9.3e/%9.3e | %s\n'], ...
        current_Re, phi_fd2_list(idx), phi_fit_list(idx), u_star_fit_list(idx), ...
        data.diagnostics.mean_k(1), data.diagnostics.mean_k(end), transition_note(idx));
end

%% Spalart-style reference curve retained for comparison.
A_theory = 3.0;
B_theory = 0.4;
kappa = 0.41;
Re_theory = linspace(400, 1500, 200);
phi_theory_deg = nan(size(Re_theory));

for i = 1:length(Re_theory)
    target_Re = Re_theory(i);
    implicit_func = @(x) (kappa/2) * (1/x) * cos(asin(A_theory*x)) + ...
        log(1/x) - log(target_Re) + B_theory;
    try
        x_sol = fzero(implicit_func, 0.04);
        if A_theory * x_sol <= 1
            phi_theory_deg(i) = rad2deg(asin(A_theory * x_sol));
        end
    catch
        phi_theory_deg(i) = NaN;
    end
end

figure('Position', [180, 120, 980, 720], 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile([1 2]);
hold on; box on; grid on;
plot(Re_theory, phi_theory_deg, 'r-', 'LineWidth', 2.2, ...
    'DisplayName', 'Spalart reference');
yline(45.0, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Laminar Stokes 45 deg');
plot(Re_delta_list, phi_fd2_list, 'o--', 'LineWidth', 1.8, 'MarkerSize', 9, ...
    'DisplayName', 'DNS tau\_fd2');
plot(Re_delta_list, phi_fit_list, 's-', 'LineWidth', 2.2, 'MarkerSize', 9, ...
    'DisplayName', 'DNS tau\_fit');
xlabel('Re_\delta');
ylabel('|wall-shear phase| (deg)');
title('Wall-shear phase: finite-difference vs robust fit');
xlim([300, 1000]);
ylim([0, 60]);
legend('Location', 'best');

nexttile;
hold on; box on; grid on;
plot(Re_delta_list, u_star_fd2_list, 'o--', 'LineWidth', 1.8, 'DisplayName', 'tau\_fd2');
plot(Re_delta_list, u_star_fit_list, 's-', 'LineWidth', 2.0, 'DisplayName', 'tau\_fit');
xlabel('Re_\delta');
ylabel('u_* / U_0');
title('Friction velocity proxy');
legend('Location', 'best');

nexttile;
hold on; box on; grid on;
for idx = 1:length(Re_delta_list)
    current_Re = Re_delta_list(idx);
    case_dir = fullfile(base_dir, sprintf('Re_delta=%d', current_Re));
    if ~exist(fullfile(case_dir, 'stat.dat'), 'file')
        continue;
    end
    data = stokes_load_case(case_dir, current_Re, opts);
    plot(data.diagnostics.period_index, data.diagnostics.mean_k, '-o', ...
        'LineWidth', 1.8, 'DisplayName', sprintf('Re=%d', current_Re));
end
set(gca, 'YScale', 'log');
xlabel('Period index');
ylabel('near-wall mean(k)');
title('Transition check');
legend('Location', 'best');

disp('绘图完成。若扰动能量单调衰减，当前数据不能判定为转捩。');

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
