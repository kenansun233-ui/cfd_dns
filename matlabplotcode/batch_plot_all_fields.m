function summary = batch_plot_all_fields(case_dir)
%BATCH_PLOT_ALL_FIELDS Batch-process all field_*.dat files in one DNS case.
%   SUMMARY = BATCH_PLOT_ALL_FIELDS() asks for a folder containing field
%   files. The script creates per-snapshot slice and Q-criterion figures,
%   aggregate turbulence profiles, time histories, and analysis_summary.mat.
%
%   SUMMARY = BATCH_PLOT_ALL_FIELDS(CASE_DIR) processes CASE_DIR directly.
%
%   The field layout must match output_velocity(): u, v, w, p with x as the
%   fastest index, followed by z and y. This script is written for the current
%   half-channel mesh and excludes the two y ghost layers from statistics.

if nargin < 1 || isempty(case_dir)
    case_dir = uigetdir(pwd, 'Select a folder containing field_*.dat');
    if isequal(case_dir, 0)
        error('No case folder was selected.');
    end
end
case_dir = char(case_dir);

cfg.gamma_mesh = 2.0;          % Must match init_mesh_para().
cfg.q_percentile = 99.5;       % Fixed threshold is obtained from first field.
cfg.q_plot_stride = [1, 1, 1]; % x, z, y visualization stride.
cfg.figure_dpi = 180;
cfg.overwrite = true;

[field_files, case_dir] = find_field_files(case_dir);
info_path = find_nearest_file(case_dir, 'run_info.dat');
if isempty(info_path)
    error('run_info.dat was not found in the field folder or its parents.');
end
info = read_run_info(info_path);

nxp = require_value(info, 'nxp');
nyp = require_value(info, 'nyp');
nzp = require_value(info, 'nzp');
lx = require_value(info, 'xlength');
h = require_value(info, 'ylength');
lz = require_value(info, 'zlength');

sx = get_value(info, 'field_xskip', 1);
sy = get_value(info, 'field_yskip', 1);
sz = get_value(info, 'field_zskip', 1);
gamma_mesh = get_value(info, 'gamma_mesh', cfg.gamma_mesh);
mesh_sx = get_value(info, 'restart_xskip', sx);
mesh_sy = get_value(info, 'restart_yskip', sy);
mesh_sz = get_value(info, 'restart_zskip', sz);

ix = 0:sx:(nxp - 1);
iy = 0:sy:nyp;
iz = 0:sz:(nzp - 1);
nx = numel(ix);
ny = numel(iy);
nz = numel(iz);
field_size = [nx, nz, ny];
expected_rows = double(nx) * double(nz) * double(ny);

[x, y, z, mesh_path] = load_grid_coordinates(case_dir, info, ...
    ix, iy, iz, mesh_sx, mesh_sy, mesh_sz, gamma_mesh);
physical = find(iy >= 1 & iy <= nyp - 1);
if numel(physical) < 3
    error('At least three physical y locations are required.');
end
y_phys = y(physical);
y_weights = cell_center_weights(y_phys, h);

dx = sx * lx / nxp;
dz = sz * lz / nzp;
delta = get_value(info, 'stokes_delta', NaN);
if isfinite(delta) && delta > 0
    [~, near_local] = min(abs(y_phys - delta));
else
    near_local = min(2, numel(y_phys));
end
near_index = physical(near_local);

steps = zeros(numel(field_files), 1);
for i = 1:numel(field_files)
    steps(i) = field_step(field_files(i).name);
end
[steps, order] = sort(steps);
field_files = field_files(order);
times = field_times(steps, info);

output_dir = fullfile(case_dir, 'field_figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

nfiles = numel(field_files);
sum_um = zeros(numel(physical), 1);
sum_vm = zeros(numel(physical), 1);
sum_wm = zeros(numel(physical), 1);
sum_urms2 = zeros(numel(physical), 1);
sum_vrms2 = zeros(numel(physical), 1);
sum_wrms2 = zeros(numel(physical), 1);

ub_history = nan(nfiles, 1);
k_history = nan(nfiles, 1);
qmax_history = nan(nfiles, 1);
enstrophy_history = nan(nfiles, 1);
q_iso = NaN;
q_threshold_initialized = false;

fprintf('Found %d field files in %s\n', nfiles, case_dir);
fprintf('Field dimensions: %d x %d x %d; expected rows: %.0f\n', ...
    nx, nz, ny, expected_rows);

for i = 1:nfiles
    field_path = fullfile(field_files(i).folder, field_files(i).name);
    fprintf('[%d/%d] Reading %s\n', i, nfiles, field_files(i).name);
    [u, v, w, p] = read_field_single(field_path, field_size, expected_rows);

    u_phys = u(:, :, physical);
    v_phys = v(:, :, physical);
    w_phys = w(:, :, physical);

    um = plane_mean(u_phys);
    vm = plane_mean(v_phys);
    wm = plane_mean(w_phys);
    urms2 = max(plane_mean(u_phys .^ 2) - um .^ 2, 0);
    vrms2 = max(plane_mean(v_phys .^ 2) - vm .^ 2, 0);
    wrms2 = max(plane_mean(w_phys .^ 2) - wm .^ 2, 0);
    k_profile = 0.5 * (urms2 + vrms2 + wrms2);

    ub_history(i) = sum(um .* y_weights) / h;
    k_history(i) = sum(k_profile .* y_weights) / h;

    sum_um = sum_um + um;
    sum_vm = sum_vm + vm;
    sum_wm = sum_wm + wm;
    sum_urms2 = sum_urms2 + urms2;
    sum_vrms2 = sum_vrms2 + vrms2;
    sum_wrms2 = sum_wrms2 + wrms2;

    [q, omega_mag] = q_and_vorticity(u, v, w, y, dx, dz);
    q_phys = q(:, :, physical);
    omega_phys = omega_mag(:, :, physical);
    qmax_history(i) = max(q_phys, [], 'all');
    omega2_profile = plane_mean(omega_phys .^ 2);
    enstrophy_history(i) = 0.5 * sum(omega2_profile .* y_weights) / h;

    if ~q_threshold_initialized
        q_iso = fixed_q_threshold(q_phys, cfg.q_percentile);
        q_threshold_initialized = true;
        fprintf('Fixed Q threshold: %.8e (%.2f percentile of positive Q)\n', ...
            q_iso, cfg.q_percentile);
    end

    slice_file = fullfile(output_dir, sprintf('slices_%08d.png', steps(i)));
    if cfg.overwrite || ~exist(slice_file, 'file')
        plot_snapshot_slices(x, y_phys, z, u, p, omega_mag, physical, ...
            near_index, ub_history(i), steps(i), times(i), info, slice_file, cfg);
    end

    q_file = fullfile(output_dir, sprintf('q_%08d.png', steps(i)));
    if cfg.overwrite || ~exist(q_file, 'file')
        plot_q_structure(x, y_phys, z, u_phys, q_phys, um, q_iso, ...
            steps(i), times(i), info, q_file, cfg);
    end

    clear u v w p u_phys v_phys w_phys q q_phys omega_mag omega_phys
end

mean_um = sum_um / nfiles;
mean_vm = sum_vm / nfiles;
mean_wm = sum_wm / nfiles;
urms = sqrt(sum_urms2 / nfiles);
vrms = sqrt(sum_vrms2 / nfiles);
wrms = sqrt(sum_wrms2 / nfiles);
k_profile = 0.5 * (urms .^ 2 + vrms .^ 2 + wrms .^ 2);
ub_mean = mean(ub_history, 'omitnan');

profile_file = fullfile(output_dir, 'statistics_profiles.png');
plot_summary_profiles(y_phys, h, mean_um, mean_vm, mean_wm, ...
    urms, vrms, wrms, k_profile, ub_mean, profile_file, cfg);

history_file = fullfile(output_dir, 'time_histories.png');
plot_time_histories(times, steps, ub_history, k_history, qmax_history, ...
    enstrophy_history, info, history_file, cfg);

summary = struct();
summary.case_dir = case_dir;
summary.run_info_path = info_path;
summary.mesh_path = mesh_path;
summary.steps = steps;
summary.times = times;
summary.y = y_phys;
summary.y_weights = y_weights;
summary.mean_u = mean_um;
summary.mean_v = mean_vm;
summary.mean_w = mean_wm;
summary.urms = urms;
summary.vrms = vrms;
summary.wrms = wrms;
summary.tke = k_profile;
summary.ubulk = ub_history;
summary.volume_tke = k_history;
summary.max_q = qmax_history;
summary.enstrophy = enstrophy_history;
summary.q_iso = q_iso;
summary.near_wall_y = y(near_index);
summary.config = cfg;
save(fullfile(output_dir, 'analysis_summary.mat'), 'summary');

fprintf('Finished. Figures and summary saved in:\n%s\n', output_dir);
end


function [files, field_dir] = find_field_files(field_dir)
files = dir(fullfile(field_dir, 'field_*.dat'));
if ~isempty(files)
    return;
end

files = dir(fullfile(field_dir, '**', 'field_*.dat'));
if isempty(files)
    error('No field_*.dat files were found under %s.', field_dir);
end
folders = unique({files.folder});
if numel(folders) ~= 1
    error(['Field files were found in multiple folders. Select one field ', ...
        'folder at a time so incompatible cases are not mixed.']);
end
field_dir = folders{1};
end


function path_out = find_nearest_file(start_dir, file_name)
path_out = '';
current = start_dir;
for level = 1:4
    candidate = fullfile(current, file_name);
    if exist(candidate, 'file')
        path_out = candidate;
        return;
    end
    parent = fileparts(current);
    if strcmp(parent, current)
        return;
    end
    current = parent;
end
end


function [x, y, z, mesh_path] = load_grid_coordinates(case_dir, info, ...
    ix, iy, iz, mesh_sx, mesh_sy, mesh_sz, gamma_mesh)
nxp = require_value(info, 'nxp');
nyp = require_value(info, 'nyp');
nzp = require_value(info, 'nzp');
lx = require_value(info, 'xlength');
h = require_value(info, 'ylength');
lz = require_value(info, 'zlength');

mesh_ix = 0:mesh_sx:(nxp - 1);
mesh_iy = 0:mesh_sy:nyp;
mesh_iz = 0:mesh_sz:(nzp - 1);
expected_count = numel(mesh_ix) + numel(mesh_iy) + numel(mesh_iz);
mesh_path = find_compatible_mesh(case_dir, expected_count);

if isempty(mesh_path)
    x = ix * (lx / nxp);
    z = iz * (lz / nzp);
    y_full = build_half_channel_y(nyp, h, gamma_mesh);
    y = y_full(iy + 1);
    fprintf('mesh.dat not found; reconstructed coordinates with gamma_mesh=%.6g.\n', ...
        gamma_mesh);
    return;
end

raw = readmatrix(mesh_path, 'FileType', 'text');
raw = raw(:);
offset_x = numel(mesh_ix);
offset_y = offset_x + numel(mesh_iy);
x_mesh = raw(1:offset_x);
y_mesh = raw(offset_x + 1:offset_y);
z_mesh = raw(offset_y + 1:end);

x = interp1(mesh_ix, x_mesh, ix, 'linear', 'extrap');
y = interp1(mesh_iy, y_mesh, iy, 'pchip', 'extrap');
z = interp1(mesh_iz, z_mesh, iz, 'linear', 'extrap');
x = x(:)';
y = y(:)';
z = z(:)';
fprintf('Grid coordinates read from: %s\n', mesh_path);
end


function mesh_path = find_compatible_mesh(start_dir, expected_count)
mesh_path = '';
candidates = {};
current = start_dir;
for level = 1:4
    local_file = fullfile(current, 'mesh.dat');
    if exist(local_file, 'file')
        candidates{end + 1} = local_file; %#ok<AGROW>
    end
    child_files = dir(fullfile(current, '*', 'mesh.dat'));
    for i = 1:numel(child_files)
        candidates{end + 1} = fullfile(child_files(i).folder, ...
            child_files(i).name); %#ok<AGROW>
    end
    parent = fileparts(current);
    if strcmp(parent, current)
        break;
    end
    current = parent;
end

if isempty(candidates)
    return;
end
candidates = unique(candidates, 'stable');
for i = 1:numel(candidates)
    fid = fopen(candidates{i}, 'r');
    if fid < 0
        continue;
    end
    cleanup = onCleanup(@() fclose(fid));
    values = textscan(fid, '%f');
    count = numel(values{1});
    clear cleanup
    if count == expected_count
        mesh_path = candidates{i};
        return;
    end
end
end


function info = read_run_info(path_in)
fid = fopen(path_in, 'r');
if fid < 0
    error('Unable to open %s.', path_in);
end
cleanup = onCleanup(@() fclose(fid));
data = textscan(fid, '%s%s');
info = struct();
for i = 1:numel(data{1})
    key = matlab.lang.makeValidName(data{1}{i});
    value = str2double(data{2}{i});
    if isfinite(value)
        info.(key) = value;
    else
        info.(key) = data{2}{i};
    end
end
end


function value = require_value(info, key)
if ~isfield(info, key) || ~isnumeric(info.(key))
    error('Required run_info.dat value "%s" is missing.', key);
end
value = info.(key);
end


function value = get_value(info, key, default_value)
if isfield(info, key) && isnumeric(info.(key))
    value = info.(key);
else
    value = default_value;
end
end


function y = build_half_channel_y(nyp, h, gamma_mesh)
nyc = nyp - 1;
xi = (0:nyc) / nyc;
yp = h * (1 + tanh(gamma_mesh * (xi - 1)) / tanh(gamma_mesh));
yp(1) = 0;
yp(end) = h;

y = zeros(1, nyp + 1);
y(2:nyp) = 0.5 * (yp(1:end - 1) + yp(2:end));
y(1) = 2 * yp(1) - y(2);
y(end) = 2 * yp(end) - y(end - 1);
end


function weights = cell_center_weights(y, h)
y = y(:);
edges = [0; 0.5 * (y(1:end - 1) + y(2:end)); h];
weights = diff(edges);
if any(weights <= 0) || abs(sum(weights) - h) > 1e-10 * max(h, 1)
    error('The reconstructed physical y grid is invalid.');
end
end


function step = field_step(file_name)
token = regexp(file_name, '^field_(\d+)\.dat$', 'tokens', 'once');
if isempty(token)
    error('Cannot extract a time step from %s.', file_name);
end
step = str2double(token{1});
end


function times = field_times(steps, info)
dt = get_value(info, 'dt', 1);
wall_on = get_value(info, 'enable_wall_oscillation', 0) ~= 0;
continues = get_value(info, 'restart_continues_oscillation_time', 0) ~= 0;
restart_step = get_value(info, 'restart_start_step', 0);
if wall_on && ~continues
    times = (steps - restart_step) * dt;
else
    times = steps * dt;
end
end


function [u, v, w, p] = read_field_single(path_in, field_size, expected_rows)
fid = fopen(path_in, 'r');
if fid < 0
    error('Unable to open %s.', path_in);
end
cleanup = onCleanup(@() fclose(fid));
columns = textscan(fid, '%f32%f32%f32%f32', ...
    'Delimiter', {' ', '\t'}, 'MultipleDelimsAsOne', true, ...
    'CollectOutput', true);
data = columns{1};
if size(data, 1) ~= expected_rows || size(data, 2) ~= 4
    error('%s contains %d x %d values; expected %.0f x 4.', ...
        path_in, size(data, 1), size(data, 2), expected_rows);
end
u = reshape(data(:, 1), field_size);
v = reshape(data(:, 2), field_size);
w = reshape(data(:, 3), field_size);
p = reshape(data(:, 4), field_size);

% v is wall-normal and stored on the y faces. The solver enforces v=0 at
% j=1 and j=nyp, but the unused output layer j=0 can contain uninitialized
% values in existing field files. Replace only these nonphysical v layers.
v(:, :, 1) = 0;
v(:, :, end) = 0;

if any(~isfinite(u), 'all') || any(~isfinite(v), 'all') || ...
        any(~isfinite(w), 'all') || any(~isfinite(p), 'all')
    error('%s contains NaN or Inf in data used by the post-processing.', path_in);
end
end


function profile = plane_mean(field)
profile = double(squeeze(mean(mean(field, 1), 2)));
profile = profile(:);
end


function [q, omega_mag] = q_and_vorticity(u, v, w, y, dx, dz)
du_dx = derivative_periodic(u, dx, 1);
dv_dy = derivative_y(v, y);
dw_dz = derivative_periodic(w, dz, 2);
q = -0.5 * (du_dx .^ 2 + dv_dy .^ 2 + dw_dz .^ 2);
clear du_dx dv_dy dw_dz

du_dy = derivative_y(u, y);
dv_dx = derivative_periodic(v, dx, 1);
q = q - du_dy .* dv_dx;

du_dz = derivative_periodic(u, dz, 2);
dw_dx = derivative_periodic(w, dx, 1);
q = q - du_dz .* dw_dx;

dv_dz = derivative_periodic(v, dz, 2);
dw_dy = derivative_y(w, y);
q = q - dv_dz .* dw_dy;

omega_x = dw_dy - dv_dz;
omega_y = du_dz - dw_dx;
omega_z = dv_dx - du_dy;
omega_mag = sqrt(omega_x .^ 2 + omega_y .^ 2 + omega_z .^ 2);
end


function derivative = derivative_periodic(field, spacing, dimension)
derivative = (circshift(field, 2, dimension) ...
    - 8 * circshift(field, 1, dimension) ...
    + 8 * circshift(field, -1, dimension) ...
    - circshift(field, -2, dimension)) / single(12 * spacing);
end


function derivative = derivative_y(field, y)
derivative = zeros(size(field), 'like', field);
hm = y(2:end - 1) - y(1:end - 2);
hp = y(3:end) - y(2:end - 1);
cm = reshape(single(-hp ./ (hm .* (hm + hp))), 1, 1, []);
cc = reshape(single((hp - hm) ./ (hm .* hp)), 1, 1, []);
cp = reshape(single(hm ./ (hp .* (hm + hp))), 1, 1, []);
derivative(:, :, 2:end - 1) = field(:, :, 1:end - 2) .* cm ...
    + field(:, :, 2:end - 1) .* cc + field(:, :, 3:end) .* cp;
derivative(:, :, 1) = (field(:, :, 2) - field(:, :, 1)) / single(y(2) - y(1));
derivative(:, :, end) = (field(:, :, end) - field(:, :, end - 1)) / ...
    single(y(end) - y(end - 1));
end


function threshold = fixed_q_threshold(q, percentile)
sample = q(1:4:end, 1:4:end, 1:2:end);
sample = double(sample(isfinite(sample) & sample > 0));
if isempty(sample)
    threshold = NaN;
    warning('No positive Q values were found in the first field.');
else
    threshold = local_percentile(sample, percentile);
end
end


function plot_snapshot_slices(x, y, z, u, p, omega_mag, physical, ...
    near_index, ub, step, time, info, output_file, cfg)
z_mid = max(1, round(numel(z) / 2));
u_scale = max(abs(ub), eps);
lx = get_value(info, 'xlength', x(end));
h = get_value(info, 'ylength', y(end));
lz = get_value(info, 'zlength', z(end));

u_xy = double(squeeze(u(:, z_mid, physical)))' / u_scale;
p_xy = double(squeeze(p(:, z_mid, physical)))';
p_xy = p_xy - mean(p_xy, 2);
u_xz = double(u(:, :, near_index));
u_xz = (u_xz - mean(u_xz, 'all')) / u_scale;
omega_xz = double(omega_mag(:, :, near_index)) * ...
    get_value(info, 'ylength', 1) / u_scale;

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50, 50, 1500, 900]);
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile(layout);
imagesc(x, y, u_xy);
set(gca, 'YDir', 'normal'); axis tight;
xlim([0, lx]); ylim([0, h]);
xlabel('x'); ylabel('y'); title('Mid-span u / U_b'); colorbar;
apply_robust_limits(u_xy, false);

nexttile(layout);
imagesc(x, y, p_xy);
set(gca, 'YDir', 'normal'); axis tight;
xlim([0, lx]); ylim([0, h]);
xlabel('x'); ylabel('y'); title("Mid-span p'"); colorbar;
apply_robust_limits(p_xy, true);

nexttile(layout);
imagesc(x, z, u_xz');
set(gca, 'YDir', 'normal'); axis equal tight;
xlim([0, lx]); ylim([0, lz]);
xlabel('x'); ylabel('z'); title("Near-wall u' / U_b"); colorbar;
apply_robust_limits(u_xz, true);

nexttile(layout);
imagesc(x, z, omega_xz');
set(gca, 'YDir', 'normal'); axis equal tight;
xlim([0, lx]); ylim([0, lz]);
xlabel('x'); ylabel('z'); title('|omega| h / U_b'); colorbar;
apply_robust_limits(omega_xz, false);

colormap(fig, parula(256));
sgtitle(layout, snapshot_title(step, time, info));
save_figure(fig, output_file, cfg.figure_dpi);
close(fig);
end


function plot_q_structure(x, y, z, u, q, um, q_iso, ...
    step, time, info, output_file, cfg)
stride = cfg.q_plot_stride;
ix = 1:stride(1):numel(x);
iz = 1:stride(2):numel(z);
iy = 1:stride(3):numel(y);

xq = x(ix);
zq = z(iz);
yq = y(iy);
% isosurface/isocolors use meshgrid ordering: [z, x, y]. The DNS arrays
% use [x, z, y], so transpose only the temporary visualization arrays.
q_plot = permute(q(ix, iz, iy), [2, 1, 3]);
u_prime = u - reshape(single(um), 1, 1, []);
u_plot = permute(u_prime(ix, iz, iy), [2, 1, 3]);
[X, Z, Y] = meshgrid(xq, zq, yq);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50, 50, 1300, 800]);
ax = axes(fig);
hold(ax, 'on');

if isfinite(q_iso) && max(q_plot, [], 'all') >= q_iso
    surface = isosurface(X, Z, Y, q_plot, q_iso);
    if ~isempty(surface.vertices)
        colors = isocolors(X, Z, Y, u_plot, surface.vertices);
        hp = patch(ax, surface, 'FaceVertexCData', colors, ...
            'FaceColor', 'interp', 'EdgeColor', 'none');
        isonormals(X, Z, Y, q_plot, hp);
        colorbar(ax);
    else
        text(ax, 0.5, 0.5, 0.5, 'No Q isosurface at the fixed threshold.', ...
            'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
else
    text(ax, 0.5, 0.5, 0.5, 'No positive Q isosurface.', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center');
end

xlabel(ax, 'x'); ylabel(ax, 'z'); zlabel(ax, 'y');
title(ax, sprintf('%s, Q_{iso}=%.4e, color=u''', ...
    snapshot_title(step, time, info), q_iso));
axis(ax, 'tight'); axis(ax, 'vis3d');
daspect(ax, [1, 1, 1]); view(ax, -35, 22);
grid(ax, 'on'); box(ax, 'on');
camlight(ax, 'headlight'); lighting(ax, 'gouraud');
colormap(fig, parula(256));
save_figure(fig, output_file, cfg.figure_dpi);
close(fig);
end


function plot_summary_profiles(y, h, um, vm, wm, urms, vrms, wrms, ...
    k, ub, output_file, cfg)
scale = max(abs(ub), eps);
eta = y / h;
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50, 50, 1400, 850]);
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile(layout);
plot(um / scale, eta, 'LineWidth', 1.8); hold on;
plot(vm / scale, eta, '--', 'LineWidth', 1.3);
plot(wm / scale, eta, ':', 'LineWidth', 1.5);
xlabel('Mean velocity / U_b'); ylabel('y / h');
legend('U', 'V', 'W', 'Location', 'best'); grid on;

nexttile(layout);
plot(urms / scale, eta, 'LineWidth', 1.8); hold on;
plot(vrms / scale, eta, 'LineWidth', 1.8);
plot(wrms / scale, eta, 'LineWidth', 1.8);
xlabel('RMS / U_b'); ylabel('y / h');
legend('u_{rms}', 'v_{rms}', 'w_{rms}', 'Location', 'best'); grid on;

nexttile(layout);
plot(k / scale ^ 2, eta, 'LineWidth', 1.8);
xlabel('k / U_b^2'); ylabel('y / h'); grid on;

nexttile(layout);
plot(urms / scale, eta, 'LineWidth', 1.8); hold on;
plot(sqrt(2 * k) / scale, eta, '--', 'LineWidth', 1.8);
xlabel('Turbulence intensity'); ylabel('y / h');
legend('u_{rms}/U_b', 'sqrt(2k)/U_b', 'Location', 'best'); grid on;

sgtitle(layout, sprintf('Statistics from all fields, mean U_b = %.6e', ub));
save_figure(fig, output_file, cfg.figure_dpi);
close(fig);
end


function plot_time_histories(times, steps, ub, k, qmax, enstrophy, ...
    info, output_file, cfg)
omega = get_value(info, 'omega', 0);
if omega > 0 && get_value(info, 'enable_wall_oscillation', 0) ~= 0
    horizontal = times * omega / (2 * pi);
    horizontal_label = 'Oscillation cycles';
else
    horizontal = times;
    horizontal_label = 'Time';
end

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50, 50, 1400, 850]);
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile(layout); plot(horizontal, ub, '-o', 'LineWidth', 1.3, 'MarkerSize', 4);
xlabel(horizontal_label); ylabel('U_b'); grid on;

nexttile(layout); plot(horizontal, k, '-o', 'LineWidth', 1.3, 'MarkerSize', 4);
xlabel(horizontal_label); ylabel('Volume-mean k'); grid on;

nexttile(layout); plot(horizontal, qmax, '-o', 'LineWidth', 1.3, 'MarkerSize', 4);
xlabel(horizontal_label); ylabel('max(Q)'); grid on;

nexttile(layout); plot(horizontal, enstrophy, '-o', 'LineWidth', 1.3, 'MarkerSize', 4);
xlabel(horizontal_label); ylabel('Volume-mean enstrophy'); grid on;

sgtitle(layout, sprintf('Field evolution, steps %d to %d', steps(1), steps(end)));
save_figure(fig, output_file, cfg.figure_dpi);
close(fig);
end


function title_text = snapshot_title(step, time, info)
omega = get_value(info, 'omega', 0);
if omega > 0 && get_value(info, 'enable_wall_oscillation', 0) ~= 0
    phase = mod(omega * time, 2 * pi) * 180 / pi;
    title_text = sprintf('step=%d, t=%.6g, phase=%.1f deg', step, time, phase);
else
    title_text = sprintf('step=%d, t=%.6g', step, time);
end
end


function apply_robust_limits(values, symmetric)
values = double(values(isfinite(values)));
if isempty(values)
    return;
end
if symmetric
    bound = local_percentile(abs(values), 99.5);
    if bound > 0
        set(gca, 'CLim', [-bound, bound]);
    end
else
    low = local_percentile(values, 0.5);
    high = local_percentile(values, 99.5);
    if high > low
        set(gca, 'CLim', [low, high]);
    end
end
end


function value = local_percentile(values, percentile)
values = sort(values(:));
if isempty(values)
    value = NaN;
    return;
end
position = 1 + (numel(values) - 1) * percentile / 100;
lower = floor(position);
upper = ceil(position);
if lower == upper
    value = values(lower);
else
    fraction = position - lower;
    value = values(lower) * (1 - fraction) + values(upper) * fraction;
end
end


function save_figure(fig, output_file, dpi)
if exist('exportgraphics', 'file') == 2
    exportgraphics(fig, output_file, 'Resolution', dpi);
else
    print(fig, output_file, '-dpng', sprintf('-r%d', dpi));
end
end
