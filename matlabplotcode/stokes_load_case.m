function data = stokes_load_case(case_dir, Re_delta, opts)
%STOKES_LOAD_CASE Load Stokes DNS output and build robust diagnostics.
%   The legacy stat.dat format stores 11 rows per snapshot. This helper keeps
%   that format intact, fixes the inferred snapshot time, and computes both a
%   two-point wall shear estimate and a less fragile near-wall multi-point
%   estimate.

if nargin < 3
    opts = struct();
end

defaults = struct();
defaults.U_osc = 1.0;
defaults.omega = 2.0 * pi;
defaults.dt = 0.0005;
defaults.output_interval = 10;
defaults.x_length = 192;
defaults.z_length = 128;
defaults.fit_points = 12;
defaults.start_periods = 5;

names = fieldnames(defaults);
for ii = 1:numel(names)
    name = names{ii};
    if ~isfield(opts, name)
        opts.(name) = defaults.(name);
    end
end

fname_mesh = fullfile(case_dir, 'mesh.dat');
fname_stat = fullfile(case_dir, 'stat.dat');

if ~exist(fname_mesh, 'file') || ~exist(fname_stat, 'file')
    error('Missing mesh.dat or stat.dat in %s', case_dir);
end

xyz = importdata(fname_mesh);
y_length = length(xyz) - opts.x_length - opts.z_length;
if y_length <= 2
    error('mesh.dat length is inconsistent with x_length/z_length in %s', case_dir);
end

y = xyz(opts.x_length + 1 : opts.x_length + y_length);
yc = y(2:end-1);
nyc = length(yc);

raw = importdata(fname_stat);
data_num = 11;
num_steps = floor(size(raw, 1) / data_num);
if num_steps < 1
    error('stat.dat has no complete snapshots in %s', case_dir);
end
if size(raw, 2) ~= nyc
    warning('stat.dat columns (%d) differ from inferred nyc (%d).', size(raw, 2), nyc);
end

um = zeros(nyc, num_steps);
vm = zeros(nyc, num_steps);
wm = zeros(nyc, num_steps);
um2 = zeros(nyc, num_steps);
vm2 = zeros(nyc, num_steps);
wm2 = zeros(nyc, num_steps);

for i = 1:num_steps
    base_idx = data_num * (i - 1);
    um(:, i) = raw(base_idx + 4, :)';
    vm(:, i) = raw(base_idx + 5, :)';
    wm(:, i) = raw(base_idx + 6, :)';
    um2(:, i) = raw(base_idx + 8, :)';
    vm2(:, i) = raw(base_idx + 9, :)';
    wm2(:, i) = raw(base_idx + 10, :)';
end

nu = (2.0 * opts.U_osc^2) / (opts.omega * Re_delta^2);
delta = sqrt(2.0 * nu / opts.omega);

% Existing data were written every output_interval steps, and the velocity
% field corresponds to the just-finished RK step. This is also the convention
% used by the updated CUDA tau_wall.dat output.
time = ((1:num_steps) * opts.output_interval + 1) * opts.dt;
u_wall = opts.U_osc * cos(opts.omega * time);

y1 = yc(1);
y2 = yc(2);
c0 = (y1 + y2) / (y1 * y2);
c1 = -y2 / (y1 * (y2 - y1));
c2 = y1 / (y2 * (y2 - y1));
tau_fd2 = nu * (c0 * u_wall + c1 * um(1, :) + c2 * um(2, :));

fit_points = min([opts.fit_points, nyc, 24]);
dudy_fit = wall_derivative_fit(yc(1:fit_points), u_wall, um(1:fit_points, :), delta);
tau_fit = -nu * dudy_fit;

tau_file = fullfile(case_dir, 'tau_wall.dat');
tau_cuda = [];
if exist(tau_file, 'file')
    tau_cuda = importdata(tau_file);
end

urms = sqrt(max(um2 - um.^2, 0.0));
vrms = sqrt(max(vm2 - vm.^2, 0.0));
wrms = sqrt(max(wm2 - wm.^2, 0.0));
k_pert = 0.5 * (urms.^2 + vrms.^2 + wrms.^2);

snaps_per_period = round((2.0 * pi / opts.omega) / (opts.output_interval * opts.dt));
start_snap = max(1, opts.start_periods * snaps_per_period);
if num_steps < 10 * snaps_per_period
    start_snap = max(1, floor(num_steps / 2));
end

diagnostics = period_diagnostics(k_pert, vrms, wrms, snaps_per_period);

data = struct();
data.case_dir = case_dir;
data.Re_delta = Re_delta;
data.y = y;
data.yc = yc;
data.nyc = nyc;
data.nu = nu;
data.delta = delta;
data.time = time;
data.u_wall = u_wall;
data.um = um;
data.vm = vm;
data.wm = wm;
data.urms = urms;
data.vrms = vrms;
data.wrms = wrms;
data.k_pert = k_pert;
data.tau_fd2 = tau_fd2;
data.tau_fit = tau_fit;
data.tau_cuda = tau_cuda;
data.snaps_per_period = snaps_per_period;
data.start_snap = start_snap;
data.diagnostics = diagnostics;
data.options = opts;
end

function dudy = wall_derivative_fit(yc, u_wall, u_near, delta)
eta = [0; yc(:) ./ delta];
degree = min(3, length(eta) - 1);
X = zeros(length(eta), degree + 1);
for p = 0:degree
    X(:, p + 1) = eta.^p;
end

num_steps = length(u_wall);
dudy = zeros(1, num_steps);
for i = 1:num_steps
    values = [u_wall(i); u_near(:, i)];
    coeff = X \ values;
    dudy(i) = coeff(2) / delta;
end
end

function diagnostics = period_diagnostics(k_pert, vrms, wrms, snaps_per_period)
num_steps = size(k_pert, 2);
num_periods = floor(num_steps / snaps_per_period);
valid_len = num_periods * snaps_per_period;

diagnostics = struct();
diagnostics.period_index = 1:num_periods;
diagnostics.max_k = [];
diagnostics.mean_k = [];
diagnostics.max_vrms = [];
diagnostics.max_wrms = [];
diagnostics.growth_flags = [];

if num_periods < 1
    return;
end

near_count = min(80, size(k_pert, 1));
k_near = k_pert(1:near_count, 1:valid_len);
v_near = vrms(1:near_count, 1:valid_len);
w_near = wrms(1:near_count, 1:valid_len);

k_period = reshape(k_near, near_count, snaps_per_period, num_periods);
v_period = reshape(v_near, near_count, snaps_per_period, num_periods);
w_period = reshape(w_near, near_count, snaps_per_period, num_periods);

diagnostics.max_k = squeeze(max(max(k_period, [], 1), [], 2)).';
diagnostics.mean_k = squeeze(mean(mean(k_period, 1), 2)).';
diagnostics.max_vrms = squeeze(max(max(v_period, [], 1), [], 2)).';
diagnostics.max_wrms = squeeze(max(max(w_period, [], 1), [], 2)).';

if num_periods > 1
    ratios = diagnostics.mean_k(2:end) ./ max(diagnostics.mean_k(1:end-1), eps);
    diagnostics.growth_flags = find(ratios > 1.02) + 1;
end
end
