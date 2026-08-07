clear; clc; close all;

case_dir = 'E:\file\zd\stokes2\Re_delta=700';
files = dir(fullfile(case_dir, 'tau_wall_map_*.dat'));
if isempty(files)
    error('No tau_wall_map_*.dat files found in %s', case_dir);
end

% Show a small sequence of maps so vortex tubes and turbulent spots can be tracked.
max_maps = min(6, numel(files));
pick = round(linspace(1, numel(files), max_maps));

figure('Position', [120, 80, 1200, 720], 'Color', 'w');
tiledlayout(2, ceil(max_maps / 2), 'TileSpacing', 'compact', 'Padding', 'compact');

for ii = 1:max_maps
    fname = fullfile(files(pick(ii)).folder, files(pick(ii)).name);
    header = readmatrix(fname, 'FileType', 'text', 'Range', [1 1 1 4]);
    tau = readmatrix(fname, 'FileType', 'text', 'NumHeaderLines', 1);

    current_time = header(1);
    u_wall = header(2);

    nexttile;
    imagesc(tau);
    axis image tight;
    colorbar;
    title(sprintf('t=%.4f, U_w=%.3f', current_time, u_wall));
    xlabel('x index');
    ylabel('z index');
end

sgtitle('Local wall-shear maps: bands indicate vortex tubes, localized spikes indicate spots');
