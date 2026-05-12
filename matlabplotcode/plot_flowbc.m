function plot_flowbc
    clc; close all;

    pi_val = 3.1415926535;
    xlength = 4 * pi_val;
    ylength = 2.0;
    ubulk   = 1.0;
    V_wall_blowing = 0.0034 * ubulk;

    figure('Units', 'pixels', 'Position', [100, 100, 1200, 500], 'Color', 'w');
    hold on; box on;

    % 上下壁面
    plot([0, xlength], [0, 0], 'k-', 'LineWidth', 3.5);
    plot([0, xlength], [ylength, ylength], 'k-', 'LineWidth', 3.5);

    % 余弦波填色
    amp_vis = 0.22;
    x_fine = linspace(0, xlength, 1200);
    wave = amp_vis * cos(x_fine);

    [x_seg_pos, y_seg_pos] = build_segments(x_fine, wave, 'positive');
    [x_seg_neg, y_seg_neg] = build_segments(x_fine, wave, 'negative');

    fill(x_seg_pos, y_seg_pos, [1.0, 0.42, 0.42], ...
         'FaceAlpha', 0.55, 'EdgeColor', 'none');
    fill(x_seg_neg, y_seg_neg, [0.42, 0.52, 1.0], ...
         'FaceAlpha', 0.55, 'EdgeColor', 'none');
    fill(x_seg_pos, ylength + y_seg_pos, [0.42, 0.52, 1.0], ...
         'FaceAlpha', 0.55, 'EdgeColor', 'none');
    fill(x_seg_neg, ylength + y_seg_neg, [1.0, 0.42, 0.42], ...
         'FaceAlpha', 0.55, 'EdgeColor', 'none');

    plot(x_fine, wave, 'b-', 'LineWidth', 2.2);
    plot(x_fine, ylength + wave, 'b-', 'LineWidth', 2.2);

    % 吹吸法向箭头
    n_arr = 46;
    x_arr = linspace(0.10, xlength - 0.10, n_arr);
    for i = 1:n_arr
        v_loc = V_wall_blowing * cos(x_arr(i));
        if abs(v_loc) > 0.01 * V_wall_blowing
            if v_loc > 0
                quiver(x_arr(i), 0.08, 0,  0.18, 0, 'r', ...
                       'LineWidth', 1.0, 'MaxHeadSize', 0.28, 'AutoScale', 'off');
                quiver(x_arr(i), ylength - 0.08, 0,  0.18, 0, 'b', ...
                       'LineWidth', 1.0, 'MaxHeadSize', 0.28, 'AutoScale', 'off');
            else
                quiver(x_arr(i), 0.08, 0, -0.18, 0, 'b', ...
                       'LineWidth', 1.0, 'MaxHeadSize', 0.28, 'AutoScale', 'off');
                quiver(x_arr(i), ylength - 0.08, 0, -0.18, 0, 'r', ...
                       'LineWidth', 1.0, 'MaxHeadSize', 0.28, 'AutoScale', 'off');
            end
        end
    end

    % 流向箭头
    h = ylength / 2;
    for yi = 1:3
        ya = 0.45 + (yi-1) * (ylength - 0.9) / 2;
        for xi = 1:6
            xa = 0.5 + (xi-1) * 2.0;
            quiver(xa, ya, 0.9, 0, 0, 'k', ...
                   'LineWidth', 0.6, 'MaxHeadSize', 0.24, 'AutoScale', 'off');
        end
    end

    % 标注
    text(xlength - 0.5, h, 'Flow \rightarrow', ...
         'FontSize', 12, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'right', 'Interpreter', 'tex');

    text(-0.35, 0, 'y=0', 'FontSize', 11, ...
         'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'Interpreter', 'tex');
    text(-0.35, ylength, 'y=2h', 'FontSize', 11, ...
         'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'Interpreter', 'tex');

    % 图例
    text(pi_val, ylength + 0.50, 'Lower: Blowing / Upper: Suction', 'FontSize', 10, ...
         'Color', [0.7, 0.15, 0.15], 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Interpreter', 'none');
    text(3*pi_val, ylength + 0.50, 'Lower: Suction / Upper: Blowing', 'FontSize', 10, ...
         'Color', [0.15, 0.15, 0.7], 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Interpreter', 'none');

    xlabel('x  (Streamwise)', 'Interpreter', 'none', 'FontSize', 14);
    ylabel('y  (Wall-normal)', 'Interpreter', 'none', 'FontSize', 14);

    xlim([-0.6, xlength + 1.0]);
    ylim([-0.55, ylength + 0.70]);
    set(gca, 'FontSize', 12, 'FontName', 'Times New Roman', ...
             'YTick', [0, 0.5, 1.0, 1.5, 2.0], ...
             'YTickLabel', {'0', '', 'h', '', '2h'});

    saveas(gcf, 'plot_flowbc.png');
end

function [x_seg, y_seg] = build_segments(x, y, sign_type)
    if strcmp(sign_type, 'positive')
        mask = (y >= 0);
    else
        mask = (y < 0);
    end
    diff_mask = diff([false, mask, false]);
    seg_start = find(diff_mask == 1);
    seg_end   = find(diff_mask == -1) - 1;
    n_seg = length(seg_start);
    total_len = sum(seg_end - seg_start + 1) + n_seg;
    x_seg = zeros(1, total_len);
    y_seg = zeros(1, total_len);
    idx = 1;
    for s = 1:n_seg
        len = seg_end(s) - seg_start(s) + 1;
        x_seg(idx:idx+len-1) = x(seg_start(s):seg_end(s));
        y_seg(idx:idx+len-1) = y(seg_start(s):seg_end(s));
        idx = idx + len;
        if s < n_seg
            x_seg(idx) = NaN;
            y_seg(idx) = NaN;
            idx = idx + 1;
        end
    end
    x_seg = x_seg(1:idx-1);
    y_seg = y_seg(1:idx-1);
end
