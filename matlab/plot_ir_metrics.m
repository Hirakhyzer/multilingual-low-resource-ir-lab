function plot_ir_metrics(outputs_dir)
%PLOT_IR_METRICS Plot synthetic multilingual IR robustness metrics.
% Usage: plot_ir_metrics('outputs')

if nargin < 1
    outputs_dir = fullfile('..', 'outputs');
end
input_path = fullfile(outputs_dir, 'results', 'synthetic_retrieval_metrics.csv');
if ~isfile(input_path)
    error('Missing %s. Run scripts/run_synthetic_ir_lab.py first.', input_path);
end
T = readtable(input_path);
figure('Color', 'w', 'Position', [100 100 900 430]);
methods = unique(T.method, 'stable');
hold on;
for i = 1:numel(methods)
    rows = T(strcmp(T.method, methods{i}), :);
    plot(rows.noise_rate, rows.top3_accuracy, '-o', 'LineWidth', 1.2, 'DisplayName', methods{i});
end
grid on;
xlabel('Synthetic OCR noise rate'); ylabel('Top-3 accuracy');
title('Synthetic multilingual retrieval robustness');
legend('Location', 'best');
exportgraphics(gcf, fullfile(outputs_dir, 'figures', 'synthetic_retrieval_robustness_matlab.png'), 'Resolution', 250);
end
