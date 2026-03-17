% Cross-language test: MATLAB vs Python Picard standard extended
% Generates synthetic data, runs both implementations with identical settings,
% and compares results.

rng(42);

%% Parameters — identical for both languages
N_super = 32;  % super-gaussian (Laplace)
N_sub   = 32;  % sub-gaussian (Uniform)
N = N_super + N_sub;
T = 10000;

tol = 1e-7;
maxiter = 500;
m_lbfgs = 7;
lambda_min = 0.01;
ls_tries = 10;

fprintf('=== Cross-Language Picard Test ===\n');
fprintf('N=%d (super=%d, sub=%d), T=%d\n\n', N, N_super, N_sub, T);

%% Generate sources
S = zeros(N, T);
for i = 1:N_super
    S(i, :) = laplace_rnd([1, T]);
    S(i, :) = S(i, :) / std(S(i, :));
end
for i = 1:N_sub
    S(N_super+i, :) = (rand(1, T) - 0.5) * sqrt(12);
    S(N_super+i, :) = S(N_super+i, :) / std(S(N_super+i, :));
end

%% Mixing matrix — well-conditioned by construction
% QR gives random orthogonal matrix; controlled singular values guarantee cond ~5
[Q1, ~] = qr(randn(N));
[Q2, ~] = qr(randn(N));
singular_values = linspace(1, 5, N);  % cond(A) = 5 by design
A = Q1 * diag(singular_values) * Q2';
X = A * S;

%% Whiten (using picard's whitening — spherical)
[X_white, W_white] = whitening(X, 'sph', N);

%% Save data for Python
data_file = fullfile(fileparts(mfilename('fullpath')), 'cross_test_data.mat');
save(data_file, 'X_white', 'S', 'A', 'W_white', ...
     'tol', 'maxiter', 'm_lbfgs', 'lambda_min', 'ls_tries');
fprintf('Saved whitened data to %s\n\n', data_file);

%% Run MATLAB picard_standard with extended
fprintf('--- MATLAB: Running picard_standard (extended=true, logcosh) ---\n');
tic;
mat_log = evalc('[Y_mat, W_mat] = picard_standard(X_white, m_lbfgs, maxiter, 2, tol, lambda_min, ls_tries, true, ''logcosh'', ''pythonlike'', true);');
t_mat = toc;
fprintf('MATLAB finished in %.3f sec\n\n', t_mat);

% Parse MATLAB convergence log
mat_lines = strsplit(mat_log, '\n');
mat_iter = []; mat_gnorm = []; mat_loss = [];
for k = 1:length(mat_lines)
    tokens = regexp(mat_lines{k}, 'iteration (\d+), gradient norm = ([\d.e+-]+) loss = ([\d.e+-]+)', 'tokens');
    if ~isempty(tokens)
        mat_iter(end+1) = str2double(tokens{1}{1});
        mat_gnorm(end+1) = str2double(tokens{1}{2});
        mat_loss(end+1) = str2double(tokens{1}{3});
    end
end
fprintf('MATLAB converged in %d iterations\n', length(mat_iter));

%% Run Python
fprintf('--- Python: Running core_picard (extended=True, Tanh) ---\n');
py_script = fullfile(fileparts(mfilename('fullpath')), 'test_cross_language.py');
python_exe = 'c:\Users\m218089\Desktop\gitlab_repos\cscs_dynamics\.venv\Scripts\python.exe';
python_cmd = sprintf('"%s" "%s" "%s"', python_exe, py_script, data_file);
tic;
[status, output] = system(python_cmd);
t_py = toc;
fprintf('%s\n', output);
if status ~= 0
    error('Python script failed with status %d', status);
end
fprintf('Python finished in %.3f sec\n\n', t_py);

%% Load Python results
py_results_file = strrep(data_file, '.mat', '_python_results.mat');
py = load(py_results_file);

%% Compare
fprintf('=== Comparison ===\n\n');

% 1. Source recovery quality
corr_mat = abs(corr(S', Y_mat'));
corr_py  = abs(corr(S', py.Y_py'));
recovery_mat = mean(max(corr_mat, [], 2));
recovery_py  = mean(max(corr_py, [], 2));
fprintf('Source recovery (mean max |corr|):\n');
fprintf('  MATLAB:  %.6f\n', recovery_mat);
fprintf('  Python:  %.6f\n', recovery_py);
fprintf('  Diff:    %.2e\n\n', abs(recovery_mat - recovery_py));

% 2. W matrix similarity
% Align permutation and sign: for each Python row, find best MATLAB row
W_mat_norm = W_mat ./ vecnorm(W_mat, 2, 2);
W_py_norm  = py.W_py ./ vecnorm(py.W_py, 2, 2);
W_sim = abs(W_mat_norm * W_py_norm');
fprintf('W matrix similarity (|cos angle| between matched rows):\n');
for i = 1:N
    [best_cos, best_j] = max(W_sim(i, :));
    fprintf('  MATLAB row %d <-> Python row %d: cos=%.6f\n', i, best_j, best_cos);
end

% 3. Overall match
matched_cos = zeros(N, 1);
used = false(N, 1);
for i = 1:N
    avail = find(~used);
    [best_cos, idx] = max(W_sim(i, avail));
    matched_cos(i) = best_cos;
    used(avail(idx)) = true;
end
fprintf('\nMean matched |cos|: %.6f\n', mean(matched_cos));
fprintf('Min  matched |cos|: %.6f\n\n', min(matched_cos));

if min(matched_cos) > 0.99 && recovery_mat > 0.95 && recovery_py > 0.95
    fprintf('PASS: MATLAB and Python produce equivalent results.\n');
else
    fprintf('REVIEW: Results differ — check settings alignment.\n');
end
%% Plot convergence trajectories
py_gnorms = py.py_gnorms(:)';
py_losses = py.py_losses(:)';
py_iters  = py.py_iters(:)';

figure('Position', [100 100 1200 500]);

subplot(1,2,1);
semilogy(mat_iter, mat_gnorm, 'b-o', 'MarkerSize', 3, 'LineWidth', 1.5, 'DisplayName', 'MATLAB');
hold on;
semilogy(py_iters, py_gnorms, 'r-s', 'MarkerSize', 3, 'LineWidth', 1.5, 'DisplayName', 'Python');
yline(tol, 'k--', 'LineWidth', 1, 'DisplayName', 'Tolerance');
xlabel('Iteration'); ylabel('Gradient Norm');
title(sprintf('Gradient Norm Convergence (N=%d, T=%d)', N, T));
legend('Location', 'best'); grid on;

subplot(1,2,2);
plot(mat_iter, mat_loss, 'b-o', 'MarkerSize', 3, 'LineWidth', 1.5, 'DisplayName', 'MATLAB');
hold on;
plot(py_iters, py_losses, 'r-s', 'MarkerSize', 3, 'LineWidth', 1.5, 'DisplayName', 'Python');
xlabel('Iteration'); ylabel('Loss');
title(sprintf('Loss Convergence (N=%d, T=%d)', N, T));
legend('Location', 'best'); grid on;

sgtitle('MATLAB vs Python Picard Standard Extended — Convergence Comparison');

%% Cleanup
delete(data_file);
delete(py_results_file);
