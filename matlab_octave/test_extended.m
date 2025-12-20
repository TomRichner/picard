% Test script for Extended Picard in MATLAB
% This script creates synthetic data with a mix of super- and sub-gaussian
% sources, then compares standard Picard vs extended Picard.
%
% This is a more challenging test with 10 sources and 10000 samples.

rng(42); % For reproducibility

% Parameters
N_super = 5;  % Number of super-gaussian sources (Laplace)
N_sub = 5;    % Number of sub-gaussian sources (Uniform)
N = N_super + N_sub;  % Total number of sources
T = 10000; % Number of samples

fprintf('=== Extended Mode ICA Test ===\n');
fprintf('Sources: %d super-gaussian (Laplace) + %d sub-gaussian (Uniform) = %d total\n', N_super, N_sub, N);
fprintf('Samples: %d\n\n', T);

% Create super-gaussian sources (Laplace distribution)
S_super = zeros(N_super, T);
for i = 1:N_super
    S_super(i, :) = laplace_rnd([1, T]);
    S_super(i, :) = S_super(i, :) / std(S_super(i, :)); % Normalize to unit variance
end

% Create sub-gaussian sources (Uniform distribution)
S_sub = zeros(N_sub, T);
for i = 1:N_sub
    S_sub(i, :) = (rand(1, T) - 0.5) * sqrt(12); % Uniform scaled to unit variance
    S_sub(i, :) = S_sub(i, :) / std(S_sub(i, :)); % Normalize to unit variance
end

% Combine sources (interleave super and sub for added difficulty)
S = zeros(N, T);
for i = 1:N_super
    S(2*i-1, :) = S_super(i, :);
end
for i = 1:N_sub
    S(2*i, :) = S_sub(i, :);
end

% Mixing matrix (random, well-conditioned)
A = randn(N, N);
% Ensure mixing matrix is well-conditioned
while cond(A) > 10
    A = randn(N, N);
end
X = A * S;

if any(isinf(X(:))) || any(isnan(X(:)))
    error('Input X contains Nan or Inf');
end

% Run Standard Picard (Non-extended)
fprintf('Running Standard Picard (extended=false)...\n');
[Y_std, ~] = picard(X, 'mode', 'standard', 'extended', false, 'verbose', true, 'distribution', 'logcosh');

% Run Extended Picard
fprintf('\nRunning Extended Picard (extended=true)...\n');
[Y_ext, ~] = picard(X, 'mode', 'standard', 'extended', true, 'verbose', true, 'distribution', 'logcosh');

% Run Standard Picard Ortho (Non-extended)
fprintf('\nRunning Standard Picard Ortho (extended=false)...\n');
[Y_std_ortho, ~] = picard(X, 'mode', 'ortho', 'extended', false, 'verbose', true);

% Run Extended Picard Ortho
fprintf('\nRunning Extended Picard Ortho (extended=true)...\n');
[Y_ext_ortho, ~] = picard(X, 'mode', 'ortho', 'extended', true, 'verbose', true);

perf_std = check_recovery(S, Y_std);
perf_ext = check_recovery(S, Y_ext);
perf_std_ortho = check_recovery(S, Y_std_ortho);
perf_ext_ortho = check_recovery(S, Y_ext_ortho);

fprintf('\nResults:\n');
fprintf('Standard Picard Mean Max Correlation: %.4f\n', perf_std);
fprintf('Extended Picard Mean Max Correlation: %.4f\n', perf_ext);
fprintf('Standard Picard Ortho Mean Max Correlation: %.4f\n', perf_std_ortho);
fprintf('Extended Picard Ortho Mean Max Correlation: %.4f\n', perf_ext_ortho);

if perf_ext > perf_std && perf_ext > 0.95
    fprintf('SUCCESS: Extended mode recovered sources better and accurately.\n');
else
    fprintf('FAILURE: Extended mode did not perform as expected.\n');
end

if perf_ext_ortho > perf_std_ortho && perf_ext_ortho > 0.95
    fprintf('SUCCESS: Extended Ortho mode recovered sources better and accurately.\n');
else
    fprintf('FAILURE: Extended Ortho mode did not perform as expected.\n');
end

% Function to compute correlation with permutation/sign handling
function max_corr = check_recovery(S, Y)
    corrs = abs(corr(S', Y'));
    % Greedy assignment for simplicity or Hungarian alg.
    % For checking, just taking max corr for each source matches
    max_corr = mean(max(corrs, [], 2));
end
