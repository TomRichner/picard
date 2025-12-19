function [Y, W] = picard_standard(X, m, maxiter, precon, tol, lambda_min, ls_tries, verbose, distribution, renormalization, extended)
% Runs the Picard algorithm
%
% The algorithm is detailed in::
%
%     Pierre Ablin, Jean-Francois Cardoso, and Alexandre Gramfort
%     Faster independent component analysis by preconditioning with Hessian
%     approximations
%     IEEE Transactions on Signal Processing, 2018
%     https://arxiv.org/abs/1706.08171
%
% Parameters
% ----------
% X : array, shape (N, T)
%     Matrix containing the signals that have to be unmixed. N is the
%     number of signals, T is the number of samples. X has to be centered
%
% m : int
%     Size of L-BFGS's memory. Typical values for m are in the range 3-15
%
% maxiter : int
%     Maximal number of iterations for the algorithm
%
% precon : 1 or 2
%     Chooses which Hessian approximation is used as preconditioner.
%     1 -> H1
%     2 -> H2
%     H2 is more costly to compute but can greatly accelerate convergence
%     (See the paper for details).
%
% tol : float
%     tolerance for the stopping criterion. Iterations stop when the norm
%     of the gradient gets smaller than tol.
%
% lambda_min : float
%     Constant used to regularize the Hessian approximations. The
%     eigenvalues of the approximation that are below lambda_min are
%     shifted to lambda_min.
%
% ls_tries : int
%     Number of tries allowed for the backtracking line-search. When that
%     number is exceeded, the direction is thrown away and the gradient
%     is used instead.
%
% verbose : boolean
%     If true, prints the informations about the algorithm.
%
% distribution : string, 'logistic' or 'logcosh'
%     The distribution to use for the score function.
%
% renormalization : string, 'original' or 'pythonlike'
%     The method for Hessian renormalization.
%
% Returns
% -------
% Y : array, shape (N, T)
%     The estimated source matrix
%
% W : array, shape (N, N)
%     The estimated unmixing matrix, such that Y = WX.

% Authors: Pierre Ablin <pierre.ablin@inria.fr>
%          Alexandre Gramfort <alexandre.gramfort@inria.fr>
%          Jean-Francois Cardoso <cardoso@iap.fr>
%
% License: BSD (3-clause)

% Set defaults for new parameters
if nargin < 9 || isempty(distribution)
    distribution = 'logistic';
end
if nargin < 10 || isempty(renormalization)
    renormalization = 'original';
end
if nargin < 11 || isempty(extended)
    extended = false;
end

% Init
[N, T] = size(X);
if extended
    C = (X * X') / T;
end
W = eye(N);
Y = X;
s_list = {};
y_list = {};
r_list = {};
signs = ones(N, 1);
old_signs = signs;
sign_change = false;
current_loss = loss(Y, W, signs, distribution, extended);

for n_top = 1:maxiter
    % Compute the score function
    if strcmp(distribution, 'logistic')
        psiY = tanh(Y / 2.);
    else % logcosh
        psiY = tanh(Y);
    end
    % Compute the relative gradient
    % Compute the relative gradient
    if extended
        C = (Y * Y') / T;
        if strcmp(distribution, 'logistic')
            psidY = (- psiY.^2 + 1.) / 2.;
        else % logcosh
            psidY = 1 - psiY.^2;
        end
        G = (psiY * Y') / T;
        K = mean(psidY, 2) .* diag(C) - diag(G);
        signs = sign(K);
        if n_top > 1
            sign_change = any(signs ~= old_signs);
        end
        old_signs = signs;
        G = diag(signs) * G;
        psiY = diag(signs) * psiY;
        % python: psidY *= signs[:, None]. In matlab signs is col vector.
        % We don't use psidY later in loop except for hessian (which recomp it).
        % But we need to update G.
        G = G + C;
        % python: psidY += 1. 
        % This psidY update is not used for G here? 
        % Wait, in python 'G' is computed using 'psidY' in ortho case for H_off?
        % In non-ortho (this file): 
        % G -= eye(N).
        % if extended: G += C (which accounts for the 'eye(N)' term if C is identity for white data?)
        % Actually python: 'if not ortho: G += C'. Then later 'G -= eye(N)'.
        % If data is white, C=I, so G += I followed by G -= I cancels out? 
        % No, G += C is done BEFORE G -= I. 
        % So effective G = G_orig + C - I.
    else
        G = (psiY * Y') / T;
    end
    G = G - eye(N);
    
    % Stopping criterion
    G_norm = max(max(abs(G)));
    if G_norm < tol
        break
    end
    % Update the memory
    if n_top > 1
        s_list{end + 1} = direction;
        y = G - G_old;
        y_list{end + 1} = y;
        r_list{end + 1} = 1. / sum(sum(direction .* y));
        if length(s_list) > m
            s_list = s_list(2:end);
            y_list = y_list(2:end);
            r_list = r_list(2:end);
        end
    end
    G_old = G;
    % Flush the memory if there is a sign change.
    if extended && sign_change
        current_loss = NaN; % Force recompute
        s_list = {};
        y_list = {};
        r_list = {};
    end
    % Find the L-BFGS direction
    % Note: l_bfgs_direction recomputes psidY internally for Hessian. 
    % We should pass signs to it?
    % Python: h = _regularize_hessian(h, h_off, lambda_min) where h = inner(psidY, Y_square).
    % Python psidY was updated with signs. 
    % So we MUST pass signs to l_bfgs_direction or update Y effectively?
    % Actually: Jacobian is diag(signs) * J_original. 
    % The code in l_bfgs_direction computes psidY from Y using 'distribution'.
    % If signs are mixed, simple 'Y' computation in l_bfgs is wrong if it assumes all positive?
    % Wait, psiY = tanh(Y/2). psidY = (1-psiY^2)/2. depends on Y.
    % If we changed sign of source, Y -> -Y? No, signs vector is just for the score function adaptation.
    % We are effectively optimizing J(Y) = sum_i( log_lik_i(y_i) ). 
    % extended means log_lik_i can be swapped.
    % So we need to pass 'signs' to l_bfgs_direction to compute correct psidY?
    % Yes. In python, psidY is computed once and reused. Here it is recomputed inside solve_hessian.
    
    direction = l_bfgs_direction(Y, psiY, G, s_list, y_list, r_list, precon, lambda_min, signs, distribution, renormalization, extended);
    % Do a line_search in that direction:
    [converged, new_Y, new_W, new_loss, direction] = line_search(Y, W, direction, current_loss, ls_tries, verbose, signs, distribution, extended);
    if ~converged
        direction = -G;
        s_list = {};
        y_list = {};
        r_list = {};
        [~, new_Y, new_W, new_loss, direction] = line_search(Y, W, direction, current_loss, 10, false, signs, distribution, extended);
    end
    Y = new_Y;
    W = new_W;
    current_loss = new_loss;
    if verbose
        fprintf('iteration %d, gradient norm = %.6g loss = %.6g\n', n_top, G_norm, current_loss)
    end
end

function [loss_val] = loss(Y, W, signs, distribution, extended)
    %
    % Computes the loss function for Y, W
    %
    N = size(Y, 1);
    loss_val = - log(abs(det(W)));
    val_nll = 0;
    if strcmp(distribution, 'logistic')
        for n=1:N
            y = Y(n, :);
            val_nll = val_nll + signs(n) * mean(abs(y) + 2. * log1p(exp(-abs(y))));
        end
    else % logcosh
        for k = 1:N
            y = Y(k, :);
            val_nll = val_nll + signs(k) * mean( abs(y) + log1p( exp(-2*abs(y)) ) );
        end
    end
    loss_val = loss_val + val_nll;
    
    val_reg = 0;
    if extended
        val_reg = 0.5 * sum(mean(Y.^2, 2));
        loss_val = loss_val + val_reg;
    end
end

function [converged, Y_new, W_new, new_loss, rel_step] = line_search(Y, W, direction, current_loss, ls_tries, verbose, signs, distribution, extended)
    %
    % Performs a backtracking line search, starting from Y and W, in the
    % direction direction. I
    %
    N = size(Y, 1);
    projected_W = direction * W;
    alpha = 1.;
    if isnan(current_loss)
        current_loss = loss(Y, W, signs, distribution, extended);
    end
    for iter_ls=1:ls_tries
        Y_new = (eye(N) + alpha * direction) * Y;
        W_new = W + alpha * projected_W;
        new_loss = loss(Y_new, W_new, signs, distribution, extended);
        if new_loss < current_loss
            converged = true;
            rel_step = alpha * direction;
            return
        end
        alpha = alpha / 2.;
    end
    if verbose
        fprintf('line search failed, falling back to gradient.\n');
    end
    converged = false;
    rel_step = alpha * direction;
end

function [direction] = l_bfgs_direction(Y, psiY, G, s_list, y_list, r_list, precon, lambda_min, signs, distribution, renormalization, extended)
    q = G;
    a_list = {};
    for ii=1:length(s_list)
        s = s_list{end - ii + 1};
        y = y_list{end - ii + 1};
        r = r_list{end - ii + 1};
        alpha = r * sum(sum(s .* q));
        a_list{end + 1} = alpha;
        q = q - alpha * y;
    end
    z = solve_hessian(q, Y, psiY, precon, lambda_min, signs, distribution, renormalization, extended);
    for ii=1:length(s_list)
        s = s_list{ii};
        y = y_list{ii};
        r = r_list{ii};
        alpha = a_list{end - ii + 1};
        beta = r * sum(sum(y .* z));
        z = z + (alpha - beta) * s;
    end
    direction = -z;
end

function a = regularize_hessian_pythonlike(a, lam)
    N = size(a, 1);
    % compute the smaller eigenvalue of each 2×2 block
    e = 0.5*(a + a' - sqrt((a - a').^2 + 4));
    % find entries where that eigenvalue is below lam (excluding diagonal)
    mask = e < lam;
    mask(1:(N+1):N*N) = false;
    [i, j] = find(mask);
    if ~isempty(i)
        idx = sub2ind([N, N], i, j);
        % shift those entries so that their block eigenvalue equals lam
        a(idx) = a(idx) + (lam - e(idx));
    end
end

function [out] = solve_hessian(G, Y, psiY, precon, lambda_min, signs, distribution, renormalization, extended)
    [N, T] = size(Y);
    % Compute the derivative of the score
    if strcmp(distribution, 'logistic')
        psidY = (- psiY.^2 + 1.) / 2.;
    else % logcosh
        psidY = 1 - psiY.^2;
    end
    if extended
        psidY = diag(signs) * psidY;
        psidY = psidY + 1;
    end
    % Build the diagonal of the Hessian, a.
    Y_squared = Y.^2;
    if precon == 2
        a = (psidY * Y_squared') / T;
    elseif precon == 1
        sigma2 = mean(Y_squared, 2);
        psidY_mean = mean(psidY, 2);
        a = psidY_mean * sigma2';
        diagonal_term = mean(mean(Y_squared .* psidY)) + 1.;
        a(1:(N+1):N*N) = diagonal_term;
    else
        error('precon should be 1 or 2')
    end

    % Regularize
    if strcmp(renormalization, 'original')
        eigenvalues = 0.5 * (a + a' - sqrt((a - a').^2 + 4.));
        problematic_locs = eigenvalues < lambda_min;
        problematic_locs(1:(N+1):N*N) = false;
        [i_pb, j_pb] = find(problematic_locs);
        a(i_pb, j_pb) = a(i_pb, j_pb) + lambda_min - eigenvalues(i_pb, j_pb);
    else % pythonlike
        a = regularize_hessian_pythonlike(a, lambda_min);
    end
    
    % Invert the transform
    out = (G .* a' - G') ./ (a .* a' - 1.);
end

end
