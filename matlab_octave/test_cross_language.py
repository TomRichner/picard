"""
Cross-language test: Run Python Picard on the same data saved by MATLAB.
Reads whitened data from a .mat file, runs Picard standard extended,
and saves results back for MATLAB to compare.
"""
import sys
import os
import numpy as np
import scipy.io as sio

# Import the modules directly without going through picard/__init__.py (which requires sklearn)
import importlib.util
import types

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

picard_root = r'c:\Users\m218089\Desktop\github_repos\picard\picard'

# Load densities first
densities = _load_module('picard.densities', os.path.join(picard_root, 'densities.py'))

# Create a fake picard package so relative imports in _core_picard.py work
fake_pkg = types.ModuleType('picard')
fake_pkg.__path__ = [picard_root]
sys.modules['picard'] = fake_pkg
sys.modules['picard.densities'] = densities

# Now load _core_picard (its `from .densities import Tanh` will resolve)
core = _load_module('picard._core_picard', os.path.join(picard_root, '_core_picard.py'))

Tanh = densities.Tanh
core_picard = core.core_picard

def main():
    mat_file = sys.argv[1] if len(sys.argv) > 1 else 'cross_test_data.mat'
    
    data = sio.loadmat(mat_file)
    X_white = data['X_white']  # Already whitened data
    
    # Extract settings
    tol = float(data['tol'].flat[0])
    maxiter = int(data['maxiter'].flat[0])
    m_lbfgs = int(data['m_lbfgs'].flat[0])
    lambda_min = float(data['lambda_min'].flat[0])
    ls_tries = int(data['ls_tries'].flat[0])
    
    N = X_white.shape[0]
    
    # Use identity w_init (same as MATLAB default)
    w_init = np.eye(N)
    X_init = w_init @ X_white
    
    # Covariance for extended mode (whitened data -> identity)
    covariance = np.eye(N)
    
    # Run core_picard directly with standard (ortho=False), extended=True
    # Using Tanh(alpha=1) which is the 'logcosh' equivalent
    density = Tanh()
    
    print(f"Python: Running core_picard (standard, extended=True)")
    print(f"  N={N}, T={X_white.shape[1]}, tol={tol}, maxiter={maxiter}, m={m_lbfgs}")
    print(f"  lambda_min={lambda_min}, ls_tries={ls_tries}")
    
    Y_py, W_core, infos = core_picard(
        X_init,
        density=density,
        ortho=False,
        extended=True,
        m=m_lbfgs,
        max_iter=maxiter,
        tol=tol,
        lambda_min=lambda_min,
        ls_tries=ls_tries,
        verbose=True,
        covariance=covariance
    )
    
    # W_core is relative to X_init. Full W = W_core @ w_init
    W_py = W_core @ w_init
    
    print(f"\nPython converged: {infos['converged']}")
    print(f"Python iterations: {infos['n_iterations']}")
    print(f"Python gradient norm: {infos['gradient_norm']:.6e}")
    print(f"Python signs: {infos['signs']}")
    
    # Save results
    out_file = mat_file.replace('.mat', '_python_results.mat')
    sio.savemat(out_file, {
        'Y_py': Y_py,
        'W_py': W_py,
        'n_iterations': infos['n_iterations'],
        'gradient_norm': infos['gradient_norm'],
        'signs_py': infos['signs'],
        'converged': infos['converged']
    })
    print(f"\nResults saved to {out_file}")

if __name__ == '__main__':
    main()
