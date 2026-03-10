"""
Optimized Adaptive Heun Solver - Drop-in replacement for torchdiffeq

Replace your existing torchdiffeq/adaptive_heun.py with this file.
Also add adaptive_heun_fast.py to the same directory.

CRITICAL BUG FIX:
The original torchdiffeq incorrectly uses FSAL for Heun, reusing k2 as k1 for
the next step. Since k2 is evaluated at y_euler (not y1), this introduces O(h) 
error per step, degrading Heun from 2nd order to effectively 1st order!

This implementation provides:
- correct_fsal=True (default): Correct 2nd order accuracy, 2 evals/step
- correct_fsal=False: Match original torchdiffeq behavior (faster, 1st order)

Other optimizations:
- Specialized Heun step (no Butcher tableau loop)
- Horner's rule for interpolation
- Fused operations
"""

import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver

# Try to import the fast implementation
try:
    from .adaptive_heun_fast import AdaptiveHeunSolver as _FastSolver
    _USE_FAST = True
except ImportError:
    _USE_FAST = False

# Butcher tableau for API compatibility
_ADAPTIVE_HEUN_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1.], dtype=torch.float64),
    beta=[
        torch.tensor([1.], dtype=torch.float64),
    ],
    c_sol=torch.tensor([0.5, 0.5], dtype=torch.float64),
    c_error=torch.tensor([0.5, -0.5], dtype=torch.float64),
)

_AH_C_MID = torch.tensor([0.5, 0.], dtype=torch.float64)


if _USE_FAST:
    # Use the optimized implementation
    class AdaptiveHeunSolver(_FastSolver):
        """Optimized Adaptive Heun solver with JIT compilation.
        
        Options:
            correct_fsal: bool (default True)
                - True: Compute k1 fresh each step (2nd order accurate)
                - False: Reuse k2 as k1 (faster, 1st order)
            
            use_compile: bool (default False)
                - True: Use torch.compile for additional speedup (PyTorch 2.0+)
                - False: Use JIT-compiled functions only
            
            compile_mode: str (default 'reduce-overhead')
                - 'default': Balanced compilation
                - 'reduce-overhead': Best for GPU
                - 'max-autotune': Slower compile, faster run
        
        Statistics via solver.get_statistics():
            - n_steps, n_accepted, n_rejected, n_func_evals
            - acceptance_rate, evals_per_step
            - jit_enabled, compile_enabled
        """
        pass  # Inherits everything from _FastSolver
        
else:
    # Fallback to original RK-based implementation
    class AdaptiveHeunSolver(RKAdaptiveStepsizeODESolver):
        """Standard Adaptive Heun solver (fallback).
        
        Using generic RK implementation because adaptive_heun_fast.py
        was not found. This has the original FSAL bug.
        """
        order = 2
        tableau = _ADAPTIVE_HEUN_TABLEAU
        mid = _AH_C_MID
