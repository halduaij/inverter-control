"""
Optimized Adaptive Heun Solver with JIT Compilation Support

Performance optimizations:
1. Specialized Heun step (no k tensor allocation)
2. torch.no_grad() for error/accept logic
3. Branchless step size control (no CPU-GPU sync)
4. JIT-compilable core functions
5. Optional torch.compile for full solver (PyTorch 2.0+)
6. Horner's rule interpolation

JIT Compilation:
- Core step functions are torch.jit.script compatible
- Full solver can be wrapped with torch.compile
- Use compile_mode='reduce-overhead' for best results on GPU
"""

import torch
import bisect
from typing import Tuple, Optional, Callable, List, Union
import collections
import warnings


# =============================================================================
# Butcher Tableau (for API compatibility)
# =============================================================================

_ButcherTableau = collections.namedtuple('_ButcherTableau', 'alpha, beta, c_sol, c_error')

_ADAPTIVE_HEUN_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1.], dtype=torch.float64),
    beta=[torch.tensor([1.], dtype=torch.float64)],
    c_sol=torch.tensor([0.5, 0.5], dtype=torch.float64),
    c_error=torch.tensor([0.5, -0.5], dtype=torch.float64),
)

_AH_C_MID = torch.tensor([0.5, 0.], dtype=torch.float64)


# =============================================================================
# JIT-Compilable Core Functions
# =============================================================================

@torch.jit.script
def _heun_step_jit(
    y0: torch.Tensor,
    k1: torch.Tensor,
    t1: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """JIT-compiled Heun step core (after k1 and k2 are computed).
    
    Note: func calls are outside this function since torch.jit.script
    cannot handle arbitrary Python callables.
    
    Returns: (y1, error, y_mid)
    """
    dt_y = dt.to(y0.dtype)
    half_dt = 0.5 * dt_y
    
    # Euler prediction point (for computing k2 outside)
    y_euler = y0 + dt_y * k1
    
    return y_euler, half_dt, dt_y


@torch.jit.script
def _heun_finalize_jit(
    y0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    half_dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """JIT-compiled Heun finalization.
    
    Returns: (y1, error, y_mid)
    """
    y1 = y0 + half_dt * (k1 + k2)
    error = half_dt * (k2 - k1)
    y_mid = y0 + half_dt * k1
    return y1, error, y_mid


@torch.jit.script
def _compute_error_ratio_jit(
    error: torch.Tensor,
    y0: torch.Tensor,
    y1: torch.Tensor,
    rtol: torch.Tensor,
    atol: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled error ratio computation."""
    tol = atol + rtol * torch.maximum(y0.abs(), y1.abs())
    return (error / tol).square().mean().sqrt()


@torch.jit.script
def _optimal_step_size_jit(
    dt: torch.Tensor,
    error_ratio: torch.Tensor,
    safety: torch.Tensor,
    ifactor: torch.Tensor,
    dfactor: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled branchless step size update.
    
    Order 2 hardcoded (exponent = 0.5).
    """
    # Base factor
    base_factor = safety * torch.pow(error_ratio + 1e-10, -0.5)
    
    # Handle edge cases with torch.where (branchless)
    factor = torch.where(error_ratio == 0, ifactor, base_factor)
    factor = torch.where(torch.isfinite(error_ratio), factor, dfactor)
    
    # Clamp based on acceptance
    one = torch.ones_like(factor)
    factor_accepted = torch.clamp(factor, min=one, max=ifactor)
    factor_rejected = torch.clamp(factor, min=dfactor, max=one)
    factor = torch.where(error_ratio <= 1, factor_accepted, factor_rejected)
    
    return dt * factor


@torch.jit.script
def _sanitize_dt_jit(dt: torch.Tensor, min_step: torch.Tensor) -> torch.Tensor:
    """JIT-compiled dt sanitization."""
    return torch.where(torch.isfinite(dt), dt, min_step)


@torch.jit.script
def _interp_evaluate_horner_jit(
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c3: torch.Tensor,
    c4: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled Horner interpolation.
    
    Note: Takes coefficients as separate args since TorchScript
    has issues with List[Tensor] in some contexts.
    """
    x = ((t - t0) / (t1 - t0)).to(c0.dtype)
    
    r = c4
    r = torch.addcmul(c3, r, x)
    r = torch.addcmul(c2, r, x)
    r = torch.addcmul(c1, r, x)
    r = torch.addcmul(c0, r, x)
    return r


@torch.jit.script
def _interp_fit_jit(
    y0: torch.Tensor,
    y1: torch.Tensor,
    y_mid: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """JIT-compiled interpolation fit.
    
    Returns coefficients as tuple (c0, c1, c2, c3, c4).
    """
    dt = dt.to(y0.dtype)
    
    c4 = 2*dt*(k2 - k1) - 8*(y1 + y0) + 16*y_mid
    c3 = dt*(5*k1 - 3*k2) + 18*y0 + 14*y1 - 32*y_mid
    c2 = dt*(k2 - 4*k1) - 11*y0 - 5*y1 + 16*y_mid
    c1 = dt * k1
    c0 = y0
    
    return c0, c1, c2, c3, c4


@torch.jit.script
def _accept_step_jit(
    error_ratio: torch.Tensor,
    dt: torch.Tensor,
    min_step: torch.Tensor,
    max_step: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled accept decision."""
    accept_by_error = error_ratio <= 1
    accept_by_min = dt <= min_step
    reject_by_max = dt > max_step
    return (accept_by_error | accept_by_min) & ~reject_by_max


# =============================================================================
# Combined JIT Step (for torch.compile)
# =============================================================================

@torch.jit.script
def _adaptive_heun_step_core_jit(
    y0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    dt: torch.Tensor,
    rtol: torch.Tensor,
    atol: torch.Tensor,
    safety: torch.Tensor,
    ifactor: torch.Tensor,
    dfactor: torch.Tensor,
    min_step: torch.Tensor,
    max_step: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """JIT-compiled core of adaptive Heun step.
    
    Everything except func() calls, which must be outside.
    
    Returns: (y1, error, y_mid, error_ratio, dt_next, accept, half_dt)
    """
    half_dt = 0.5 * dt.to(y0.dtype)
    
    # Heun solution
    y1 = y0 + half_dt * (k1 + k2)
    error = half_dt * (k2 - k1)
    y_mid = y0 + half_dt * k1
    
    # Error ratio
    tol = atol + rtol * torch.maximum(y0.abs(), y1.abs())
    error_ratio = (error / tol).square().mean().sqrt()
    
    # Step size update (branchless)
    base_factor = safety * torch.pow(error_ratio + 1e-10, -0.5)
    factor = torch.where(error_ratio == 0, ifactor, base_factor)
    factor = torch.where(torch.isfinite(error_ratio), factor, dfactor)
    
    one = torch.ones_like(factor)
    factor_accepted = torch.clamp(factor, min=one, max=ifactor)
    factor_rejected = torch.clamp(factor, min=dfactor, max=one)
    factor = torch.where(error_ratio <= 1, factor_accepted, factor_rejected)
    
    dt_next = torch.clamp(dt * factor, min_step, max_step)
    
    # Accept decision
    accept = ((error_ratio <= 1) | (dt <= min_step)) & ~(dt > max_step)
    
    return y1, error, y_mid, error_ratio, dt_next, accept, half_dt


# =============================================================================
# Non-JIT Helper Functions (need Python features)
# =============================================================================

def _heun_step_correct(func, y0, t0, dt, t1):
    """Correct Heun step with JIT-compiled internals."""
    k1 = func(t0, y0)
    dt_y = dt.to(y0.dtype)
    y_euler = y0 + dt_y * k1
    k2 = func(t1, y_euler)
    
    half_dt = 0.5 * dt_y
    y1, error, y_mid = _heun_finalize_jit(y0, k1, k2, half_dt)
    
    return y1, k1, k2, error, y_mid


def _heun_step_fsal(func, y0, f0, t0, dt, t1):
    """FSAL Heun step with JIT-compiled internals."""
    k1 = f0
    dt_y = dt.to(y0.dtype)
    y_euler = y0 + dt_y * k1
    k2 = func(t1, y_euler)
    
    half_dt = 0.5 * dt_y
    y1, error, y_mid = _heun_finalize_jit(y0, k1, k2, half_dt)
    
    return y1, k1, k2, error, y_mid


def _select_initial_step(func, t0, y0, rtol, atol, f0=None):
    """Select initial step size."""
    if f0 is None:
        f0 = func(t0, y0)
    
    scale = atol + rtol * y0.abs()
    d0 = (y0 / scale).square().mean().sqrt()
    d1 = (f0 / scale).square().mean().sqrt()
    
    h0_default = torch.tensor(1e-6, dtype=y0.dtype, device=y0.device)
    h0_computed = 0.01 * d0 / (d1 + 1e-10)
    h0 = torch.where((d0 < 1e-5) | (d1 < 1e-5), h0_default, h0_computed)
    
    y1 = y0 + h0 * f0
    f1 = func(t0 + h0, y1)
    
    d2 = ((f1 - f0) / scale).square().mean().sqrt() / (h0 + 1e-10)
    
    max_d = torch.maximum(d1, d2)
    h1 = torch.where(
        max_d <= 1e-15,
        torch.maximum(h0_default, h0 * 1e-3),
        torch.pow(0.01 / (max_d + 1e-10), 1.0 / 3.0)
    )
    
    return torch.minimum(100 * h0, h1).abs(), f0


# =============================================================================
# Main Solver Class
# =============================================================================

class AdaptiveHeunSolver:
    """Optimized Adaptive Heun Solver with JIT support.
    
    Performance features:
    - JIT-compiled core functions (@torch.jit.script)
    - Optional torch.compile for full solver (PyTorch 2.0+)
    - Specialized Heun step (no k tensor)
    - torch.no_grad() for control logic
    - Branchless operations
    
    Args:
        func: ODE function f(t, y)
        y0: Initial condition
        correct_fsal: If True, compute k1 fresh (2nd order). If False, reuse (faster).
        use_compile: If True, use torch.compile on step function (PyTorch 2.0+)
        compile_mode: Mode for torch.compile ('default', 'reduce-overhead', 'max-autotune')
    """
    
    order = 2
    tableau = _ADAPTIVE_HEUN_TABLEAU
    mid = _AH_C_MID
    
    def __init__(
        self,
        func: Callable,
        y0: torch.Tensor,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        min_step: float = 0.0,
        max_step: float = float('inf'),
        first_step: Optional[float] = None,
        step_t: Optional[torch.Tensor] = None,
        jump_t: Optional[torch.Tensor] = None,
        safety: float = 0.9,
        ifactor: float = 10.0,
        dfactor: float = 0.2,
        max_num_steps: int = 2**31 - 1,
        dtype: torch.dtype = torch.float64,
        norm: Optional[Callable] = None,
        correct_fsal: bool = True,
        use_compile: bool = False,
        compile_mode: str = 'reduce-overhead',
        **kwargs,
    ):
        self._raw_func = func
        self.func = self._wrap_func(func)
        
        self.y0 = y0
        self.correct_fsal = correct_fsal
        
        dtype = torch.promote_types(dtype, y0.dtype)
        device = y0.device
        
        self.dtype = dtype
        self.device = device
        
        # Store as tensors on device
        self.rtol = torch.as_tensor(rtol, dtype=dtype, device=device)
        self.atol = torch.as_tensor(atol, dtype=dtype, device=device)
        self.min_step = torch.as_tensor(min_step, dtype=dtype, device=device)
        self.max_step = torch.as_tensor(max_step, dtype=dtype, device=device)
        self.safety = torch.as_tensor(safety, dtype=dtype, device=device)
        self.ifactor = torch.as_tensor(ifactor, dtype=dtype, device=device)
        self.dfactor = torch.as_tensor(dfactor, dtype=dtype, device=device)
        
        # Python int to avoid sync
        self.max_num_steps = int(max_num_steps)
        
        self.first_step = first_step
        self.step_t = step_t
        self.jump_t = jump_t
        self.norm = norm
        
        # torch.compile support (PyTorch 2.0+)
        self.use_compile = use_compile
        self._compiled_step = None
        if use_compile:
            self._setup_compile(compile_mode)
        
        # Statistics
        self._n_steps = 0
        self._n_accepted = 0
        self._n_rejected = 0
        self._n_func_evals = 0
    
    def _setup_compile(self, mode: str):
        """Setup torch.compile for the step function."""
        try:
            # Create a compiled version of the core step
            @torch.compile(mode=mode, fullgraph=False)
            def compiled_step_core(y0, k1, k2, dt, rtol, atol, safety, ifactor, dfactor, min_step, max_step):
                return _adaptive_heun_step_core_jit(
                    y0, k1, k2, dt, rtol, atol, safety, ifactor, dfactor, min_step, max_step
                )
            
            self._compiled_step = compiled_step_core
            
        except Exception as e:
            warnings.warn(f"torch.compile failed, using JIT-only mode: {e}")
            self.use_compile = False
    
    def _wrap_func(self, func):
        """Wrap function to track evaluations."""
        solver = self
        
        class FuncWrapper:
            def __init__(self, base_func):
                self.base_func = base_func
                for attr in ['callback_step', 'callback_accept_step', 'callback_reject_step']:
                    if hasattr(base_func, attr):
                        setattr(self, attr, getattr(base_func, attr))
                    else:
                        setattr(self, attr, lambda *a, **k: None)
            
            def __call__(self, t, y, perturb=None):
                solver._n_func_evals += 1
                if perturb == 'next':
                    t = torch.nextafter(t.float(), torch.tensor(float('inf'))).to(t.dtype)
                elif perturb == 'prev':
                    t = torch.nextafter(t.float(), torch.tensor(float('-inf'))).to(t.dtype)
                return self.base_func(t, y)
        
        return FuncWrapper(func)
    
    @classmethod
    def valid_callbacks(cls):
        return {'callback_step', 'callback_accept_step', 'callback_reject_step'}
    
    def get_statistics(self):
        """Return solver statistics."""
        return {
            'n_steps': self._n_steps,
            'n_accepted': self._n_accepted,
            'n_rejected': self._n_rejected,
            'n_func_evals': self._n_func_evals,
            'acceptance_rate': self._n_accepted / max(1, self._n_steps),
            'evals_per_step': self._n_func_evals / max(1, self._n_accepted),
            'jit_enabled': True,
            'compile_enabled': self.use_compile,
        }
    
    def integrate(self, t: torch.Tensor) -> torch.Tensor:
        """Integrate ODE over time points t."""
        t = t.to(dtype=self.dtype, device=self.device)
        n_times = len(t)
        
        # Pre-allocate output
        solution = torch.empty(n_times, *self.y0.shape, dtype=self.y0.dtype, device=self.device)
        solution[0] = self.y0
        
        # Setup discontinuities
        if self.step_t is not None:
            step_t = self.step_t[self.step_t >= t[0]]
            step_t = torch.sort(step_t).values.to(self.dtype)
        else:
            step_t = torch.tensor([], dtype=self.dtype, device=self.device)
        
        if self.jump_t is not None:
            jump_t = self.jump_t[self.jump_t >= t[0]]
            jump_t = torch.sort(jump_t).values.to(self.dtype)
        else:
            jump_t = torch.tensor([], dtype=self.dtype, device=self.device)
        
        next_step_idx = bisect.bisect(step_t.tolist(), t[0].item()) if len(step_t) > 0 else 0
        next_jump_idx = bisect.bisect(jump_t.tolist(), t[0].item()) if len(jump_t) > 0 else 0
        
        # Initialize
        y = self.y0
        t_cur = t[0]
        
        # Initial step size
        if self.first_step is not None:
            dt = torch.as_tensor(self.first_step, dtype=self.dtype, device=self.device)
            f_cached = self.func(t_cur, y)
        else:
            dt, f_cached = _select_initial_step(self.func, t_cur, y, self.rtol, self.atol)
        
        dt = _sanitize_dt_jit(dt, self.min_step)
        dt = torch.clamp(dt, self.min_step, self.max_step)
        
        # Interpolation state (store as tuple for JIT compatibility)
        t_prev = t_cur
        y_prev = y
        c0, c1, c2, c3, c4 = y, y, y, y, y  # Placeholder coefficients
        
        for i in range(1, n_times):
            target = t[i]
            
            while t_cur < target:
                if self._n_steps >= self.max_num_steps:
                    raise RuntimeError(f"max_num_steps ({self.max_num_steps}) exceeded")
                
                dt = _sanitize_dt_jit(dt, self.min_step)
                dt = torch.clamp(dt, self.min_step, self.max_step)
                
                t1 = t_cur + dt
                
                # Handle discontinuities
                on_step_t = False
                on_jump_t = False
                
                if len(step_t) > 0 and next_step_idx < len(step_t):
                    next_step = step_t[next_step_idx]
                    if t_cur < next_step < t1:
                        on_step_t = True
                        t1 = next_step
                        dt = t1 - t_cur
                
                if len(jump_t) > 0 and next_jump_idx < len(jump_t):
                    next_jump = jump_t[next_jump_idx]
                    if t_cur < next_jump < t1:
                        on_jump_t = True
                        on_step_t = False
                        t1 = next_jump
                        dt = t1 - t_cur
                
                # Callback
                self.func.callback_step(t_cur, y, dt)
                
                # === HEUN STEP (compute k1, k2) ===
                if self.correct_fsal:
                    k1 = self.func(t_cur, y)
                else:
                    k1 = f_cached
                
                dt_y = dt.to(y.dtype)
                y_euler = y + dt_y * k1
                k2 = self.func(t1, y_euler)
                
                # === CORE STEP (JIT or compiled) ===
                with torch.no_grad():
                    if self.use_compile and self._compiled_step is not None:
                        y1, error, y_mid, err_ratio, dt_next, accept, _ = self._compiled_step(
                            y, k1, k2, dt, self.rtol, self.atol,
                            self.safety, self.ifactor, self.dfactor,
                            self.min_step, self.max_step
                        )
                    else:
                        y1, error, y_mid, err_ratio, dt_next, accept, _ = _adaptive_heun_step_core_jit(
                            y, k1, k2, dt, self.rtol, self.atol,
                            self.safety, self.ifactor, self.dfactor,
                            self.min_step, self.max_step
                        )
                    
                    # Custom norm override
                    if self.norm is not None:
                        tol = self.atol + self.rtol * torch.maximum(y.abs(), y1.abs())
                        err_ratio = self.norm(error / tol)
                        accept = _accept_step_jit(err_ratio, dt, self.min_step, self.max_step)
                        dt_next = _optimal_step_size_jit(
                            dt, err_ratio, self.safety, self.ifactor, self.dfactor
                        )
                        dt_next = torch.clamp(dt_next, self.min_step, self.max_step)
                
                # === RECOMPUTE y1 WITH GRAD (outside no_grad) ===
                # We need y1 to be differentiable
                half_dt = 0.5 * dt_y
                y1_grad = y + half_dt * (k1 + k2)
                y_mid_grad = y + half_dt * k1
                
                self._n_steps += 1
                
                if accept.item():
                    self._n_accepted += 1
                    self.func.callback_accept_step(t_cur, y, dt)
                    
                    # Save for interpolation
                    t_prev = t_cur
                    y_prev = y
                    c0, c1, c2, c3, c4 = _interp_fit_jit(y, y1_grad, y_mid_grad, k1, k2, dt)
                    
                    # Update state (differentiable)
                    y = y1_grad
                    t_cur = t1
                    f_cached = k2
                    
                    if on_jump_t:
                        f_cached = self.func(t_cur, y, perturb='next')
                        if next_jump_idx < len(jump_t) - 1:
                            next_jump_idx += 1
                    if on_step_t:
                        if next_step_idx < len(step_t) - 1:
                            next_step_idx += 1
                else:
                    self._n_rejected += 1
                    self.func.callback_reject_step(t_cur, y, dt)
                
                dt = dt_next
            
            # Store solution
            if t_cur == target:
                solution[i] = y
            else:
                solution[i] = _interp_evaluate_horner_jit(c0, c1, c2, c3, c4, t_prev, t_cur, target)
        
        return solution
    
    def integrate_until_event(self, t0: torch.Tensor, event_fn: Callable):
        """Integrate until event function crosses zero."""
        t0 = t0.to(self.device, self.dtype)
        
        y = self.y0
        t_cur = t0
        
        if self.first_step is not None:
            dt = torch.as_tensor(self.first_step, dtype=self.dtype, device=self.device)
            f_cached = self.func(t_cur, y)
        else:
            dt, f_cached = _select_initial_step(self.func, t_cur, y, self.rtol, self.atol)
        
        dt = _sanitize_dt_jit(dt, self.min_step)
        dt = torch.clamp(dt, self.min_step, self.max_step)
        
        sign0 = torch.sign(event_fn(t_cur, y))
        
        t_prev = t_cur
        y_prev = y
        c0, c1, c2, c3, c4 = y, y, y, y, y
        
        while True:
            if self._n_steps >= self.max_num_steps:
                raise RuntimeError("max_num_steps exceeded in event detection")
            
            dt = _sanitize_dt_jit(dt, self.min_step)
            dt = torch.clamp(dt, self.min_step, self.max_step)
            t1 = t_cur + dt
            
            # Compute k1, k2
            if self.correct_fsal:
                k1 = self.func(t_cur, y)
            else:
                k1 = f_cached
            
            dt_y = dt.to(y.dtype)
            y_euler = y + dt_y * k1
            k2 = self.func(t1, y_euler)
            
            # Core step
            with torch.no_grad():
                if self.use_compile and self._compiled_step is not None:
                    _, error, _, err_ratio, dt_next, accept, _ = self._compiled_step(
                        y, k1, k2, dt, self.rtol, self.atol,
                        self.safety, self.ifactor, self.dfactor,
                        self.min_step, self.max_step
                    )
                else:
                    _, error, _, err_ratio, dt_next, accept, _ = _adaptive_heun_step_core_jit(
                        y, k1, k2, dt, self.rtol, self.atol,
                        self.safety, self.ifactor, self.dfactor,
                        self.min_step, self.max_step
                    )
            
            # Recompute with gradients
            half_dt = 0.5 * dt_y
            y1 = y + half_dt * (k1 + k2)
            y_mid = y + half_dt * k1
            
            self._n_steps += 1
            
            if accept.item():
                self._n_accepted += 1
                t_prev, y_prev = t_cur, y
                c0, c1, c2, c3, c4 = _interp_fit_jit(y, y1, y_mid, k1, k2, dt)
                
                y, t_cur = y1, t1
                f_cached = k2
                
                sign1 = torch.sign(event_fn(t_cur, y))
                if sign1 != sign0:
                    t_lo, t_hi = t_prev, t_cur
                    for _ in range(50):
                        t_mid = 0.5 * (t_lo + t_hi)
                        y_mid_interp = _interp_evaluate_horner_jit(c0, c1, c2, c3, c4, t_prev, t_cur, t_mid)
                        if torch.sign(event_fn(t_mid, y_mid_interp)) == sign0:
                            t_lo = t_mid
                        else:
                            t_hi = t_mid
                        if (t_hi - t_lo).abs() < self.atol:
                            break
                    
                    event_t = 0.5 * (t_lo + t_hi)
                    event_y = _interp_evaluate_horner_jit(c0, c1, c2, c3, c4, t_prev, t_cur, event_t)
                    return event_t, torch.stack([self.y0, event_y])
            else:
                self._n_rejected += 1
            
            dt = dt_next
