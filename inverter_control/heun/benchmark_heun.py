"""
BENCHMARK: Original vs Optimized AdaptiveHeunSolver

Run this in your environment with torchdiffeq installed.
Place adaptive_heun_fast.py in the same directory.

Usage:
    python benchmark_heun.py
"""

import torch
import time
import sys
import os

# Try to import torchdiffeq
try:
    from torchdiffeq import odeint as original_odeint
    HAS_ORIGINAL = True
except ImportError:
    HAS_ORIGINAL = False
    print("WARNING: torchdiffeq not installed, can't compare to original")

# Import optimized solver (assumes adaptive_heun_fast.py is in same directory)
try:
    from adaptive_heun_fast import AdaptiveHeunSolver as OptimizedSolver
    HAS_OPTIMIZED = True
except ImportError:
    HAS_OPTIMIZED = False
    print("ERROR: adaptive_heun_fast.py not found")
    sys.exit(1)


# =============================================================================
# Test Functions
# =============================================================================

def simple_ode(t, y):
    """dy/dt = -0.5 * y"""
    return -0.5 * y


def spiral_ode(t, y):
    """2D spiral"""
    A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]], dtype=y.dtype, device=y.device)
    return y @ A.T


class NeuralODE(torch.nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, dim)
        )
    
    def forward(self, t, y):
        return self.net(y)


# =============================================================================
# Benchmark
# =============================================================================

def benchmark(fn, n_warmup=5, n_runs=50):
    """Benchmark a function."""
    # Warmup
    for _ in range(n_warmup):
        fn()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return sum(times) / len(times) * 1000  # ms


def run_original(func, y0, t):
    return original_odeint(func, y0, t, method='adaptive_heun')


def run_optimized(func, y0, t):
    solver = OptimizedSolver(func, y0)
    return solver.integrate(t)


def run_optimized_fsal_off(func, y0, t):
    solver = OptimizedSolver(func, y0, correct_fsal=False)
    return solver.integrate(t)


def main():
    print("=" * 70)
    print("BENCHMARK: Original torchdiffeq vs Optimized AdaptiveHeunSolver")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64
    
    print(f"\nDevice: {device}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    configs = [
        ("Simple ODE (dim=10, 50 pts)", simple_ode, 10, 50),
        ("Simple ODE (dim=100, 50 pts)", simple_ode, 100, 50),
        ("Simple ODE (dim=500, 50 pts)", simple_ode, 500, 50),
        ("Spiral ODE (dim=2, 100 pts)", spiral_ode, 2, 100),
        ("Neural ODE (dim=32, 50 pts)", NeuralODE(32), 32, 50),
        ("Neural ODE (dim=128, 50 pts)", NeuralODE(128), 128, 50),
    ]
    
    print("-" * 70)
    header = f"{'Test':<35} {'Original':>10} {'Optimized':>10} {'Speedup':>8}"
    print(header)
    print("-" * 70)
    
    for name, func, dim, n_pts in configs:
        if isinstance(func, torch.nn.Module):
            func = func.to(device, dtype)
            y0 = torch.randn(dim, device=device, dtype=dtype)
        else:
            y0 = torch.randn(dim, device=device, dtype=dtype)
        
        t = torch.linspace(0, 1, n_pts, device=device, dtype=dtype)
        
        # Original
        if HAS_ORIGINAL:
            try:
                orig_ms = benchmark(lambda: run_original(func, y0, t), n_warmup=3, n_runs=30)
            except Exception as e:
                orig_ms = None
                print(f"  Original failed: {e}")
        else:
            orig_ms = None
        
        # Optimized
        try:
            opt_ms = benchmark(lambda: run_optimized(func, y0, t), n_warmup=3, n_runs=30)
        except Exception as e:
            opt_ms = None
            print(f"  Optimized failed: {e}")
        
        # Print results
        orig_str = f"{orig_ms:.2f}ms" if orig_ms else "N/A"
        opt_str = f"{opt_ms:.2f}ms" if opt_ms else "N/A"
        
        if orig_ms and opt_ms:
            speedup = orig_ms / opt_ms
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        
        print(f"{name:<35} {orig_str:>10} {opt_str:>10} {speedup_str:>8}")
    
    print("-" * 70)
    
    # Gradient test
    print("\n" + "=" * 70)
    print("GRADIENT FLOW TEST")
    print("=" * 70)
    
    func = NeuralODE(16).to(device, dtype)
    y0 = torch.randn(16, device=device, dtype=dtype, requires_grad=True)
    t = torch.linspace(0, 1, 20, device=device, dtype=dtype)
    
    solver = OptimizedSolver(func, y0)
    sol = solver.integrate(t)
    loss = sol[-1].sum()
    loss.backward()
    
    print(f"\ny0.grad exists: {y0.grad is not None}")
    print(f"All param grads exist: {all(p.grad is not None for p in func.parameters())}")
    
    stats = solver.get_statistics()
    print(f"\nSolver stats:")
    print(f"  Accepted: {stats['n_accepted']}, Rejected: {stats['n_rejected']}")
    print(f"  Func evals: {stats['n_func_evals']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.1%}")
    print(f"  Evals per step: {stats['evals_per_step']:.2f}")


if __name__ == '__main__':
    main()
