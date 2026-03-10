"""
Generate real benchmark data for heun_paper.tex Tables 1 and 3.

Table 1: Convergence order (fixed-step Heun, original FSAL vs corrected)
Table 3: JIT timing (adaptive solver on stiff-like ODE system)
"""

import torch
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from inverter_control.heun.adaptive_heun_fast import AdaptiveHeunSolver


# =============================================================================
# Table 1: Convergence Order Verification
# =============================================================================

def fixed_step_heun_correct(func, y0, t0, t_end, h):
    """Fixed-step Heun (correct: fresh k1 each step)."""
    y = y0.clone()
    t = t0
    n_steps = 0
    while t < t_end - 1e-14:
        h_actual = min(h, t_end - t)
        k1 = func(t, y)
        y_euler = y + h_actual * k1
        k2 = func(t + h_actual, y_euler)
        y = y + 0.5 * h_actual * (k1 + k2)
        t += h_actual
        n_steps += 1
    return y, n_steps


def fixed_step_heun_fsal_bug(func, y0, t0, t_end, h):
    """Fixed-step Heun with FSAL bug (reuse k2 as k1)."""
    y = y0.clone()
    t = t0
    k1 = func(t, y)  # Initial k1
    n_steps = 0
    while t < t_end - 1e-14:
        h_actual = min(h, t_end - t)
        y_euler = y + h_actual * k1
        k2 = func(t + h_actual, y_euler)
        y = y + 0.5 * h_actual * (k1 + k2)
        t += h_actual
        k1 = k2  # FSAL bug: reuse k2 (evaluated at y_euler, not y_new)
        n_steps += 1
    return y, n_steps


def test_convergence_order():
    """Run convergence order test for Table 1."""
    print("=" * 70)
    print("TABLE 1: Convergence Order Verification")
    print("  ODE: dy/dt = -0.5*y, y(0)=1, t in [0,4]")
    print("  Exact: y(4) = exp(-2)")
    print("=" * 70)

    def simple_ode(t, y):
        return -0.5 * y

    y0 = torch.tensor([1.0], dtype=torch.float64)
    t0, t_end = 0.0, 4.0
    exact = torch.tensor([np.exp(-2.0)], dtype=torch.float64)

    step_sizes = [0.4, 0.2, 0.1, 0.05, 0.025]

    print(f"\n{'h':>8s}  {'Original (FSAL bug)':>20s}  {'Corrected':>20s}  {'Orig ratio':>11s}  {'Corr ratio':>11s}")
    print("-" * 75)

    prev_err_orig = None
    prev_err_corr = None

    results = []
    for h in step_sizes:
        y_orig, _ = fixed_step_heun_fsal_bug(simple_ode, y0, t0, t_end, h)
        y_corr, _ = fixed_step_heun_correct(simple_ode, y0, t0, t_end, h)

        err_orig = (y_orig - exact).abs().item()
        err_corr = (y_corr - exact).abs().item()

        ratio_orig = f"{prev_err_orig / err_orig:.2f}" if prev_err_orig else "--"
        ratio_corr = f"{prev_err_corr / err_corr:.2f}" if prev_err_corr else "--"

        print(f"{h:8.3f}  {err_orig:20.6e}  {err_corr:20.6e}  {ratio_orig:>11s}  {ratio_corr:>11s}")
        results.append((h, err_orig, err_corr))

        prev_err_orig = err_orig
        prev_err_corr = err_corr

    print()
    print("Expected ratios: ~2.0 for 1st order, ~4.0 for 2nd order")

    # Compute average convergence orders
    orig_orders = []
    corr_orders = []
    for i in range(1, len(results)):
        h_ratio = results[i-1][0] / results[i][0]
        if results[i][1] > 0 and results[i-1][1] > 0:
            orig_orders.append(np.log(results[i-1][1] / results[i][1]) / np.log(h_ratio))
        if results[i][2] > 0 and results[i-1][2] > 0:
            corr_orders.append(np.log(results[i-1][2] / results[i][2]) / np.log(h_ratio))

    print(f"\nAverage convergence order (original): {np.mean(orig_orders):.2f}")
    print(f"Average convergence order (corrected): {np.mean(corr_orders):.2f}")

    # LaTeX table output
    print("\n--- LaTeX for Table 1 ---")
    for h, err_orig, err_corr in results:
        # Format in scientific notation for LaTeX
        def fmt(x):
            exp = int(np.floor(np.log10(abs(x))))
            mantissa = x / 10**exp
            return f"${mantissa:.1f} \\times 10^{{{exp}}}$"
        print(f"{h:.3f} & {fmt(err_orig)} & {fmt(err_corr)} \\\\")

    return results


# =============================================================================
# Table 3: JIT Timing Benchmark
# =============================================================================

def test_jit_timing():
    """Run JIT timing benchmark for Table 3."""
    print("\n" + "=" * 70)
    print("TABLE 3: Wall-clock Time per 100ms Integration")
    print("=" * 70)

    # Stiff-ish ODE system mimicking PLL + power sharing dynamics
    # 18 states (like a 3-converter system)
    dim = 18

    # Build a system matrix with fast and slow modes
    # Fast modes: eigenvalues near -450 +/- 780j (PLL-like)
    # Slow modes: eigenvalues near -1 to -10 (power sharing)
    torch.manual_seed(42)

    # Construct A matrix with controlled eigenvalues
    A = torch.zeros(dim, dim, dtype=torch.float64)
    # Fast subsystem (6 states, 3 PLLs with Kp=60, Ki=900)
    for i in range(3):
        idx = 2*i
        A[idx, idx] = -30.0      # damping
        A[idx, idx+1] = 780.0    # oscillatory
        A[idx+1, idx] = -780.0
        A[idx+1, idx+1] = -30.0
    # Medium subsystem (6 states, filter dynamics ~100 rad/s)
    for i in range(3):
        idx = 6 + 2*i
        A[idx, idx] = -50.0
        A[idx, idx+1] = 100.0
        A[idx+1, idx] = -100.0
        A[idx+1, idx+1] = -50.0
    # Slow subsystem (6 states, power sharing ~1-10 rad/s)
    for i in range(6):
        idx = 12 + i
        A[idx, idx] = -(1.0 + i * 2.0)
    # Cross-coupling
    A[0, 12] = 0.5
    A[12, 0] = -0.3
    A[6, 13] = 0.4
    A[13, 6] = -0.2

    def stiff_ode(t, y):
        return y @ A.T

    y0 = torch.randn(dim, dtype=torch.float64)
    y0 = y0 / y0.norm() * 0.1  # Small initial condition for stability

    t_span = torch.tensor([0.0, 0.1], dtype=torch.float64)  # 100ms
    t_eval = torch.linspace(0.0, 0.1, 50, dtype=torch.float64)

    n_warmup = 10
    n_runs = 100

    # --- Method 1: Pure Python (no JIT) ---
    # We simulate this by reimplementing without JIT functions
    def run_pure_python():
        """Manual Heun integration without JIT."""
        y = y0.clone()
        t = 0.0
        dt = 1e-4
        rtol, atol = 1e-6, 1e-9
        safety = 0.9
        results = [y0.clone()]

        while t < 0.1 - 1e-14:
            dt_actual = min(dt, 0.1 - t)
            k1 = stiff_ode(t, y)
            y_euler = y + dt_actual * k1
            k2 = stiff_ode(t + dt_actual, y_euler)

            y1 = y + 0.5 * dt_actual * (k1 + k2)
            error = 0.5 * dt_actual * (k2 - k1)

            tol = atol + rtol * torch.maximum(y.abs(), y1.abs())
            err_ratio = (error / tol).square().mean().sqrt().item()

            if err_ratio <= 1.0 or dt_actual <= 1e-8:
                y = y1
                t += dt_actual

            # Step size update
            if err_ratio > 0:
                factor = safety * (err_ratio + 1e-10) ** (-0.5)
            else:
                factor = 10.0
            factor = max(0.2, min(factor, 10.0))
            if err_ratio <= 1:
                factor = max(1.0, factor)
            else:
                factor = min(1.0, factor)
            dt = dt_actual * factor
            dt = max(1e-10, min(dt, 0.01))

        return y

    # --- Method 2: JIT-compiled (default) ---
    def run_jit():
        solver = AdaptiveHeunSolver(
            stiff_ode, y0, rtol=1e-6, atol=1e-9, correct_fsal=True
        )
        return solver.integrate(t_eval)

    # --- Method 3: JIT + torch.compile ---
    def run_compiled():
        solver = AdaptiveHeunSolver(
            stiff_ode, y0, rtol=1e-6, atol=1e-9, correct_fsal=True,
            use_compile=True, compile_mode='reduce-overhead'
        )
        return solver.integrate(t_eval)

    configs = [
        ("Pure Python loop", run_pure_python),
        ("torch.jit.script", run_jit),
    ]

    # Check if torch.compile is available
    try:
        test_solver = AdaptiveHeunSolver(
            stiff_ode, y0, rtol=1e-6, atol=1e-9, use_compile=True
        )
        _ = test_solver.integrate(t_eval)
        configs.append(("+ torch.compile", run_compiled))
    except Exception as e:
        print(f"  torch.compile not available: {e}")

    print(f"\nSystem: {dim}-state stiff ODE (PLL-like, stiffness ratio ~1000)")
    print(f"Integration: 0 to 100ms, rtol=1e-6, atol=1e-9")
    print(f"Warmup: {n_warmup}, Runs: {n_runs}")
    print()

    results = {}
    for name, fn in configs:
        # Warmup
        for _ in range(n_warmup):
            fn()

        # Timed runs
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            fn()
            times.append((time.perf_counter() - start) * 1000)  # ms

        mean_ms = np.mean(times)
        std_ms = np.std(times)
        results[name] = mean_ms
        print(f"  {name:<25s}  {mean_ms:7.2f} ms  (+/- {std_ms:.2f})")

    # Speedups
    baseline = results.get("Pure Python loop")
    if baseline:
        print()
        for name, ms in results.items():
            speedup = baseline / ms
            print(f"  {name:<25s}  speedup: {speedup:.1f}x")

    # LaTeX output
    print("\n--- LaTeX for Table 3 ---")
    for name, ms in results.items():
        speedup = baseline / ms if baseline else 1.0
        print(f"{name} & {ms:.1f} & {speedup:.1f}$\\times$ \\\\")

    return results


# =============================================================================
# Additional: Adaptive step count comparison
# =============================================================================

def test_adaptive_step_count():
    """Compare step counts: original FSAL vs corrected."""
    print("\n" + "=" * 70)
    print("ADAPTIVE STEP COUNT COMPARISON (stiff system)")
    print("=" * 70)

    dim = 18
    A = torch.zeros(dim, dim, dtype=torch.float64)
    for i in range(3):
        idx = 2*i
        A[idx, idx] = -30.0
        A[idx, idx+1] = 780.0
        A[idx+1, idx] = -780.0
        A[idx+1, idx+1] = -30.0
    for i in range(3):
        idx = 6 + 2*i
        A[idx, idx] = -50.0
        A[idx, idx+1] = 100.0
        A[idx+1, idx] = -100.0
        A[idx+1, idx+1] = -50.0
    for i in range(6):
        idx = 12 + i
        A[idx, idx] = -(1.0 + i * 2.0)
    A[0, 12] = 0.5
    A[12, 0] = -0.3

    def stiff_ode(t, y):
        return y @ A.T

    y0 = torch.randn(dim, dtype=torch.float64)
    y0 = y0 / y0.norm() * 0.1

    t_eval = torch.linspace(0.0, 0.1, 50, dtype=torch.float64)

    for rtol in [1e-4, 1e-6, 1e-8]:
        # Corrected
        solver_corr = AdaptiveHeunSolver(
            stiff_ode, y0, rtol=rtol, atol=rtol*1e-3, correct_fsal=True
        )
        sol_corr = solver_corr.integrate(t_eval)
        stats_corr = solver_corr.get_statistics()

        # Original (FSAL bug)
        solver_orig = AdaptiveHeunSolver(
            stiff_ode, y0, rtol=rtol, atol=rtol*1e-3, correct_fsal=False
        )
        sol_orig = solver_orig.integrate(t_eval)
        stats_orig = solver_orig.get_statistics()

        ratio = stats_orig['n_accepted'] / max(1, stats_corr['n_accepted'])
        print(f"  rtol={rtol:.0e}: Corrected {stats_corr['n_accepted']:4d} steps "
              f"({stats_corr['acceptance_rate']:.1%} accept), "
              f"Original {stats_orig['n_accepted']:4d} steps "
              f"({stats_orig['acceptance_rate']:.1%} accept), "
              f"ratio {ratio:.1f}x")


if __name__ == '__main__':
    results_order = test_convergence_order()
    results_timing = test_jit_timing()
    test_adaptive_step_count()
    print("\n" + "=" * 70)
    print("BENCHMARKS COMPLETE")
    print("=" * 70)
