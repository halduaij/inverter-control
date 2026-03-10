"""
Heun FSAL benchmark v2: test multiple ODEs (linear + nonlinear + stiff)
to verify convergence order behavior.
"""
import torch
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from inverter_control.heun.adaptive_heun_fast import AdaptiveHeunSolver


def fixed_step_heun_correct(func, y0, t0, t_end, h):
    y = y0.clone()
    t = t0
    while t < t_end - 1e-14:
        h_actual = min(h, t_end - t)
        k1 = func(t, y)
        k2 = func(t + h_actual, y + h_actual * k1)
        y = y + 0.5 * h_actual * (k1 + k2)
        t += h_actual
    return y


def fixed_step_heun_fsal_bug(func, y0, t0, t_end, h):
    y = y0.clone()
    t = t0
    k1 = func(t, y)
    while t < t_end - 1e-14:
        h_actual = min(h, t_end - t)
        y_euler = y + h_actual * k1
        k2 = func(t + h_actual, y_euler)
        y = y + 0.5 * h_actual * (k1 + k2)
        t += h_actual
        k1 = k2  # FSAL bug
    return y


def convergence_test(name, func, y0, t_end, exact_fn, step_sizes):
    print(f"\n--- {name} ---")
    exact = exact_fn(t_end)

    print(f"{'h':>8s}  {'FSAL-bug err':>14s}  {'Corrected err':>14s}  {'Bug ratio':>10s}  {'Corr ratio':>10s}")

    prev_orig, prev_corr = None, None
    for h in step_sizes:
        y_orig = fixed_step_heun_fsal_bug(func, y0, 0.0, t_end, h)
        y_corr = fixed_step_heun_correct(func, y0, 0.0, t_end, h)
        err_orig = (y_orig - exact).norm().item()
        err_corr = (y_corr - exact).norm().item()

        r_orig = f"{prev_orig/err_orig:.2f}" if prev_orig else "--"
        r_corr = f"{prev_corr/err_corr:.2f}" if prev_corr else "--"
        print(f"{h:8.4f}  {err_orig:14.6e}  {err_corr:14.6e}  {r_orig:>10s}  {r_corr:>10s}")
        prev_orig, prev_corr = err_orig, err_corr


# Test 1: Linear ODE (y' = -0.5y)
def test_linear():
    func = lambda t, y: -0.5 * y
    y0 = torch.tensor([1.0], dtype=torch.float64)
    exact = lambda t: torch.tensor([np.exp(-0.5*t)], dtype=torch.float64)
    convergence_test("Linear: y'=-0.5y, y(0)=1", func, y0, 4.0, exact,
                     [0.4, 0.2, 0.1, 0.05, 0.025])


# Test 2: Nonlinear ODE (y' = -y^2, y(0)=1, exact: y=1/(1+t))
def test_nonlinear():
    func = lambda t, y: -y**2
    y0 = torch.tensor([1.0], dtype=torch.float64)
    exact = lambda t: torch.tensor([1.0/(1.0+t)], dtype=torch.float64)
    convergence_test("Nonlinear: y'=-y^2, y(0)=1", func, y0, 2.0, exact,
                     [0.2, 0.1, 0.05, 0.025, 0.0125])


# Test 3: Oscillatory (y'' + y = 0 as system)
def test_oscillatory():
    def func(t, y):
        return torch.stack([ y[1], -y[0] ])
    y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    exact = lambda t: torch.tensor([np.cos(t), -np.sin(t)], dtype=torch.float64)
    convergence_test("Oscillatory: y''+y=0", func, y0, 6.283, exact,
                     [0.2, 0.1, 0.05, 0.025, 0.0125])


# Test 4: Stiff-ish (Van der Pol mu=10)
def test_vdp():
    mu = 10.0
    def func(t, y):
        return torch.stack([ y[1], mu*(1-y[0]**2)*y[1] - y[0] ])
    y0 = torch.tensor([2.0, 0.0], dtype=torch.float64)
    # No exact solution, use very fine step as reference
    ref = fixed_step_heun_correct(func, y0, 0.0, 1.0, 1e-5)
    exact = lambda t: ref
    convergence_test("Van der Pol (mu=10), t=1", func, y0, 1.0, exact,
                     [0.01, 0.005, 0.0025, 0.00125, 0.000625])


# Test 5: Lotka-Volterra (nonlinear, oscillatory)
def test_lotka():
    def func(t, y):
        return torch.stack([ 1.5*y[0] - y[0]*y[1], -3.0*y[1] + y[0]*y[1] ])
    y0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    ref = fixed_step_heun_correct(func, y0, 0.0, 2.0, 1e-5)
    exact = lambda t: ref
    convergence_test("Lotka-Volterra, t=2", func, y0, 2.0, exact,
                     [0.05, 0.025, 0.0125, 0.00625, 0.003125])


# Adaptive step count comparison
def test_adaptive():
    print("\n\n=== ADAPTIVE STEP COUNT: corrected vs FSAL-bug ===")
    dim = 18
    A = torch.zeros(dim, dim, dtype=torch.float64)
    for i in range(3):
        idx = 2*i
        A[idx, idx] = -30.0; A[idx, idx+1] = 780.0
        A[idx+1, idx] = -780.0; A[idx+1, idx+1] = -30.0
    for i in range(3):
        idx = 6 + 2*i
        A[idx, idx] = -50.0; A[idx, idx+1] = 100.0
        A[idx+1, idx] = -100.0; A[idx+1, idx+1] = -50.0
    for i in range(6):
        A[12+i, 12+i] = -(1.0 + i*2.0)
    A[0,12]=0.5; A[12,0]=-0.3

    func = lambda t, y: y @ A.T
    torch.manual_seed(0)
    y0 = torch.randn(dim, dtype=torch.float64) * 0.1
    t_eval = torch.linspace(0, 0.1, 50, dtype=torch.float64)

    for rtol in [1e-3, 1e-5, 1e-7]:
        s_corr = AdaptiveHeunSolver(func, y0, rtol=rtol, atol=rtol*1e-3, correct_fsal=True)
        s_corr.integrate(t_eval)
        sc = s_corr.get_statistics()

        s_bug = AdaptiveHeunSolver(func, y0, rtol=rtol, atol=rtol*1e-3, correct_fsal=False)
        s_bug.integrate(t_eval)
        sb = s_bug.get_statistics()

        print(f"  rtol={rtol:.0e}: Corrected {sc['n_accepted']:5d} accepted, "
              f"{sc['n_func_evals']:5d} evals | "
              f"FSAL-bug {sb['n_accepted']:5d} accepted, "
              f"{sb['n_func_evals']:5d} evals | "
              f"step ratio {sb['n_accepted']/max(1,sc['n_accepted']):.2f}x, "
              f"eval ratio {sb['n_func_evals']/max(1,sc['n_func_evals']):.2f}x")


# Timing: corrected vs FSAL-bug adaptive solver
def test_timing():
    print("\n\n=== TIMING: corrected vs FSAL-bug (18-state stiff system, 100ms) ===")
    dim = 18
    A = torch.zeros(dim, dim, dtype=torch.float64)
    for i in range(3):
        idx = 2*i
        A[idx, idx] = -30.0; A[idx, idx+1] = 780.0
        A[idx+1, idx] = -780.0; A[idx+1, idx+1] = -30.0
    for i in range(3):
        idx = 6 + 2*i
        A[idx, idx] = -50.0; A[idx, idx+1] = 100.0
        A[idx+1, idx] = -100.0; A[idx+1, idx+1] = -50.0
    for i in range(6):
        A[12+i, 12+i] = -(1.0 + i*2.0)
    A[0,12]=0.5; A[12,0]=-0.3

    func = lambda t, y: y @ A.T
    torch.manual_seed(0)
    y0 = torch.randn(dim, dtype=torch.float64) * 0.1
    t_eval = torch.linspace(0, 0.1, 50, dtype=torch.float64)

    n_warmup, n_runs = 5, 50

    for label, fsal in [("Corrected (2 evals/step)", True), ("FSAL-bug (1 eval/step)", False)]:
        for _ in range(n_warmup):
            s = AdaptiveHeunSolver(func, y0, rtol=1e-6, atol=1e-9, correct_fsal=fsal)
            s.integrate(t_eval)

        times = []
        for _ in range(n_runs):
            s = AdaptiveHeunSolver(func, y0, rtol=1e-6, atol=1e-9, correct_fsal=fsal)
            t0 = time.perf_counter()
            s.integrate(t_eval)
            times.append((time.perf_counter() - t0)*1000)

        stats = s.get_statistics()
        mean_ms = np.mean(times)
        print(f"  {label:<35s}: {mean_ms:7.1f} ms, "
              f"{stats['n_accepted']} accepted, {stats['n_func_evals']} evals")


if __name__ == '__main__':
    print("=" * 70)
    print("CONVERGENCE ORDER TESTS (fixed step)")
    print("=" * 70)
    test_linear()
    test_nonlinear()
    test_oscillatory()
    test_vdp()
    test_lotka()
    test_adaptive()
    test_timing()
