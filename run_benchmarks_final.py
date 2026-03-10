#!/usr/bin/env python3
"""Final Heun FSAL benchmark - convergence order + timing."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np
import time
from inverter_control.heun.adaptive_heun_fast import AdaptiveHeunSolver


def fixed_heun_correct(func, y0, t_end, h):
    y, t = y0.clone(), 0.0
    while t < t_end - 1e-14:
        ha = min(h, t_end - t)
        k1 = func(t, y)
        k2 = func(t + ha, y + ha * k1)
        y = y + 0.5 * ha * (k1 + k2)
        t += ha
    return y


def fixed_heun_fsal(func, y0, t_end, h):
    y, t = y0.clone(), 0.0
    k1 = func(t, y)
    while t < t_end - 1e-14:
        ha = min(h, t_end - t)
        k2 = func(t + ha, y + ha * k1)
        y = y + 0.5 * ha * (k1 + k2)
        t += ha
        k1 = k2  # FSAL bug
    return y


print("=" * 70)
print("TABLE 1: Convergence Order (y'=-0.5y, y(0)=1, t=4)")
print("=" * 70)

func = lambda t, y: -0.5 * y
y0 = torch.tensor([1.0], dtype=torch.float64)
exact = np.exp(-2.0)
hs = [0.4, 0.2, 0.1, 0.05, 0.025]

print(f"{'h':>8s} | {'FSAL-bug':>14s} | {'Corrected':>14s} | {'Bug/Corr':>8s}")
print("-" * 55)

results = []
for h in hs:
    e_bug = abs(fixed_heun_fsal(func, y0, 4.0, h).item() - exact)
    e_cor = abs(fixed_heun_correct(func, y0, 4.0, h).item() - exact)
    ratio = e_bug / e_cor if e_cor > 0 else float('inf')
    print(f"{h:8.3f} | {e_bug:14.6e} | {e_cor:14.6e} | {ratio:8.1f}x")
    results.append((h, e_bug, e_cor))

# Compute convergence orders
print("\nConvergence orders (log ratio when h halves):")
for i in range(1, len(results)):
    r = results[i-1][0] / results[i][0]
    o_bug = np.log(results[i-1][1] / results[i][1]) / np.log(r)
    o_cor = np.log(results[i-1][2] / results[i][2]) / np.log(r)
    print(f"  h={results[i][0]:.3f}: FSAL-bug order={o_bug:.2f}, Corrected order={o_cor:.2f}")

# Nonlinear test: y' = -y^2
print("\n" + "=" * 70)
print("Nonlinear: y'=-y^2, y(0)=1, t=2 (exact=1/3)")
print("=" * 70)

func2 = lambda t, y: -y**2
y02 = torch.tensor([1.0], dtype=torch.float64)
exact2 = 1.0/3.0

print(f"{'h':>8s} | {'FSAL-bug':>14s} | {'Corrected':>14s} | {'Bug/Corr':>8s}")
print("-" * 55)

results2 = []
for h in [0.1, 0.05, 0.025, 0.0125, 0.00625]:
    e_bug = abs(fixed_heun_fsal(func2, y02, 2.0, h).item() - exact2)
    e_cor = abs(fixed_heun_correct(func2, y02, 2.0, h).item() - exact2)
    ratio = e_bug / e_cor if e_cor > 0 else float('inf')
    print(f"{h:8.4f} | {e_bug:14.6e} | {e_cor:14.6e} | {ratio:8.1f}x")
    results2.append((h, e_bug, e_cor))

print("\nConvergence orders:")
for i in range(1, len(results2)):
    r = results2[i-1][0] / results2[i][0]
    o_bug = np.log(results2[i-1][1] / results2[i][1]) / np.log(r)
    o_cor = np.log(results2[i-1][2] / results2[i][2]) / np.log(r)
    print(f"  h={results2[i][0]:.4f}: FSAL-bug order={o_bug:.2f}, Corrected order={o_cor:.2f}")

# Oscillatory: y'' + y = 0
print("\n" + "=" * 70)
print("Oscillatory: y''+y=0, y(0)=1, y'(0)=0, t=2pi")
print("=" * 70)

func3 = lambda t, y: torch.stack([y[1], -y[0]])
y03 = torch.tensor([1.0, 0.0], dtype=torch.float64)
exact3 = torch.tensor([1.0, 0.0], dtype=torch.float64)  # at t=2pi

print(f"{'h':>8s} | {'FSAL-bug':>14s} | {'Corrected':>14s} | {'Bug/Corr':>8s}")
print("-" * 55)

results3 = []
for h in [0.1, 0.05, 0.025, 0.0125, 0.00625]:
    e_bug = (fixed_heun_fsal(func3, y03, 2*np.pi, h) - exact3).norm().item()
    e_cor = (fixed_heun_correct(func3, y03, 2*np.pi, h) - exact3).norm().item()
    ratio = e_bug / e_cor if e_cor > 0 else float('inf')
    print(f"{h:8.4f} | {e_bug:14.6e} | {e_cor:14.6e} | {ratio:8.1f}x")
    results3.append((h, e_bug, e_cor))

print("\nConvergence orders:")
for i in range(1, len(results3)):
    r = results3[i-1][0] / results3[i][0]
    o_bug = np.log(results3[i-1][1] / results3[i][1]) / np.log(r)
    o_cor = np.log(results3[i-1][2] / results3[i][2]) / np.log(r)
    print(f"  h={results3[i][0]:.4f}: FSAL-bug order={o_bug:.2f}, Corrected order={o_cor:.2f}")


# Adaptive step count + timing
print("\n" + "=" * 70)
print("ADAPTIVE SOLVER: corrected vs FSAL-bug")
print("  18-state stiff system (PLL-like), 100ms integration")
print("=" * 70)

dim = 18
A = torch.zeros(dim, dim, dtype=torch.float64)
for i in range(3):
    j = 2*i
    A[j,j] = -30.; A[j,j+1] = 780.; A[j+1,j] = -780.; A[j+1,j+1] = -30.
for i in range(3):
    j = 6+2*i
    A[j,j] = -50.; A[j,j+1] = 100.; A[j+1,j] = -100.; A[j+1,j+1] = -50.
for i in range(6):
    A[12+i, 12+i] = -(1. + i*2.)
A[0,12]=0.5; A[12,0]=-0.3

stiff_func = lambda t, y: y @ A.T
torch.manual_seed(0)
stiff_y0 = torch.randn(dim, dtype=torch.float64) * 0.1
t_eval = torch.linspace(0, 0.1, 50, dtype=torch.float64)

for rtol in [1e-4, 1e-6, 1e-8]:
    sc = AdaptiveHeunSolver(stiff_func, stiff_y0, rtol=rtol, atol=rtol*1e-3, correct_fsal=True)
    sc.integrate(t_eval)
    c = sc.get_statistics()

    sb = AdaptiveHeunSolver(stiff_func, stiff_y0, rtol=rtol, atol=rtol*1e-3, correct_fsal=False)
    sb.integrate(t_eval)
    b = sb.get_statistics()

    print(f"  rtol={rtol:.0e}: Corrected {c['n_accepted']:5d} steps ({c['n_func_evals']:5d} evals) | "
          f"FSAL-bug {b['n_accepted']:5d} steps ({b['n_func_evals']:5d} evals)")

# Timing
print("\nTiming (50 runs, rtol=1e-6):")
for label, fsal in [("Corrected", True), ("FSAL-bug", False)]:
    # warmup
    for _ in range(5):
        AdaptiveHeunSolver(stiff_func, stiff_y0, rtol=1e-6, atol=1e-9, correct_fsal=fsal).integrate(t_eval)
    times = []
    for _ in range(50):
        s = AdaptiveHeunSolver(stiff_func, stiff_y0, rtol=1e-6, atol=1e-9, correct_fsal=fsal)
        t0 = time.perf_counter()
        s.integrate(t_eval)
        times.append((time.perf_counter()-t0)*1000)
    print(f"  {label:12s}: {np.mean(times):7.1f} ms (+/- {np.std(times):.1f})")

print("\n=== LaTeX Table 1 data ===")
for h, eb, ec in results:
    def fmt(x):
        if x == 0: return "$0$"
        exp = int(np.floor(np.log10(abs(x))))
        man = x / 10**exp
        return f"${man:.1f} \\times 10^{{{exp}}}$"
    print(f"{h:.3f} & {fmt(eb)} & {fmt(ec)} \\\\")

print("\nDone.")
