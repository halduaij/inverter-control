# Inverter Control: Distributed Adaptive Control for Multi-Converter Power Systems

A PyTorch-based optimization package for tuning adaptive distributed control parameters in multi-converter power systems (grid-forming inverters). Uses neural network local gain schedulers with stability constraints from the nested Lyapunov framework of Subotic et al. (2021), which decomposes the closed-loop converter dynamics into temporally nested subsystems (dVOC, network, voltage control, current control) and provides explicit gain bounds for almost global asymptotic stability.

## Features

- Per-unit power system framework with 3-converter network model
- Distributed voltage-oriented control (dVOC) with PI voltage and current loops
- Local neural network gain schedulers (per-converter, O(1) complexity)
- Nested Lyapunov stability constraints (Condition 4/5/6 from Subotic et al., 2021)
- Adaptive Heun ODE solver with FSAL bug fix and JIT compilation
- Batch simulation for parallel scenario optimization on GPU
- Augmented Lagrangian constrained optimization with cosine annealing

## Package Structure

```
inverter_control/
├── core.py               # Per-unit system, setpoints, Newton solver
├── network.py            # Network topology, transmission lines, algebraic solver
├── converter.py          # Converter state management, voltage/current control
├── simulation.py         # ODE integration, batch simulation orchestration
├── constraints.py        # Stability constraints, Lagrange multiplier updates
├── losses.py             # Frequency deviation loss, control action penalty
├── optimization.py       # Main training loop, checkpointing, scheduling
├── local_schedulers.py   # Neural gain schedulers, adaptive eta controller
└── heun/
    ├── __init__.py
    ├── adaptive_heun.py      # Solver wrapper (optimized or fallback)
    ├── adaptive_heun_fast.py # JIT-optimized adaptive Heun solver
    └── benchmark_heun.py     # Solver performance benchmarks
```

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- PyTorch >= 2.0
- NumPy
- SciPy
- torchdiffeq
- Matplotlib

## Key Components

### Local Gain Schedulers

Each converter uses a small neural network that maps local operating conditions to PI control gains:

```
Input: [p_k*, q_k*, v_k*, rL_k, v_error_k]  (5 features)
Output: [Kp_v, Ki_v, Kp_f, Ki_f] adjustments
```

Only local information is used -- no communication between converters.

### Adaptive Heun Solver

Includes a corrected adaptive Heun ODE solver that fixes a FSAL (First Same As Last) bug in the original torchdiffeq implementation. The bug causes k2 (evaluated at y_euler) to be incorrectly reused as k1, inflating the error constant by 2--3x. Combined with JIT compilation and branchless step-size control, the optimized solver achieves 2.1x speedup over torchdiffeq.

### Stability Constraints

Implements the three stability conditions from the nested Lyapunov framework of Subotic et al. (2021). The closed-loop system is decomposed into four temporally nested subsystems (dVOC reference model -> transmission line dynamics -> voltage PI controller -> current PI controller), ordered slow to fast. Stability is guaranteed when each faster layer converges fast enough relative to the slower layers, which translates to explicit bounds on the control gains:
- **Condition 4**: Network loading bound -- links dVOC gain eta to graph Laplacian lambda_2(L), branch powers, and network load margin c_L
- **Condition 5**: Voltage controller time-scale separation -- bounds voltage PI gains (Kp_v, Ki_v) relative to eta, ensuring the voltage loop is fast enough vs. dVOC
- **Condition 6**: Current controller time-scale separation -- bounds current PI gains (Kp_f, Ki_f) relative to voltage gains, ensuring the inner current loop is fastest

## License

MIT License. See [LICENSE](LICENSE).
