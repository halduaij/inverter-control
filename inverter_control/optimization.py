"""
Optimization routines for power system parameter tuning.

Contains:
- run_multi_scenario_optimization: Main optimization loop with CosineAnnealingWarmRestarts
- _run_batch_scenario: Batch scenario execution
- plot_optimization_results: Visualization of optimization progress
- _plot_lr_schedule: Learning rate schedule visualization
"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from pathlib import Path
from torchdiffeq import odeint

from typing import Dict, List, Optional, Tuple, Union

# Import from sibling modules
from .losses import compute_loss_batch, print_loss_components
from .constraints import (
    check_stability_conditions,
    compute_batch_constraint_violations,
    update_lagrange_multipliers,
    update_lagrange_multipliers_batch,
    project_parameters,
    clear_constraint_cache
)


def run_multi_scenario_optimization(
    sim,
    num_epochs: int = 500,
    learning_rate: float = 0.005,
    multiplier_step_size: float = 0.1,
    batch_size: Optional[int] = None,
    load_factors: Optional[torch.Tensor] = None,
    plot_results: bool = True,
    use_warm_restarts: bool = True,
    T_0: int = 100,
    T_mult: int = 2,
    save_every: int = 10,
    plot_every: int = 40,
    show_inline_plots: bool = False,
    experiment_name: Optional[str] = None
) -> Dict:
    """
    Run multi-scenario optimization with CosineAnnealingWarmRestarts scheduler.
    
    This is the main optimization loop that tunes the control parameters to minimize
    the combined loss across scenarios while satisfying stability constraints.
    
    Args:
        sim: MultiConverterSimulation instance
        num_epochs: Number of optimization epochs
        learning_rate: Base learning rate
        multiplier_step_size: Step size for Lagrange multiplier updates
        batch_size: Batch size for parallel trajectories (None = use sim's batch_size)
        load_factors: Load scaling factors for batch simulation
        plot_results: Whether to plot final results
        use_warm_restarts: Use CosineAnnealingWarmRestarts scheduler
        T_0: Initial period length for cosine annealing
        T_mult: Period multiplier after each restart
        save_every: Save checkpoint every N epochs
        plot_every: Plot progress every N epochs
        show_inline_plots: Display plots inline during training
        experiment_name: Name for experiment logging
        
    Returns:
        Dictionary containing optimization results, parameter history, and metrics
    """
    # Initialize logger for incremental saving
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = Path("optimization_results") / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = save_dir / "checkpoint.pkl"

    # Initialize results dictionary
    saved_results = {
        'losses': [],
        'per_scenario_losses': {s: [] for s in ["load_change"]},
        'constraint_satisfaction': [],
        'parameter_history': {
            'eta': [], 'eta_a': [], 'Kp_v': [],
            'Ki_v': [], 'Kp_f': [], 'Ki_f': []
        },
        'constraint_history': {
            'g4': [], 'g5': [], 'g6': [],
            'lambda4': [], 'lambda5': [], 'lambda6': []
        },
        'lr_history': {},
        'load_history': [],
        'restart_epochs': [],
        'restart_losses': [],
        'config': {
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'use_warm_restarts': use_warm_restarts,
            'T_0': T_0,
            'T_mult': T_mult,
            'batch_size': batch_size or sim.batch_size
        }
    }

    # Reinitialize if batch size changed
    if batch_size is not None and batch_size != sim.batch_size:
        sim.__init__(batch_size=batch_size, device=sim.device, dtype=sim.dtype)

    # Setup scenarios
    use_pcgrad = False
    if load_factors is None:
        scenarios = ["load_change"]
        scenario_weights = {s: 1.0/len(scenarios) for s in scenarios}
        use_pcgrad = len(scenarios) > 1
    else:
        scenarios = ["load_change"]
        scenario_weights = {"load_change": 1.0}

    print(f"Running optimization with batch_size={sim.batch_size}")
    print(f"Using scheduler: {'CosineAnnealingWarmRestarts' if use_warm_restarts else 'Fixed LR'}")

    # Create optimizer with parameter groups
    primal_params = [p for n, p in sim.named_parameters() if not n.startswith('lambda_')]

    # Group parameters by behavior
    param_groups = [
        {
            'params': [sim.eta],
            'lr': learning_rate * 0.2,
            'name': 'eta'
        },
        {
            'params': [sim.eta_a, sim.Kp_f, sim.Kp_v],
            'lr': learning_rate * 7,
            'name': 'linear_growth'
        },
        {
            'params': [sim.Ki_f, sim.Ki_v],
            'lr': learning_rate,
            'name': 'voltage_gains'
        }
    ]

    if use_pcgrad:
        try:
            from pcgrad import PCGrad
            inner_optim = torch.optim.Adam(param_groups, weight_decay=0.0)
            optimizer = PCGrad(inner_optim)
        except ImportError:
            print("PCGrad not available, using standard Adam optimizer")
            optimizer = torch.optim.Adam(param_groups, weight_decay=0.0)
            use_pcgrad = False
    else:
        optimizer = torch.optim.Adam(param_groups, weight_decay=0.0)

    # Create scheduler if requested
    restart_epochs = []
    restart_losses = []

    if use_warm_restarts:
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        T_mult_int = int(T_mult) if T_mult > 1 else 1
        if T_mult != T_mult_int:
            print(f"Note: T_mult={T_mult} converted to {T_mult_int} for PyTorch compatibility")
        gamma = 0.9
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult_int,
            eta_min=1e-5
        )

    # Tracking
    all_losses = []
    per_scenario_losses = {s: [] for s in scenarios}
    constraint_satisfaction = []
    lr_history = {name: [] for name in ['eta', 'linear_growth', 'voltage_gains']}

    parameter_history = {
        'eta': [], 'eta_a': [], 'Kp_v': [],
        'Ki_v': [], 'Kp_f': [], 'Ki_f': []
    }

    constraint_history = {
        'g4': [], 'g5': [], 'g6': [],
        'lambda4': [], 'lambda5': [], 'lambda6': []
    }

    load_history = []

    # Store initial parameters (in SI units)
    initial_params_si = _get_params_si(sim)

    print("Initial parameters (SI units):")
    for name, value in initial_params_si.items():
        print(f"  {name}: {value:.6f}")

    # Main optimization loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        task_losses = {}

        # Handle load factors for this epoch
        if load_factors is not None:
            epoch_loads = _process_load_factors(load_factors, sim.batch_size, sim.device, epoch)
            sim.network.update_batch_loads(epoch_loads)
            load_history.append(epoch_loads.clone().cpu())

        # Process each scenario
        constraint_info = None
        for scen in scenarios:
            if sim.batch_size > 1 and load_factors is not None:
                loss = _run_batch_scenario(sim, scen, epoch_loads, epoch)
                task_losses[scen] = loss
                per_scenario_losses[scen].append(loss.item())
            else:
                t_vec, sol = sim.run_simulation_for_scenario(scen)
                from .constraints import compute_lagrangian_loss
                loss, perf_loss, constraint_terms, constraint_info = compute_lagrangian_loss(sim, t_vec, sol)
                task_losses[scen] = loss
                per_scenario_losses[scen].append(loss.item())

        # Track learning rates before step
        for param_group in optimizer.param_groups:
            lr_history[param_group['name']].append(param_group['lr'])

        # Track parameters (convert to SI)
        _update_parameter_history(sim, parameter_history, saved_results)

        # Track constraints
        if sim.batch_size > 1 and load_factors is not None and 'epoch_loads' in locals():
            violations = compute_batch_constraint_violations(sim, epoch_loads)
            constraint_history['g4'].append(float(violations['g4_worst']))
            constraint_history['g5'].append(float(violations['g5_worst']))
            constraint_history['g6'].append(float(violations['g6_worst']))
            constraint_history['lambda4'].append(sim.lambda_cond4.item())
            constraint_history['lambda5'].append(sim.lambda_cond5.item())
            constraint_history['lambda6'].append(sim.lambda_cond6.item())
        elif constraint_info is not None:
            constraint_history['g4'].append(constraint_info['g4'])
            constraint_history['g5'].append(constraint_info['g5'])
            constraint_history['g6'].append(constraint_info['g6'])
            constraint_history['lambda4'].append(constraint_info['lambda4'])
            constraint_history['lambda5'].append(constraint_info['lambda5'])
            constraint_history['lambda6'].append(constraint_info['lambda6'])

        # Backward pass
        if use_pcgrad and hasattr(optimizer, 'pc_backward'):
            optimizer.pc_backward([task_losses[s] for s in scenarios])
            if use_warm_restarts:
                torch.nn.utils.clip_grad_norm_(primal_params, max_norm=1.0)
            optimizer.step()
        else:
            combined_loss = sum(scenario_weights[s] * task_losses[s] for s in scenarios)
            combined_loss.backward()
            if use_warm_restarts:
                torch.nn.utils.clip_grad_norm_(primal_params, max_norm=1.0)
            optimizer.step()

        # Project parameters to valid bounds
        project_parameters(sim)

        # Compute combined loss value for tracking
        combined_loss_val = sum(scenario_weights[s] * task_losses[s].item() for s in scenarios)
        all_losses.append(combined_loss_val)

        # Scheduler step and restart detection
        if use_warm_restarts:
            prev_T_cur = scheduler.T_cur
            scheduler.step()
            if scheduler.T_cur == 0 and epoch > 0:
                for group in optimizer.param_groups:
                    group['lr'] *= gamma
                scheduler.base_lrs = [lr * gamma for lr in scheduler.base_lrs]
            if prev_T_cur > 0 and scheduler.T_cur == 0:
                restart_epochs.append(epoch)
                restart_losses.append(combined_loss_val)
                print(f"\n=== RESTART at epoch {epoch} ===")
                print(f"Loss: {combined_loss_val:.6f}")
                print(f"Current LRs: eta={optimizer.param_groups[0]['lr']:.6f}, "
                      f"linear={optimizer.param_groups[1]['lr']:.6f}")

        # Update multipliers
        if load_factors is not None and sim.batch_size > 1 and 'epoch_loads' in locals():
            update_lagrange_multipliers_batch(sim, step_size=multiplier_step_size, load_factors=epoch_loads)
        else:
            update_lagrange_multipliers(sim, step_size=multiplier_step_size)

        # Check constraints
        if load_factors is not None and sim.batch_size > 1 and 'epoch_loads' in locals():
            violations = compute_batch_constraint_violations(sim, epoch_loads)
            all_satisfied = (violations['g4_worst'] <= 0 and
                           violations['g5_worst'] <= 0 and
                           violations['g6_worst'] <= 0)
        else:
            stab = check_stability_conditions(sim)
            all_satisfied = stab["all_satisfied"]

        constraint_satisfaction.append(all_satisfied)

        # Periodic output
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={combined_loss_val:.6f}")
            if load_factors is not None and 'epoch_loads' in locals():
                print(f"  Load range: [{epoch_loads.min():.2f}, {epoch_loads.max():.2f}]")
            print(f"  Constraints satisfied: {all_satisfied}")
            if use_warm_restarts:
                print(f"  LRs: eta={optimizer.param_groups[0]['lr']:.6f}, "
                      f"linear={optimizer.param_groups[1]['lr']:.6f}, "
                      f"voltage={optimizer.param_groups[2]['lr']:.6f}")

        # Periodic plotting
        if plot_every > 0 and epoch > 0 and epoch % plot_every == 0:
            _plot_progress(
                epoch, all_losses, constraint_history, parameter_history,
                lr_history, all_satisfied, restart_epochs, save_dir, show_inline_plots
            )

        # Incremental save
        if epoch % save_every == 0 or (epoch == num_epochs - 1):
            _save_checkpoint(
                saved_results, all_losses, per_scenario_losses, constraint_satisfaction,
                parameter_history, constraint_history, lr_history, load_history,
                restart_epochs, restart_losses, epoch, checkpoint_file
            )
            if epoch % save_every == 0 and epoch > 0:
                print(f"  [Checkpoint saved at epoch {epoch}]")

        # Early stopping check
        if len(all_losses) > 50:
            recent_losses = all_losses[-20:]
            if max(recent_losses) - min(recent_losses) < 1e-6:
                print(f"\nEarly stopping at epoch {epoch}: Loss converged")
                break

    # Final report
    final_stability = check_stability_conditions(sim, verbose=True)
    print(f"\nFinal constraint satisfaction: {final_stability['all_satisfied']}")

    final_params_si = _get_params_si(sim)

    print("\nFinal parameters (SI units):")
    for name, value in final_params_si.items():
        print(f"  {name}: {value:.6f} (changed by {value-initial_params_si[name]:+.6f})")

    # Prepare results
    results = {
        'losses': all_losses,
        'per_scenario_losses': per_scenario_losses,
        'constraint_satisfaction': constraint_satisfaction,
        'parameter_history': parameter_history,
        'constraint_history': constraint_history,
        'load_history': load_history,
        'initial_params': initial_params_si,
        'final_params': final_params_si,
        'scenarios': scenarios,
        'lr_history': lr_history,
        'save_dir': str(save_dir),
        'checkpoint_file': str(checkpoint_file)
    }

    if use_warm_restarts:
        results['restart_epochs'] = restart_epochs
        results['restart_losses'] = restart_losses

    # Final save
    _save_checkpoint(
        saved_results, all_losses, per_scenario_losses, constraint_satisfaction,
        parameter_history, constraint_history, lr_history, load_history,
        restart_epochs, restart_losses, len(all_losses) - 1, checkpoint_file,
        initial_params_si, final_params_si
    )

    print(f"\nResults saved to: {checkpoint_file}")

    # Plot results if requested
    if plot_results:
        plot_optimization_results(results)
        if use_warm_restarts and lr_history['eta']:
            _plot_lr_schedule(lr_history, results.get('restart_epochs', []))

    return results


def _run_batch_scenario(sim, scenario: str, load_factors: torch.Tensor, epoch: int) -> torch.Tensor:
    """
    Run a scenario with batch processing.
    
    Args:
        sim: MultiConverterSimulation instance
        scenario: Scenario name
        load_factors: Load scaling factors [batch_size]
        epoch: Current epoch number
        
    Returns:
        Total loss for the batch
    """
    # Set scenario
    sim.scenario = scenario
    sim.network.update_batch_loads(load_factors)
    sim.integrate_line_dynamics = (scenario == "load_change")

    # Initialize states for batch
    x0_batch = torch.zeros(sim.batch_size, 5 * 2 * sim.network.Nc,
                          dtype=sim.dtype, device=sim.device)

    # Initialize based on scenario type
    if scenario == "black_start":
        for b in range(sim.batch_size):
            for i in range(2):
                idx_vhat = 2 * i
                x0_batch[b, idx_vhat:idx_vhat+2] = torch.tensor(
                    [0.01/120, 0.0], dtype=sim.dtype, device=sim.device
                )
    else:
        x0_single = sim.initialize_from_equilibrium()
        x0_batch = x0_single.unsqueeze(0).expand(sim.batch_size, -1).contiguous()

    # Time span
    steps = int(sim.T_sim / sim.dt) + 1
    t_span = torch.linspace(0.0, sim.T_sim, steps, dtype=sim.dtype, device=sim.device)

    # Run PARALLEL batch ODE integration
    sol_batch = odeint(
        func=sim,
        y0=x0_batch,
        t=t_span,
        rtol=1e-3,
        atol=1e-3,
        method='dopri5'
    )

    # Compute performance losses
    performance_losses, components = compute_loss_batch(
        sim, t_span, sol_batch,
        include_frequency=True,
        freq_weight=0.1,
        include_action=True,
        action_weight=0.05,
        verbose=True
    )

    # Print loss components
    print_loss_components(components, epoch=epoch, scenario='load_change')

    # Average performance loss
    avg_performance_loss = performance_losses.mean()

    # Compute worst-case constraint violations across batch
    constraint_violations = compute_batch_constraint_violations(sim, load_factors)

    # Lagrangian terms for worst-case constraints
    g4_worst = constraint_violations['g4_worst']
    g5_worst = constraint_violations['g5_worst']
    g6_worst = constraint_violations['g6_worst']

    lagrangian_term4 = sim.lambda_cond4 * torch.relu(g4_worst)
    lagrangian_term5 = sim.lambda_cond5 * torch.relu(g5_worst)
    lagrangian_term6 = sim.lambda_cond6 * torch.relu(g6_worst)

    aug_term4 = 0.5 * torch.relu(g4_worst) ** 2
    aug_term5 = 0.5 * torch.relu(g5_worst) ** 2
    aug_term6 = 0.5 * torch.relu(g6_worst) ** 2

    constraint_terms = (lagrangian_term4 + lagrangian_term5 + lagrangian_term6 +
                       aug_term4 + aug_term5 + aug_term6)

    # Total loss
    total_loss = avg_performance_loss + constraint_terms

    return total_loss


def _get_params_si(sim) -> Dict[str, float]:
    """Get parameters in SI units."""
    return {
        'eta': sim.eta.item() * (sim.network.pu.Zb * sim.network.pu.ωb),
        'eta_a': sim.eta_a.item() / sim.network.pu.Zb,
        'Kp_v': sim.Kp_v.item() / sim.network.pu.Zb,
        'Ki_v': sim.Ki_v.item() / (sim.network.pu.Zb / sim.network.pu.ωb),
        'Kp_f': sim.Kp_f.item() * sim.network.pu.Zb,
        'Ki_f': sim.Ki_f.item() * (sim.network.pu.Zb * sim.network.pu.ωb)
    }


def _process_load_factors(load_factors, batch_size: int, device: str, epoch: int) -> torch.Tensor:
    """Process load factors for an epoch."""
    if isinstance(load_factors, torch.Tensor):
        epoch_loads = load_factors[:batch_size] if len(load_factors) > batch_size else load_factors
    elif isinstance(load_factors, tuple) and len(load_factors) == 2:
        min_load, max_load = load_factors
        epoch_loads = min_load + (max_load - min_load) * torch.rand(batch_size, device=device)
    elif isinstance(load_factors, list):
        load_set_idx = epoch % len(load_factors)
        epoch_loads = load_factors[load_set_idx]
        if len(epoch_loads) > batch_size:
            epoch_loads = epoch_loads[:batch_size]
    else:
        epoch_loads = torch.ones(batch_size, device=device)
    return epoch_loads


def _update_parameter_history(sim, parameter_history: Dict, saved_results: Dict):
    """Update parameter tracking."""
    params_si = _get_params_si(sim)
    for name in ['eta', 'eta_a', 'Kp_v', 'Ki_v', 'Kp_f', 'Ki_f']:
        parameter_history[name].append(params_si[name])
        saved_results['parameter_history'][name].append(params_si[name])


def _save_checkpoint(
    saved_results: Dict,
    all_losses: List,
    per_scenario_losses: Dict,
    constraint_satisfaction: List,
    parameter_history: Dict,
    constraint_history: Dict,
    lr_history: Dict,
    load_history: List,
    restart_epochs: List,
    restart_losses: List,
    epoch: int,
    checkpoint_file: Path,
    initial_params: Optional[Dict] = None,
    final_params: Optional[Dict] = None
):
    """Save checkpoint to file."""
    saved_results['losses'] = all_losses.copy()
    saved_results['per_scenario_losses'] = {k: v.copy() for k, v in per_scenario_losses.items()}
    saved_results['constraint_satisfaction'] = constraint_satisfaction.copy()
    saved_results['constraint_history'] = {k: v.copy() for k, v in constraint_history.items()}
    saved_results['lr_history'] = {k: v.copy() for k, v in lr_history.items()}
    saved_results['load_history'] = [x.clone().cpu() if torch.is_tensor(x) else x for x in load_history]
    saved_results['restart_epochs'] = restart_epochs.copy()
    saved_results['restart_losses'] = restart_losses.copy()
    saved_results['epochs_completed'] = epoch + 1

    if initial_params is not None:
        saved_results['initial_params'] = initial_params
    if final_params is not None:
        saved_results['final_params'] = final_params

    with open(checkpoint_file, 'wb') as f:
        pickle.dump(saved_results, f)


def _plot_progress(
    epoch: int,
    all_losses: List,
    constraint_history: Dict,
    parameter_history: Dict,
    lr_history: Dict,
    all_satisfied: bool,
    restart_epochs: List,
    save_dir: Path,
    show_inline: bool
):
    """Plot training progress."""
    print(f"\n[Plotting progress at epoch {epoch}...]")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    current_epochs = np.arange(len(all_losses))

    # 1. Loss
    ax = axes[0, 0]
    ax.plot(current_epochs, all_losses, 'b-', linewidth=2)
    for restart_epoch in restart_epochs:
        ax.axvline(x=restart_epoch, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss (Current: {all_losses[-1]:.4f})')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. Constraints
    ax = axes[0, 1]
    if constraint_history['g4']:
        ax.plot(constraint_history['g4'], 'r-', label='g4', alpha=0.8)
        ax.plot(constraint_history['g5'], 'g-', label='g5', alpha=0.8)
        ax.plot(constraint_history['g6'], 'b-', label='g6', alpha=0.8)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Constraint Value')
    ax.set_title(f'Constraints (Satisfied: {"✓" if all_satisfied else "✗"})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. All parameters (normalized)
    ax = axes[1, 0]
    param_names = ['eta', 'eta_a', 'Kp_v', 'Ki_v', 'Kp_f', 'Ki_f']
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']

    for i, (param, color) in enumerate(zip(param_names, colors)):
        if parameter_history[param]:
            values = np.array(parameter_history[param])
            if len(values) > 0 and values[0] != 0:
                normalized = (values / values[0] - 1) * 100
                ax.plot(normalized, color=color, linewidth=1.5,
                       label=f'{param} ({values[-1]:.3f})', alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Change from Initial (%)')
    ax.set_title('All Parameters Evolution')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # 4. Learning rates
    ax = axes[1, 1]
    for group_name, lr_vals in lr_history.items():
        if lr_vals:
            ax.plot(lr_vals, linewidth=2, label=group_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Training Progress - Epoch {epoch}', fontsize=14)
    plt.tight_layout()

    plot_path = save_dir / f"progress_epoch_{epoch}.png"
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')

    if show_inline:
        plt.show()
    else:
        plt.close()

    print(f"[Progress plot saved to {plot_path}]")


def plot_optimization_results(results: Dict):
    """
    Plot comprehensive optimization results.
    
    Args:
        results: Dictionary containing optimization results
    """
    epochs = np.arange(len(results['losses']))

    fig = plt.figure(figsize=(16, 12))

    # 1. Loss convergence
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(epochs, results['losses'], 'b-', linewidth=2, label='Combined Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Combined Loss Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 2. Per-scenario losses
    ax2 = plt.subplot(3, 3, 2)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (scenario, losses) in enumerate(results['per_scenario_losses'].items()):
        ax2.plot(epochs, losses, colors[i % len(colors)],
                label=scenario.replace('_', ' ').title(), alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Per-Scenario Losses')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # 3. Constraint satisfaction
    ax3 = plt.subplot(3, 3, 3)
    constraint_array = np.array([x.cpu().numpy() if torch.is_tensor(x) else x
                                for x in results['constraint_satisfaction']])
    constraint_epochs = np.where(constraint_array)[0]
    violation_epochs = np.where(np.logical_not(constraint_array))[0]

    ax3.scatter(violation_epochs, np.ones_like(violation_epochs),
               color='red', s=30, alpha=0.6, label='Violated')
    ax3.scatter(constraint_epochs, np.ones_like(constraint_epochs),
               color='green', s=30, alpha=0.6, label='Satisfied')
    ax3.set_xlabel('Epoch')
    ax3.set_ylim(0.5, 1.5)
    ax3.set_yticks([])
    ax3.set_title('Constraint Satisfaction')
    ax3.legend()
    ax3.grid(True, axis='x', alpha=0.3)

    # 4-9. Parameter evolution
    param_names = ['eta', 'eta_a', 'Kp_v', 'Ki_v', 'Kp_f', 'Ki_f']
    param_units = ['(rad/s)·Ω', 'S', 'V⁻¹', '(V·s)⁻¹', 'A⁻¹', '(A·s)⁻¹']

    for i, (param, unit) in enumerate(zip(param_names, param_units)):
        ax = plt.subplot(3, 3, i + 4)
        values = results['parameter_history'][param]
        ax.plot(epochs, values, 'b-', linewidth=2)
        ax.axhline(y=results['initial_params'][param], color='g',
                  linestyle='--', alpha=0.5, label='Initial')
        ax.axhline(y=results['final_params'][param], color='r',
                  linestyle='--', alpha=0.5, label='Final')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{param} [{unit}]')
        ax.set_title(f'{param} Evolution')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # Second figure for constraints and load factors
    if results['constraint_history']['g4']:
        fig2 = plt.figure(figsize=(12, 8))

        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(epochs, results['constraint_history']['g4'], label='g4 (condition 4)')
        ax1.plot(epochs, results['constraint_history']['g5'], label='g5 (condition 5)')
        ax1.plot(epochs, results['constraint_history']['g6'], label='g6 (condition 6)')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Constraint Violation')
        ax1.set_title('Constraint Violations (negative = satisfied)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(epochs, results['constraint_history']['lambda4'], label='λ4')
        ax2.plot(epochs, results['constraint_history']['lambda5'], label='λ5')
        ax2.plot(epochs, results['constraint_history']['lambda6'], label='λ6')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Lagrange Multiplier')
        ax2.set_title('Lagrange Multipliers Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        if results['load_history']:
            ax3 = plt.subplot(2, 1, 2)
            load_list = [x.cpu() if torch.is_tensor(x) else x for x in results['load_history']]
            load_array = torch.stack(load_list).numpy()

            if load_array.shape[1] > 1:
                mean_loads = np.mean(load_array, axis=1)
                min_loads = np.min(load_array, axis=1)
                max_loads = np.max(load_array, axis=1)

                ax3.plot(epochs, mean_loads, 'b-', linewidth=2, label='Mean')
                ax3.fill_between(epochs, min_loads, max_loads, alpha=0.3, label='Range')
            else:
                ax3.plot(epochs, load_array.squeeze(), 'b-', linewidth=2)

            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Load Factor')
            ax3.set_title('Load Factors Used During Optimization')
            ax3.grid(True, alpha=0.3)
            ax3.legend()

        plt.tight_layout()
        plt.show()


def _plot_lr_schedule(lr_history: Dict, restart_epochs: List):
    """
    Plot learning rate schedule with restart markers.
    
    Args:
        lr_history: Dictionary of learning rate histories by parameter group
        restart_epochs: List of epochs where restarts occurred
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = np.arange(len(lr_history['eta']))

    for name, lrs in lr_history.items():
        ax.plot(epochs, lrs, label=name, linewidth=2)

    for restart_epoch in restart_epochs:
        ax.axvline(x=restart_epoch, color='red', linestyle='--', alpha=0.5)
        ax.text(restart_epoch, ax.get_ylim()[1]*0.9, 'Restart',
                rotation=90, va='top', ha='right', color='red')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule with Warm Restarts')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
