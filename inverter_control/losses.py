"""
Loss Functions for Power System Optimization.

Contains:
- compute_loss: Single trajectory loss computation
- compute_loss_batch: Batched loss computation
- Frequency loss computation (dynamics-based, differentiable)
- Action loss computation (control effort penalty)

FIXES APPLIED:
- _compute_frequency_loss_batch: Uses batched forward pass (handles per-batch rL correctly)
- _compute_action_loss_simple_batch: Uses scalar gains directly (gradient flow preserved)
"""

import math
import torch
import numpy as np
from typing import Tuple, Optional, Dict


def compute_frequency_differentiable(sim, t, state):
    """
    Compute frequency directly from dynamics (differentiable).

    Formula: ω = (vhat_α * dvhat_β - vhat_β * dvhat_α) / ||vhat||²

    The dynamics use per-unit time, so dvhat is in pu/pu-time.
    omega_dev in per-unit means deviation from ω0 = 1 pu.
    f = f0 * (1 + omega_dev) where f0 = 60 Hz.

    Args:
        sim: MultiConverterSimulation instance
        t: Time (scalar)
        state: State tensor [state_size]

    Returns:
        frequency: Tensor [Nc] in Hz
    """
    Nc = sim.network.Nc
    n_conv = 2 * Nc
    f0 = 60.0

    # Get dvhat from dynamics (differentiable)
    dstate = sim.forward(t, state)

    # vhat is always the first n_conv elements
    vhat = state[0:n_conv].view(Nc, 2)
    dvhat = dstate[0:n_conv].view(Nc, 2)

    # ω_dev = (vhat_α * dvhat_β - vhat_β * dvhat_α) / ||vhat||²
    # This is the angular frequency deviation in per-unit
    vhat_mag_sq = (vhat ** 2).sum(dim=-1) + 1e-12
    omega_dev_pu = (vhat[:, 0] * dvhat[:, 1] - vhat[:, 1] * dvhat[:, 0]) / vhat_mag_sq

    # Convert to Hz: f = f0 * (1 + omega_dev_pu)
    f = f0 * (1.0 + omega_dev_pu)

    return f


def compute_frequency_differentiable_batch(sim, t, state_batch):
    """
    Compute frequency directly from dynamics for BATCHED states (differentiable).

    Formula: ω = (vhat_α * dvhat_β - vhat_β * dvhat_α) / ||vhat||²

    Args:
        sim: MultiConverterSimulation instance
        t: Time (scalar)
        state_batch: State tensor [B, state_size]

    Returns:
        frequency: Tensor [B, Nc] in Hz
    """
    B = state_batch.shape[0]
    Nc = sim.network.Nc
    n_conv = 2 * Nc
    f0 = 60.0

    # Get dvhat from dynamics (differentiable) - calls batched version
    dstate_batch = sim.forward(t, state_batch)  # [B, state_size]

    # vhat is always the first n_conv elements
    vhat = state_batch[:, 0:n_conv].view(B, Nc, 2)      # [B, Nc, 2]
    dvhat = dstate_batch[:, 0:n_conv].view(B, Nc, 2)    # [B, Nc, 2]

    # ω_dev = (vhat_α * dvhat_β - vhat_β * dvhat_α) / ||vhat||²
    vhat_mag_sq = (vhat ** 2).sum(dim=-1) + 1e-12  # [B, Nc]
    omega_dev_pu = (vhat[:, :, 0] * dvhat[:, :, 1] - vhat[:, :, 1] * dvhat[:, :, 0]) / vhat_mag_sq

    # Convert to Hz: f = f0 * (1 + omega_dev_pu)
    f = f0 * (1.0 + omega_dev_pu)  # [B, Nc]

    return f


def compute_frequency_trajectory(sim, t_vec, sol):
    """
    Compute frequency over entire trajectory (differentiable).

    Args:
        sim: MultiConverterSimulation instance
        t_vec: Time vector [T]
        sol: Solution tensor [T, state_size]

    Returns:
        freq: Tensor [T, Nc] in Hz
    """
    T = len(t_vec)
    Nc = sim.network.Nc
    freq = torch.zeros(T, Nc, dtype=sol.dtype, device=sol.device)

    for k in range(T):
        freq[k] = compute_frequency_differentiable(sim, t_vec[k], sol[k])

    return freq


def compute_loss(sim, t_vec, sol, include_frequency=True, freq_weight=0.1,
                include_action=True, action_weight=0.05, verbose=False):
    """
    Compute performance loss with simple action penalty on control efforts.

    Args:
        sim: MultiConverterSimulation instance
        t_vec: Time vector [T]
        sol: Solution tensor [T, state_size]
        include_frequency: If True, includes frequency regulation loss
        freq_weight: Weight for frequency loss term
        include_action: If True, includes penalty on control efforts
        action_weight: Weight for the action loss term
        verbose: If True, returns (loss, components_dict) instead of just loss
        
    Returns:
        loss tensor, or (loss, components_dict) if verbose=True
    """
    if sol is None:
        if verbose:
            return torch.tensor(1e6, dtype=sim.dtype, device=sim.device), {}
        return torch.tensor(1e6, dtype=sim.dtype, device=sim.device)

    Nc, Nt = sim.network.Nc, sim.network.Nt
    n_conv, n_line = 2 * Nc, 2 * Nt

    # Extract voltages based on state format
    if sim.integrate_line_dynamics:
        vhat_indices = slice(0, n_conv)
        v_indices = slice(n_conv + n_line, 2*n_conv + n_line)
    else:
        vhat_indices = slice(0, n_conv)
        v_indices = slice(n_conv, 2*n_conv)

    vhat_sol = sol[:, vhat_indices].reshape(-1, Nc, 2)
    v_sol = sol[:, v_indices].reshape(-1, Nc, 2)

    # Voltage magnitudes
    vhat_mag = torch.norm(vhat_sol, dim=2)
    v_mag = torch.norm(v_sol, dim=2)

    # Deviations from voltage setpoint
    v_star = sim.converter.setpoints.v_star  # Use actual setpoint, not hardcoded 1.0
    vhat_dev = (vhat_mag - v_star.unsqueeze(0)) / v_star.unsqueeze(0)
    v_dev = (v_mag - v_star.unsqueeze(0)) / v_star.unsqueeze(0)

    # Loss components - L_inf (max) norm
    vhat_loss_inf = torch.max(torch.abs(vhat_dev))
    v_loss_inf = torch.max(torch.abs(v_dev))

    # Loss components - L_2 (RMS) norm
    vhat_loss_l2 = torch.sqrt(torch.mean(vhat_dev ** 2))
    v_loss_l2 = torch.sqrt(torch.mean(v_dev ** 2))

    # Oscillation penalty
    num_steps = sol.shape[0]
    latter_half_idx = num_steps // 2
    v_mag_latter = v_mag[latter_half_idx:, :]
    peak_to_peak = torch.max(v_mag_latter, dim=0)[0] - torch.min(v_mag_latter, dim=0)[0]
    oscillation_penalty = torch.max(peak_to_peak)

    # Individual voltage loss components (L_inf + L_2)
    vhat_loss_weighted = 3.0 * vhat_loss_inf + 1.0 * vhat_loss_l2
    v_loss_weighted = 10.0 * v_loss_inf + 5.0 * v_loss_l2
    oscillation_loss_weighted = 0.1 * oscillation_penalty

    # Base voltage loss
    voltage_loss = vhat_loss_weighted + v_loss_weighted + oscillation_loss_weighted

    # Initialize components dict
    components = {
        'vhat_loss_inf': vhat_loss_inf.item(),
        'vhat_loss_l2': vhat_loss_l2.item(),
        'v_loss_inf': v_loss_inf.item(),
        'v_loss_l2': v_loss_l2.item(),
        'v_loss': v_loss_inf.item(),  # Keep for backward compat
        'oscillation_penalty': oscillation_penalty.item(),
        'vhat_loss_weighted': vhat_loss_weighted.item(),
        'v_loss_weighted': v_loss_weighted.item(),
        'oscillation_loss_weighted': oscillation_loss_weighted.item(),
        'voltage_loss_total': voltage_loss.item()
    }

    # Add frequency loss if requested (uses vhat dynamics for exact oscillator frequency)
    if include_frequency and num_steps > 10:
        freq_loss, freq_components = _compute_frequency_loss(sim, t_vec, sol, return_components=True)
        freq_loss_weighted = freq_weight * freq_loss
        voltage_loss = voltage_loss + freq_loss_weighted

        components['freq_loss'] = freq_loss.item()
        components['freq_loss_weighted'] = freq_loss_weighted.item()
        components.update(freq_components)
    else:
        components['freq_loss'] = 0.0
        components['freq_loss_weighted'] = 0.0

    # Add simple action loss if requested
    if include_action and num_steps > 10:
        action_loss = _compute_action_loss_simple(sim, sol, t_vec)
        action_loss_weighted = action_weight * action_loss
        total_loss = voltage_loss + action_loss_weighted

        components['action_loss'] = action_loss.item()
        components['action_loss_weighted'] = action_loss_weighted.item()
        components['action_weight'] = action_weight
    else:
        total_loss = voltage_loss
        components['action_loss'] = 0.0
        components['action_loss_weighted'] = 0.0

    components['total_loss'] = total_loss.item()

    # Maintain gradients
    if sol.requires_grad and not total_loss.requires_grad:
        total_loss = total_loss + 0.0 * sol.sum()

    if verbose:
        return total_loss, components
    return total_loss


def _compute_action_loss_simple(sim, sol, t_vec):
    """
    Compute simple action loss based on control efforts (single trajectory).
    Penalizes u_v = Kp_v * e_v + Ki_v * zeta_v and u_f = Kp_f * e_f + Ki_f * zeta_f
    """
    T = sol.shape[0]
    Nc = sim.network.Nc
    n_conv = 2 * Nc

    # Skip initial transient (20% of simulation)
    skip_idx = max(1, int(0.2 * T))

    # Extract states
    if sim.integrate_line_dynamics:
        n_line = 2 * sim.network.Nt
        vhat_sol = sol[:, 0:n_conv].reshape(T, Nc, 2)
        v_sol = sol[:, n_conv + n_line:2*n_conv + n_line].reshape(T, Nc, 2)
        zeta_v_sol = sol[:, 2*n_conv + n_line:3*n_conv + n_line].reshape(T, Nc, 2)
        i_f_sol = sol[:, 3*n_conv + n_line:4*n_conv + n_line].reshape(T, Nc, 2)
        zeta_f_sol = sol[:, 4*n_conv + n_line:5*n_conv + n_line].reshape(T, Nc, 2)
    else:
        vhat_sol = sol[:, 0:n_conv].reshape(T, Nc, 2)
        v_sol = sol[:, n_conv:2*n_conv].reshape(T, Nc, 2)
        zeta_v_sol = sol[:, 2*n_conv:3*n_conv].reshape(T, Nc, 2)
        i_f_sol = sol[:, 3*n_conv:4*n_conv].reshape(T, Nc, 2)
        zeta_f_sol = sol[:, 4*n_conv:5*n_conv].reshape(T, Nc, 2)

    # Only consider steady state
    vhat_steady = vhat_sol[skip_idx:]
    v_steady = v_sol[skip_idx:]
    zeta_v_steady = zeta_v_sol[skip_idx:]
    zeta_f_steady = zeta_f_sol[skip_idx:]

    # Get control matrices - extract 2x2 diagonal blocks from (2*Nc, 2*Nc) block-diagonal matrices
    Kp_v_mat = torch.zeros((Nc, 2, 2), dtype=sim.dtype, device=sim.device)
    Ki_v_mat = torch.zeros((Nc, 2, 2), dtype=sim.dtype, device=sim.device)
    Kp_f_mat = torch.zeros((Nc, 2, 2), dtype=sim.dtype, device=sim.device)
    Ki_f_mat = torch.zeros((Nc, 2, 2), dtype=sim.dtype, device=sim.device)
    for c in range(Nc):
        idx = slice(2*c, 2*c + 2)
        Kp_v_mat[c] = sim.converter.Kp_v_mat[idx, idx]
        Ki_v_mat[c] = sim.converter.Ki_v_mat[idx, idx]
        Kp_f_mat[c] = sim.converter.Kp_f_mat[idx, idx]
        Ki_f_mat[c] = sim.converter.Ki_f_mat[idx, idx]

    # 1. Voltage control effort: u_v = Kp_v * e_v + Ki_v * zeta_v
    e_v = v_steady - vhat_steady  # voltage error
    u_v = torch.zeros_like(e_v)

    for c in range(Nc):
        if (sim.converter.converter_states[c]['active'] and
            sim.converter.converter_states[c]['voltage_control']):
            u_v[:, c] = (
                torch.matmul(Kp_v_mat[c], e_v[:, c].T).T +
                torch.matmul(Ki_v_mat[c], zeta_v_steady[:, c].T).T
            )

    # RMS with epsilon for numerical stability
    u_v_rms = torch.sqrt(torch.mean(u_v ** 2) + 1e-8)

    # 2. Current control effort: u_f = Kp_f * (i_f - i_ref) + Ki_f * zeta_f
    u_f = torch.zeros_like(zeta_f_steady)

    for c in range(Nc):
        if sim.converter.converter_states[c]['active']:
            u_f[:, c] = (
                torch.matmul(Kp_f_mat[c], zeta_f_steady[:, c].T).T * 0.1 +
                torch.matmul(Ki_f_mat[c], zeta_f_steady[:, c].T).T
            )

    # RMS with epsilon for numerical stability
    u_f_rms = torch.sqrt(torch.mean(u_f ** 2) + 1e-8)

    # 3. Combined action penalty with tunable parameters
    eref_noise = 0.01  # 1% p.u. sensor noise
    ki_eff_rms = torch.abs(sim.converter.Ki_f * eref_noise)

    # Unified coefficients
    α, β, κ = 0.1, 0.1, 0.1       # linear weights
    γ, δ, κs = 1.0, 1.0, 1.0      # spike weights
    thr_u = 0.20                   # 20% p.u. for u_v / u_f
    thr_ki = 0.002                 # 0.2% p.u. for Ki_f branch

    action_loss = (
        α * u_v_rms
        + γ * torch.relu(u_v_rms - thr_u)
        + β * u_f_rms
        + δ * torch.relu(u_f_rms - thr_u)
        + κ * ki_eff_rms
        + κs * torch.relu(ki_eff_rms - thr_ki)
    )

    return action_loss


def _compute_frequency_loss(sim, t_vec, sol, return_components=False):
    """
    Single trajectory frequency loss using dynamics-based frequency computation.

    Uses vhat and dvhat from sim.forward() for exact differentiable frequency.

    Args:
        sim: MultiConverterSimulation instance
        t_vec: Time vector [T]
        sol: Full solution tensor [T, state_size]
        return_components: If True, return (loss, components_dict)

    Returns:
        Loss tensor or (loss, components_dict)
    """
    T = len(t_vec)
    Nc = sim.network.Nc
    f0_hz = sim.network.pu.fb

    if T < 10:
        zero_loss = torch.tensor(0.0, dtype=sim.dtype, device=sim.device)
        if return_components:
            return zero_loss, {}
        return zero_loss

    # Compute frequency at each time step using dynamics
    freq_hz = compute_frequency_trajectory(sim, t_vec, sol)  # [T, Nc]

    # Skip transient period (first 10%)
    transient_steps = int(0.1 * T)
    if transient_steps >= T:
        zero_loss = torch.tensor(0.0, dtype=sim.dtype, device=sim.device)
        return (zero_loss, {}) if return_components else zero_loss

    freq_steady = freq_hz[transient_steps:]  # [T_steady, Nc]
    freq_steady_flat = freq_steady.reshape(-1)  # Flatten for stats

    # Loss metrics
    max_dev_hz = (freq_steady_flat - f0_hz).abs().max()
    mse_dev_hz2 = ((freq_steady_flat - f0_hz)**2).mean()
    std_dev_hz = freq_steady_flat.std()
    max_freq = freq_steady_flat.max()
    min_freq = freq_steady_flat.min()
    extreme_hz = torch.relu(max_freq - 63.0) + torch.relu(57.0 - min_freq)

    # Combined L_inf + L_2 loss (captures both peak deviation and total deviation)
    # L_2 captures duration of deviation, creating eta tradeoff: faster settling vs larger peak
    loss = 0.5 * max_dev_hz + 0.5 * torch.sqrt(mse_dev_hz2 + 1e-12)

    if return_components:
        comps = {
            'freq_max_dev_hz': max_dev_hz.item(),
            'freq_mse_dev_hz2': mse_dev_hz2.item(),
            'freq_std_hz': std_dev_hz.item(),
            'freq_extreme_hz': extreme_hz.item(),
            'freq_min_in_batch': min_freq.item(),
            'freq_max_in_batch': max_freq.item()
        }
        return loss, comps

    return loss


def _compute_frequency_loss_batch(sim, t_vec, sol_batch, return_components=False):
    """
    Batch frequency loss using dynamics-based frequency computation.

    FIXED: Uses batched forward pass which properly handles per-batch rL.

    Args:
        sim: MultiConverterSimulation instance
        t_vec: Time vector [T]
        sol_batch: Full solution tensor [T, B, state_size]
        return_components: If True, return (loss, components_dict)

    Returns:
        Loss tensor [B] or (loss [B], components_dict)
    """
    T, B, state_size = sol_batch.shape
    Nc = sim.network.Nc
    f0_hz = sim.network.pu.fb

    if T < 10:
        zero_losses = torch.zeros(B, dtype=sim.dtype, device=sim.device)
        if return_components:
            return zero_losses, {}
        return zero_losses

    # Compute frequency for all batches at each time step
    # [T, B, Nc]
    freq_hz = torch.zeros(T, B, Nc, dtype=sim.dtype, device=sim.device)

    for k in range(T):
        # sol_batch[k] is [B, state_size] - batched forward handles rL correctly
        freq_hz[k] = compute_frequency_differentiable_batch(sim, t_vec[k], sol_batch[k])

    # Skip transient period
    transient_steps = int(0.1 * T)
    if transient_steps >= T:
        zero_losses = torch.zeros(B, dtype=sim.dtype, device=sim.device)
        return (zero_losses, {}) if return_components else zero_losses

    freq_steady = freq_hz[transient_steps:]  # [T_steady, B, Nc]
    freq_steady_flat = freq_steady.permute(1, 0, 2).reshape(B, -1)  # [B, T_steady*Nc]

    # Per-trajectory metrics
    max_dev_hz = (freq_steady_flat - f0_hz).abs().max(dim=1)[0]
    mse_dev_hz2 = ((freq_steady_flat - f0_hz)**2).mean(dim=1)
    std_dev_hz = freq_steady_flat.std(dim=1)
    max_freq_per_traj = freq_steady_flat.max(dim=1)[0]
    min_freq_per_traj = freq_steady_flat.min(dim=1)[0]
    extreme_hz = torch.relu(max_freq_per_traj - 63.0) + torch.relu(57.0 - min_freq_per_traj)

    # Combined loss
    loss_batch = (2.0 * max_dev_hz / 1.0 +
                  1.0 * mse_dev_hz2 / 0.25 +
                  2.0 * std_dev_hz / 0.1 +
                  10.0 * extreme_hz / 1.0)

    if return_components:
        comps = {
            'freq_max_dev_hz': max_dev_hz.mean().item(),
            'freq_mse_dev_hz2': mse_dev_hz2.mean().item(),
            'freq_std_hz': std_dev_hz.mean().item(),
            'freq_extreme_hz': extreme_hz.mean().item(),
            'freq_min_in_batch': freq_steady.min().item(),
            'freq_max_in_batch': freq_steady.max().item()
        }
        return loss_batch, comps

    return loss_batch


# torch.compile disabled: causes graph breaks from .item() calls in compute_loss path
# @torch.compile(mode="reduce-overhead")
def _compute_loss_batch_fast(sim, t_vec, sol_batch, include_frequency,
                            freq_weight, include_action, action_weight):
    """Fast path - returns both loss and component tensors (no .item() calls)."""
    if sol_batch.dim() == 2:
        return compute_loss(sim, t_vec, sol_batch, include_frequency, freq_weight,
                          include_action, action_weight, verbose=False)

    if sol_batch is None:
        return torch.full((sol_batch.shape[1],), 1e6, dtype=sim.dtype, device=sim.device)

    batch_size = sol_batch.shape[1]
    Nc, Nt = sim.network.Nc, sim.network.Nt
    n_conv, n_line = 2 * Nc, 2 * Nt

    if sim.integrate_line_dynamics:
        vhat_indices = slice(0, n_conv)
        v_indices = slice(n_conv + n_line, 2*n_conv + n_line)
    else:
        vhat_indices = slice(0, n_conv)
        v_indices = slice(n_conv, 2*n_conv)

    vhat_sol = sol_batch[:, :, vhat_indices].reshape(-1, batch_size, Nc, 2)
    v_sol = sol_batch[:, :, v_indices].reshape(-1, batch_size, Nc, 2)

    vhat_mag = torch.norm(vhat_sol, dim=3)
    v_mag = torch.norm(v_sol, dim=3)

    v_star = torch.ones(Nc, dtype=sim.dtype, device=sim.device)
    v_star_expanded = v_star.unsqueeze(0).unsqueeze(0)

    vhat_dev = (vhat_mag - v_star_expanded) / v_star_expanded
    v_dev = (v_mag - v_star_expanded) / v_star_expanded

    # L_inf (max) norm per batch
    vhat_loss_inf = torch.max(torch.abs(vhat_dev).reshape(-1, batch_size, Nc).max(dim=2)[0], dim=0)[0]
    v_loss_inf = torch.max(torch.abs(v_dev).reshape(-1, batch_size, Nc).max(dim=2)[0], dim=0)[0]

    # L_2 (RMS) norm per batch
    vhat_loss_l2 = torch.sqrt(torch.mean(vhat_dev ** 2, dim=(0, 2)))
    v_loss_l2 = torch.sqrt(torch.mean(v_dev ** 2, dim=(0, 2)))

    num_steps = sol_batch.shape[0]
    latter_half_idx = num_steps // 2
    v_mag_latter = v_mag[latter_half_idx:, :, :]

    peak_to_peak = (torch.max(v_mag_latter, dim=0)[0] -
                    torch.min(v_mag_latter, dim=0)[0])
    oscillation_penalty = torch.max(peak_to_peak, dim=1)[0]

    voltage_losses = (3.0 * vhat_loss_inf + 1.0 * vhat_loss_l2 +
                      10.0 * v_loss_inf + 5.0 * v_loss_l2 +
                      0.1 * oscillation_penalty)

    # Store component tensors (NOT scalars)
    component_tensors = {
        'vhat_loss_inf': vhat_loss_inf.max(),
        'vhat_loss_l2': vhat_loss_l2.max(),
        'v_loss_inf': v_loss_inf.max(),
        'v_loss_l2': v_loss_l2.max(),
        'v_loss': v_loss_inf.max(),  # backward compat
        'oscillation_penalty': oscillation_penalty.max(),
        'voltage_loss_total': voltage_losses.max(),
        'voltage_loss_mean': voltage_losses.mean()
    }

    if include_frequency and num_steps > 10:
        # Use full sol_batch for dynamics-based frequency (vhat oscillator)
        freq_losses = _compute_frequency_loss_batch(sim, t_vec, sol_batch, return_components=False)
        freq_losses_weighted = freq_weight * freq_losses
        total_losses = voltage_losses + freq_losses_weighted
        component_tensors['freq_loss'] = freq_losses.mean()
        component_tensors['freq_loss_weighted'] = freq_losses_weighted.mean()
    else:
        total_losses = voltage_losses

    if include_action and num_steps > 10:
        action_losses = _compute_action_loss_simple_batch(sim, sol_batch, t_vec)
        action_losses_weighted = action_weight * action_losses
        total_losses = total_losses + action_losses_weighted
        component_tensors['action_loss'] = action_losses.mean()
        component_tensors['action_loss_max'] = action_losses.max()

    component_tensors['total_loss_mean'] = total_losses.mean()

    if sol_batch.requires_grad and not total_losses.requires_grad:
        total_losses = total_losses + 0.0 * sol_batch.sum()

    return total_losses, component_tensors


def compute_loss_batch(sim, t_vec, sol_batch, include_frequency=True, freq_weight=0.1,
                      include_action=True, action_weight=0.05,
                      include_power=False, power_weight=0.1, verbose=False):
    """
    Compute loss for batched solutions.

    Args:
        sim: MultiConverterSimulation instance
        t_vec: Time vector [T]
        sol_batch: Solution tensor [T, B, state_size]
        include_frequency: Include frequency loss term
        freq_weight: Weight for frequency loss
        include_action: Include action/control effort loss
        action_weight: Weight for action loss
        include_power: Include P,Q power tracking loss
        power_weight: Weight for power tracking loss (default 0.1)
        verbose: If True, return (losses, components_dict)

    Returns:
        losses [B] or (losses [B], components_dict)
    """
    losses, component_tensors = _compute_loss_batch_fast(
        sim, t_vec, sol_batch, include_frequency, freq_weight,
        include_action, action_weight
    )

    # Add power tracking loss if requested
    num_steps = sol_batch.shape[0]
    if include_power and num_steps > 10 and sim.integrate_line_dynamics:
        power_losses, power_comps = _compute_power_tracking_loss_batch(
            sim, sol_batch, t_vec, return_components=True
        )
        power_losses_weighted = power_weight * power_losses
        losses = losses + power_losses_weighted

        # Add to component tensors
        component_tensors['power_loss'] = power_losses.mean()
        component_tensors['power_loss_weighted'] = power_losses_weighted.mean()
        component_tensors['P_loss_inf'] = torch.tensor(power_comps['P_loss_inf'])
        component_tensors['Q_loss_inf'] = torch.tensor(power_comps['Q_loss_inf'])
        component_tensors['P_loss_l2'] = torch.tensor(power_comps['P_loss_l2'])
        component_tensors['Q_loss_l2'] = torch.tensor(power_comps['Q_loss_l2'])
        # Update total_loss_mean to include power
        component_tensors['total_loss_mean'] = losses.mean()

    if verbose:
        # Convert tensors to scalars only when verbose=True
        components = {k: v.item() if torch.is_tensor(v) else v
                    for k, v in component_tensors.items()}
        components['batch_size'] = sol_batch.shape[1]
        return losses, components
    else:
        return losses


def _compute_action_loss_simple_batch(sim, sol_batch, t_vec):
    """
    Fully vectorized simple action loss for batched trajectories.
    Penalizes control efforts to prevent gain explosion.

    FIXED: Uses scalar gains directly for proper gradient flow.
    """
    T, B = sol_batch.shape[0], sol_batch.shape[1]
    Nc = sim.network.Nc
    n_conv = 2 * Nc
    n_line = 2 * sim.network.Nt if sim.integrate_line_dynamics else 0

    # Skip initial transient
    skip_idx = max(1, int(0.2 * T))

    # Efficient state extraction
    slc = lambda a, b: slice(a, a + b)

    # Reshape all states
    vhat = sol_batch[..., slc(0, n_conv)].view(T, B, Nc, 2)

    if sim.integrate_line_dynamics:
        v = sol_batch[..., slc(n_conv + n_line, n_conv)].view(T, B, Nc, 2)
        zeta_v = sol_batch[..., slc(2*n_conv + n_line, n_conv)].view(T, B, Nc, 2)
        i_f = sol_batch[..., slc(3*n_conv + n_line, n_conv)].view(T, B, Nc, 2)
        zeta_f = sol_batch[..., slc(4*n_conv + n_line, n_conv)].view(T, B, Nc, 2)
    else:
        v = sol_batch[..., slc(n_conv, n_conv)].view(T, B, Nc, 2)
        zeta_v = sol_batch[..., slc(2*n_conv, n_conv)].view(T, B, Nc, 2)
        i_f = sol_batch[..., slc(3*n_conv, n_conv)].view(T, B, Nc, 2)
        zeta_f = sol_batch[..., slc(4*n_conv, n_conv)].view(T, B, Nc, 2)

    # Steady state only
    vhat_steady = vhat[skip_idx:]
    v_steady = v[skip_idx:]
    zeta_v_steady = zeta_v[skip_idx:]
    zeta_f_steady = zeta_f[skip_idx:]

    # Get active converter masks
    active_mask = torch.tensor([
        sim.converter.converter_states[c]['active'] for c in range(Nc)
    ], dtype=sim.dtype, device=sim.device)

    voltage_control_mask = torch.tensor([
        sim.converter.converter_states[c]['voltage_control'] for c in range(Nc)
    ], dtype=sim.dtype, device=sim.device)

    # FIXED: Use scalar gains directly (preserves gradient flow to sim.Kp_v etc.)
    # Scalars broadcast to [T_ss, B, Nc, 2]

    # Voltage control effort: u_v = Kp_v * e_v + Ki_v * zeta_v
    e_v = v_steady - vhat_steady  # [T_ss, B, Nc, 2]
    u_v = sim.Kp_v * e_v + sim.Ki_v * zeta_v_steady

    # Mask inactive converters
    active_v_mask = (active_mask * voltage_control_mask).view(1, 1, Nc, 1)
    u_v = u_v * active_v_mask

    # RMS over time, converters, and dq components -> [B]
    u_v_rms = torch.sqrt(torch.mean(u_v ** 2, dim=(0, 2, 3)) + 1e-8)

    # Current control effort: u_f = Kp_f * (approx) + Ki_f * zeta_f
    u_f = sim.Kp_f * zeta_f_steady * 0.1 + sim.Ki_f * zeta_f_steady

    active_f_mask = active_mask.view(1, 1, Nc, 1)
    u_f = u_f * active_f_mask

    u_f_rms = torch.sqrt(torch.mean(u_f ** 2, dim=(0, 2, 3)) + 1e-8)

    # Combined action penalty
    eref_noise = 0.01
    ki_noise_rms = torch.abs(sim.Ki_f * eref_noise)

    α, β, κ = 0.1, 0.1, 0.1
    γ, δ, κs = 1.0, 1.0, 1.0
    thr_u, thr_ki = 0.20, 0.002

    action_losses = (
        α * u_v_rms
        + γ * torch.relu(u_v_rms - thr_u)
        + β * u_f_rms
        + δ * torch.relu(u_f_rms - thr_u)
        + κ * ki_noise_rms
        + κs * torch.relu(ki_noise_rms - thr_ki)
    )
    return action_losses


def print_loss_components(components, epoch, scenario):
    """
    Print formatted loss components for monitoring.
    
    Args:
        components: Dictionary of loss components
        epoch: Current epoch number
        scenario: Current scenario name
    """
    header = f"\n--- Epoch {epoch} | Scenario: {scenario} | Loss Components ---"
    print(header)

    if not components:
        print("  No components to display.")
        return

    print(f"  Voltage Loss Total:  {components.get('voltage_loss_total', 0):.4f}")

    # Print action loss if available
    action_loss = components.get('action_loss', 0)
    if action_loss > 0:
        print(f"  Action Loss (mean): {action_loss:.4f}")
        print(f"  Action Loss (max):  {components.get('action_loss_max', 0):.4f}")

    # Print frequency loss if available
    if 'freq_loss' in components or 'freq_max_dev_hz' in components:
        print("  Frequency Stats:")
        if 'freq_loss' in components and components['freq_loss'] > 0:
            print(f"    Frequency Loss: {components['freq_loss']:.4f}")
            print(f"    Frequency Loss (weighted): {components.get('freq_loss_weighted', 0):.4f}")
        if 'freq_max_dev_hz' in components:
            print(f"    Max Deviation (Hz):   {components.get('freq_max_dev_hz', 0):.4f}")
            print(f"    Mean Squared Dev (Hz^2): {components.get('freq_mse_dev_hz2', 0):.4f}")
            print(f"    Std Dev (Hz):         {components.get('freq_std_hz', 0):.4f}")
            print(f"    Extreme Penalty (Hz): {components.get('freq_extreme_hz', 0):.4f}")
            print(f"    Freq Range in Batch:  [{components.get('freq_min_in_batch', 60):.2f}, {components.get('freq_max_in_batch', 60):.2f}] Hz")

    print(f"  ----------------------------------------------------")
    print(f"  AVG TOTAL LOSS (Perf.): {components.get('total_loss_mean', 0):.4f}")
    print("-" * len(header))


def _compute_power_tracking_loss_batch(sim, sol_batch, t_vec, return_components=False):
    """
    Compute power (P, Q) tracking loss for batched trajectories.

    Uses io = B @ i_line for output power computation.
    Normalized by setpoint magnitude to keep loss scale-invariant.

    Args:
        sim: MultiConverterSimulation instance
        sol_batch: Solution tensor [T, B, state_size]
        t_vec: Time vector [T]
        return_components: If True, return (loss, components_dict)

    Returns:
        Loss tensor [B] or (loss [B], components_dict)
    """
    T, B = sol_batch.shape[0], sol_batch.shape[1]
    Nc = sim.network.Nc
    n_conv = 2 * Nc
    n_line = 2 * sim.network.Nt

    # Need line currents for power computation
    if not sim.integrate_line_dynamics:
        zero_loss = torch.zeros(B, dtype=sim.dtype, device=sim.device)
        if return_components:
            return zero_loss, {}
        return zero_loss

    # Skip initial transient (20%) - same as action loss
    skip_idx = max(1, int(0.2 * T))

    # State indices for differential mode
    v_idx = n_conv + n_line
    i_line_idx = n_conv

    # Extract steady-state v and i_line
    v_sol = sol_batch[skip_idx:, :, v_idx:v_idx+n_conv]  # [T_ss, B, 2*Nc]
    i_line_sol = sol_batch[skip_idx:, :, i_line_idx:i_line_idx+n_line]  # [T_ss, B, 2*Nt]

    T_ss = v_sol.shape[0]

    # Compute output current: io = B @ i_line
    B_mat = sim.network.B  # [2*Nc, 2*Nt]
    io_sol = torch.einsum('ij,tbj->tbi', B_mat, i_line_sol)  # [T_ss, B, 2*Nc]

    # Reshape to [T_ss, B, Nc, 2]
    v_reshaped = v_sol.view(T_ss, B, Nc, 2)
    io_reshaped = io_sol.view(T_ss, B, Nc, 2)

    # Compute P, Q in per-unit (already in pu since states are in pu)
    # P = v_α * io_α + v_β * io_β
    # Q = v_β * io_α - v_α * io_β
    P_pu = v_reshaped[..., 0] * io_reshaped[..., 0] + v_reshaped[..., 1] * io_reshaped[..., 1]
    Q_pu = v_reshaped[..., 1] * io_reshaped[..., 0] - v_reshaped[..., 0] * io_reshaped[..., 1]

    # Get setpoints (per-unit) - use final setpoints after t_setpoint_change
    # Check for batch setpoints
    if hasattr(sim, 'batch_p_star') and sim.batch_p_star is not None:
        p_star = sim.batch_p_star  # [B, Nc]
        q_star = sim.batch_q_star
    else:
        # Fall back to converter setpoints
        p_star = sim.converter.setpoints.p_star  # [Nc]
        q_star = sim.converter.setpoints.q_star

    # Expand if needed
    if p_star.dim() == 1:
        p_star = p_star.unsqueeze(0).expand(B, -1)  # [B, Nc]
        q_star = q_star.unsqueeze(0).expand(B, -1)

    # Compute absolute errors
    P_err = P_pu - p_star.unsqueeze(0)  # [T_ss, B, Nc]
    Q_err = Q_pu - q_star.unsqueeze(0)

    # Normalize by setpoint scale (prevents large power = large loss)
    # Use mean of absolute setpoints, clamped to avoid division by small numbers
    P_scale = torch.abs(p_star).mean(dim=1, keepdim=True).clamp(min=0.05)  # [B, 1]
    Q_scale = torch.abs(q_star).mean(dim=1, keepdim=True).clamp(min=0.05)

    # Alternative: use max for better normalization when setpoints vary a lot
    # P_scale = torch.abs(p_star).max(dim=1, keepdim=True)[0].clamp(min=0.05)

    # Relative errors (dimensionless, ~O(1) when tracking well)
    P_rel_err = P_err / P_scale.unsqueeze(0)  # [T_ss, B, Nc]
    Q_rel_err = Q_err / Q_scale.unsqueeze(0)

    # --- Loss computation (mirrors voltage loss structure) ---

    # L_inf (max) over time and converters -> worst-case tracking error
    P_loss_inf = torch.abs(P_rel_err).view(T_ss * Nc, B).max(dim=0)[0]  # [B]
    Q_loss_inf = torch.abs(Q_rel_err).view(T_ss * Nc, B).max(dim=0)[0]

    # L_2 (RMS) over time and converters -> average tracking error
    P_loss_l2 = torch.sqrt((P_rel_err ** 2).mean(dim=(0, 2)) + 1e-8)  # [B]
    Q_loss_l2 = torch.sqrt((Q_rel_err ** 2).mean(dim=(0, 2)) + 1e-8)

    # Combined loss: L2 (RMS) only for smooth gradients
    power_loss = P_loss_l2 + Q_loss_l2
    if return_components:
        # Compute actual P, Q in SI for monitoring
        Sb = sim.network.pu.Sb
        P_final_si = P_pu[-1].mean(dim=1) * Sb  # [B] mean over converters
        Q_final_si = Q_pu[-1].mean(dim=1) * Sb

        comps = {
            'P_loss_inf': P_loss_inf.mean().item(),
            'P_loss_l2': P_loss_l2.mean().item(),
            'Q_loss_inf': Q_loss_inf.mean().item(),
            'Q_loss_l2': Q_loss_l2.mean().item(),
            'P_final_mean_W': P_final_si.mean().item(),
            'Q_final_mean_VAr': Q_final_si.mean().item(),
            'power_loss_mean': power_loss.mean().item()
        }
        return power_loss, comps

    return power_loss


def _compute_power_tracking_loss_simple(sim, sol, t_vec):
    """
    Compute power tracking loss for single trajectory.
    Wrapper around batch version.
    """
    sol_batch = sol.unsqueeze(1)  # [T, 1, state_size]
    loss = _compute_power_tracking_loss_batch(sim, sol_batch, t_vec)
    return loss.squeeze(0)


def get_frequency_stats_hz(sim, t_vec, sol):
    """
    Helper method to get frequency statistics in Hz for monitoring.

    Args:
        sim: MultiConverterSimulation instance
        t_vec: Time vector
        sol: Solution tensor
        
    Returns:
        dict: Contains min, max, mean frequency in Hz
    """
    Nc = sim.network.Nc
    n_conv = 2 * Nc

    # Extract voltages
    if sim.integrate_line_dynamics:
        v_indices = slice(n_conv + 2*sim.network.Nt, 2*n_conv + 2*sim.network.Nt)
    else:
        v_indices = slice(n_conv, 2*n_conv)

    v_sol = sol[:, v_indices].reshape(-1, Nc, 2)

    # Compute phase angles
    phases = torch.atan2(v_sol[:, :, 1], v_sol[:, :, 0])

    # Compute angular frequency
    omega = torch.zeros((len(phases)-1, Nc), dtype=sim.dtype, device=sim.device)
    for k in range(1, len(phases)):
        dt = (t_vec[k] - t_vec[k-1]).clamp(min=1e-8)
        phase_diff = phases[k] - phases[k-1]
        # Simple unwrapping
        phase_diff = torch.where(phase_diff > np.pi, phase_diff - 2*np.pi, phase_diff)
        phase_diff = torch.where(phase_diff < -np.pi, phase_diff + 2*np.pi, phase_diff)
        omega[k-1] = phase_diff / dt

    # Convert to Hz: f = ω_pu × 60 Hz
    freq_hz = omega * 60.0

    # Skip initial transient
    skip_idx = max(1, int(0.1 * len(omega)))
    freq_steady = freq_hz[skip_idx:]

    return {
        'min_hz': freq_steady.min().item(),
        'max_hz': freq_steady.max().item(),
        'mean_hz': freq_steady.mean().item(),
        'std_hz': freq_steady.std().item()
    }