"""
Multi-Converter Power System Simulation - OPTIMIZED VERSION.

Contains:
- MultiConverterSimulation: Main simulation class with ODE integration and batch support

OPTIMIZATIONS APPLIED:
1. Parameter sync with smart caching - rebuild only when parameters change
2. Scenario-level cache invalidation for load changes
3. Streamlined state update flow

GRADIENT SAFETY CRITICAL NOTES:
- Parameter sync MUST remain in forward pass (system_equations_*)
- The converter.rebuild_control_matrices() uses LIVE nn.Parameters
- Caching only affects WHEN rebuild happens, not WHAT values are used
- All gradient paths: Loss → ODE solution → system_equations → control matrices → nn.Parameters

The key insight is that during a single optimizer step:
1. Parameters are fixed (we're computing loss for current parameters)
2. ODE solver calls forward() many times (thousands of evaluations)
3. Parameters only change AFTER optimizer.step()

So within one forward pass, caching the Kronecker products is safe because
parameters don't change during ODE integration. The cache is invalidated
when parameters change (detected by rebuild_control_matrices).
"""

import math
import torch
import numpy as np
from torchdiffeq import odeint_adjoint, odeint
from typing import Tuple, Optional

from .core import PerUnitSystem, Setpoints, super_safe_solve
from .network import PowerSystemNetwork
from .converter import ConverterControl, ConverterState
from .local_schedulers import (
    LocalGainScheduler, LocalEtaScheduler, LocalEtaAAdapter,
    SupervisoryLayer, DistributedAdaptiveController
)

class MultiConverterSimulation(torch.nn.Module):
    """
    Complete multi-converter simulation with per-unit implementation and batch support.
    
    This is the main simulation class that orchestrates the power system simulation,
    including ODE integration, scenario management, and optimization support.
    
    Args:
        batch_size: Number of parallel trajectories for batch simulation
        device: Torch device ('cuda' or 'cpu')
        dtype: Torch data type
    """
    def __init__(self, batch_size=1, device='cuda', dtype=torch.float64):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.integrate_line_dynamics = False

        # Initialize network with batch support
        self.network = PowerSystemNetwork(batch_size=batch_size, device=device, dtype=dtype)

        # Convert to per-unit
        Vb = self.network.pu.Vb
        Ib = self.network.pu.Ib
        ωb = self.network.pu.ωb

        # Voltage control gains
        Kp_v_SI = 1
        Ki_v_SI = .01
        self.Kp_v = torch.nn.Parameter(
            torch.tensor(Kp_v_SI * self.network.pu.Zb, dtype=dtype, device=device)
        )
        self.Ki_v = torch.nn.Parameter(
            torch.tensor(Ki_v_SI * self.network.pu.Zb / ωb, dtype=dtype, device=device)
        )

        Kp_f_SI = 50
        Ki_f_SI = 0.01  # Must be >> lf_SI (0.001 H) for Condition 6 (ratio ~10 makes LHS small)
        self.Kp_f = torch.nn.Parameter(
            torch.tensor(Kp_f_SI / self.network.pu.Zb, dtype=dtype, device=device)
        )
        self.Ki_f = torch.nn.Parameter(
            torch.tensor(Ki_f_SI / (self.network.pu.Zb * self.network.pu.ωb), dtype=dtype, device=device)
        )

        eta_SI = 9.66
        eta_a_SI = 4.6256

        # Convert to per-unit
        self.eta = torch.nn.Parameter(
            torch.tensor(eta_SI / (self.network.pu.Zb * self.network.pu.ωb), dtype=dtype, device=device)
        )
        self.eta_a = torch.nn.Parameter(
            torch.tensor(eta_a_SI * self.network.pu.Zb, dtype=dtype, device=device)
        )

        # Lagrange multipliers
        self.lambda_cond4 = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))
        self.lambda_cond5 = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))
        self.lambda_cond6 = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))

        # Initialize converter control
        self.converter = ConverterControl(
            self.network,
            {'eta': self.eta, 'eta_a': self.eta_a,
             'Kp_v': self.Kp_v, 'Ki_v': self.Ki_v,
             'Kp_f': self.Kp_f, 'Ki_f': self.Ki_f}
        )
        self.state_handler = ConverterState(self.converter)

        # Simulation parameters
        self.dt = 0.01
        self.T_sim = 0.1

        # Scenario handling
        self.scenario = "black_start"
        self.disturbance = None
        self.disturbance_applied = False
        self.disturbance_time = 0.05

        # Equilibrium tracking
        self.default_equilibrium_target = None
        self.scenario_equilibrium_targets = {}

        # Constraint cache
        self._constraint_cache = {}

        # Store original load value
        self.original_rL = self.network.rL

        # =====================================================================
        # Adaptive Control Configuration (Local/Distributed Architecture)
        # =====================================================================
        self.enable_adaptive_control = True  # Set to False to disable

        # Bounds for adaptive parameters (registered as buffers for device handling)
        self.register_buffer("eta_min", torch.tensor(1e-4, dtype=dtype, device=device))
        self.register_buffer("eta_a_min", torch.tensor(0.01, dtype=dtype, device=device))
        self.register_buffer("eta_a_max", torch.tensor(100.0, dtype=dtype, device=device))

        # Initialize distributed adaptive controller
        Nc = self.network.Nc
        if self.enable_adaptive_control:
            self.adaptive_controller = DistributedAdaptiveController(
                n_converters=Nc,
                hidden_dim=32,
                share_weights=True,  # Shared weights = scalable O(1) parameters
                theta=0.5,           # Safety factor for passivity
                device=device,
                dtype=dtype
            )
        else:
            self.adaptive_controller = None

        # Storage for scheduled eta (preserves gradient graph for constraint training)
        # This allows Lagrangian penalties to train scheduler networks, not just sim.eta
        self._scheduled_eta_for_constraints = None

    def _sync_converter_params(self):
        """
        Sync nn.Parameters to converter control.
        
        GRADIENT SAFETY: This syncs the REFERENCES to nn.Parameters.
        The rebuild_control_matrices() uses these live parameters,
        ensuring gradients flow correctly through the Kronecker products.
        
        The caching in rebuild_control_matrices() only determines WHETHER
        to rebuild (based on parameter values), not WHAT values to use.
        """
        self.converter.eta = self.eta
        self.converter.eta_a = self.eta_a
        self.converter.Kp_v = self.Kp_v
        self.converter.Ki_v = self.Ki_v
        self.converter.Kp_f = self.Kp_f
        self.converter.Ki_f = self.Ki_f
        
        # This will check if rebuild is needed (using cached param check)
        # and only rebuild if parameters have actually changed
        self.converter.rebuild_control_matrices()

    def configure_anti_windup(self, enabled: bool = True, 
                              vm_limit: float = 1.15,
                              vm_sharpness: float = 50.0,
                              i_ref_limit: float = 1.5,
                              i_ref_sharpness: float = 20.0,
                              Tt_current: float = None,
                              Tt_voltage: float = None):
        """
        Configure anti-windup parameters.
        
        Args:
            enabled: Enable/disable anti-windup
            vm_limit: Modulation voltage limit (p.u.)
            vm_sharpness: Saturation sharpness (higher = sharper transition)
            i_ref_limit: Current reference limit (p.u.)
            i_ref_sharpness: Current ref saturation sharpness
            Tt_current: Tracking time constant for current loop (None = 1/Ki_f optimal)
            Tt_voltage: Tracking time constant for voltage loop (None = 1/Ki_v optimal)
        """
        self.converter.anti_windup_enabled = enabled
        self.converter.vm_limit = vm_limit
        self.converter.vm_sat_sharpness = vm_sharpness
        self.converter.i_ref_limit = i_ref_limit
        self.converter.i_ref_sat_sharpness = i_ref_sharpness
        self.converter.Tt_current = Tt_current
        self.converter.Tt_voltage = Tt_voltage
        
        print(f"Anti-windup configuration:")
        print(f"  Enabled: {enabled}")
        print(f"  VM limit: {vm_limit} p.u.")
        print(f"  VM sharpness: {vm_sharpness}")
        print(f"  I_ref limit: {i_ref_limit} p.u.")
        print(f"  Tt_current: {Tt_current if Tt_current else '1/Ki_f (optimal)'}")
        print(f"  Tt_voltage: {Tt_voltage if Tt_voltage else '1/Ki_v (optimal)'}")

    def get_state_size(self) -> int:
        """
        Get total state size.

        With adaptive control, each converter has its own η_{a,k} state.
        """
        Nc = self.network.Nc
        Nt = self.network.Nt
        n_conv = 2 * Nc

        base_size = 5 * n_conv  # [v̂, v, ζ_v, i_f, ζ_f]

        if self.integrate_line_dynamics:
            base_size += 2 * Nt  # Add i_line

        if self.enable_adaptive_control:
            base_size += Nc  # Add η_a per converter

        return base_size

    def extract_eta_a_states(self, state: torch.Tensor) -> torch.Tensor:
        """Extract η_a states [Nc] from end of state vector."""
        Nc = self.network.Nc
        if self.enable_adaptive_control:
            if state.dim() == 1:
                return state[-Nc:]
            else:
                return state[:, -Nc:]
        else:
            # Return fixed parameter values
            if state.dim() == 1:
                return self.eta_a.expand(Nc)
            else:
                return self.eta_a.expand(state.shape[0], Nc)

    def extract_core_state(self, state: torch.Tensor) -> torch.Tensor:
        """Extract core state (everything except η_a)."""
        Nc = self.network.Nc
        if self.enable_adaptive_control:
            if state.dim() == 1:
                return state[:-Nc]
            else:
                return state[:, :-Nc]
        else:
            return state

    def clear_adaptive_cache(self):
        """Clear supervisor cache and scheduled eta when operating point changes."""
        if self.adaptive_controller is not None:
            self.adaptive_controller.invalidate_cache()
        # Clear scheduled eta to force recomputation on next forward pass
        self._scheduled_eta_for_constraints = None

    def get_eta_for_constraints(self) -> torch.Tensor:
        """
        Return eta for constraint checking (Conditions 4, 5, 6).

        Returns scheduled eta (with gradients) if available, enabling
        Lagrangian penalties to train scheduler networks directly.
        Falls back to sim.eta when adaptive control is disabled or
        before the first ODE forward pass.

        For stability conditions, we use min(scheduled_eta) as the
        conservative value (worst-case for stability analysis).
        """
        if (self.enable_adaptive_control and
            self._scheduled_eta_for_constraints is not None):
            scheduled = self._scheduled_eta_for_constraints
            # Return min for conservative constraint check (worst-case for stability)
            if scheduled.dim() > 0 and scheduled.numel() > 1:
                return torch.min(scheduled)
            return scheduled
        return self.eta

    def forward(self, t: torch.Tensor, state: torch.Tensor):
        """
        ODE forward pass - handles batch dimension.
        
        Args:
            t: Current time
            state: System state [state_size] or [batch, state_size]
            
        Returns:
            Time derivative of state
        """
        # Disturbance injection
        if (self.scenario == "disturbance" and
            not self.disturbance_applied and
            t >= self.disturbance_time):
            Nc = self.network.Nc
            n_conv = 2 * Nc
            i_f_off = 3*n_conv if not self.integrate_line_dynamics else 3*n_conv + 2*self.network.Nt
            state = state.clone()

            if state.dim() == 1:
                state[i_f_off:i_f_off+2*Nc] += self.disturbance
            else:
                state[:, i_f_off:i_f_off+2*Nc] += self.disturbance.unsqueeze(0)

            self.disturbance_applied = True

        # Choose dynamics based on state dimension
        if state.dim() == 1:
            if self.integrate_line_dynamics:
                return self.system_equations_differential(t, state)
            else:
                return self.system_equations_algebraic(t, state)
        else:
            if self.integrate_line_dynamics:
                return self.system_equations_differential_batch(t, state)
            else:
                return self.system_equations_algebraic_batch(t, state)

    def system_equations_algebraic(self, t, state):
        """
        System dynamics with algebraic line currents.

        MODIFIED for local adaptive control with per-converter η_{a,k}.

        State layout: [v̂, v, ζ_v, i_f, ζ_f, η_a₁, ..., η_aₙ]
        """
        with torch.no_grad():
            self.state_handler.update_states(float(t))

        Nc = self.network.Nc
        n_conv = 2 * Nc

        # Extract η_a states if adaptive control enabled
        if self.enable_adaptive_control:
            eta_a_states = state[-Nc:]  # [Nc]
            core_state = state[:-Nc]
        else:
            eta_a_states = self.eta_a.expand(Nc)
            core_state = state

        # Unpack core state
        vhat = core_state[0:n_conv]
        v = core_state[n_conv:2*n_conv]
        zeta_v = core_state[2*n_conv:3*n_conv]
        i_f = core_state[3*n_conv:4*n_conv]
        zeta_f = core_state[4*n_conv:5*n_conv]

        # Schedule parameters using distributed controller
        if self.enable_adaptive_control:
            scheduled = self.adaptive_controller.schedule_all_converters(
                vhat=vhat,
                eta_a_states=eta_a_states,
                setpoints=self.converter.setpoints,
                rL=self.network.rL,
                gains_base={
                    'Kp_v': self.Kp_v,
                    'Ki_v': self.Ki_v,
                    'Kp_f': self.Kp_f,
                    'Ki_f': self.Ki_f,
                    'eta': self.eta
                },
                cf=self.converter.cf,
                lf=self.converter.lf,
                network=self.network,
                v_actual=v  # Pass actual voltage for adaptation signal
            )

            # Get per-converter parameters
            Kp_v_all = scheduled['Kp_v']  # [Nc]
            Ki_v_all = scheduled['Ki_v']  # [Nc]
            Kp_f_all = scheduled['Kp_f']  # [Nc]
            Ki_f_all = scheduled['Ki_f']  # [Nc]
            eta_all = scheduled['eta']    # [Nc]
            d_eta_a = scheduled['eta_a_dot']  # [Nc]

            # Store scheduled eta for constraint checking (preserves gradient graph)
            self._scheduled_eta_for_constraints = eta_all
        else:
            # Use fixed parameters
            Kp_v_all = self.Kp_v.expand(Nc)
            Ki_v_all = self.Ki_v.expand(Nc)
            Kp_f_all = self.Kp_f.expand(Nc)
            Ki_f_all = self.Ki_f.expand(Nc)
            eta_all = self.eta.expand(Nc)
            d_eta_a = torch.zeros(Nc, dtype=self.dtype, device=self.device)

        # Compute algebraic line currents
        i_line, v_common = self.network.compute_algebraic_line_currents(v)

        # Converter dynamics with PER-CONVERTER gains
        dvhat_list, dv_list, dzeta_v_list, dif_list, dzeta_f_list = [], [], [], [], []

        for idx in range(Nc):
            # Update converter parameters for THIS converter
            self.converter.eta = eta_all[idx]
            self.converter.eta_a = eta_a_states[idx]
            self.converter.Kp_v = Kp_v_all[idx]
            self.converter.Ki_v = Ki_v_all[idx]
            self.converter.Kp_f = Kp_f_all[idx]
            self.converter.Ki_f = Ki_f_all[idx]
            self.converter.rebuild_control_matrices()

            dvh, dv, dzv, dif, dzf = self.converter.compute_converter_dynamics(
                idx, torch.cat([vhat, v, zeta_v, i_f, zeta_f], 0),
                self.converter.setpoints, i_line
            )
            dvhat_list.append(dvh)
            dv_list.append(dv)
            dzeta_v_list.append(dzv)
            dif_list.append(dif)
            dzeta_f_list.append(dzf)

        dx_core = torch.cat([
            torch.cat(dvhat_list),
            torch.cat(dv_list),
            torch.cat(dzeta_v_list),
            torch.cat(dif_list),
            torch.cat(dzeta_f_list)
        ], 0)

        # Append η_a dynamics for all converters
        if self.enable_adaptive_control:
            return torch.cat([dx_core, d_eta_a])
        else:
            return dx_core

    def _update_states_no_compile(self, t):
        """Non-compiled state updates."""
        with torch.no_grad():
            self.state_handler.update_states(float(t))

    def system_equations_algebraic_batch(self, t, state_batch):
        """System dynamics - batch version with per-converter adaptive control."""
        self._update_states_no_compile(t)

        batch_size = state_batch.shape[0]
        Nc = self.network.Nc
        n_conv = 2 * Nc

        # Extract η_a states
        if self.enable_adaptive_control:
            eta_a_states = state_batch[:, -Nc:]  # [batch, Nc]
            core_state = state_batch[:, :-Nc]
        else:
            # Expand scalar eta_a to [batch, Nc]
            eta_a_states = torch.ones(batch_size, Nc, dtype=self.dtype, device=self.device) * self.eta_a
            core_state = state_batch

        # Unpack batched state
        vhat = core_state[:, 0:n_conv]
        v = core_state[:, n_conv:2*n_conv]
        zeta_v = core_state[:, 2*n_conv:3*n_conv]
        i_f = core_state[:, 3*n_conv:4*n_conv]
        zeta_f = core_state[:, 4*n_conv:5*n_conv]

        # Schedule gains - expand to [batch, Nc] format for unified handling
        if self.enable_adaptive_control:
            scheduled = self.adaptive_controller.schedule_all_converters(
                vhat=vhat[0],
                eta_a_states=eta_a_states[0],
                setpoints=self.converter.setpoints,
                rL=self.network.rL,
                gains_base={
                    'Kp_v': self.Kp_v,
                    'Ki_v': self.Ki_v,
                    'Kp_f': self.Kp_f,
                    'Ki_f': self.Ki_f,
                    'eta': self.eta
                },
                cf=self.converter.cf,
                lf=self.converter.lf,
                network=self.network,
                v_actual=v[0]
            )
            # Expand to [batch, Nc] for unified handling
            Kp_v_batch = scheduled['Kp_v'].unsqueeze(0).expand(batch_size, -1)
            Ki_v_batch = scheduled['Ki_v'].unsqueeze(0).expand(batch_size, -1)
            Kp_f_batch = scheduled['Kp_f'].unsqueeze(0).expand(batch_size, -1)
            Ki_f_batch = scheduled['Ki_f'].unsqueeze(0).expand(batch_size, -1)
            eta_batch = scheduled['eta'].unsqueeze(0).expand(batch_size, -1)
            d_eta_a = scheduled['eta_a_dot'].unsqueeze(0).expand(batch_size, -1)

            # Store scheduled eta for constraint checking (preserves gradient graph)
            self._scheduled_eta_for_constraints = scheduled['eta']
        else:
            # Expand scalar parameters to [batch, Nc]
            ones_batch_nc = torch.ones(batch_size, Nc, dtype=self.dtype, device=self.device)
            Kp_v_batch = ones_batch_nc * self.Kp_v
            Ki_v_batch = ones_batch_nc * self.Ki_v
            Kp_f_batch = ones_batch_nc * self.Kp_f
            Ki_f_batch = ones_batch_nc * self.Ki_f
            eta_batch = ones_batch_nc * self.eta
            d_eta_a = torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)

        # Compute algebraic line currents for batch
        i_line, v_common = self.network.compute_algebraic_line_currents(v)

        # Initialize derivative storage
        dvhat_all = torch.zeros_like(vhat)
        dv_all = torch.zeros_like(v)
        dzeta_v_all = torch.zeros_like(zeta_v)
        dif_all = torch.zeros_like(i_f)
        dzeta_f_all = torch.zeros_like(zeta_f)

        # Build full state once
        full_state_batch = torch.cat([vhat, v, zeta_v, i_f, zeta_f], dim=1)

        for idx in range(Nc):
            # Set per-batch gains for this converter: [batch] shaped tensors
            self.converter.eta = eta_batch[:, idx]
            self.converter.eta_a = eta_a_states[:, idx]
            self.converter.Kp_v = Kp_v_batch[:, idx]
            self.converter.Ki_v = Ki_v_batch[:, idx]
            self.converter.Kp_f = Kp_f_batch[:, idx]
            self.converter.Ki_f = Ki_f_batch[:, idx]

            dvh, dv, dzv, dif, dzf = self.converter.compute_converter_dynamics(
                idx, full_state_batch, self.converter.setpoints, i_line
            )
            idx_slice = slice(2*idx, 2*(idx+1))
            dvhat_all[:, idx_slice] = dvh
            dv_all[:, idx_slice] = dv
            dzeta_v_all[:, idx_slice] = dzv
            dif_all[:, idx_slice] = dif
            dzeta_f_all[:, idx_slice] = dzf

        dx_core = torch.cat([dvhat_all, dv_all, dzeta_v_all, dif_all, dzeta_f_all], dim=1)

        if self.enable_adaptive_control:
            return torch.cat([dx_core, d_eta_a], dim=1)
        else:
            return dx_core

    def system_equations_differential(self, t, state):
        """System dynamics with differential line currents - adaptive version."""
        with torch.no_grad():
            self.state_handler.update_states(float(t))

        Nc = self.network.Nc
        Nt = self.network.Nt
        n_conv = 2 * Nc
        n_line = 2 * Nt

        # Extract η_a states
        if self.enable_adaptive_control:
            eta_a_states = state[-Nc:]
            core_state = state[:-Nc]
        else:
            eta_a_states = self.eta_a.expand(Nc)
            core_state = state

        # Unpack state [vhat, i_line, v, zeta_v, i_f, zeta_f]
        vhat = core_state[0:n_conv]
        i_line = core_state[n_conv:n_conv+n_line]
        v = core_state[n_conv+n_line:2*n_conv+n_line]
        zeta_v = core_state[2*n_conv+n_line:3*n_conv+n_line]
        i_f = core_state[3*n_conv+n_line:4*n_conv+n_line]
        zeta_f = core_state[4*n_conv+n_line:5*n_conv+n_line]

        # Schedule parameters
        if self.enable_adaptive_control:
            scheduled = self.adaptive_controller.schedule_all_converters(
                vhat=vhat,
                eta_a_states=eta_a_states,
                setpoints=self.converter.setpoints,
                rL=self.network.rL,
                gains_base={
                    'Kp_v': self.Kp_v,
                    'Ki_v': self.Ki_v,
                    'Kp_f': self.Kp_f,
                    'Ki_f': self.Ki_f,
                    'eta': self.eta
                },
                cf=self.converter.cf,
                lf=self.converter.lf,
                network=self.network,
                v_actual=v  # Pass actual voltage for adaptation signal
            )
            Kp_v_all = scheduled['Kp_v']
            Ki_v_all = scheduled['Ki_v']
            Kp_f_all = scheduled['Kp_f']
            Ki_f_all = scheduled['Ki_f']
            eta_all = scheduled['eta']
            d_eta_a = scheduled['eta_a_dot']

            # Store scheduled eta for constraint checking (preserves gradient graph)
            self._scheduled_eta_for_constraints = eta_all
        else:
            Kp_v_all = self.Kp_v.expand(Nc)
            Ki_v_all = self.Ki_v.expand(Nc)
            Kp_f_all = self.Kp_f.expand(Nc)
            Ki_f_all = self.Ki_f.expand(Nc)
            eta_all = self.eta.expand(Nc)
            d_eta_a = torch.zeros(Nc, dtype=self.dtype, device=self.device)

        # Line current dynamics (unchanged)
        di_line = self.network.line_dynamics(v, i_line)

        # Converter dynamics with per-converter gains
        dvhat_list, dv_list, dzeta_v_list, dif_list, dzeta_f_list = [], [], [], [], []

        for idx in range(Nc):
            self.converter.eta = eta_all[idx]
            self.converter.eta_a = eta_a_states[idx]
            self.converter.Kp_v = Kp_v_all[idx]
            self.converter.Ki_v = Ki_v_all[idx]
            self.converter.Kp_f = Kp_f_all[idx]
            self.converter.Ki_f = Ki_f_all[idx]
            self.converter.rebuild_control_matrices()

            full_state = torch.cat([vhat, i_line, v, zeta_v, i_f, zeta_f], 0)
            dvh, dv, dzv, dif, dzf = self.converter.compute_converter_dynamics(
                idx, torch.cat([vhat, v, zeta_v, i_f, zeta_f], 0),
                self.converter.setpoints, i_line
            )
            dvhat_list.append(dvh)
            dv_list.append(dv)
            dzeta_v_list.append(dzv)
            dif_list.append(dif)
            dzeta_f_list.append(dzf)

        dx_core = torch.cat([
            torch.cat(dvhat_list),
            di_line,
            torch.cat(dv_list),
            torch.cat(dzeta_v_list),
            torch.cat(dif_list),
            torch.cat(dzeta_f_list)
        ], 0)

        if self.enable_adaptive_control:
            return torch.cat([dx_core, d_eta_a])
        else:
            return dx_core

    def system_equations_differential_batch(self, t, state_batch):
        """
        System dynamics with differential line currents - batch processing.

        GRADIENT FLOW: Parameters synced here for correct gradient computation.
        State layout: [vhat, i_line, v, zeta_v, i_f, zeta_f, η_a₁...η_aₙ]

        Supports per-batch setpoints after t_setpoint_change when batch_p_star is set.
        """
        # Convert time to float for comparison
        t_float = float(t.detach().item()) if hasattr(t, 'item') else float(t)

        with torch.no_grad():
            self.state_handler.update_states(t_float)

        batch_size = state_batch.shape[0]
        Nc = self.network.Nc
        Nt = self.network.Nt
        n_conv = 2 * Nc
        n_line = 2 * Nt

        # Check if we should use per-batch setpoints (final setpoints)
        # batch_p_star contains FINAL setpoints per load factor
        # Only use them AFTER t_setpoint_change
        rL_is_batched = isinstance(self.network.rL, torch.Tensor) and self.network.rL.dim() > 0
        has_batch_setpoints = hasattr(self, 'batch_p_star') and self.batch_p_star is not None
        after_setpoint_change = t_float >= self.state_handler.t_setpoint_change

        use_batch_setpoints = has_batch_setpoints and after_setpoint_change

        # Extract η_a states if adaptive control enabled
        if self.enable_adaptive_control:
            eta_a_states = state_batch[:, -Nc:]  # [batch, Nc]
            core_state = state_batch[:, :-Nc]
        else:
            # Expand scalar eta_a to [batch, Nc]
            eta_a_states = torch.ones(batch_size, Nc, dtype=self.dtype, device=self.device) * self.eta_a
            core_state = state_batch

        # Unpack batched core state
        vhat = core_state[:, 0:n_conv]
        i_line = core_state[:, n_conv:n_conv+n_line]
        v = core_state[:, n_conv+n_line:2*n_conv+n_line]
        zeta_v = core_state[:, 2*n_conv+n_line:3*n_conv+n_line]
        i_f = core_state[:, 3*n_conv+n_line:4*n_conv+n_line]
        zeta_f = core_state[:, 4*n_conv+n_line:5*n_conv+n_line]

        # Import Setpoints for creating batch-specific setpoints
        from .core import Setpoints

        # Schedule gains - per-batch scheduling for correct d_eta_a
        if self.enable_adaptive_control:
            rL_is_batched = isinstance(self.network.rL, torch.Tensor) and self.network.rL.dim() > 0

            if rL_is_batched:
                if use_batch_setpoints:
                    batch_p_star = self.batch_p_star
                    batch_q_star = self.batch_q_star
                    batch_v_star = self.batch_v_star
                else:
                    # BEFORE setpoint change: expand base setpoints to batch
                    base_sp = self.converter.setpoints
                    batch_p_star = base_sp.p_star.unsqueeze(0).expand(batch_size, -1)
                    batch_q_star = base_sp.q_star.unsqueeze(0).expand(batch_size, -1)
                    batch_v_star = base_sp.v_star.unsqueeze(0).expand(batch_size, -1)




                # VECTORIZED BATCH SCHEDULING: Single call handles all batches
                scheduled = self.adaptive_controller.schedule_all_converters_batched(
                    vhat=vhat,  # [batch, 2*Nc]
                    eta_a_states=eta_a_states,  # [batch, Nc]
                    batch_p_star=self.batch_p_star,  # [batch, Nc]
                    batch_q_star=self.batch_q_star,
                    batch_v_star=self.batch_v_star,
                    rL_batch=self.network.rL,  # [batch]
                    gains_base={
                        'Kp_v': self.Kp_v,
                        'Ki_v': self.Ki_v,
                        'Kp_f': self.Kp_f,
                        'Ki_f': self.Ki_f,
                        'eta': self.eta
                    },
                    cf=self.converter.cf,
                    lf=self.converter.lf,
                    network=self.network,
                    v_actual=v  # [batch, 2*Nc]
                )
                Kp_v_batch = scheduled['Kp_v']  # [batch, Nc]
                Ki_v_batch = scheduled['Ki_v']
                Kp_f_batch = scheduled['Kp_f']
                Ki_f_batch = scheduled['Ki_f']
                eta_batch = scheduled['eta']
                d_eta_a = scheduled['eta_a_dot']

                # Store scheduled eta for constraint checking (use first batch, all share same scheduler)
                self._scheduled_eta_for_constraints = eta_batch[0]

                # Flag for per-batch gains
                use_per_batch_gains = True
            else:
                # SHARED SCHEDULING: Use first batch element
                rL_for_sched = self.network.rL[0] if rL_is_batched else self.network.rL
                scheduled = self.adaptive_controller.schedule_all_converters(
                    vhat=vhat[0],
                    eta_a_states=eta_a_states[0],
                    setpoints=self.converter.setpoints,
                    rL=rL_for_sched,
                    gains_base={
                        'Kp_v': self.Kp_v,
                        'Ki_v': self.Ki_v,
                        'Kp_f': self.Kp_f,
                        'Ki_f': self.Ki_f,
                        'eta': self.eta
                    },
                    cf=self.converter.cf,
                    lf=self.converter.lf,
                    network=self.network,
                    v_actual=v[0]
                )
                Kp_v_batch = scheduled['Kp_v'].unsqueeze(0).expand(batch_size, -1)  # [batch, Nc]
                Ki_v_batch = scheduled['Ki_v'].unsqueeze(0).expand(batch_size, -1)
                Kp_f_batch = scheduled['Kp_f'].unsqueeze(0).expand(batch_size, -1)
                Ki_f_batch = scheduled['Ki_f'].unsqueeze(0).expand(batch_size, -1)
                eta_batch = scheduled['eta'].unsqueeze(0).expand(batch_size, -1)
                d_eta_a = scheduled['eta_a_dot'].unsqueeze(0).expand(batch_size, -1)

                # Store scheduled eta for constraint checking (preserves gradient graph)
                self._scheduled_eta_for_constraints = scheduled['eta']

                use_per_batch_gains = False  # Same gains for all batches (but still in batch format)
        else:
            # No adaptive control - use base parameters expanded to batch format
            # Scalars need to be expanded to [batch, Nc] - use ones() * scalar
            ones_batch_nc = torch.ones(batch_size, Nc, dtype=self.dtype, device=self.device)
            Kp_v_batch = ones_batch_nc * self.Kp_v
            Ki_v_batch = ones_batch_nc * self.Ki_v
            Kp_f_batch = ones_batch_nc * self.Kp_f
            Ki_f_batch = ones_batch_nc * self.Ki_f
            eta_batch = ones_batch_nc * self.eta
            d_eta_a = torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)
            use_per_batch_gains = False

        # Line dynamics for batch
        di_line = self.network.line_dynamics(v, i_line)

        # Prepare batched setpoints [batch, Nc]
        if use_batch_setpoints:
            batched_setpoints = Setpoints(
                p_star=self.batch_p_star,      # [batch, Nc]
                q_star=self.batch_q_star,      # [batch, Nc]
                v_star=self.batch_v_star,      # [batch, Nc]
                theta_star=torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)
            )
        else:
            # Expand base setpoints to [batch, Nc]
            base_sp = self.converter.setpoints
            batched_setpoints = Setpoints(
                p_star=base_sp.p_star.unsqueeze(0).expand(batch_size, -1),
                q_star=base_sp.q_star.unsqueeze(0).expand(batch_size, -1),
                v_star=base_sp.v_star.unsqueeze(0).expand(batch_size, -1),
                theta_star=torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)
            )

        # Build full state once
        full_state_batch = torch.cat([vhat, v, zeta_v, i_f, zeta_f], dim=1)

        # Vectorized: compute ALL converters at once (no loop)
        gains_batch = {
            'Kp_v': Kp_v_batch,  # [batch, Nc]
            'Ki_v': Ki_v_batch,
            'Kp_f': Kp_f_batch,
            'Ki_f': Ki_f_batch,
            'eta': eta_batch,
        }

        dvhat_all, dv_all, dzeta_v_all, dif_all, dzeta_f_all = \
            self.converter.compute_all_converters_vectorized(
                full_state=full_state_batch,
                setpoints=batched_setpoints,
                i_line=i_line,
                gains_batch=gains_batch,
                eta_a_batch=eta_a_states,
            )

        # Stack all derivatives including η_a
        dx_core = torch.cat([dvhat_all, di_line, dv_all, dzeta_v_all, dif_all, dzeta_f_all], dim=1)

        if self.enable_adaptive_control:
            return torch.cat([dx_core, d_eta_a], dim=1)
        else:
            return dx_core

    def initialize_state(self, scenario: str):
        """
        Initialize state for scenario.
        
        Args:
            scenario: Scenario name
            
        Returns:
            Initial state tensor
        """
        if scenario == "black_start":
            return self.initialize_black_start()
        elif scenario in ["load_change", "setpoint_change", "disturbance"]:
            return self.initialize_from_equilibrium()
        else:
            return self.initialize_from_equilibrium()

    def initialize_black_start(self):
        """Initialize for black start with per-converter η_a."""
        Nc = self.network.Nc
        n_conv = 2 * Nc

        if self.enable_adaptive_control:
            state_size = 5 * n_conv + Nc  # Add Nc η_a values
        else:
            state_size = 5 * n_conv

        x0 = torch.zeros(state_size, dtype=self.dtype, device=self.device)

        # Small initial voltage perturbation
        for i in range(min(2, Nc)):
            idx_vhat = 2 * i
            x0[idx_vhat:idx_vhat+2] = torch.tensor([0.01/120, 0.0],
                                                    dtype=self.dtype, device=self.device)

        # Initialize all η_a to base parameter value
        if self.enable_adaptive_control:
            # Use eta_a directly (no detach) so it can be trained
            x0[-Nc:] = self.eta_a.expand(Nc)

        return x0
    def initialize_from_equilibrium(self):
        """Return an initial state in *pu* that matches the expected layout."""
        Nc = self.network.Nc
        n_conv = 2 * Nc          # 6
        n_line = 2 * self.network.Nt  # 6

        if self.integrate_line_dynamics:
            base_size = 5 * n_conv + n_line     # 36
        else:
            base_size = 5 * n_conv              # 30

        # Add η_a states if adaptive control is enabled
        if self.enable_adaptive_control:
            state_size = base_size + Nc
        else:
            state_size = base_size

        # -------- raw equilibrium in SI (length 36) --------------------------
        equilibrium_si = torch.tensor([ 1.19999761e+02,  1.47205573e-02,  1.19999364e+02,  1.31420683e-02,
        1.19999766e+02,  1.27254732e-02,  3.59544743e-01,  6.03823651e-03,
        3.42582820e-01,  4.64092451e-05,  3.41196500e-01, -6.19499773e-03,
        1.19999789e+02,  1.47204781e-02,  1.19999393e+02,  1.31419874e-02,
        1.19999794e+02,  1.27253910e-02, -2.86868963e-03, -2.20470373e-05,
       -2.86834341e-03, -2.18902413e-05, -2.86832572e-03, -2.17560862e-05,
        3.59411556e-01,  1.09177105e+00,  3.42463914e-01,  1.08577564e+00,
        3.41081364e-01,  1.07953786e+00, -7.18813876e-06, -2.18351406e-05,
       -6.84918951e-06, -2.17152338e-05, -6.82153811e-06, -2.15904794e-05], dtype=self.dtype, device=self.device)
        pu = self.network.pu
        x_pu = torch.zeros(state_size, dtype=self.dtype, device=self.device)

        # -------- common blocks ----------------------------------------------
        # v̂ (0-5)
        x_pu[0:6] = pu.to_pu(equilibrium_si[0:6], "voltage")

        if self.integrate_line_dynamics:
            # i_line (6-11)
            x_pu[6:12] = pu.to_pu(equilibrium_si[6:12], "current")

            # v (12-17)
            x_pu[12:18] = pu.to_pu(equilibrium_si[12:18], "voltage")

            # ζ_v (18-23)   V·s ➜ pu
            x_pu[18:24] = equilibrium_si[18:24] * pu.ωb / pu.Vb

            # i_f (24-29)
            x_pu[24:30] = pu.to_pu(equilibrium_si[24:30], "current")

            # ζ_f (30-35)   A·s ➜ pu
            x_pu[30:36] = equilibrium_si[30:36] * pu.ωb / pu.Ib
        else:
            # algebraic mode: skip i_line
            # v (6-11) comes from 12-17 in SI vector
            x_pu[6:12]  = pu.to_pu(equilibrium_si[12:18], "voltage")

            # ζ_v (12-17) comes from 18-23
            x_pu[12:18] = equilibrium_si[18:24] * pu.ωb / pu.Vb

            # i_f (18-23) comes from 24-29
            x_pu[18:24] = pu.to_pu(equilibrium_si[24:30], "current")

            # ζ_f (24-29) comes from 30-35
            x_pu[24:30] = equilibrium_si[30:36] * pu.ωb / pu.Ib

        # Initialize η_a states to the base eta_a parameter value
        # This ensures proper dVOC dynamics from the start
        if self.enable_adaptive_control:
            x_pu[-Nc:] = self.eta_a

        return x_pu

    def initialize_from_equilibrium_high_load(self):
        """Return an initial state in *pu* that matches the expected layout."""
        Nc = self.network.Nc
        n_conv = 2 * Nc          # 6
        n_line = 2 * self.network.Nt  # 6

        if self.integrate_line_dynamics:
            base_size = 5 * n_conv + n_line     # 36
        else:
            base_size = 5 * n_conv              # 30

        # Add η_a states if adaptive control is enabled
        if self.enable_adaptive_control:
            state_size = base_size + Nc
        else:
            state_size = base_size

        # -------- raw equilibrium in SI (length 36) --------------------------
        equilibrium_si = torch.tensor([1.19991731e+02, -1.57738882e+00,  1.19987099e+02, -1.58860393e+00,
        1.19987495e+02, -1.58902566e+00,  2.86731109e+00, -2.22627709e-02,
        2.73570041e+00, -4.81013489e-02,  2.73423694e+00, -5.43291816e-02,
        1.19992990e+02, -1.57738893e+00,  1.19988358e+02, -1.58860408e+00,
        1.19988754e+02, -1.58902582e+00, -5.64050504e-04,  1.78481308e-05,
       -5.62836202e-04,  1.82334232e-05, -5.62667349e-04,  1.82913725e-05,
        2.87623102e+00,  6.56282400e-01,  2.74468377e+00,  6.30417627e-01,
        2.74322268e+00,  6.24192036e-01, -6.26904355e+01, -4.40352882e-02,
       -6.26824614e+01, -4.21463199e-02, -6.26906055e+01, -4.16569973e-02], dtype=self.dtype, device=self.device)
        pu = self.network.pu
        x_pu = torch.zeros(state_size, dtype=self.dtype, device=self.device)

        # -------- common blocks ----------------------------------------------
        # v̂ (0-5)
        x_pu[0:6] = pu.to_pu(equilibrium_si[0:6], "voltage")

        if self.integrate_line_dynamics:
            # i_line (6-11)
            x_pu[6:12] = pu.to_pu(equilibrium_si[6:12], "current")

            # v (12-17)
            x_pu[12:18] = pu.to_pu(equilibrium_si[12:18], "voltage")

            # ζ_v (18-23)   V·s ➜ pu
            x_pu[18:24] = equilibrium_si[18:24] * pu.ωb / pu.Vb

            # i_f (24-29)
            x_pu[24:30] = pu.to_pu(equilibrium_si[24:30], "current")

            # ζ_f (30-35)   A·s ➜ pu
            x_pu[30:36] = equilibrium_si[30:36] * pu.ωb / pu.Ib
        else:
            # algebraic mode: skip i_line
            # v (6-11) comes from 12-17 in SI vector
            x_pu[6:12]  = pu.to_pu(equilibrium_si[12:18], "voltage")

            # ζ_v (12-17) comes from 18-23
            x_pu[12:18] = equilibrium_si[18:24] * pu.ωb / pu.Vb

            # i_f (18-23) comes from 24-29
            x_pu[18:24] = pu.to_pu(equilibrium_si[24:30], "current")

            # ζ_f (24-29) comes from 30-35
            x_pu[24:30] = equilibrium_si[30:36] * pu.ωb / pu.Ib

        # Initialize η_a states to the base eta_a parameter value
        # This ensures proper dVOC dynamics from the start
        if self.enable_adaptive_control:
            x_pu[-Nc:] = self.eta_a

        return x_pu
    def setup_scenario(self, scenario: str):
        """
        Setup scenario parameters.
        
        Args:
            scenario: Scenario name
        """
        self.scenario = scenario
        print(f"\n--- Setting up scenario: {scenario} ---")

        # Restore nominal load
        self.network.rL = self.original_rL
        self.disturbance_applied = False
        self.disturbance = None

        # Clear adaptive control cache when operating point changes
        self.clear_adaptive_cache()

        # OPTIMIZATION #9: Invalidate line current cache when scenario changes
        self.network.invalidate_line_current_cache()

        # Set integration mode
        self.integrate_line_dynamics = (scenario == "load_change")

        # Sync converter parameters and rebuild control matrices ONCE per scenario
        self._sync_converter_params()

        # Get default equilibrium if needed
        if self.default_equilibrium_target is None:
            print("Calculating default equilibrium target...")
            guess = self.initialize_from_equilibrium()
            self.default_equilibrium_target = self.compute_equilibrium_point(0.0, guess)
            self.scenario_equilibrium_targets["default"] = self.default_equilibrium_target

        target_equilibrium = self.default_equilibrium_target

        # Scenario-specific setup
        if scenario == "load_change":
            new_rL_si = 115.0 * .1
            self.network.rL = self.network.pu.to_pu(new_rL_si, 'resistance')
            self.network.setup_network()
            # Cache is invalidated in setup_network()

        elif scenario == "black_start":
            print("Black start scenario - algebraic line currents")

        elif scenario == "setpoint_change":
            print("Setpoint change scenario")

        elif scenario == "disturbance":
            print("Disturbance scenario")
            self.disturbance = torch.randn(2 * self.network.Nc, dtype=self.dtype, device=self.device) * 0.1

        self.scenario_equilibrium_targets[scenario] = target_equilibrium

    def update_batch_setpoints(self, load_factors: torch.Tensor, perturb_scale: float = 0.0):
        """
        Update per-batch setpoints based on load factors.

        Computes power setpoints that match the load at each operating point.
        The setpoints are scaled according to the load factor.

        Args:
            load_factors: Tensor of load scaling factors [batch_size]
            perturb_scale: Optional perturbation scale for randomization
        """
        batch_size = load_factors.shape[0]
        Nc = self.network.Nc

        # Base setpoints (from initial setpoints)
        base_p_star = self.state_handler.initial_setpoints['p_star']  # [Nc]
        base_q_star = self.state_handler.initial_setpoints['q_star']  # [Nc]
        base_v_star = self.state_handler.initial_setpoints['v_star']  # [Nc]

        # Compute per-batch setpoints scaled by load factors
        # Higher load factor -> higher power consumption -> need higher power generation
        self.batch_p_star = torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)
        self.batch_q_star = torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)
        self.batch_v_star = torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)

        for b in range(batch_size):
            lf = load_factors[b]
            # Scale power setpoints by load factor
            self.batch_p_star[b] = base_p_star * lf
            self.batch_q_star[b] = base_q_star * lf
            self.batch_v_star[b] = base_v_star.clone()  # Voltage setpoint usually stays at 1.0 pu

            # Optional perturbation for exploration
            if perturb_scale > 0:
                self.batch_p_star[b] += perturb_scale * torch.randn(Nc, dtype=self.dtype, device=self.device) * base_p_star.abs().mean()
                self.batch_q_star[b] += perturb_scale * torch.randn(Nc, dtype=self.dtype, device=self.device) * base_q_star.abs().mean()

        # Also update final_setpoints in state_handler to the nominal (middle) batch
        # This is used when not in batch mode
        mid_idx = batch_size // 2
        self.state_handler.final_setpoints = {
            'p_star': self.batch_p_star[mid_idx].clone(),
            'q_star': self.batch_q_star[mid_idx].clone(),
            'v_star': self.batch_v_star[mid_idx].clone()
        }

    def run_simulation_for_scenario(self, scenario):
        """
        Run simulation for a scenario.
        
        Args:
            scenario: Scenario name
            
        Returns:
            Tuple of (time_vector, solution)
        """
        self.setup_scenario(scenario)

        # Time span
        steps = int(self.T_sim / self.dt) + 1
        t_span = torch.linspace(0.0, self.T_sim, steps, dtype=self.dtype, device=self.device)

        # Initial state
        x0 = self.initialize_state(scenario)

        # Run ODE
        sol = odeint(
            func=self,
            y0=x0,
            t=t_span,
            rtol=1e-5,
            atol=1e-5,
            method='dopri5'
        )

        return t_span, sol

    def compute_equilibrium_point(self, t_steady_val=0.0, x0_guess=None):
        """
        Compute equilibrium point.
        
        Args:
            t_steady_val: Steady-state time value
            x0_guess: Initial guess for equilibrium
            
        Returns:
            Equilibrium state tensor or None if failed
        """
        t_steady = torch.tensor(t_steady_val, dtype=self.dtype, device=self.device)

        # Make sure control states are properly set for equilibrium
        self.state_handler.update_states(t_steady_val)

        # Prepare initial guess
        if x0_guess is None:
            x0_guess_tensor = self.initialize_from_equilibrium()
            if self.integrate_line_dynamics:
                x0_guess_tensor = self.map_states_with_differential_line_currents(x0_guess_tensor)
            x0_guess = x0_guess_tensor.cpu().numpy()
        elif isinstance(x0_guess, torch.Tensor):
            if self.integrate_line_dynamics:
                expected_size = 5 * 2 * self.network.Nc + 2 * self.network.Nt
                if x0_guess.shape[0] < expected_size:
                    x0_guess = self.map_states_with_differential_line_currents(x0_guess).cpu().numpy()
                else:
                    x0_guess = x0_guess.cpu().numpy()
            else:
                x0_guess = x0_guess.cpu().numpy()

        # Residual function
        def F(x_np):
            x_t = torch.as_tensor(x_np, dtype=self.dtype, device=self.device)
            with torch.no_grad():
                if self.integrate_line_dynamics:
                    dx = self.system_equations_differential(t_steady, x_t)
                else:
                    dx = self.system_equations_algebraic(t_steady, x_t)
            return dx.cpu().numpy()

        # Solve for equilibrium
        x_eq_np, ok, residual, msg = super_safe_solve(F, x0_guess)

        if not ok:
            print(f"[equilibrium] Failed to find equilibrium: {msg}, residual={residual:.3e}")
            return None

        print(f"[equilibrium] Found equilibrium with residual={residual:.3e}")
        return torch.as_tensor(x_eq_np, dtype=self.dtype, device=self.device)

    def map_states_with_differential_line_currents(self, state_without_i_line):
        """Add line currents to state vector, preserving η_a states at end."""
        Nc = self.network.Nc
        n_conv = 2 * Nc

        # Check if η_a states are present
        if self.enable_adaptive_control:
            eta_a_vals = state_without_i_line[-Nc:]  # [Nc]
            core_state = state_without_i_line[:-Nc]
        else:
            eta_a_vals = None
            core_state = state_without_i_line

        vhat = core_state[0:n_conv]
        v = core_state[n_conv:2*n_conv]
        zeta_v = core_state[2*n_conv:3*n_conv]
        i_f = core_state[3*n_conv:4*n_conv]
        zeta_f = core_state[4*n_conv:5*n_conv]

        i_line, _ = self.network.compute_algebraic_line_currents(v)
        full_state = torch.cat([vhat, i_line, v, zeta_v, i_f, zeta_f], dim=0)

        # Append η_a states
        if self.enable_adaptive_control and eta_a_vals is not None:
            full_state = torch.cat([full_state, eta_a_vals])

        return full_state

    # Constraint-related methods delegated to constraints module
    def check_stability_conditions(self, verbose=False):
        """Check all stability conditions."""
        from .constraints import check_stability_conditions
        return check_stability_conditions(self, verbose)

    def clear_constraint_cache(self):
        """Clear constraint cache."""
        from .constraints import clear_constraint_cache
        clear_constraint_cache(self)

    def project_parameters(self):
        """Project parameters to valid bounds."""
        from .constraints import project_parameters
        project_parameters(self)

    def compute_lagrangian_loss(self, t_vec, sol, check_constraints_every=1):
        """Compute Lagrangian loss."""
        from .constraints import compute_lagrangian_loss
        return compute_lagrangian_loss(self, t_vec, sol, check_constraints_every)

    def update_lagrange_multipliers(self, step_size=0.1):
        """Update Lagrange multipliers."""
        from .constraints import update_lagrange_multipliers
        update_lagrange_multipliers(self, step_size)

    def update_lagrange_multipliers_batch(self, step_size=0.1, load_factors=None):
        """Update Lagrange multipliers for batch."""
        from .constraints import update_lagrange_multipliers_batch
        update_lagrange_multipliers_batch(self, step_size, load_factors)

    def compute_batch_constraint_violations(self, load_factors):
        """Compute batch constraint violations."""
        from .constraints import compute_batch_constraint_violations
        return compute_batch_constraint_violations(self, load_factors)

    # Loss-related methods delegated to losses module
    def compute_loss(self, t_vec, sol, **kwargs):
        """Compute loss."""
        from .losses import compute_loss
        return compute_loss(self, t_vec, sol, **kwargs)

    def compute_loss_batch(self, t_vec, sol_batch, **kwargs):
        """Compute batch loss."""
        from .losses import compute_loss_batch
        return compute_loss_batch(self, t_vec, sol_batch, **kwargs)

    # Optimization delegated to optimization module
    def run_multi_scenario_optimization(self, **kwargs):
        """Run multi-scenario optimization."""
        from .optimization import run_multi_scenario_optimization
        return run_multi_scenario_optimization(self, **kwargs)

    # =========================================================================
    # Adaptive Control Methods
    # =========================================================================

    def adaptive_parameters(self):
        """Return adaptive control neural network parameters for optimization."""
        if self.adaptive_controller is not None:
            return list(self.adaptive_controller.parameters())
        return []

    def all_trainable_parameters(self):
        """Return ALL trainable parameters including adaptive control."""
        params = list(self.parameters())
        params.extend(self.adaptive_parameters())
        return params

    def get_eta_a_states_from_solution(self, sol: torch.Tensor) -> torch.Tensor:
        """
        Extract η_a trajectories from solution.

        Args:
            sol: Solution tensor [T, state_size] or [T, batch, state_size]

        Returns:
            η_a trajectories [T, Nc] or [T, batch, Nc]
        """
        Nc = self.network.Nc
        if self.enable_adaptive_control:
            if sol.dim() == 2:
                return sol[:, -Nc:]
            else:
                return sol[:, :, -Nc:]
        else:
            return self.eta_a.expand(sol.shape[0], Nc)

    def verify_stability_conditions(self, verbose: bool = False):
        """
        Verify all stability conditions using conservative η.
        """
        from .constraints import check_condition4, check_condition5, check_condition6

        # Temporarily set eta to conservative value
        original_eta = self.eta.clone()

        if self.enable_adaptive_control:
            conservative_eta = self.get_eta_for_constraints()
            self.converter.eta = conservative_eta

        results = {
            'condition4': check_condition4(self),
            'condition5': check_condition5(self),
            'condition6': check_condition6(self)
        }

        # Restore
        self.converter.eta = original_eta

        all_satisfied = all(r['satisfied'] for r in results.values())
        min_margin = min(
            r['margin'].item() if hasattr(r['margin'], 'item') else r['margin']
            for r in results.values()
        )

        results['all_satisfied'] = all_satisfied
        results['min_margin'] = min_margin

        if verbose:
            print("=" * 50)
            print("Stability Verification (Conservative η)")
            print("=" * 50)
            for name, res in results.items():
                if name not in ['all_satisfied', 'min_margin']:
                    status = '✓' if res['satisfied'] else '✗'
                    margin = res['margin']
                    if hasattr(margin, 'item'):
                        margin = margin.item()
                    print(f"{name}: {status} (margin: {margin:.6f})")
            print("-" * 50)
            print(f"Overall: {'✓' if all_satisfied else '✗'}")
            print("=" * 50)

        return results

    def get_supervisor_bounds(self) -> dict:
        """Get the global bounds computed by supervisor."""
        if self.adaptive_controller is not None:
            return self.adaptive_controller.supervisor.compute_bounds(
                self.network, self.converter.setpoints
            )
        return {}
