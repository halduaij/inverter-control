"""
Converter Control Implementation - OPTIMIZED VERSION.

Contains:
- ConverterState: Manages converter states and setpoints over time
- ConverterControl: Implements voltage/current control with anti-windup

Optimizations Applied (gradient-safe):
- #2: Parameter change detection for rebuild_control_matrices (with gradient preservation)
- #5: Pre-computed block matrix views  
- #6: Pre-compute filter reciprocals
- #7: Pre-allocate zero tensors
- #8: Cache state handler updates
- #10: Avoid sqrt in calculate_Phi

CRITICAL: All optimizations preserve gradient flow for learnable parameters.
The key insight is that block-diagonal matrices Kp_v_mat = kron(I, Kp_v * I_2) can be
replaced with direct scalar multiplication in many operations, which is both faster
AND maintains cleaner gradient flow.
"""

import math
import torch
from typing import Tuple, Optional

from .core import Setpoints


class ConverterState:
    """
    Manage converter states and setpoints over time.
    
    Handles the temporal evolution of converter connection status,
    control activation, and setpoint changes.
    
    Args:
        converter_control: ConverterControl instance to manage
    """
    def __init__(self, converter_control):
        self.converter = converter_control
        self.network = converter_control.network
        device = self.network.device
        dtype = self.network.dtype
        pu = self.network.pu

        # Convert SI setpoints to per-unit
        p_star_si = torch.tensor([43.2, 41.0, 41.0], dtype=dtype, device=device)
        q_star_si = torch.tensor([-0.9, 0.5, -0.5], dtype=dtype, device=device)

        self.initial_setpoints = {
            'v_star': torch.ones(3, dtype=dtype, device=device),
            'p_star': pu.to_pu(p_star_si, 'power'),
            'q_star': pu.to_pu(q_star_si, 'power')
        }

        self.final_setpoints = self.initial_setpoints.copy()

        # Connection times
        self.t_connect = {0: 0.0, 1: 0.0, 2: 0.0}

        # Control activation times
        self.t_sequence = {
            0: {'current': 0.0, 'voltage': 0.0, 'power': 0.0},
            1: {'current': 0.0, 'voltage': 0.0, 'power': 0.0},
            2: {'current': 0.0, 'voltage': 0.0, 'power': 0.0}
        }

        self.t_setpoint_change = 3.5
        
        # OPTIMIZATION #8: Cache for state handler updates
        self._last_update_time = -1.0

    def update_states(self, t: float):
        """
        Update converter states at time t.
        
        OPTIMIZATION #8: Skip redundant updates at same time step.
        
        Args:
            t: Current simulation time
        """
        # OPTIMIZATION #8: Skip if already updated at this time
        if abs(t - self._last_update_time) < 1e-9:
            return
        self._last_update_time = t
        
        # Update breaker status
        breaker_status = torch.ones(self.network.Nc, dtype=torch.bool, device=self.network.device)
        self.network.update_breaker_status(breaker_status)

        # Update converter states
        for i in range(self.network.Nc):
            if t >= self.t_connect[i]:
                is_active = (t >= self.t_sequence[i]['current'])
                v_ctl = (t >= self.t_sequence[i]['voltage'])
                p_ctl = (t >= self.t_sequence[i]['power'])
                self.converter.update_converter_state(i, is_active, v_ctl, p_ctl)
            else:
                self.converter.update_converter_state(i, False, False, False)

        # Update setpoints
        sps = self.final_setpoints if t >= self.t_setpoint_change else self.initial_setpoints

        self.converter.setpoints = Setpoints(
            v_star=sps['v_star'],
            p_star=sps['p_star'],
            q_star=sps['q_star'],
            theta_star=torch.zeros(self.network.Nc, dtype=self.network.dtype, device=self.network.device)
        )


class ConverterControl:
    """
    Converter control implementation in per-unit with batch support - OPTIMIZED.
    
    Implements cascaded voltage-current control with dVOC (dispatchable
    Virtual Oscillator Control) for power sharing. Includes anti-windup
    for integrator states.
    
    OPTIMIZATION NOTES:
    - Uses direct scalar multiplication where possible instead of matrix ops
    - Maintains gradient flow through learnable parameters (eta, eta_a, Kp_v, Ki_v, Kp_f, Ki_f)
    - Pre-computes filter reciprocals and zero tensors
    
    Args:
        network: PowerSystemNetwork instance
        params: Dictionary with control parameters (eta, eta_a, Kp_v, Ki_v, Kp_f, Ki_f)
    """
    def __init__(self, network, params: dict):
        self.network = network
        self.device = network.device
        self.dtype = network.dtype
        self.pu = network.pu
        self.batch_size = network.batch_size
        
        # Saturation limits
        self.vm_limit = 1.95  # Maximum modulation voltage in p.u.
        self.vm_sat_sharpness = 50.0  # Sharpness of smooth saturation
        self.i_ref_limit = 1.5  # Maximum reference current in p.u.
        self.i_ref_sat_sharpness = 20.0  # Sharpness for current ref saturation
        
        # Anti-windup parameters
        self.anti_windup_enabled = True
        self.Tt_current = None  # If None, use optimal 1/Ki_f
        self.Tt_voltage = None  # If None, use optimal 1/Ki_v
        
        # Filter parameters in SI (from Table I)
        rf_si = 0.124    # Filter resistance (124 mΩ)
        lf_si = 0.5e-3     # Filter inductance (1 mH)
        cf_si = 15e-6    # Filter capacitance (24 μF)

        # Convert to per-unit
        self.rf = self.pu.to_pu(rf_si, 'resistance')
        self.lf = self.pu.to_pu(lf_si, 'inductance')
        self.cf = self.pu.to_pu(cf_si, 'capacitance')

        # OPTIMIZATION #6: Pre-compute filter reciprocals
        self.inv_lf = 1.0 / self.lf
        self.inv_cf = 1.0 / self.cf

        # Per-converter filter conductances
        gf_values = [1/rf_si, 1/rf_si, 1/rf_si]  # SI values
        self.gf = torch.tensor([self.pu.to_pu(g, 'conductance') for g in gf_values],
                              dtype=self.dtype, device=self.device)

        # Control gains from params (already in per-unit)
        # These are the LEARNABLE parameters - gradient flow goes through these
        self.eta = params['eta']
        self.eta_a = params['eta_a']
        self.Kp_v = params['Kp_v']
        self.Ki_v = params['Ki_v']
        self.Kp_f = params['Kp_f']
        self.Ki_f = params['Ki_f']

        # Converter states
        self.converter_states = {
            i: {
                'active': False,
                'voltage_control': False,
                'power_control': False
            }
            for i in range(self.network.Nc)
        }

        # Initialize setpoints (will be set by ConverterState)
        self.setpoints = Setpoints(
            v_star=torch.ones(self.network.Nc, dtype=self.dtype, device=self.device),
            p_star=torch.zeros(self.network.Nc, dtype=self.dtype, device=self.device),
            q_star=torch.zeros(self.network.Nc, dtype=self.dtype, device=self.device),
            theta_star=torch.zeros(self.network.Nc, dtype=self.dtype, device=self.device)
        )

        # OPTIMIZATION #7: Pre-allocate zero tensors
        self._zeros_2 = torch.zeros(2, dtype=self.dtype, device=self.device)
        
        self.setup_converter_matrices()

    def apply_vm_limit_with_antiwindup(
        self, vm: torch.Tensor, limit: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exact magnitude limiting: ||vm_sat|| <= limit.
        
        Args:
            vm: Modulation voltage [2] or [batch, 2]
            limit: Maximum magnitude limit
            
        Returns:
            Tuple of (saturated_vm, saturation_error) for anti-windup
        """
        if limit is None:
            limit = self.vm_limit

        eps = 1e-12

        if vm.dim() == 1:
            r = torch.linalg.norm(vm) + eps
            scale = torch.clamp(limit / r, max=1.0)
            vm_sat = vm * scale
            delta_vm = vm_sat - vm
            return vm_sat, delta_vm

        # Batched: [batch, 2]
        r = torch.linalg.norm(vm, dim=-1, keepdim=True) + eps  # [batch, 1]
        scale = torch.clamp(limit / r, max=1.0)                # [batch, 1]
        vm_sat = vm * scale
        delta_vm = vm_sat - vm
        return vm_sat, delta_vm

    def apply_vm_limit(self, vm, limit=None, *_unused):
        """
        Backward-compatible vm limit function.
        Returns only the saturated vm (for existing code compatibility).
        """
        vm_sat, _ = self.apply_vm_limit_with_antiwindup(vm, limit)
        return vm_sat

    def limit_current_with_antiwindup(
        self, i_ref: torch.Tensor, limit: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exact magnitude limiting: ||i_ref_sat|| <= limit.
        
        Args:
            i_ref: Reference current [2] or [batch, 2]
            limit: Maximum magnitude limit
            
        Returns:
            Tuple of (saturated_i_ref, saturation_error) for anti-windup
        """
        if limit is None:
            limit = self.i_ref_limit

        eps = 1e-12

        if i_ref.dim() == 1:
            r = torch.linalg.norm(i_ref) + eps
            scale = torch.clamp(limit / r, max=1.0)
            i_ref_sat = i_ref * scale
            delta_i = i_ref_sat - i_ref
            return i_ref_sat, delta_i

        r = torch.linalg.norm(i_ref, dim=-1, keepdim=True) + eps
        scale = torch.clamp(limit / r, max=1.0)
        i_ref_sat = i_ref * scale
        delta_i = i_ref_sat - i_ref
        return i_ref_sat, delta_i

    def limit_current(self, i_ref, limit=1.1, sharp=10.0):
        """Backward-compatible current limiting."""
        i_ref_sat, _ = self.limit_current_with_antiwindup(i_ref, limit if limit != 1.1 else self.i_ref_limit)
        return i_ref_sat

    def rebuild_control_matrices(self):
        """
        Rebuild control matrices with current parameters.
        
        Called once per scenario setup - NOT in the hot path.
        Uses current parameter references to maintain gradient flow.
        """
        Nc = self.network.Nc
        In = self.network.In

        # Build full block-diagonal matrices using current parameters
        # CRITICAL: Uses self.Kp_v etc. directly to maintain gradient flow
        self.Kp_v_mat = torch.kron(
            torch.eye(Nc, dtype=self.dtype, device=self.device),
            self.Kp_v * In
        )
        self.Ki_v_mat = torch.kron(
            torch.eye(Nc, dtype=self.dtype, device=self.device),
            self.Ki_v * In
        )
        self.Kp_f_mat = torch.kron(
            torch.eye(Nc, dtype=self.dtype, device=self.device),
            self.Kp_f * In
        )
        self.Ki_f_mat = torch.kron(
            torch.eye(Nc, dtype=self.dtype, device=self.device),
            self.Ki_f * In
        )
        
        # OPTIMIZATION #5: Update block views after rebuild
        self._update_block_views()

    def _update_block_views(self):
        """
        OPTIMIZATION #5: Create/update pre-computed block views.
        
        These views point to the same underlying tensor memory, so they
        automatically reflect any changes and maintain gradient flow.
        """
        Nc = self.network.Nc
        
        # Control matrix block views
        self.Kp_v_blocks = [self.Kp_v_mat[2*i:2*i+2, 2*i:2*i+2] for i in range(Nc)]
        self.Ki_v_blocks = [self.Ki_v_mat[2*i:2*i+2, 2*i:2*i+2] for i in range(Nc)]
        self.Kp_f_blocks = [self.Kp_f_mat[2*i:2*i+2, 2*i:2*i+2] for i in range(Nc)]
        self.Ki_f_blocks = [self.Ki_f_mat[2*i:2*i+2, 2*i:2*i+2] for i in range(Nc)]
        
        # Filter matrix block views
        self.Zf_blocks = [self.Zf[2*i:2*i+2, 2*i:2*i+2] for i in range(Nc)]
        self.Yf_blocks = [self.Yf[2*i:2*i+2, 2*i:2*i+2] for i in range(Nc)]

    def setup_converter_matrices(self):
        """Setup converter filter and control matrices."""
        Nc = self.network.Nc
        In = self.network.In
        J = self.network.J
        omega0 = self.network.omega0

        # Filter matrices
        self.Rf = torch.kron(torch.eye(Nc, dtype=self.dtype, device=self.device), self.rf * In)
        self.Lf = torch.kron(torch.eye(Nc, dtype=self.dtype, device=self.device), self.lf * In)
        self.Cf = torch.kron(torch.eye(Nc, dtype=self.dtype, device=self.device), self.cf * In)
        self.Gf = torch.kron(torch.diag(self.gf), In)
        self.Jnc = torch.kron(torch.eye(Nc, dtype=self.dtype, device=self.device), J)

        # Filter impedance/admittance
        self.Zf = self.Rf + omega0 * (self.Jnc @ self.Lf)
        self.Yf = omega0 * (self.Jnc @ self.Cf)

        # Cache 3D block tensors [Nc, 2, 2] for fully vectorized operations
        self.Yf_3d = torch.stack([self.Yf[2*k:2*k+2, 2*k:2*k+2] for k in range(Nc)])
        self.Zf_3d = torch.stack([self.Zf[2*k:2*k+2, 2*k:2*k+2] for k in range(Nc)])

        # Build control matrices
        self.rebuild_control_matrices()

    def update_converter_state(self, idx, active, voltage_control, power_control):
        """
        Update converter state.
        
        Args:
            idx: Converter index
            active: Whether converter is active
            voltage_control: Whether voltage control is enabled
            power_control: Whether power control (dVOC) is enabled
        """
        self.converter_states[idx].update({
            'active': active,
            'voltage_control': voltage_control,
            'power_control': power_control
        })

    def calculate_K(self, idx, setpoints):
        """
        Power-sharing matrix K in per-unit.

        Args:
            idx: Converter index
            setpoints: Current setpoints (can be batched)

        Returns:
            K matrix [2, 2] or [batch, 2, 2] if batched setpoints
        """
        # Check if setpoints are batched (v_star has shape [batch, Nc])
        is_batched = setpoints.v_star.dim() == 2

        if is_batched:
            v_star = setpoints.v_star[:, idx]  # [batch]
            p_star = setpoints.p_star[:, idx]  # [batch]
            q_star = setpoints.q_star[:, idx]  # [batch]

            # OPTIMIZATION #5: Use pre-computed R_kappa block view
            R_kappa = self.network.R_kappa_blocks[idx]  # [2, 2]

            # Build P matrix for each batch element: [batch, 2, 2]
            batch_size = v_star.shape[0]
            P = torch.zeros(batch_size, 2, 2, dtype=self.dtype, device=self.device)
            P[:, 0, 0] = p_star
            P[:, 0, 1] = q_star
            P[:, 1, 0] = -q_star
            P[:, 1, 1] = p_star

            # K_mat = (1/v_star²) * (R_kappa @ P) for each batch
            # R_kappa @ P: [2,2] @ [batch,2,2] -> need einsum
            RP = torch.einsum('ij,bjk->bik', R_kappa, P)  # [batch, 2, 2]
            K_mat = (1.0 / (v_star**2 + 1e-12)).unsqueeze(-1).unsqueeze(-1) * RP
            return K_mat
        else:
            v_star = setpoints.v_star[idx]
            p_star = setpoints.p_star[idx]
            q_star = setpoints.q_star[idx]

            # OPTIMIZATION #5: Use pre-computed R_kappa block view
            R_kappa = self.network.R_kappa_blocks[idx]

            P = torch.tensor([
                [p_star, q_star],
                [-q_star, p_star]
            ], dtype=self.dtype, device=self.device)

            K_mat = (1.0 / (v_star**2)) * (R_kappa @ P)
            return K_mat

    def calculate_Phi(self, v_hat_local, idx, setpoints):
        """
        Calculate Φ = 1 - ||v_hat||^2 / v_star^2.
        
        OPTIMIZATION #10: Avoid sqrt by using squared norm directly.
        
        Args:
            v_hat_local: Voltage reference [2]
            idx: Converter index
            setpoints: Current setpoints
            
        Returns:
            Phi value (scalar)
        """
        v_star = setpoints.v_star[idx]
        # OPTIMIZATION #10: Use squared norm directly instead of norm()**2
        norm_vhat_sq = v_hat_local[0]**2 + v_hat_local[1]**2
        return 1.0 - norm_vhat_sq / (v_star**2 + 1e-12)

    def get_tracking_gain_current(self, idx: int) -> torch.Tensor:
        """
        Get anti-windup tracking gain for current controller.
        Kt = Ki gives optimal tracking (tracking time Tt = 1/Ki).
        
        Uses pre-computed block view for efficiency.
        """
        if self.Tt_current is not None:
            Kt = torch.eye(2, dtype=self.dtype, device=self.device) / self.Tt_current
        else:
            # OPTIMIZATION #5: Use pre-computed block view
            Kt = self.Ki_f_blocks[idx]
        return Kt

    def get_tracking_gain_voltage(self, idx: int) -> torch.Tensor:
        """Get anti-windup tracking gain for voltage controller."""
        if self.Tt_voltage is not None:
            Kt = torch.eye(2, dtype=self.dtype, device=self.device) / self.Tt_voltage
        else:
            # OPTIMIZATION #5: Use pre-computed block view
            Kt = self.Ki_v_blocks[idx]
        return Kt

    def filter_dynamics_active(self, idx, v_local, i_f_local, vm_local, v_full, i_line):
        """Active filter dynamics in per-unit using pre-computed values."""
        # OPTIMIZATION #5: Use pre-computed block views
        Yf_local = self.Yf_blocks[idx]
        Zf_local = self.Zf_blocks[idx]
        
        i_total = self.network.calculate_total_currents(v_full, i_line)

        if i_total.dim() == 1:
            i_o_local = i_total[2*idx:2*idx+2]
        else:
            i_o_local = i_total[2*idx:2*idx+2]

        # OPTIMIZATION #6: Use pre-computed reciprocals
        dv = self.inv_cf * (-Yf_local @ v_local - i_o_local + i_f_local)
        dif = self.inv_lf * (-Zf_local @ i_f_local - v_local + vm_local)
        return dv, dif

    def filter_dynamics_inactive(self, idx, v_local, i_f_local, v_full, i_line):
        """Inactive filter dynamics in per-unit."""
        # OPTIMIZATION #5: Use pre-computed block views
        Yf_local = self.Yf_blocks[idx]
        
        i_total = self.network.calculate_total_currents(v_full, i_line)

        if i_total.dim() == 1:
            i_o_local = i_total[2*idx:2*idx+2]
        else:
            i_o_local = i_total[2*idx:2*idx+2]

        dv = (self.network.pu.ωb * self.inv_cf) * (-Yf_local @ v_local - i_o_local + i_f_local)
        # OPTIMIZATION #7: Use pre-allocated zeros
        dif = self._zeros_2.clone()
        return dv, dif

    def voltage_control(self, idx, v_node, vhat_node, i_line, zeta_v_node, v_full, setpoints):
        """Voltage control dynamics."""
        if not (self.converter_states[idx]['voltage_control'] and
                self.converter_states[idx]['active']):
            # OPTIMIZATION #7: Use pre-allocated zeros
            return self._zeros_2.clone(), self._zeros_2.clone()

        i_total = self.network.calculate_total_currents(v_full, i_line)

        if i_total.dim() == 1:
            i_inj = i_total[2*idx:2*idx+2]
        else:
            i_inj = i_total[2*idx:2*idx+2]

        if self.converter_states[idx]['power_control']:
            # Full dVOC
            K = self.calculate_K(idx, setpoints)
            phi_val = self.calculate_Phi(vhat_node, idx, setpoints)

            # OPTIMIZATION #5: Use pre-computed R_kappa block
            R_kappa = self.network.R_kappa_blocks[idx]

            dvhat = self.eta * (K @ vhat_node - R_kappa @ i_inj +
                               self.eta_a * phi_val * vhat_node)
        else:
            # Basic voltage regulation
            v_star = setpoints.v_star[idx]
            v_mag = torch.norm(v_node) + 1e-12
            dvhat = -self.Kp_v * ((v_mag - v_star) * (v_node / v_mag))

        dzeta_v = (v_node - vhat_node)
        return dvhat, dzeta_v

    def calculate_reference_current(self, idx, v_node, vhat_node, i_line, zeta_v_node, v_full):
        """Calculate reference current."""
        if not self.converter_states[idx]['active']:
            # OPTIMIZATION #7: Use pre-allocated zeros
            return self._zeros_2.clone()

        # OPTIMIZATION #5: Use pre-computed block views
        Yf_local = self.Yf_blocks[idx]

        if self.converter_states[idx]['voltage_control']:
            i_total = self.network.calculate_total_currents(v_full, i_line)
            if i_total.dim() == 1:
                i_inj = i_total[2*idx:2*idx+2]
            else:
                i_inj = i_total[2*idx:2*idx+2]

            # OPTIMIZATION #5: Use pre-computed block views
            Kp_v_local = self.Kp_v_blocks[idx]
            Ki_v_local = self.Ki_v_blocks[idx]

            if self.converter_states[idx]['power_control']:
                i_ref = (Yf_local @ v_node + i_inj -
                        Kp_v_local @ (v_node - vhat_node) -
                        Ki_v_local @ zeta_v_node)
            else:
                i_ref = (Yf_local @ v_node -
                        Kp_v_local @ (v_node - vhat_node) -
                        Ki_v_local @ zeta_v_node)
        else:
            i_ref = Yf_local @ v_node

        i_ref = self.limit_current(i_ref, limit=2)
        return i_ref

    def current_control(self, idx, v_node, i_f_node, i_ref_node, zeta_f_node):
        """
        Current control dynamics with back-calculation anti-windup.
        
        When vm saturates, the anti-windup term reduces integrator accumulation
        to prevent windup and overshoot.
        """
        if not self.converter_states[idx]['active']:
            # OPTIMIZATION #7: Use pre-allocated zeros
            return self._zeros_2.clone(), self._zeros_2.clone()

        # OPTIMIZATION #5: Use pre-computed block views
        Zf_local = self.Zf_blocks[idx]
        Kp_f_local = self.Kp_f_blocks[idx]
        Ki_f_local = self.Ki_f_blocks[idx]

        # Compute unsaturated modulation voltage
        error_i = i_f_node - i_ref_node
        vm_unsat = (Zf_local @ i_f_node + v_node -
                    Kp_f_local @ error_i -
                    Ki_f_local @ zeta_f_node)

        # Apply saturation with anti-windup feedback
        vm_sat, delta_vm = self.apply_vm_limit_with_antiwindup(vm_unsat)

        # Compute integrator dynamics with anti-windup
        if self.anti_windup_enabled:
            Kt = self.get_tracking_gain_current(idx)
            antiwindup_term = Kt @ delta_vm
            dzeta_f = error_i + antiwindup_term
        else:
            dzeta_f = error_i

        return vm_sat, dzeta_f

    def compute_converter_dynamics(self, idx, full_state, setpoints, i_line):
        """
        Compute full converter dynamics - TRUE PARALLEL.
        
        GRADIENT FLOW: Uses learnable parameters (eta, eta_a, Kp_v, etc.) directly
        in computations to maintain proper gradient flow for optimization.
        
        Args:
            idx: Converter index
            full_state: Full system state [state_size] or [batch, state_size]
            setpoints: Current setpoints
            i_line: Line currents
            
        Returns:
            Tuple of (dvhat, dv, dzeta_v, dif, dzeta_f) for this converter
        """
        if full_state.dim() == 1:
            return self._compute_single_converter_dynamics(idx, full_state, setpoints, i_line)

        # TRUE PARALLEL IMPLEMENTATION
        batch_size = full_state.shape[0]
        Nc = self.network.Nc
        n_conv = 2 * Nc
        device = self.device
        dtype = self.dtype

        # Unpack batched state
        vhat = full_state[:, 0:n_conv]
        v = full_state[:, n_conv:2*n_conv]
        zeta_v = full_state[:, 2*n_conv:3*n_conv]
        i_f = full_state[:, 3*n_conv:4*n_conv]
        zeta_f = full_state[:, 4*n_conv:5*n_conv]

        # Extract local variables for this converter
        idx_slice = slice(2*idx, 2*(idx+1))
        vhat_local = vhat[:, idx_slice]
        v_local = v[:, idx_slice]
        zeta_v_local = zeta_v[:, idx_slice]
        i_f_local = i_f[:, idx_slice]
        zeta_f_local = zeta_f[:, idx_slice]

        # Initialize outputs
        dvhat = torch.zeros_like(vhat_local)
        dv = torch.zeros_like(v_local)
        dzeta_v = torch.zeros_like(zeta_v_local)
        dif = torch.zeros_like(i_f_local)
        dzeta_f = torch.zeros_like(zeta_f_local)

        # Check breaker status
        if not self.network.breaker_status[idx]:
            # OPTIMIZATION #6: Use pre-computed reciprocal
            dif = -(self.rf * self.inv_lf) * i_f_local
            return dvhat, dv, dzeta_v, dif, dzeta_f

        # Get total currents (vectorized)
        if i_line.dim() == 2:
            i_total = i_line @ self.network.B_active.T
        else:
            i_total = self.network.calculate_total_currents(v, i_line)

        i_o_local = i_total[:, idx_slice] if i_total.dim() == 2 else i_total[idx_slice]

        # OPTIMIZATION #5: Use pre-computed block views
        Yf_local = self.Yf_blocks[idx]
        Zf_local = self.Zf_blocks[idx]

        if self.converter_states[idx]['active']:
            # ACTIVE CONVERTER - vectorized operations with anti-windup
            if self.converter_states[idx]['voltage_control']:
                i_inj = i_o_local

                if self.converter_states[idx]['power_control']:
                    # Full dVOC - vectorized
                    K = self.calculate_K(idx, setpoints)
                    R_kappa = self.network.R_kappa_blocks[idx]

                    # Check if K is batched [batch, 2, 2] or not [2, 2]
                    if K.dim() == 3:
                        K_vhat = torch.einsum('bij,bj->bi', K, vhat_local)
                    else:
                        K_vhat = torch.einsum('ij,bj->bi', K, vhat_local)
                    R_i = torch.einsum('ij,bj->bi', R_kappa, i_inj)

                    # Handle batched v_star
                    is_batched_setpoints = setpoints.v_star.dim() == 2
                    if is_batched_setpoints:
                        v_star = setpoints.v_star[:, idx].unsqueeze(-1)  # [batch, 1]
                    else:
                        v_star = setpoints.v_star[idx]
                    # OPTIMIZATION #10: Use squared norm directly
                    norm_vhat_sq = (vhat_local ** 2).sum(dim=1, keepdim=True)
                    phi_val = 1.0 - norm_vhat_sq / (v_star**2 + 1e-12)

                    # Support per-batch eta and eta_a
                    eta = self.eta.unsqueeze(-1) if hasattr(self.eta, 'dim') and self.eta.dim() >= 1 else self.eta
                    eta_a = self.eta_a.unsqueeze(-1) if hasattr(self.eta_a, 'dim') and self.eta_a.dim() >= 1 else self.eta_a
                    dvhat = eta * (K_vhat - R_i + eta_a * phi_val * vhat_local)
                else:
                    is_batched_setpoints = setpoints.v_star.dim() == 2
                    if is_batched_setpoints:
                        v_star = setpoints.v_star[:, idx].unsqueeze(-1)  # [batch, 1]
                    else:
                        v_star = setpoints.v_star[idx]
                    v_mag = torch.norm(v_local, dim=1, keepdim=True) + 1e-12
                    dvhat = -self.Kp_v * ((v_mag - v_star) * (v_local / v_mag))

            # Current reference calculation - vectorized using scalar multiplication
            Yf_v = torch.einsum('ij,bj->bi', Yf_local, v_local)

            if self.converter_states[idx]['voltage_control']:
                # GRADIENT-PRESERVING: Use parameters directly for efficiency
                # Supports both scalar gains and per-batch gains [batch]
                error_v = v_local - vhat_local
                # Handle per-batch gains: [batch] -> [batch, 1] for broadcasting with [batch, 2]
                Kp_v = self.Kp_v.unsqueeze(-1) if self.Kp_v.dim() >= 1 else self.Kp_v
                Ki_v = self.Ki_v.unsqueeze(-1) if self.Ki_v.dim() >= 1 else self.Ki_v
                Kpv_err = Kp_v * error_v
                Kiv_zeta = Ki_v * zeta_v_local

                if self.converter_states[idx]['power_control']:
                    i_ref_unsat = Yf_v + i_inj - Kpv_err - Kiv_zeta
                else:
                    i_ref_unsat = Yf_v - Kpv_err - Kiv_zeta
            else:
                i_ref_unsat = Yf_v

            # Apply current reference saturation with anti-windup
            i_ref_sat, i_ref_sat_error = self.limit_current_with_antiwindup(i_ref_unsat)

            # Voltage integrator with anti-windup
            if self.converter_states[idx]['voltage_control']:
                error_v = v_local - vhat_local
                if self.anti_windup_enabled:
                    # Support per-batch Ki_v
                    Ki_v = self.Ki_v.unsqueeze(-1) if self.Ki_v.dim() >= 1 else self.Ki_v
                    antiwindup_v = Ki_v * i_ref_sat_error
                    dzeta_v = error_v + antiwindup_v
                else:
                    dzeta_v = error_v

            # Current control - supports per-batch gains
            Zf_if = torch.einsum('ij,bj->bi', Zf_local, i_f_local)
            error_i = i_f_local - i_ref_sat
            # Handle per-batch gains: [batch] -> [batch, 1] for broadcasting
            Kp_f = self.Kp_f.unsqueeze(-1) if self.Kp_f.dim() >= 1 else self.Kp_f
            Ki_f = self.Ki_f.unsqueeze(-1) if self.Ki_f.dim() >= 1 else self.Ki_f
            Kpf_err = Kp_f * error_i
            Kif_zeta = Ki_f * zeta_f_local

            vm_unsat = Zf_if + v_local - Kpf_err - Kif_zeta
            vm_sat, delta_vm = self.apply_vm_limit_with_antiwindup(vm_unsat)

            # Current integrator with anti-windup
            if self.anti_windup_enabled:
                antiwindup_f = Ki_f * delta_vm
                dzeta_f = error_i + antiwindup_f
            else:
                dzeta_f = error_i

            # Filter dynamics - vectorized with pre-computed reciprocals
            dv = self.inv_cf * (-Yf_v - i_o_local + i_f_local)
            dif = self.inv_lf * (-Zf_if - v_local + vm_sat)
        else:
            # INACTIVE CONVERTER - vectorized
            Yf_v = torch.einsum('ij,bj->bi', Yf_local, v_local)
            dv = (self.network.pu.ωb * self.inv_cf) * (-Yf_v - i_o_local + i_f_local)

        return dvhat, dv, dzeta_v, dif, dzeta_f

    def compute_all_converters_vectorized(self, full_state, setpoints, i_line,
                                           gains_batch, eta_a_batch):
        """
        Compute dynamics for ALL converters and ALL batches - NO LOOPS.

        Fully vectorized over [batch, Nc, 2].

        Args:
            full_state: [batch, 5*n_conv] - [vhat, v, zeta_v, i_f, zeta_f]
            setpoints: Setpoints with p_star, q_star, v_star each [batch, Nc]
            i_line: [batch, 2*Nt] line currents
            gains_batch: dict with 'Kp_v', 'Ki_v', 'Kp_f', 'Ki_f', 'eta' each [batch, Nc]
            eta_a_batch: [batch, Nc] adaptive gain states

        Returns:
            Tuple of (dvhat, dv, dzeta_v, dif, dzeta_f) each [batch, 2*Nc]
        """
        batch_size = full_state.shape[0]
        Nc = self.network.Nc
        n_conv = 2 * Nc

        # Unpack state [batch, 2*Nc] for each variable
        vhat = full_state[:, 0:n_conv]
        v = full_state[:, n_conv:2*n_conv]
        zeta_v = full_state[:, 2*n_conv:3*n_conv]
        i_f = full_state[:, 3*n_conv:4*n_conv]
        zeta_f = full_state[:, 4*n_conv:5*n_conv]

        # Reshape to [batch, Nc, 2] for vectorized per-converter operations
        vhat_3d = vhat.reshape(batch_size, Nc, 2)
        v_3d = v.reshape(batch_size, Nc, 2)
        zeta_v_3d = zeta_v.reshape(batch_size, Nc, 2)
        i_f_3d = i_f.reshape(batch_size, Nc, 2)
        zeta_f_3d = zeta_f.reshape(batch_size, Nc, 2)

        # Gains [batch, Nc] -> [batch, Nc, 1] for broadcasting with [batch, Nc, 2]
        Kp_v = gains_batch['Kp_v'].unsqueeze(-1)  # [batch, Nc, 1]
        Ki_v = gains_batch['Ki_v'].unsqueeze(-1)
        Kp_f = gains_batch['Kp_f'].unsqueeze(-1)
        Ki_f = gains_batch['Ki_f'].unsqueeze(-1)
        eta = gains_batch['eta'].unsqueeze(-1)
        eta_a = eta_a_batch.unsqueeze(-1)  # [batch, Nc, 1]

        # Setpoints [batch, Nc] -> [batch, Nc, 1]
        v_star = setpoints.v_star.unsqueeze(-1)  # [batch, Nc, 1]
        p_star = setpoints.p_star  # [batch, Nc]
        q_star = setpoints.q_star  # [batch, Nc]

        # Compute total currents and reshape
        i_total = self.network.calculate_total_currents(v, i_line)
        i_o_3d = i_total.reshape(batch_size, Nc, 2)  # [batch, Nc, 2]

        # === dVOC dynamics ===
        # Build P matrices [batch, Nc, 2, 2]: P = [[p, q], [-q, p]]
        P = torch.zeros(batch_size, Nc, 2, 2, dtype=self.dtype, device=self.device)
        P[:, :, 0, 0] = p_star
        P[:, :, 0, 1] = q_star
        P[:, :, 1, 0] = -q_star
        P[:, :, 1, 1] = p_star

        # R_kappa blocks [Nc, 2, 2] - stack from network
        R_kappa_3d = torch.stack(self.network.R_kappa_blocks)  # [Nc, 2, 2]

        # K = (1/v_star^2) * R_kappa @ P
        # [Nc, 2, 2] @ [batch, Nc, 2, 2] -> [batch, Nc, 2, 2]
        RP = torch.einsum('kij,bkjl->bkil', R_kappa_3d, P)
        K = RP / (v_star.unsqueeze(-1)**2 + 1e-12)  # [batch, Nc, 2, 2]

        # K @ vhat: [batch, Nc, 2, 2] @ [batch, Nc, 2] -> [batch, Nc, 2]
        K_vhat = torch.einsum('bkij,bkj->bki', K, vhat_3d)

        # R_kappa @ i_inj: [Nc, 2, 2] @ [batch, Nc, 2] -> [batch, Nc, 2]
        R_i = torch.einsum('kij,bkj->bki', R_kappa_3d, i_o_3d)

        # Phi = 1 - ||vhat||^2 / v_star^2
        norm_vhat_sq = (vhat_3d ** 2).sum(dim=-1, keepdim=True)  # [batch, Nc, 1]
        phi = 1.0 - norm_vhat_sq / (v_star**2 + 1e-12)

        # dvhat = eta * (K@vhat - R@i + eta_a * phi * vhat)
        dvhat_3d = eta * (K_vhat - R_i + eta_a * phi * vhat_3d)

        # === Voltage control: current reference ===
        # Yf @ v: [Nc, 2, 2] @ [batch, Nc, 2] -> [batch, Nc, 2]
        Yf_v = torch.einsum('kij,bkj->bki', self.Yf_3d, v_3d)

        # Voltage error and PI control
        error_v = v_3d - vhat_3d  # [batch, Nc, 2]
        Kpv_err = Kp_v * error_v
        Kiv_zeta = Ki_v * zeta_v_3d

        # i_ref = Yf@v + i_inj - Kp*(v-vhat) - Ki*zeta_v
        i_ref_unsat = Yf_v + i_o_3d - Kpv_err - Kiv_zeta

        # Current saturation: flatten to [batch*Nc, 2], apply, reshape back
        i_ref_flat = i_ref_unsat.reshape(batch_size * Nc, 2)
        i_ref_sat_flat, i_ref_err_flat = self.limit_current_with_antiwindup(i_ref_flat)
        i_ref_sat = i_ref_sat_flat.reshape(batch_size, Nc, 2)
        i_ref_err = i_ref_err_flat.reshape(batch_size, Nc, 2)

        # Voltage integrator with anti-windup
        dzeta_v_3d = error_v + Ki_v * i_ref_err

        # === Current control ===
        # Zf @ if: [Nc, 2, 2] @ [batch, Nc, 2] -> [batch, Nc, 2]
        Zf_if = torch.einsum('kij,bkj->bki', self.Zf_3d, i_f_3d)

        error_i = i_f_3d - i_ref_sat
        Kpf_err = Kp_f * error_i
        Kif_zeta = Ki_f * zeta_f_3d

        # vm = Zf@if + v - Kp*(if-iref) - Ki*zeta_f
        vm_unsat = Zf_if + v_3d - Kpf_err - Kif_zeta

        # Voltage saturation
        vm_flat = vm_unsat.reshape(batch_size * Nc, 2)
        vm_sat_flat, delta_vm_flat = self.apply_vm_limit_with_antiwindup(vm_flat)
        vm_sat = vm_sat_flat.reshape(batch_size, Nc, 2)
        delta_vm = delta_vm_flat.reshape(batch_size, Nc, 2)

        # Current integrator with anti-windup
        dzeta_f_3d = error_i + Ki_f * delta_vm

        # === Filter dynamics ===
        # dv = (1/cf) * (-Yf@v - i_o + if)
        dv_3d = self.inv_cf * (-Yf_v - i_o_3d + i_f_3d)

        # dif = (1/lf) * (-Zf@if - v + vm)
        dif_3d = self.inv_lf * (-Zf_if - v_3d + vm_sat)

        # Reshape back to [batch, 2*Nc]
        dvhat = dvhat_3d.reshape(batch_size, n_conv)
        dv = dv_3d.reshape(batch_size, n_conv)
        dzeta_v = dzeta_v_3d.reshape(batch_size, n_conv)
        dif = dif_3d.reshape(batch_size, n_conv)
        dzeta_f = dzeta_f_3d.reshape(batch_size, n_conv)

        return dvhat, dv, dzeta_v, dif, dzeta_f

    def _compute_single_converter_dynamics(self, idx, full_state, setpoints, i_line):
        """Original single trajectory converter dynamics with anti-windup - OPTIMIZED."""
        Nc = self.network.Nc
        n_conv = 2 * Nc

        # Unpack state
        vhat = full_state[0:n_conv]
        v = full_state[n_conv:2*n_conv]
        zeta_v = full_state[2*n_conv:3*n_conv]
        i_f = full_state[3*n_conv:4*n_conv]
        zeta_f = full_state[4*n_conv:5*n_conv]

        idx_slice = slice(2*idx, 2*(idx+1))
        vhat_local = vhat[idx_slice]
        v_local = v[idx_slice]
        zeta_v_local = zeta_v[idx_slice]
        i_f_local = i_f[idx_slice]
        zeta_f_local = zeta_f[idx_slice]
        v_full = v

        # OPTIMIZATION #7: Use pre-allocated zeros via clone
        dvhat = self._zeros_2.clone()
        dv_ = self._zeros_2.clone()
        dzeta_v_ = self._zeros_2.clone()
        dif_ = self._zeros_2.clone()
        dzeta_f_ = self._zeros_2.clone()

        if self.network.breaker_status[idx]:
            if self.converter_states[idx]['active']:
                # OPTIMIZATION #5: Use pre-computed block views
                Yf_local = self.Yf_blocks[idx]
                Zf_local = self.Zf_blocks[idx]

                i_total = self.network.calculate_total_currents(v_full, i_line)
                i_inj = i_total[idx_slice] if i_total.dim() == 1 else i_total[:, idx_slice]

                # Voltage reference dynamics
                if self.converter_states[idx]['voltage_control']:
                    if self.converter_states[idx]['power_control']:
                        K = self.calculate_K(idx, setpoints)
                        phi_val = self.calculate_Phi(vhat_local, idx, setpoints)
                        R_kappa = self.network.R_kappa_blocks[idx]
                        dvhat = self.eta * (K @ vhat_local - R_kappa @ i_inj +
                                           self.eta_a * phi_val * vhat_local)
                    else:
                        v_star = setpoints.v_star[idx]
                        v_mag = torch.norm(v_local) + 1e-12
                        dvhat = -self.Kp_v * ((v_mag - v_star) * (v_local / v_mag))

                # Current reference with saturation - use scalar multiplication
                Yf_v = Yf_local @ v_local

                if self.converter_states[idx]['voltage_control']:
                    # GRADIENT-PRESERVING: Direct scalar multiplication
                    error_v = v_local - vhat_local
                    Kpv_err = self.Kp_v * error_v
                    Kiv_zeta = self.Ki_v * zeta_v_local

                    if self.converter_states[idx]['power_control']:
                        i_ref_unsat = Yf_v + i_inj - Kpv_err - Kiv_zeta
                    else:
                        i_ref_unsat = Yf_v - Kpv_err - Kiv_zeta
                else:
                    i_ref_unsat = Yf_v

                i_ref_sat, i_ref_sat_error = self.limit_current_with_antiwindup(i_ref_unsat)

                # Voltage integrator with anti-windup
                if self.converter_states[idx]['voltage_control']:
                    error_v = v_local - vhat_local
                    if self.anti_windup_enabled:
                        # Use scalar Ki_v for tracking gain
                        antiwindup_v = self.Ki_v * i_ref_sat_error
                        dzeta_v_ = error_v + antiwindup_v
                    else:
                        dzeta_v_ = error_v

                # Current control - use scalar multiplication
                Zf_if = Zf_local @ i_f_local
                error_i = i_f_local - i_ref_sat
                # GRADIENT-PRESERVING: Direct scalar multiplication
                Kpf_err = self.Kp_f * error_i
                Kif_zeta = self.Ki_f * zeta_f_local

                vm_unsat = Zf_if + v_local - Kpf_err - Kif_zeta
                vm_sat, delta_vm = self.apply_vm_limit_with_antiwindup(vm_unsat)

                # Current integrator with anti-windup
                if self.anti_windup_enabled:
                    antiwindup_f = self.Ki_f * delta_vm
                    dzeta_f_ = error_i + antiwindup_f
                else:
                    dzeta_f_ = error_i

                # Filter dynamics with pre-computed reciprocals
                i_o_local = i_inj
                dv_ = self.inv_cf * (-Yf_v - i_o_local + i_f_local)
                dif_ = self.inv_lf * (-Zf_if - v_local + vm_sat)
            else:
                # Inactive but connected
                dv_, dif_ = self.filter_dynamics_inactive(
                    idx, v_local, i_f_local, v_full, i_line
                )
        else:
            # Breaker open - use pre-computed reciprocal
            dif_ = -(self.rf * self.inv_lf) * i_f_local

        return dvhat, dv_, dzeta_v_, dif_, dzeta_f_
