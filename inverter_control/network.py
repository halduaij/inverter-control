"""
Power System Network Model - OPTIMIZED VERSION.

Contains:
- PowerSystemNetwork: Network topology, matrices, and line dynamics with batch support

Optimizations Applied (gradient-safe):
- #3: Cached algebraic line current solver matrices (Z_inv, S_matrix, etc.)
- #4: Vectorized line current computation with einsum
- #5: Pre-computed block matrix views
- #9: Cache invalidation hook for load changes
"""

import math
import torch
from typing import Tuple, Optional, Union

from .core import PerUnitSystem


class PowerSystemNetwork:
    """
    Power system network model in per-unit with batch support.
    
    Implements the electrical network including transmission lines, load,
    and network topology matrices. Supports both algebraic and differential
    line current formulations.
    
    Args:
        batch_size: Number of parallel trajectories
        device: Torch device ('cuda' or 'cpu')
        dtype: Torch data type
    """
    def __init__(self, batch_size=1, device='cuda', dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        # Initialize per-unit system
        self.pu = PerUnitSystem(Sb=1000.0, Vb=120.0, fb=60.0, device=device, dtype=dtype)

        # Network size
        self.Nc = 3  # Number of converters
        self.Nt = 3  # Number of transmission lines
        self.n = 2   # αβ dimension

        # Network parameters in SI (from Table I)
        rt_si = 0.05        # Line resistance (50 mΩ)
        lt_si = 0.2e-3      # Line inductance (0.2 mH)
        rL_si = 115.0       # Load resistance (115 Ω)

        # Convert to per-unit
        self.rt = torch.tensor(self.pu.to_pu(rt_si, 'resistance'),
                       dtype=dtype, device=device)
        self.lt = torch.tensor(self.pu.to_pu(lt_si, 'inductance'),
                              dtype=dtype, device=device)
        self.omega0 = torch.tensor(1.0, dtype=dtype, device=device)

        # Store original values for scenario changes
        rL_pu = self.pu.to_pu(rL_si, 'resistance')
        self.original_rL = torch.tensor(rL_pu, dtype=dtype, device=device)

        # BATCH SUPPORT: rL can be tensor for different loads per trajectory
        if batch_size > 1:
            self.rL = torch.full((batch_size,), rL_pu, dtype=dtype, device=device)
        else:
            self.rL = rL_pu

        # Calculate kappa
        self.kappa = math.atan(self.omega0 * self.lt / self.rt)

        # R(κ) matrix
        R_kappa_base = torch.tensor([
            [math.cos(self.kappa), -math.sin(self.kappa)],
            [math.sin(self.kappa), math.cos(self.kappa)]
        ], dtype=self.dtype, device=self.device)

        self.R_kappa = torch.kron(
            torch.eye(self.Nc, dtype=self.dtype, device=self.device),
            R_kappa_base
        )

        # OPTIMIZATION #5: Pre-compute R_kappa block views
        self.R_kappa_blocks = [self.R_kappa[2*i:2*i+2, 2*i:2*i+2] for i in range(self.Nc)]

        # Breaker status
        self.breaker_status = torch.ones(self.Nc, dtype=torch.bool, device=self.device)

        # Standard matrices
        self.J = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=self.dtype, device=self.device)
        self.In = torch.eye(self.n, dtype=self.dtype, device=self.device)

        # OPTIMIZATION #3: Line current cache state
        self._line_current_cache_valid = False
        self._Z_inv_list = None
        self._Z_inv_stacked = None
        self._S_matrix = None
        
        self.setup_network()

    def setup_network(self):
        """Setup network matrices in αβ frame."""
        # Network topology - radial connection
        B_lines = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=self.dtype, device=self.device)

        self.B_lines = B_lines
        self.B = torch.kron(B_lines, self.In)

        # Transmission line matrices (per-unit)
        self.Rt = torch.kron(
            torch.eye(self.Nt, dtype=self.dtype, device=self.device),
            self.rt * self.In
        )
        self.Lt = torch.kron(
            torch.eye(self.Nt, dtype=self.dtype, device=self.device),
            self.lt * self.In
        )
        self.Jnt = torch.kron(
            torch.eye(self.Nt, dtype=self.dtype, device=self.device),
            self.J
        )

        # Line impedance
        self.Zt = self.Rt + self.omega0 * (self.Jnt @ self.Lt)

        self.update_network_matrices()
        
        # OPTIMIZATION #3: Build and cache line current matrices
        self._cache_line_current_matrices()

    def _cache_line_current_matrices(self):
        """
        OPTIMIZATION #3: Cache matrices for algebraic line current computation.
        
        These matrices depend only on network topology and impedances (not learnable
        control gains), so they are safe to cache without breaking gradient flow.
        """
        Nc = self.Nc
        device = self.device
        dtype = self.dtype
        
        # Pre-compute Z inverses for each line
        self._Z_inv_list = []
        for i in range(Nc):
            idx = slice(2*i, 2*(i+1))
            Z_i = self.Zt[idx, idx]
            Z_i_inv = torch.linalg.inv(Z_i)
            self._Z_inv_list.append(Z_i_inv)
        
        # Stack for vectorized operations: [Nc, 2, 2]
        self._Z_inv_stacked = torch.stack(self._Z_inv_list, dim=0)
        
        # Sum of Z inverses (used in the algebraic solution)
        self._S_matrix = self._Z_inv_stacked.sum(dim=0)  # [2, 2]
        
        self._line_current_cache_valid = True

    def invalidate_line_current_cache(self):
        """
        OPTIMIZATION #9: Invalidate line current cache when network changes.
        Call this when load or impedance parameters change.
        """
        self._line_current_cache_valid = False

    def update_breaker_status(self, status: torch.Tensor):
        """
        Update breaker status.
        
        Args:
            status: Boolean tensor of breaker states [Nc]
        """
        self.breaker_status = status.to(device=self.device)
        self.update_network_matrices()

    def update_network_matrices(self):
        """Update active network matrices based on breaker status."""
        if torch.all(self.breaker_status):
            # All breakers closed - no need to modify
            self.B_active = self.B
        else:
            # Apply breaker mask (would implement breaker logic here)
            self.B_active = self.B.clone()

    def calculate_total_currents(self, v_nodes: torch.Tensor, i_line: torch.Tensor) -> torch.Tensor:
        """
        Calculate total current injections.
        
        Args:
            v_nodes: Node voltages [2*Nc] or [batch, 2*Nc]
            i_line: Line currents [2*Nt] or [batch, 2*Nt]
            
        Returns:
            Total current injections [2*Nc] or [batch, 2*Nc]
        """
        if v_nodes.dim() == 1:
            return self.B_active @ i_line
        else:
            # Batch case - preserve same matrix multiplication
            B_batch = self.B_active.unsqueeze(0).expand(i_line.shape[0], -1, -1)
            return torch.bmm(B_batch, i_line.unsqueeze(-1)).squeeze(-1)

    def update_batch_loads(self, load_factors: torch.Tensor):
        """
        Update load values for batch simulation.
        
        Args:
            load_factors: Tensor of load scaling factors [batch_size]
        """
        # OPTIMIZATION #9: Invalidate cache when loads change
        self.invalidate_line_current_cache()
        
        if self.batch_size > 1:
            base_rL_si = 115.0
            rL_si_batch = base_rL_si * load_factors
            self.rL = torch.tensor([self.pu.to_pu(rL_si, 'resistance') for rL_si in rL_si_batch],
                                 dtype=self.dtype, device=self.device)
        else:
            base_rL_si = 115.0 * load_factors.item() if load_factors.numel() == 1 else 115.0 * load_factors[0].item()
            self.rL = self.pu.to_pu(base_rL_si, 'resistance')

    def compute_algebraic_line_currents(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute steady-state line currents algebraically - TRUE PARALLEL.
        
        Uses algebraic equations to solve for line currents given node voltages,
        avoiding differential equation integration for the fast line dynamics.
        
        OPTIMIZATIONS APPLIED:
        - #3: Uses cached Z_inv matrices
        - #4: Vectorized computation with einsum
        
        Args:
            v: Node voltages [2*Nc] or [batch, 2*Nc]
            
        Returns:
            Tuple of (line_currents, common_voltage)
        """
        # Ensure cache is valid
        if not self._line_current_cache_valid:
            self._cache_line_current_matrices()
        
        if v.dim() == 1:
            return self._compute_single_algebraic_line_currents(v)

        # TRUE PARALLEL IMPLEMENTATION with optimizations
        batch_size = v.shape[0]
        Nc = self.Nc
        device = self.device
        dtype = self.dtype

        # Reshape voltage
        v_nodes = v.view(batch_size, Nc, 2)  # [batch, Nc, 2]

        # OPTIMIZATION #3 & #4: Use cached matrices and einsum
        Z_inv_stacked = self._Z_inv_stacked  # [Nc, 2, 2]
        S = self._S_matrix  # [2, 2]
        
        # Expand S for batch: [batch, 2, 2]
        S_batch = S.unsqueeze(0).expand(batch_size, -1, -1)

        # OPTIMIZATION #4: Vectorized T computation using einsum
        # v_nodes: [batch, Nc, 2], Z_inv_stacked: [Nc, 2, 2]
        T_batch = torch.einsum('bci,cij->bj', v_nodes, Z_inv_stacked)  # [batch, 2]

        # Handle rL for batch
        I2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0)  # [1, 2, 2]

        if self.batch_size > 1 and isinstance(self.rL, torch.Tensor):
            rL_batch = self.rL.view(batch_size, 1, 1)  # [batch, 1, 1]
            rL_vec = self.rL.view(batch_size, 1)  # [batch, 1]
        else:
            rL_val = self.rL if not isinstance(self.rL, torch.Tensor) else self.rL.item()
            rL_batch = rL_val
            rL_vec = rL_val

        # Batch solve for v_common
        A_batch = I2 + rL_batch * S_batch  # [batch, 2, 2]
        b_batch = rL_vec * T_batch  # [batch, 2]
        v_common = torch.linalg.solve(A_batch, b_batch)  # [batch, 2]

        # OPTIMIZATION #4: Vectorized line current computation using einsum
        v_common_expanded = v_common.unsqueeze(1)  # [batch, 1, 2]
        v_diff = v_nodes - v_common_expanded  # [batch, Nc, 2]

        # Compute line currents: i_line_k = Z_k^{-1} @ (v_k - v_common)
        i_line_parts = torch.einsum('cij,bcj->bci', Z_inv_stacked, v_diff)  # [batch, Nc, 2]
        i_line = i_line_parts.reshape(batch_size, -1)  # [batch, 2*Nc]

        return i_line, v_common

    def _compute_single_algebraic_line_currents(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Original single trajectory implementation - OPTIMIZED.
        
        OPTIMIZATIONS APPLIED:
        - #3: Uses cached Z_inv matrices
        - #4: Uses einsum for vectorization
        """
        Nc = self.Nc
        v_nodes = v.view(Nc, 2)

        # OPTIMIZATION #3: Use cached matrices
        if not self._line_current_cache_valid:
            self._cache_line_current_matrices()
        
        Z_inv_stacked = self._Z_inv_stacked  # [Nc, 2, 2]
        S = self._S_matrix  # [2, 2]
        
        # OPTIMIZATION #4: Use einsum for T computation
        T = torch.einsum('cij,cj->i', Z_inv_stacked, v_nodes)  # [2]

        I2 = torch.eye(2, dtype=self.dtype, device=self.device)
        # Handle rL as: Python scalar, 0-dim tensor, or 1-dim batched tensor
        if isinstance(self.rL, (int, float)):
            rL_scalar = self.rL
        elif isinstance(self.rL, torch.Tensor):
            if self.rL.dim() == 0:
                rL_scalar = self.rL.item()
            else:
                rL_scalar = self.rL[0].item()
        else:
            rL_scalar = float(self.rL)
        v_common = torch.linalg.solve(I2 + rL_scalar * S, rL_scalar * T)

        # OPTIMIZATION #4: Vectorized line current computation
        v_diff = v_nodes - v_common.unsqueeze(0)  # [Nc, 2]
        i_line_parts = torch.einsum('cij,cj->ci', Z_inv_stacked, v_diff)  # [Nc, 2]
        i_line = i_line_parts.reshape(-1)  # [2*Nc]

        return i_line, v_common

    def line_dynamics(self, v: torch.Tensor, i_line: torch.Tensor) -> torch.Tensor:
        """
        Line dynamics equation in per-unit.
        
        Computes di_line/dt for differential line current formulation.
        
        Args:
            v: Node voltages [2*Nc] or [batch, 2*Nc]
            i_line: Line currents [2*Nt] or [batch, 2*Nt]
            
        Returns:
            Time derivative of line currents
        """
        if v.dim() == 1:
            # Single trajectory - original code
            i_sum = torch.sum((self.B_active @ i_line).view(self.Nc, 2), dim=0)
            # Handle rL as: Python scalar, 0-dim tensor, or 1-dim batched tensor
            if isinstance(self.rL, (int, float)):
                rL_scalar = self.rL
            elif isinstance(self.rL, torch.Tensor):
                if self.rL.dim() == 0:
                    rL_scalar = self.rL.item()
                else:
                    rL_scalar = self.rL[0].item()
            else:
                rL_scalar = float(self.rL)
            v_common = rL_scalar * i_sum
            v_diff = (self.B_active.T @ v) - torch.cat([v_common for _ in range(self.Nt)], dim=0)
            rhs = -self.Zt @ i_line + v_diff
            di_line = torch.linalg.solve(self.Lt, rhs)
            return di_line

        # Batch implementation
        batch_size = v.shape[0]

        # Current injection: B @ i_line
        B_batch = self.B_active.unsqueeze(0).expand(batch_size, -1, -1)
        i_inj = torch.bmm(B_batch, i_line.unsqueeze(-1)).squeeze(-1)
        i_inj_reshaped = i_inj.view(batch_size, self.Nc, 2)
        i_sum = i_inj_reshaped.sum(dim=1)  # [batch, 2]

        # Common voltage
        if self.batch_size > 1 and isinstance(self.rL, torch.Tensor):
            v_common = self.rL.view(batch_size, 1) * i_sum
        else:
            rL_val = self.rL if not isinstance(self.rL, torch.Tensor) else self.rL.item()
            v_common = rL_val * i_sum

        # Line voltages: B.T @ v
        BT_batch = self.B_active.T.unsqueeze(0).expand(batch_size, -1, -1)
        v_lines = torch.bmm(BT_batch, v.unsqueeze(-1)).squeeze(-1)

        # v_common expansion
        v_common_expanded = v_common.repeat(1, self.Nt)  # [batch, 2*Nt]

        v_diff = v_lines - v_common_expanded

        # RHS computation
        Zt_batch = self.Zt.unsqueeze(0).expand(batch_size, -1, -1)
        Zt_i = torch.bmm(Zt_batch, i_line.unsqueeze(-1)).squeeze(-1)
        rhs = -Zt_i + v_diff

        # Batch solve
        Lt_batch = self.Lt.unsqueeze(0).expand(batch_size, -1, -1)
        di_line = torch.linalg.solve(Lt_batch, rhs.unsqueeze(-1)).squeeze(-1)

        return di_line
