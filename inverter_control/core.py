"""
Core utilities for per-unit power system simulation.

Contains:
- PerUnitSystem: Per-unit conversions and base values
- Setpoints: Dataclass for system setpoints
- super_safe_solve: Robust Newton solver
"""

import math
import torch
import numpy as np
from dataclasses import dataclass
from scipy.optimize import fsolve
from typing import Tuple

torch.set_float32_matmul_precision('high')


@dataclass
class Setpoints:
    """System setpoints for power system operation."""
    v_star: torch.Tensor
    p_star: torch.Tensor
    q_star: torch.Tensor
    theta_star: torch.Tensor


def super_safe_solve(F, x0, tol=1e-4, max_iter=50):
    """
    Robust Newton solver with fallback strategies.
    
    Args:
        F: Function to find roots of
        x0: Initial guess
        tol: Tolerance for convergence
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (solution, success, residual, message)
    """
    try:
        sol, info, ier, msg = fsolve(F, x0, full_output=True, xtol=tol)
        residual = np.linalg.norm(F(sol))
        if ier == 1 and residual < tol:
            return sol, True, residual, msg
        x0_perturbed = x0 + 0.1 * np.random.randn(*x0.shape)
        sol2, info2, ier2, msg2 = fsolve(F, x0_perturbed, full_output=True, xtol=tol)
        residual2 = np.linalg.norm(F(sol2))
        if residual2 < residual:
            return sol2, (residual2 < tol), residual2, msg2
        else:
            return sol, (residual < tol), residual, msg
    except Exception as e:
        return x0, False, float('inf'), str(e)


class PerUnitSystem:
    """
    Per-unit system calculations and conversions.
    
    Provides conversion between SI units and per-unit values for power system quantities.
    
    Args:
        Sb: Base power (VA)
        Vb: Base voltage (V)
        fb: Base frequency (Hz)
        device: Torch device
        dtype: Torch dtype
    """
    def __init__(self, Sb=1e3, Vb=120.0, fb=60.0, device='cpu', dtype=torch.float64):
        self.Sb = Sb
        self.Vb = Vb
        self.fb = fb
        self.ωb = 2 * math.pi * fb
        self.Ib = Sb / Vb
        self.Zb = Vb**2 / Sb
        self.Lb = self.Zb / self.ωb
        self.Cb = 1.0 / (self.ωb * self.Zb)
        self.Yb = 1.0 / self.Zb
        self.device = device
        self.dtype = dtype

    def to_pu(self, value, quantity):
        """
        Convert SI value to per-unit.
        
        Args:
            value: SI value to convert
            quantity: Type of quantity ('voltage', 'current', 'power', etc.)
            
        Returns:
            Per-unit value
        """
        conversions = {
            'voltage': value / self.Vb,
            'current': value / self.Ib,
            'power': value / self.Sb,
            'impedance': value / self.Zb,
            'resistance': value / self.Zb,
            'reactance': value / self.Zb,
            'inductance': value * self.ωb / self.Zb,
            'capacitance': value * self.Zb * self.ωb,
            'admittance': value * self.Zb,
            'conductance': value * self.Zb,
            'frequency': value / self.fb,
            'angular_frequency': value / self.ωb
        }
        return conversions.get(quantity, value)

    def from_pu(self, value, quantity):
        """
        Convert per-unit value to SI.
        
        Args:
            value: Per-unit value to convert
            quantity: Type of quantity ('voltage', 'current', 'power', etc.)
            
        Returns:
            SI value
        """
        conversions = {
            'voltage': value * self.Vb,
            'current': value * self.Ib,
            'power': value * self.Sb,
            'impedance': value * self.Zb,
            'resistance': value * self.Zb,
            'reactance': value * self.Zb,
            'inductance': value * self.Zb / self.ωb,
            'capacitance': value / (self.Zb * self.ωb),
            'admittance': value / self.Zb,
            'conductance': value / self.Zb,
            'frequency': value * self.fb,
            'angular_frequency': value * self.ωb
        }
        return conversions.get(quantity, value)


def as_finite_tensor(x, dtype, device):
    """
    Convert x to a finite tensor on the specified device/dtype.
    
    Replaces infinite values with large finite values.
    
    Args:
        x: Value to convert
        dtype: Torch dtype
        device: Torch device
        
    Returns:
        Finite tensor
    """
    t = torch.as_tensor(x, dtype=dtype, device=device)
    t = torch.where(torch.isfinite(t),
                    t,
                    torch.full_like(t, 1e12).copysign(t))
    return t
