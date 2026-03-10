"""
Local Neural Network Schedulers for Adaptive dVOC Control.

SCALABLE ARCHITECTURE: Each converter uses only local information.

Contains:
- LocalGainScheduler: PI gains from local setpoint only
- LocalEtaScheduler: η_k from local setpoint + local errors  
- LocalEtaAAdapter: η_{a,k} as per-converter dynamic state (CORRECTED)
- SupervisoryLayer: Computes global bounds at slow timescale
- DistributedAdaptiveController: Orchestrates all local schedulers

CORRECTED η_a ADAPTATION (based on rigorous Lyapunov analysis):
============================================================

The Lyapunov function for the dVOC reference model (Eq. 27 in Subotić et al.) is:

    V_v̂(v̂) = (1/2) v̂ᵀ P_S v̂ + (1/2) η η_a α₁ Σ_k Φ_k²

where:
    - P_S is a projection matrix encoding synchronization
    - Φ_k = 1 - ||v̂_k||²/v_k*² is the voltage regulation error
    - α₁ = c_L / (5η||K-L||²) from Eq. 28

When η_a is time-varying, the time derivative of V gets an additional term:

    dV/dt = (original terms) + (1/2) η α₁ (dη_a/dt) Σ_k Φ_k²

The PASSIVITY CONSTRAINT for stability when η̇_a > 0:

    σ_k · η̇_a,k ≤ θ · α₁ · ψ_k²

where:
    - σ_k = (1/2) η_k α₁ Φ_k² (coefficient of η_a in V)
    - ψ_k = η_k(||K-L|| ||v̂_k||/v_k* + η_{a,k}|Φ_k|||v̂_k||) (comparison function)

KEY CORRECTIONS FROM ORIGINAL:
1. REMOVED redundant err_mag scaling - passivity constraint already ensures stability
2. FIXED Φ→0 limit handling - ratio ψ²/σ → ∞ as Φ→0 (no constraint near equilibrium)
3. FIXED v_error_k=None handling - now properly passes None instead of zero tensor
4. ADDED option for tracking-error-based activation (operationally superior)
5. CORRECTED σ formula to match Lyapunov function exactly
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from .core import Setpoints


class LocalGainScheduler(nn.Module):
    """
    Fully local gain scheduler for a SINGLE converter.
    
    Input: [p_k*, q_k*, v_k*, rL_k] = 4 scalars (O(1) complexity)
    Output: [Kp_v_k, Ki_v_k, Kp_f_k, Ki_f_k] for this converter
    
    Can be:
    - Shared weights: One instance applied to all converters (homogeneous)
    - Per-converter: Separate instance per converter (heterogeneous)
    
    Args:
        hidden_dim: Hidden layer dimension
        delta_scale: Max log-adjustment (0.5 = ±40% variation)
    """
    
    def __init__(self, hidden_dim: int = 32, delta_scale: float = 0.5):
        super().__init__()
        
        self.delta_scale = delta_scale

        # Input: [p_k*, q_k*, v_k*, rL_k, v_error_k] = 5 features
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4)  # [adj_Kp_v, adj_Ki_v, adj_Kp_f, adj_Ki_f]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with moderate gain for visible variation from start."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.3)  # Increased from 0.1
                nn.init.zeros_(layer.bias)
    
    def forward(self, p_k: torch.Tensor, q_k: torch.Tensor,
                v_k: torch.Tensor, rL_k: torch.Tensor,
                Kp_v_base: torch.Tensor, Ki_v_base: torch.Tensor,
                Kp_f_base: torch.Tensor, Ki_f_base: torch.Tensor,
                cf: float, lf: float,
                phi_k: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        """
        Compute scheduled gains for converter k.

        Args:
            p_k, q_k, v_k: Local setpoints (scalars)
            rL_k: Local load resistance (or global if uniform)
            *_base: Base gain values
            cf, lf: Filter capacitance/inductance for constraints
            phi_k: Amplitude regulation error Φ_k = 1 - ||v̂_k||²/v_k*² (vhat-based)

        Returns:
            Tuple of (Kp_v_k, Ki_v_k, Kp_f_k, Ki_f_k)
        """
        # Use phi_k (vhat-based) instead of actual voltage error to avoid noise
        if phi_k is None:
            phi_normalized = torch.zeros_like(v_k)
        else:
            # Normalize phi_k: typical range [-0.1, 0.1], use tanh for soft saturation
            phi_normalized = torch.tanh(phi_k / 0.05)

        # Normalize inputs for network stability
        # IMPROVED: Make rL more visible - it's the most important feature!
        # rL_pu typically in [2, 15], use log scale to preserve sensitivity
        # log(2)=0.69, log(15)=2.71, so (log(rL)-1.5)/1.0 gives [-0.8, 1.2] range
        log_rL_normalized = (torch.log(rL_k + 0.1) - 1.5) / 1.0

        features = torch.stack([
            torch.tanh(p_k * 5.0),  # Amplify small power values
            torch.tanh(q_k * 5.0),  # Same for reactive power
            v_k - 1.0,  # Deviation from nominal (small but informative)
            log_rL_normalized,  # Log-scale rL - MOST IMPORTANT
            phi_normalized  # vhat-based error
        ])
        
        adjustments = self.net(features)
        
        # Bounded multiplicative adjustments
        adj_Kp_v = torch.exp(torch.tanh(adjustments[0]) * self.delta_scale)
        adj_Ki_v = torch.exp(torch.tanh(adjustments[1]) * self.delta_scale)
        adj_Kp_f = torch.exp(torch.tanh(adjustments[2]) * self.delta_scale)
        adj_Ki_f = torch.exp(torch.tanh(adjustments[3]) * self.delta_scale)
        
        Kp_v = Kp_v_base * adj_Kp_v
        Ki_v = Ki_v_base * adj_Ki_v
        Kp_f = Kp_f_base * adj_Kp_f
        Ki_f = Ki_f_base * adj_Ki_f

        # Hard constraints from Conditions 5-6 (in per-unit)
        # From constraints.py check_condition5/6:
        #   ratio_SI = omega_b² * Ki_pu / (cf_pu or lf_pu) > 1
        # So: Ki_pu > (cf_pu or lf_pu) / omega_b²
        omega_b = 2.0 * 3.141592653589793 * 60.0  # 376.99 rad/s
        omega_b_sq = omega_b * omega_b
        Ki_v = torch.clamp(Ki_v, min=cf / omega_b_sq * 1.01)
        Ki_f = torch.clamp(Ki_f, min=lf / omega_b_sq * 1.01)

        return Kp_v, Ki_v, Kp_f, Ki_f

    def forward_batch(
        self,
        p_all: torch.Tensor,      # [Nc]
        q_all: torch.Tensor,      # [Nc]
        v_all: torch.Tensor,      # [Nc]
        rL: torch.Tensor,         # scalar or [Nc]
        Kp_v_base: torch.Tensor,
        Ki_v_base: torch.Tensor,
        Kp_f_base: torch.Tensor,
        Ki_f_base: torch.Tensor,
        cf: float,
        lf: float,
        phi_all: torch.Tensor = None  # [Nc]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Vectorized forward for ALL converters at once.

        Args:
            p_all, q_all, v_all: Setpoints for all converters [Nc]
            rL: Load resistance (scalar broadcast or [Nc])
            *_base: Base gain values (scalars)
            cf, lf: Filter parameters
            phi_all: Amplitude errors for all converters [Nc]

        Returns:
            Tuple of (Kp_v_all, Ki_v_all, Kp_f_all, Ki_f_all), each [Nc]
        """
        Nc = p_all.shape[0]
        device = p_all.device
        dtype = p_all.dtype

        # Handle rL broadcasting
        if rL.dim() == 0:
            rL_all = rL.expand(Nc)
        else:
            rL_all = rL

        # Phi normalization
        if phi_all is None:
            phi_normalized = torch.zeros(Nc, dtype=dtype, device=device)
        else:
            phi_normalized = torch.tanh(phi_all / 0.05)

        # Build feature matrix [Nc, 5]
        # IMPROVED: Make rL more visible with log-scale normalization
        log_rL_normalized = (torch.log(rL_all + 0.1) - 1.5) / 1.0

        features = torch.stack([
            torch.tanh(p_all * 5.0),  # Amplify small power values
            torch.tanh(q_all * 5.0),  # Same for reactive power
            v_all - 1.0,  # Deviation from nominal
            log_rL_normalized,  # Log-scale rL - MOST IMPORTANT
            phi_normalized
        ], dim=1)  # [Nc, 5]

        # Single batched network call
        adjustments = self.net(features)  # [Nc, 4]

        # Bounded multiplicative adjustments
        adj = torch.exp(torch.tanh(adjustments) * self.delta_scale)  # [Nc, 4]

        Kp_v_all = Kp_v_base * adj[:, 0]
        Ki_v_all = Ki_v_base * adj[:, 1]
        Kp_f_all = Kp_f_base * adj[:, 2]
        Ki_f_all = Ki_f_base * adj[:, 3]

        # Hard constraints
        omega_b = 2.0 * 3.141592653589793 * 60.0
        omega_b_sq = omega_b * omega_b
        Ki_v_all = torch.clamp(Ki_v_all, min=cf / omega_b_sq * 1.01)
        Ki_f_all = torch.clamp(Ki_f_all, min=lf / omega_b_sq * 1.01)

        return Kp_v_all, Ki_v_all, Kp_f_all, Ki_f_all


class LocalEtaScheduler(nn.Module):
    """
    Per-converter η_k scheduler using only local information.
    
    NO NEW FAST STATES - maintains passivity.

    Structure: η_k(t) = η_base_param × adjustment_k(setpoints) × boost_k(errors)

    - η_base_param: Base eta from simulation (passed in)
    - adjustment_k: Multiplicative adjustment from neural network (slow, setpoint-based)
    - boost_k: Function of local error features (responds to v/v̂ during transients)

    IMPROVEMENTS OVER ORIGINAL (fixing ~1% variation issue):
    1. Soft normalization: tanh(x/scale) instead of x/(|x|+0.1) - preserves sensitivity
    2. Larger boost_max (0.5 instead of 0.3) - allows ±50% variation
    3. Direct tanh output instead of sigmoid - better gradient flow
    4. Larger weight initialization for boost_net (gain=0.5) - faster response
    5. Analytical boost term - guaranteed response even before learning
    6. Better features: signed voltage deviation instead of ratio near 1.0

    Args:
        hidden_dim: Hidden layer dimension
        boost_max: Maximum boost factor (0.5 = ±50% variation)
        eta_min: Minimum allowed η
        delta_scale: Max log-adjustment for base (0.5 = ±40% variation)
        use_analytical_boost: If True, add model-based boost term
        analytical_gain: Gain for analytical boost (higher = more responsive)
        error_scale: Scale for error normalization in tanh (default 0.02 p.u.)
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        boost_max: float = 3,           # INCREASED from 0.3
        eta_min: float = 1e-4,
        delta_scale: float = 0.5,
        use_analytical_boost: bool = True,  # Guaranteed transient response
        analytical_gain: float = 5.0,       # Gain for analytical term
        error_scale: float = 0.02           # 2% p.u. scale for normalization
    ):
        super().__init__()

        self.boost_max = boost_max
        self.eta_min = eta_min
        self.delta_scale = delta_scale
        self.use_analytical_boost = use_analytical_boost
        self.analytical_gain = analytical_gain
        self.error_scale = error_scale

        # Base network: setpoint-based adjustment (SLOW)
        # Input: [p_k*, q_k*, v_k*, rL_k] = 4 features (rL is crucial!)
        self.base_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Boost network: error-based adjustment (responds to v/v̂ during transients)
        # Input: [|v_error|_norm, Φ_k_norm, (||v̂||-v*)/v*] = 3 features
        # NO sigmoid - direct output, we apply tanh externally
        self.boost_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with appropriate gains."""
        # Base net: moderate initialization for visible load-dependent variation
        for layer in self.base_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.3)  # Increased from 0.1
                nn.init.zeros_(layer.bias)

        # Boost net: LARGER initialization for better transient response
        for layer in self.boost_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)

    def compute_eta_base(self, p_k: torch.Tensor, q_k: torch.Tensor,
                         v_k: torch.Tensor, eta_base_param: torch.Tensor,
                         rL_k: torch.Tensor = None) -> torch.Tensor:
        """Compute adjusted η_k from local operating point and base eta (SLOW)."""
        # Handle rL default
        if rL_k is None:
            rL_k = torch.tensor(8.0, dtype=p_k.dtype, device=p_k.device)

        # IMPROVED: Include rL with log-scale normalization
        log_rL_normalized = (torch.log(rL_k + 0.1) - 1.5) / 1.0

        features = torch.stack([
            torch.tanh(p_k * 5.0),  # Amplify small power values
            torch.tanh(q_k * 5.0),  # Same for reactive power
            v_k - 1.0,  # Deviation from nominal
            log_rL_normalized  # Log-scale rL - MOST IMPORTANT
        ])

        adj_raw = self.base_net(features).squeeze()
        adjustment = torch.exp(torch.tanh(adj_raw) * self.delta_scale)

        eta_base = eta_base_param * adjustment
        eta_base = torch.clamp(eta_base, min=self.eta_min)

        return eta_base
    
    def compute_boost(self, phi_k: torch.Tensor,
                      vhat_norm_k: torch.Tensor,
                      v_k: torch.Tensor,
                      eta_a_k: torch.Tensor = None) -> torch.Tensor:
        """
        State-dependent boost using LOCAL error features.

        This is where η responds to v̂ during transients.

        IMPORTANT: Uses ONLY vhat-based features (phi_k, vhat_norm_k) to avoid noise
        from actual voltage dynamics. The actual voltage v is NOT used.

        Features:
        1. phi_k: Amplitude regulation error Φ_k = 1 - ||v̂_k||²/v_k*²
        2. Signed voltage deviation: (||v̂|| - v*) / v*
        3. eta_a_k: Current eta_a state (allows eta to respond to eta_a dynamics)
        """
        # === FEATURE 1: Φ_k (amplitude regulation error from vhat only) ===
        # phi_k = 1 - ||v̂||²/v*² -> depends only on vhat, not actual v
        phi_normalized = torch.tanh(phi_k / 0.05)  # Scale ~5%

        # === FEATURE 2: Signed voltage deviation (from vhat only) ===
        # (||v̂|| - v*) / v* - positive = overvoltage, negative = undervoltage
        voltage_deviation = (vhat_norm_k - v_k) / (v_k + 1e-6)
        voltage_dev_normalized = torch.tanh(voltage_deviation / 0.05)

        # === FEATURE 3: eta_a state (allows eta to couple with eta_a) ===
        if eta_a_k is not None:
            # Normalize eta_a: typical range [0.1, 10], use log scale
            eta_a_normalized = torch.tanh((eta_a_k - 1.0) / 2.0)  # Center at 1.0
        else:
            eta_a_normalized = torch.zeros_like(phi_k)

        # === BUILD FEATURE VECTOR ===
        features = torch.stack([
            phi_normalized,          # Amplitude regulation state (from vhat)
            voltage_dev_normalized,  # Signed deviation from setpoint (from vhat)
            eta_a_normalized         # Current eta_a state
        ])

        # === NEURAL NETWORK BOOST ===
        # Direct output → tanh (NO sigmoid saturation)
        boost_raw = self.boost_net(features).squeeze()
        nn_boost = torch.tanh(boost_raw) * self.boost_max  # ∈ [-boost_max, +boost_max]

        # === ANALYTICAL BOOST ===
        # GUARANTEED response proportional to error - works even before learning
        # Uses |phi_k| as error measure (depends only on vhat)
        if self.use_analytical_boost:
            # analytical_gain=5 means: 5% phi error → tanh(5×1)≈1 → full boost contribution
            phi_error_normalized = torch.tanh(torch.abs(phi_k) / self.error_scale)
            analytical_boost = torch.tanh(
                self.analytical_gain * phi_error_normalized
            ) * self.boost_max * 0.5  # Up to boost_max/2 from analytical term
        else:
            analytical_boost = torch.tensor(0.0, device=phi_k.device, dtype=phi_k.dtype)

        # === COMBINE ===
        # nn_boost ∈ [-boost_max, +boost_max]
        # analytical_boost ∈ [0, boost_max/2] (always positive during errors)
        boost = 1.0 + nn_boost + analytical_boost

        # Safety clamp
        boost = torch.clamp(boost, min=0.5, max=1.0 + 1.5 * self.boost_max)

        return boost
    
    def forward(self, p_k: torch.Tensor, q_k: torch.Tensor, v_k: torch.Tensor,
                phi_k: torch.Tensor, vhat_norm_k: torch.Tensor,
                eta_base_param: torch.Tensor,
                rL_k: torch.Tensor = None,  # ADDED: load resistance
                eta_upper: torch.Tensor = None,
                eta_a_k: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute η_k for converter k.

        Args:
            p_k, q_k, v_k: Local setpoints
            phi_k: Local Φ_k = 1 - ||v̂_k||²/v_k*² (depends on vhat only)
            vhat_norm_k: ||v̂_k|| (depends on vhat only)
            eta_base_param: Base eta from simulation parameters
            rL_k: Load resistance (crucial for load-dependent scheduling!)
            eta_upper: Optional upper bound from stability constraints
            eta_a_k: Current eta_a state (allows eta to couple with eta_a)

        Returns:
            Tuple of (eta_k, eta_base_k)
            Use eta_base_k for constraint checking (conservative)
        """
        eta_base = self.compute_eta_base(p_k, q_k, v_k, eta_base_param, rL_k)
        boost = self.compute_boost(phi_k, vhat_norm_k, v_k, eta_a_k)

        eta_k = eta_base * boost

        # NOTE: eta_upper hard constraint is INTENTIONALLY DISABLED for offline training.
        # Stability conditions (4, 5, 6) are enforced via soft Lagrangian penalties in the
        # training loss, allowing the optimizer to converge to satisfying them rather than
        # enforcing them at every instant. This provides more flexibility for gradient descent.
        # The trained scheduler will output values that satisfy conditions at deployment.
        #
        # For online/real-time use, uncomment to enforce hard constraint:
        # if eta_upper is not None:
        #     eta_k = torch.min(eta_k, eta_upper)

        return eta_k, eta_base

    def forward_batch(
        self,
        p_all: torch.Tensor,           # [Nc]
        q_all: torch.Tensor,           # [Nc]
        v_all: torch.Tensor,           # [Nc]
        phi_all: torch.Tensor,         # [Nc]
        vhat_norm_all: torch.Tensor,   # [Nc]
        eta_base_param: torch.Tensor,  # scalar
        rL: torch.Tensor = None,       # scalar or [Nc] - ADDED: load resistance
        eta_upper: torch.Tensor = None,
        eta_a_all: torch.Tensor = None  # [Nc]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized forward for ALL converters at once.

        Returns:
            Tuple of (eta_all, eta_base_all), each [Nc]
        """
        Nc = p_all.shape[0]
        device = p_all.device
        dtype = p_all.dtype

        # Handle rL broadcasting
        if rL is None:
            rL_all = torch.ones(Nc, dtype=dtype, device=device) * 8.0  # Default
        elif rL.dim() == 0:
            rL_all = rL.expand(Nc)
        else:
            rL_all = rL

        # === BATCHED compute_eta_base ===
        # IMPROVED: Include rL with log-scale normalization
        log_rL_normalized = (torch.log(rL_all + 0.1) - 1.5) / 1.0

        base_features = torch.stack([
            torch.tanh(p_all * 5.0),  # Amplify small power values
            torch.tanh(q_all * 5.0),  # Same for reactive power
            v_all - 1.0,  # Deviation from nominal
            log_rL_normalized  # Log-scale rL - MOST IMPORTANT
        ], dim=1)  # [Nc, 4]

        adj_raw = self.base_net(base_features).squeeze(-1)  # [Nc]
        adjustment = torch.exp(torch.tanh(adj_raw) * self.delta_scale)
        eta_base_all = eta_base_param * adjustment
        eta_base_all = torch.clamp(eta_base_all, min=self.eta_min)

        # === BATCHED compute_boost ===
        phi_normalized = torch.tanh(phi_all / 0.05)
        voltage_deviation = (vhat_norm_all - v_all) / (v_all + 1e-6)
        voltage_dev_normalized = torch.tanh(voltage_deviation / 0.05)

        if eta_a_all is not None:
            eta_a_normalized = torch.tanh((eta_a_all - 1.0) / 2.0)
        else:
            eta_a_normalized = torch.zeros(Nc, dtype=dtype, device=device)

        boost_features = torch.stack([
            phi_normalized,
            voltage_dev_normalized,
            eta_a_normalized
        ], dim=1)  # [Nc, 3]

        boost_raw = self.boost_net(boost_features).squeeze(-1)  # [Nc]
        nn_boost = torch.tanh(boost_raw) * self.boost_max

        # Analytical boost
        if self.use_analytical_boost:
            phi_error_normalized = torch.tanh(torch.abs(phi_all) / self.error_scale)
            analytical_boost = torch.tanh(
                self.analytical_gain * phi_error_normalized
            ) * self.boost_max * 0.5
        else:
            analytical_boost = torch.zeros(Nc, dtype=dtype, device=device)

        boost = 1.0 + nn_boost + analytical_boost
        boost = torch.clamp(boost, min=0.5, max=1.0 + 1.5 * self.boost_max)

        # === COMBINE ===
        eta_all = eta_base_all * boost

        return eta_all, eta_base_all


class LocalEtaAAdapter(nn.Module):
    """
    CORRECTED per-converter η_{a,k} adapter with rigorous Lyapunov passivity constraint.
    
    KEY CORRECTIONS FROM ORIGINAL IMPLEMENTATION:
    
    1. REMOVED err_mag scaling that caused double-counting:
       The original multiplied η̇_a by tanh(|Φ|/φ_scale) BEFORE applying passivity cap.
       But the passivity cap already accounts for Φ_k² through σ_k.
       This double-counting caused η̇_a → 0 even when passivity allows adaptation.
    
    2. PROPER Φ→0 LIMIT HANDLING:
       As Φ_k → 0:
       - σ_k = ½ηα₁Φ_k² → 0
       - ψ_k → η||K-L||||v̂||/v* (sync term dominates)
       - ψ²/σ → ∞ (no constraint near equilibrium)
       
       This is physically correct: near equilibrium, changing η_a has negligible 
       effect on V, so adaptation should be unrestricted.
    
    3. CORRECT σ FORMULA:
       From Lyapunov function (Eq. 27): σ_k = (1/2) η_k α₁ Φ_k²
       NOT the approximate σ_k = (c_L / 10||K-L||²) Φ_k²
    
    4. TRACKING-ERROR-BASED ACTIVATION (optional):
       Instead of gating by Φ_k (amplitude regulation state), can gate by
       ||v - v̂|| (tracking error). This allows adaptation when control 
       performance is poor, not just when amplitude is off.
    
    Args:
        hidden_dim: Hidden layer dimension for desired η_a network
        theta: Safety factor for passivity constraint (0.5 = 50% of stability budget)
        eta_a_min: Minimum allowed η_a
        eta_a_max: Maximum allowed η_a  
        tau: Time constant for first-order desired dynamics
        use_tracking_error: If True, use tracking error for optional activation
        tracking_error_scale: Scale for tracking error activation (p.u.)
        enable_equilibrium_gating: If True, add soft gating near equilibrium
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        theta: float = 0.5,
        eta_a_min: float = 0.1,
        eta_a_max: float = 10.0,
        tau: float = 0.01,
        use_tracking_error: bool = False,
        tracking_error_scale: float = 0.02,
        enable_equilibrium_gating: bool = False
    ):
        super().__init__()
        
        self.theta = theta
        self.eta_a_min = eta_a_min
        self.eta_a_max = eta_a_max
        self.tau = tau
        self.use_tracking_error = use_tracking_error
        self.tracking_error_scale = tracking_error_scale
        self.enable_equilibrium_gating = enable_equilibrium_gating

        # Input: [Φ_k, ||v̂_k||, v̂_k[0], v̂_k[1], p_k*, q_k*, v_k*, v_error_k, rL_k] = 9 features
        self.des_net = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive output
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize for moderate output at start."""
        for layer in self.des_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.3)
                nn.init.zeros_(layer.bias)

    def compute_eta_a_desired(
        self,
        vhat_k: torch.Tensor,
        phi_k: torch.Tensor,
        p_k: torch.Tensor,
        q_k: torch.Tensor,
        v_k: torch.Tensor,
        v_error_k: Optional[torch.Tensor] = None,
        rL_k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute desired η_a from local features.

        Args:
            vhat_k: Reference voltage [2]
            phi_k: Φ_k = 1 - ||v̂_k||²/v_k*² (scalar)
            p_k, q_k, v_k: Local setpoints (scalars)
            v_error_k: Voltage tracking error (optional, scalar)
            rL_k: Load resistance (crucial for load-dependent adaptation!)

        Returns:
            Desired η_a value (scalar)
        """
        vhat_norm = torch.linalg.norm(vhat_k)

        # Normalize v_error for network input
        if v_error_k is None:
            v_error_normalized = torch.zeros_like(v_k)
        else:
            v_error_normalized = v_error_k / (torch.abs(v_error_k) + 0.05)

        # Handle rL default and normalize with log scale
        if rL_k is None:
            rL_k = torch.tensor(8.0, dtype=p_k.dtype, device=p_k.device)
        log_rL_normalized = (torch.log(rL_k + 0.1) - 1.5) / 1.0

        features = torch.stack([
            torch.tanh(phi_k / 0.05),           # Normalized Φ_k (improved)
            vhat_norm,                          # ||v̂_k||
            vhat_k[0],                          # v̂_k,α
            vhat_k[1],                          # v̂_k,β
            torch.tanh(p_k * 5.0),             # Normalized p_k* (improved)
            torch.tanh(q_k * 5.0),             # Normalized q_k* (improved)
            v_k - 1.0,                          # v_k* deviation from nominal
            v_error_normalized,                 # Normalized voltage error
            log_rL_normalized                   # Log-scale rL - MOST IMPORTANT
        ])

        eta_a_des = self.des_net(features).squeeze()
        return torch.clamp(eta_a_des, min=self.eta_a_min, max=self.eta_a_max)

    def compute_sigma_local(
        self,
        eta_k: torch.Tensor,
        alpha1: torch.Tensor,
        phi_k: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute local contribution to σ (coefficient of η_a in V).
        
        From the Lyapunov function (Eq. 27):
            V contains term: (1/2) η η_a α₁ Σ_k Φ_k²
        
        So the coefficient of η_a for converter k is:
            σ_k = (1/2) η_k α₁ Φ_k²
        
        Args:
            eta_k: Local η value
            alpha1: Dissipation constant from Lyapunov analysis
            phi_k: Φ_k = 1 - ||v̂_k||²/v_k*²
        
        Returns:
            σ_k (always non-negative)
        """
        return 0.5 * eta_k * alpha1 * (phi_k ** 2)
    
    def compute_psi_local(
        self,
        eta_k: torch.Tensor,
        eta_a_k: torch.Tensor,
        vhat_k: torch.Tensor,
        phi_k: torch.Tensor,
        v_k: torch.Tensor,
        norm_KL: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute local contribution to ψ (comparison function from Assumption 2).
        
        From Assumption 2 in the paper:
            ψ = η(||K-L|| ||v̂||_S + η_a ||Φ(v̂)v̂||)
        
        For local approximation:
            ψ_k ≈ η_k (||K-L||·||v̂_k||/v_k* + η_{a,k}·|Φ_k|·||v̂_k||)
        
        Args:
            eta_k: Local η value
            eta_a_k: Local η_a value
            vhat_k: Reference voltage [2]
            phi_k: Φ_k = 1 - ||v̂_k||²/v_k*²
            v_k: Voltage setpoint v_k*
            norm_KL: ||K - L|| norm
        
        Returns:
            ψ_k (always non-negative)
        """
        vhat_norm = torch.linalg.norm(vhat_k)
        
        # Synchronization contribution (using local v̂ normalized by v*)
        sync_term = norm_KL * vhat_norm / (v_k + 1e-6)
        
        # Amplitude regulation contribution
        amp_term = eta_a_k * torch.abs(phi_k) * vhat_norm
        
        psi_k = eta_k * (sync_term + amp_term)
        return psi_k
    
    def compute_passivity_max_rate(
        self,
        eta_k: torch.Tensor,
        eta_a_k: torch.Tensor,
        vhat_k: torch.Tensor,
        phi_k: torch.Tensor,
        v_k: torch.Tensor,
        alpha1: torch.Tensor,
        norm_KL: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute maximum allowed η̇_a from passivity constraint.
        
        The passivity constraint is:
            σ_k · η̇_a ≤ θ · α₁ · ψ_k²
        
        Solving for η̇_a:
            η̇_a ≤ (θ · α₁ · ψ_k²) / σ_k
        
        CRITICAL: As Φ_k → 0:
        - σ_k = ½ηα₁Φ_k² → 0
        - ψ_k → η||K-L||||v̂||/v* (sync term dominates)
        - ψ²/σ → ∞ (constraint becomes arbitrarily permissive)
        
        This is physically correct: near equilibrium, changing η_a has 
        negligible effect on V.
        
        Args:
            All local quantities for converter k
        
        Returns:
            Maximum allowed positive η̇_a
        """
        # Compute σ_k (coefficient of η_a in V)
        sigma_k = self.compute_sigma_local(eta_k, alpha1, phi_k)
        
        # Compute ψ_k (comparison function)
        psi_k = self.compute_psi_local(eta_k, eta_a_k, vhat_k, phi_k, v_k, norm_KL)
        
        eps = 1e-12
        vhat_norm = torch.linalg.norm(vhat_k)
        
        # Handle Φ→0 limit properly using L'Hôpital-style analysis
        if sigma_k.abs() < eps:
            # When Φ_k → 0:
            # ψ_k ≈ η_k · ||K-L|| · ||v̂_k||/v_k* (sync term dominates)
            # σ_k ≈ (1/2) η_k α₁ Φ_k² → 0
            # The ratio ψ_k²/σ_k → ∞, so effectively no constraint
            # 
            # Near equilibrium, the amplitude term in V is small,
            # so changing η_a has negligible effect on V.
            # Set a large but finite max_rate based on sync term only.
            max_rate = 100.0 * self.theta * alpha1 * eta_k * (vhat_norm ** 2)
        else:
            # Standard computation: η̇_a ≤ θ·α₁·ψ²/σ
            psi_sq = psi_k ** 2
            max_rate = self.theta * alpha1 * psi_sq / (sigma_k + eps)
        
        # Clamp to reasonable range [1e-6, 100.0]
        max_rate = torch.clamp(max_rate, min=1e-6, max=100.0)
        
        return max_rate

    def compute_tracking_error(
        self,
        v_actual_k: Optional[torch.Tensor],
        vhat_k: torch.Tensor,
        v_error_k: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute tracking error magnitude for optional activation gating.
        
        Uses ||v - v̂|| if actual voltage is available, otherwise uses
        v_error_k (magnitude error).
        
        Args:
            v_actual_k: Actual terminal voltage [2] (optional)
            vhat_k: Reference voltage [2]
            v_error_k: Voltage magnitude error (optional, scalar)
        
        Returns:
            Tracking error magnitude (scalar)
        """
        if v_actual_k is not None:
            # Full vector tracking error
            return torch.linalg.norm(v_actual_k - vhat_k)
        elif v_error_k is not None:
            # Magnitude error as proxy
            return torch.abs(v_error_k)
        else:
            # Fallback: small nonzero value to allow some adaptation
            return torch.tensor(0.001, device=vhat_k.device, dtype=vhat_k.dtype)

    def forward(
        self,
        eta_a_k: torch.Tensor,
        vhat_k: torch.Tensor,
        phi_k: torch.Tensor,
        p_k: torch.Tensor,
        q_k: torch.Tensor,
        v_k: torch.Tensor,
        eta_k: torch.Tensor,
        c_L: torch.Tensor,
        norm_KL: torch.Tensor,
        alpha1: torch.Tensor,
        rL_k: Optional[torch.Tensor] = None,  # ADDED: load resistance
        v_error_k: Optional[torch.Tensor] = None,
        v_actual_k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute η̇_{a,k} for converter k.
        
        This is the CORRECTED implementation that:
        1. Does NOT multiply by err_mag (avoids double-counting)
        2. Uses proper passivity constraint formulation
        3. Handles Φ→0 limit correctly
        4. Allows unrestricted decreases (always stable)
        5. Optionally uses tracking error for activation gating
        
        Args:
            eta_a_k: Current η_a value (scalar)
            vhat_k: Reference voltage [2]
            phi_k: Φ_k = 1 - ||v̂_k||²/v_k*² (scalar)
            p_k, q_k, v_k: Local setpoints (scalars)
            eta_k: Local η value (scalar)
            c_L: Load margin from Condition 4 (scalar)
            norm_KL: ||K - L|| norm (scalar)
            alpha1: Dissipation constant (scalar)
            v_error_k: Voltage magnitude error (optional, scalar)
            v_actual_k: Actual terminal voltage [2] (optional)
        
        Returns:
            η̇_{a,k} (scalar)
        """
        # Compute desired η_a from learned network
        eta_a_des = self.compute_eta_a_desired(
            vhat_k, phi_k, p_k, q_k, v_k, v_error_k, rL_k
        )
        
        # First-order desired dynamics: η̇_a = (η_a_des - η_a) / τ
        eta_a_dot_des = (eta_a_des - eta_a_k) / self.tau
        
        # =====================================================================
        # KEY CORRECTION: Do NOT multiply by err_mag here!
        # The passivity constraint already accounts for Φ_k dependence through σ.
        # The original err_mag = tanh(|Φ|) scaling double-counted this.
        # =====================================================================
        eta_a_dot = eta_a_dot_des
        
        # OPTIONAL: Tracking-error-based activation
        # This allows adaptation when control performance is poor (||v - v̂|| large)
        # but vanishes when tracking is perfect, preventing unnecessary drift.
        if self.use_tracking_error:
            tracking_error = self.compute_tracking_error(v_actual_k, vhat_k, v_error_k)
            activation = torch.tanh(tracking_error / self.tracking_error_scale)
            eta_a_dot = eta_a_dot * activation
        
        # OPTIONAL: Equilibrium gating (softer than original)
        # Only enable if you want adaptation to slow down near equilibrium
        # even when passivity allows it.
        if self.enable_equilibrium_gating:
            # Use max of Φ and tracking error to avoid freezing
            phi_mag = torch.abs(phi_k)
            tracking_error = self.compute_tracking_error(v_actual_k, vhat_k, v_error_k)
            combined_error = torch.maximum(phi_mag, tracking_error)
            equilibrium_gate = torch.tanh(combined_error / 0.05)
            eta_a_dot = eta_a_dot * equilibrium_gate
        
        # Apply passivity constraint ONLY when increasing η_a
        # Decreasing η_a is always stable (reduces the η_a term in V)
        if eta_a_dot > 0:
            max_rate = self.compute_passivity_max_rate(
                eta_k=eta_k,
                eta_a_k=eta_a_k,
                vhat_k=vhat_k,
                phi_k=phi_k,
                v_k=v_k,
                alpha1=alpha1,
                norm_KL=norm_KL
            )
            eta_a_dot = torch.minimum(eta_a_dot, max_rate)
        
        # Anti-windup: enforce state bounds
        if eta_a_k <= self.eta_a_min and eta_a_dot < 0:
            eta_a_dot = torch.zeros_like(eta_a_dot)
        if eta_a_k >= self.eta_a_max and eta_a_dot > 0:
            eta_a_dot = torch.zeros_like(eta_a_dot)
        
        return eta_a_dot

    def forward_batch(
        self,
        eta_a_all: torch.Tensor,      # [Nc]
        vhat_all: torch.Tensor,       # [2*Nc] flattened or [Nc, 2]
        phi_all: torch.Tensor,        # [Nc]
        p_all: torch.Tensor,          # [Nc]
        q_all: torch.Tensor,          # [Nc]
        v_all: torch.Tensor,          # [Nc]
        eta_all: torch.Tensor,        # [Nc]
        c_L: torch.Tensor,            # scalar
        norm_KL: torch.Tensor,        # scalar
        alpha1: torch.Tensor,         # scalar
        rL: torch.Tensor = None,      # scalar or [Nc] - ADDED: load resistance
        v_error_all: torch.Tensor = None,   # [Nc] or None
        v_actual_all: torch.Tensor = None   # [2*Nc] or None
    ) -> torch.Tensor:
        """
        Vectorized forward for ALL converters at once.

        Returns:
            eta_a_dot_all [Nc]
        """
        Nc = eta_a_all.shape[0]
        device = eta_a_all.device
        dtype = eta_a_all.dtype

        # Handle rL broadcasting
        if rL is None:
            rL_all = torch.ones(Nc, dtype=dtype, device=device) * 8.0
        elif rL.dim() == 0:
            rL_all = rL.expand(Nc)
        else:
            rL_all = rL

        # Reshape vhat if needed: [2*Nc] -> [Nc, 2]
        if vhat_all.dim() == 1:
            vhat_2d = vhat_all.view(Nc, 2)
        else:
            vhat_2d = vhat_all
        vhat_norm_all = torch.linalg.norm(vhat_2d, dim=1)  # [Nc]

        # === BATCHED compute_eta_a_desired ===
        if v_error_all is None:
            v_error_normalized = torch.zeros(Nc, dtype=dtype, device=device)
        else:
            v_error_normalized = v_error_all / (torch.abs(v_error_all) + 0.05)

        # Log-scale rL normalization
        log_rL_normalized = (torch.log(rL_all + 0.1) - 1.5) / 1.0

        features = torch.stack([
            torch.tanh(phi_all / 0.05),        # Normalized Φ (improved)
            vhat_norm_all,
            vhat_2d[:, 0],
            vhat_2d[:, 1],
            torch.tanh(p_all * 5.0),           # Normalized p (improved)
            torch.tanh(q_all * 5.0),           # Normalized q (improved)
            v_all - 1.0,                        # Deviation from nominal
            v_error_normalized,
            log_rL_normalized                   # Log-scale rL - MOST IMPORTANT
        ], dim=1)  # [Nc, 9]

        eta_a_des = self.des_net(features).squeeze(-1)  # [Nc]
        eta_a_des = torch.clamp(eta_a_des, min=self.eta_a_min, max=self.eta_a_max)

        # First-order dynamics
        eta_a_dot = (eta_a_des - eta_a_all) / self.tau

        # === BATCHED passivity constraint ===
        # Compute sigma [Nc]
        sigma_all = 0.5 * eta_all * alpha1 * (phi_all ** 2)

        # Compute psi [Nc]
        sync_term = norm_KL * vhat_norm_all / (v_all + 1e-6)
        amp_term = eta_a_all * torch.abs(phi_all) * vhat_norm_all
        psi_all = eta_all * (sync_term + amp_term)

        # Compute max_rate [Nc]
        eps = 1e-12
        # For small sigma (near equilibrium), use large max_rate
        default_max = 100.0 * self.theta * alpha1 * eta_all * (vhat_norm_all ** 2)
        standard_max = self.theta * alpha1 * (psi_all ** 2) / (sigma_all + eps)
        max_rate = torch.where(sigma_all.abs() < eps, default_max, standard_max)
        max_rate = torch.clamp(max_rate, min=1e-6, max=100.0)

        # Apply passivity constraint only when increasing (eta_a_dot > 0)
        eta_a_dot = torch.where(
            eta_a_dot > 0,
            torch.minimum(eta_a_dot, max_rate),
            eta_a_dot
        )

        # Anti-windup
        eta_a_dot = torch.where(
            (eta_a_all <= self.eta_a_min) & (eta_a_dot < 0),
            torch.zeros_like(eta_a_dot),
            eta_a_dot
        )
        eta_a_dot = torch.where(
            (eta_a_all >= self.eta_a_max) & (eta_a_dot > 0),
            torch.zeros_like(eta_a_dot),
            eta_a_dot
        )

        return eta_a_dot


class EtaANeuralODE(nn.Module):
    """
    Simplified η_a neural ODE with proper state coupling.

    η_a dynamics depend on:
    - v̂ (vhat) - reference voltage STATE
    - p*, q*, v* - setpoints (operating point)
    - rL - load resistance (operating point)

    Design:
        η_a_des = target_net(||v̂||, p*, q*, v*, log(rL))
        dη_a/dt = rate_net(η_a, ||v̂||, log(rL)) × (η_a_des - η_a)

        With passivity constraint applied when increasing.

    Inputs:
        State: v̂ (reference voltage)
        Operating point: p*, q*, v*, rL

    The network learns:
        - target_net: What η_a should be for given operating point
        - rate_net: How fast to adapt (replaces fixed τ)
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        eta_a_min: float = 10.0,
        eta_a_max: float = 100.0,
        theta: float = 0.5,  # Safety factor for passivity
        soft_constraint: bool = True,  # Use soft constraint for gradient flow
    ):
        super().__init__()

        self.eta_a_min = eta_a_min
        self.eta_a_max = eta_a_max
        self.theta = theta
        self.soft_constraint = soft_constraint

        # Target network: (||v̂||, p*, q*, v*, log(rL)) -> η_a_des
        # 5 inputs
        self.target_net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Rate network: (η_a, ||v̂||, log(rL)) -> rate (positive)
        # 3 inputs
        # Output is log(rate), then exp to get positive rate
        self.rate_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Base rate (learnable) - typical adaptation rate ~50-100
        self.log_rate_base = nn.Parameter(torch.tensor(4.0))  # exp(4) ≈ 55

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for reasonable starting behavior."""
        # Target net: initialize to output ~1.0 (middle of typical range)
        for layer in self.target_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)

        # Rate net: initialize to output moderate rate (~10-100)
        for layer in self.rate_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.3)
                nn.init.zeros_(layer.bias)

    def compute_eta_a_des(
        self,
        vhat_norm: torch.Tensor,  # ||v̂|| - state
        p_star: torch.Tensor,     # p* - operating point
        q_star: torch.Tensor,     # q* - operating point
        v_star: torch.Tensor,     # v* - operating point
        rL: torch.Tensor,         # rL - operating point
    ) -> torch.Tensor:
        """
        Compute desired η_a from state and operating point.

        Args:
            vhat_norm: Reference voltage magnitude ||v̂||
            p_star, q_star, v_star: Setpoints
            rL: Load resistance

        Returns:
            η_a_des in [eta_a_min, eta_a_max]
        """
        # Normalize inputs
        log_rL = (torch.log(rL + 0.1) - 1.5) / 1.0

        # Build feature vector
        features = torch.stack([
            vhat_norm,                    # State: ||v̂||
            torch.tanh(p_star * 5.0),     # Operating point
            torch.tanh(q_star * 5.0),
            v_star - 1.0,                 # Deviation from nominal
            log_rL,
        ], dim=-1)

        # Network output
        raw = self.target_net(features).squeeze(-1)

        # Map to [eta_a_min, eta_a_max] using sigmoid
        eta_a_des = self.eta_a_min + (self.eta_a_max - self.eta_a_min) * torch.sigmoid(raw)

        return eta_a_des

    def compute_rate(
        self,
        eta_a: torch.Tensor,      # Current η_a - state
        vhat_norm: torch.Tensor,  # ||v̂|| - state
        rL: torch.Tensor,         # rL - operating point
    ) -> torch.Tensor:
        """
        Compute adaptation rate.

        Returns positive rate using exp(log_rate_base + adjustment).
        """
        log_rL = (torch.log(rL + 0.1) - 1.5) / 1.0

        # Normalize eta_a - center around 55 (middle of [10, 100] range)
        eta_a_norm = (eta_a - 55.0) / 30.0

        features = torch.stack([
            eta_a_norm,
            vhat_norm,
            log_rL,
        ], dim=-1)

        # Network outputs adjustment to log rate
        log_rate_adj = self.rate_net(features).squeeze(-1)

        # Rate = exp(log_rate_base + tanh(adjustment) * scale)
        # tanh bounds adjustment to [-1, 1], scale allows ~3x variation
        rate = torch.exp(self.log_rate_base + torch.tanh(log_rate_adj) * 1.0)

        # Soft clamp to reasonable range [1.0, 500.0]
        rate = torch.clamp(rate, min=1.0, max=500.0)

        return rate

    def compute_passivity_max_rate(
        self,
        eta: torch.Tensor,        # Current η
        eta_a: torch.Tensor,      # Current η_a
        vhat_norm: torch.Tensor,  # ||v̂||
        phi: torch.Tensor,        # Φ = 1 - ||v̂||²/v*²
        v_star: torch.Tensor,     # v*
        alpha1: torch.Tensor,     # α₁ from supervisor
        norm_KL: torch.Tensor,    # ||K - L||
    ) -> torch.Tensor:
        """
        Compute maximum allowed η̇_a from passivity constraint.

        Constraint: σ · η̇_a ≤ θ · α₁ · ψ²

        Where:
            σ = (1/2) η α₁ φ²
            ψ = η (||K-L|| ||v̂||/v* + η_a |φ| ||v̂||)
        """
        eps = 1e-12

        # Compute σ (Lyapunov coefficient)
        sigma = 0.5 * eta * alpha1 * (phi ** 2)

        # Compute ψ (comparison function)
        sync_term = norm_KL * vhat_norm / (v_star + eps)
        amp_term = eta_a * torch.abs(phi) * vhat_norm
        psi = eta * (sync_term + amp_term)

        # Compute max rate
        # When σ → 0 (near equilibrium), use large default
        default_max = 100.0 * self.theta * alpha1 * eta * (vhat_norm ** 2)
        standard_max = self.theta * alpha1 * (psi ** 2) / (sigma + eps)

        max_rate = torch.where(
            sigma.abs() < eps,
            default_max,
            standard_max
        )

        # Clamp to reasonable range
        max_rate = torch.clamp(max_rate, min=1e-6, max=100.0)

        return max_rate

    def forward(
        self,
        eta_a: torch.Tensor,      # Current η_a state [Nc] or scalar
        vhat: torch.Tensor,       # Reference voltage [2*Nc] or [2]
        p_star: torch.Tensor,     # Power setpoint [Nc] or scalar
        q_star: torch.Tensor,     # Reactive power setpoint
        v_star: torch.Tensor,     # Voltage setpoint
        rL: torch.Tensor,         # Load resistance (scalar)
        eta: torch.Tensor,        # Current η (for passivity)
        alpha1: torch.Tensor,     # α₁ from supervisor
        norm_KL: torch.Tensor,    # ||K - L||
    ) -> torch.Tensor:
        """
        Compute dη_a/dt for all converters.

        Args:
            eta_a: Current η_a values [Nc]
            vhat: Reference voltages [2*Nc]
            p_star, q_star, v_star: Setpoints [Nc]
            rL: Load resistance (scalar)
            eta: Current η values [Nc]
            alpha1: Dissipation constant
            norm_KL: Network norm

        Returns:
            dη_a/dt [Nc]
        """
        # Handle scalar vs vector inputs
        if eta_a.dim() == 0:
            eta_a = eta_a.unsqueeze(0)

        Nc = eta_a.shape[0]

        # Reshape vhat to [Nc, 2]
        if vhat.dim() == 1:
            vhat_2d = vhat.view(Nc, 2)
        else:
            vhat_2d = vhat

        # Compute ||v̂|| for each converter
        vhat_norm = torch.linalg.norm(vhat_2d, dim=-1)  # [Nc]

        # Compute φ for each converter
        phi = 1.0 - (vhat_norm ** 2) / (v_star ** 2 + 1e-12)  # [Nc]

        # Expand rL to [Nc] if scalar
        if rL.dim() == 0:
            rL_expanded = rL.expand(Nc)
        else:
            rL_expanded = rL

        # Compute η_a_des for each converter
        eta_a_des = self.compute_eta_a_des(vhat_norm, p_star, q_star, v_star, rL_expanded)

        # Compute rate for each converter
        rate = self.compute_rate(eta_a, vhat_norm, rL_expanded)

        # Raw dynamics: rate × (η_a_des - η_a)
        d_eta_a_raw = rate * (eta_a_des - eta_a)

        # Apply passivity constraint when increasing
        max_rate = self.compute_passivity_max_rate(
            eta, eta_a, vhat_norm, phi, v_star, alpha1, norm_KL
        )

        if self.soft_constraint:
            # Soft constraint using softplus-based smooth minimum
            # This preserves gradients even when the constraint is active
            #
            # soft_min(a, b) = a - softplus(a - b)
            #   When a < b: softplus(a-b) ≈ 0, so result ≈ a
            #   When a > b: softplus(a-b) ≈ a-b, so result ≈ b
            #
            # Temperature controls sharpness (larger = softer transition)
            temperature = 0.1 * (max_rate + 1.0)

            # Soft minimum: min(d_eta_a_raw, max_rate) with gradient flow
            d_eta_a_pos = d_eta_a_raw - torch.nn.functional.softplus(
                (d_eta_a_raw - max_rate) / temperature
            ) * temperature

            d_eta_a = torch.where(d_eta_a_raw > 0, d_eta_a_pos, d_eta_a_raw)
        else:
            # Hard constraint: clamp positive derivatives to max_rate
            d_eta_a = torch.where(
                d_eta_a_raw > 0,
                torch.minimum(d_eta_a_raw, max_rate),
                d_eta_a_raw
            )

        # Anti-windup at bounds
        d_eta_a = torch.where(
            (eta_a <= self.eta_a_min) & (d_eta_a < 0),
            torch.zeros_like(d_eta_a),
            d_eta_a
        )
        d_eta_a = torch.where(
            (eta_a >= self.eta_a_max) & (d_eta_a > 0),
            torch.zeros_like(d_eta_a),
            d_eta_a
        )

        return d_eta_a

    def forward_batch(
        self,
        eta_a: torch.Tensor,      # [batch, Nc]
        vhat: torch.Tensor,       # [batch, 2*Nc]
        p_star: torch.Tensor,     # [batch, Nc]
        q_star: torch.Tensor,     # [batch, Nc]
        v_star: torch.Tensor,     # [batch, Nc]
        rL: torch.Tensor,         # [batch] or scalar
        eta: torch.Tensor,        # [batch, Nc]
        alpha1: torch.Tensor,     # scalar
        norm_KL: torch.Tensor,    # scalar
    ) -> torch.Tensor:
        """
        Batched version for multiple trajectories.

        Returns:
            dη_a/dt [batch, Nc]
        """
        batch_size, Nc = eta_a.shape

        # Reshape vhat: [batch, 2*Nc] -> [batch, Nc, 2]
        vhat_3d = vhat.view(batch_size, Nc, 2)
        vhat_norm = torch.linalg.norm(vhat_3d, dim=-1)  # [batch, Nc]

        # Compute φ
        phi = 1.0 - (vhat_norm ** 2) / (v_star ** 2 + 1e-12)  # [batch, Nc]

        # Handle rL broadcasting
        if rL.dim() == 0:
            rL_expanded = rL.expand(batch_size, Nc)
        elif rL.dim() == 1:
            rL_expanded = rL.unsqueeze(-1).expand(batch_size, Nc)
        else:
            rL_expanded = rL

        # Flatten for network calls: [batch*Nc]
        # Use reshape (not view) to handle non-contiguous tensors from expand()
        vhat_norm_flat = vhat_norm.reshape(-1)
        p_star_flat = p_star.reshape(-1)
        q_star_flat = q_star.reshape(-1)
        v_star_flat = v_star.reshape(-1)
        rL_flat = rL_expanded.reshape(-1)
        eta_a_flat = eta_a.reshape(-1)
        eta_flat = eta.reshape(-1)
        phi_flat = phi.reshape(-1)

        # Compute η_a_des
        eta_a_des = self.compute_eta_a_des(
            vhat_norm_flat, p_star_flat, q_star_flat, v_star_flat, rL_flat
        )

        # Compute rate
        rate = self.compute_rate(eta_a_flat, vhat_norm_flat, rL_flat)

        # Raw dynamics
        d_eta_a_raw = rate * (eta_a_des - eta_a_flat)

        # Passivity constraint
        max_rate = self.compute_passivity_max_rate(
            eta_flat, eta_a_flat, vhat_norm_flat, phi_flat,
            v_star_flat, alpha1, norm_KL
        )

        # Apply constraint
        if self.soft_constraint:
            # Soft constraint with tanh for gradient flow
            d_eta_a_pos = max_rate * torch.tanh(d_eta_a_raw / (max_rate + 1e-6))
            d_eta_a = torch.where(d_eta_a_raw > 0, d_eta_a_pos, d_eta_a_raw)
        else:
            # Hard constraint
            d_eta_a = torch.where(
                d_eta_a_raw > 0,
                torch.minimum(d_eta_a_raw, max_rate),
                d_eta_a_raw
            )

        # Anti-windup
        d_eta_a = torch.where(
            (eta_a_flat <= self.eta_a_min) & (d_eta_a < 0),
            torch.zeros_like(d_eta_a),
            d_eta_a
        )
        d_eta_a = torch.where(
            (eta_a_flat >= self.eta_a_max) & (d_eta_a > 0),
            torch.zeros_like(d_eta_a),
            d_eta_a
        )

        # Reshape back to [batch, Nc]
        return d_eta_a.view(batch_size, Nc)


class SupervisoryLayer:
    """
    Computes global bounds at SLOW timescale (setpoint changes only).
    
    These quantities depend on full network but change rarely.
    Computed centrally and broadcast as scalars to all converters.
    
    Computes:
    - c_L: Load margin from Condition 4
    - ||K - L||: Norm for Lyapunov bounds
    - η_upper: Upper bound from Condition 4
    - α₁: Dissipation constant (using conservative η)
    - θ*: Maximum safety factor for passivity
    """
    
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self._cache = {}
        self._cache_valid = False
    
    def invalidate_cache(self):
        """Call when operating point changes."""
        self._cache_valid = False
        self._cache = {}
    
    def compute_bounds(self, network, setpoints,
                       eta_conservative: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Compute all global bounds.

        Uses the SAME formulas as constraints.py for consistency:
        - c_L = 0.40 * λ₂(L) (design parameter for stability margin)
        - d_max = max node degree (weighted by line admittance)
        - eta_upper from Condition 4 second inequality

        Args:
            network: PowerSystemNetwork instance
            setpoints: Setpoints dataclass
            eta_conservative: Conservative η for computing α₁

        Returns:
            Dictionary of broadcast quantities
        """
        if self._cache_valid:
            return self._cache

        device = self.device
        dtype = self.dtype

        Nc = network.Nc
        Nt = network.Nt

        p_star = setpoints.p_star
        q_star = setpoints.q_star
        v_star = setpoints.v_star

        # ===== Network parameters =====
        rt = network.rt
        lt = network.lt
        omega0 = network.omega0

        # Line admittance magnitude |Y_jk|
        Z_sq = rt**2 + (omega0 * lt)**2
        Y_line = 1.0 / torch.sqrt(Z_sq)

        # ===== Compute graph Laplacian and λ₂(L) =====
        # Use B_lines (incidence matrix) not B (extended)
        B_lines = network.B_lines  # [Nc, Nt]
        Y_line_scalar = Y_line.item() if Y_line.dim() == 0 else Y_line
        Y_diag_graph = torch.diag(torch.full((Nt,), Y_line_scalar if isinstance(Y_line_scalar, float) else Y_line_scalar.item(),
                                              dtype=dtype, device=device))
        L_graph = B_lines @ Y_diag_graph @ B_lines.T  # [Nc, Nc]

        # Eigenvalues of graph Laplacian
        eigenvalues = torch.linalg.eigvalsh(L_graph)
        lambda2_L = eigenvalues[1]  # Second smallest eigenvalue

        # ===== c_L design parameter (SAME as constraints.py) =====
        # Use 0.40 × λ₂(L) to balance first inequality margin with wider η range
        c_L = 0.40 * lambda2_L
        c_L = torch.clamp(c_L, min=1e-6)

        # ===== d_max = max node degree (SAME as constraints.py) =====
        Y_line_val = Y_line.item() if Y_line.dim() == 0 else Y_line
        node_degrees = torch.abs(B_lines) @ torch.full((Nt,),
            Y_line_val if isinstance(Y_line_val, float) else Y_line_val.item(),
            dtype=dtype, device=device)
        d_max = torch.max(node_degrees)

        # ===== p_star_max = max(sqrt(p² + q²) / v²) (SAME as constraints.py) =====
        apparent_power = torch.sqrt(p_star**2 + q_star**2)
        apparent_power_normalized = apparent_power / (v_star**2)
        p_star_max = torch.max(apparent_power_normalized)

        # ===== K matrix (block diagonal) =====
        kappa = network.kappa
        cos_k = math.cos(kappa)
        sin_k = math.sin(kappa)

        K_blocks = []
        for k in range(Nc):
            p_k = p_star[k]
            q_k = q_star[k]
            v_k_sq = v_star[k]**2 + 1e-12

            a = p_k * cos_k + q_k * sin_k
            b = q_k * cos_k - p_k * sin_k

            K_k = (1.0 / v_k_sq) * torch.tensor([
                [a, b],
                [-b, a]
            ], dtype=dtype, device=device)
            K_blocks.append(K_k)

        K_full = torch.block_diag(*K_blocks)

        # ===== L matrix (extended Laplacian for 2D states) =====
        B = network.B  # [2*Nc, 2*Nt]

        if rt.dim() == 0:
            Y_mag = 1.0 / torch.sqrt(Z_sq)
            Y_diag_vals = torch.full((2*Nt,), Y_mag.item(), dtype=dtype, device=device)
        else:
            Y_mag = 1.0 / torch.sqrt(Z_sq)
            Y_diag_vals = Y_mag.repeat_interleave(2)

        L_ext = B @ torch.diag(Y_diag_vals) @ B.T

        # ===== ||K - L|| =====
        norm_KL = torch.linalg.norm(K_full - L_ext, ord=2)

        # ===== η_upper from Condition 4 (SAME as constraints.py) =====
        # η < c_L / [2ρ·d_max·(c_L + 5·max_k{S_k/v_k*²} + 10·d_max)]
        # CRITICAL FIX: ρ = L/R is a time constant in SECONDS, not per-unit.
        # In per-unit: lt_pu/rt_pu = ρ_SI × ωb. Divide by ωb to get SI.
        omega_b = network.pu.ωb
        rho = (lt / rt) / omega_b  # ρ in seconds (SI units)
        constant_term = 5.0 * p_star_max + 10.0 * d_max
        denominator = 2.0 * rho * d_max * (c_L + constant_term) + 1e-12
        eta_upper = c_L / denominator
        # No artificial clamp - use the actual bound from the constraint
        
        # ===== α₁ using conservative η =====
        eta_cons = torch.tensor(eta_conservative, dtype=dtype, device=device)
        alpha1 = c_L / (5.0 * eta_cons * (norm_KL**2) + 1e-12)
        
        # ===== θ* (maximum safety factor) =====
        # Conservative estimate
        lambda_min_S = c_L * 0.1
        mu2 = torch.tensor(1.0, dtype=dtype, device=device)
        beta2_norm = norm_KL  # Conservative
        
        numerator = lambda_min_S * alpha1
        denominator = lambda_min_S * alpha1 + (mu2**2) * (beta2_norm**2) + 1e-12
        theta_star = torch.clamp(numerator / denominator, min=0.05, max=0.95)

        # NOTE: lambda2_L already computed above from graph Laplacian (lines 433-435)
        # This matches the computation in constraints.py

        self._cache = {
            'c_L': c_L,
            'norm_KL': norm_KL,
            'd_max': d_max,
            'eta_upper': eta_upper,
            'alpha1': alpha1,
            'theta_star': theta_star,
            'lambda2_L': lambda2_L,
            'K': K_full,
            'L': L_ext
        }
        self._cache_valid = True
        
        return self._cache


class DistributedAdaptiveController(nn.Module):
    """
    Orchestrates all local schedulers for the full system.
    
    Architecture:
    - One LocalGainScheduler (shared weights across converters)
    - One LocalEtaScheduler (shared weights across converters)  
    - One LocalEtaAAdapter (shared weights across converters)
    - One SupervisoryLayer (computes global bounds)
    
    For heterogeneous converters, can use per-converter instances instead.
    
    Args:
        n_converters: Number of converters
        hidden_dim: Hidden layer dimension for all networks
        share_weights: If True, use one network for all converters
        theta: Safety factor for passivity constraint
        use_tracking_error: If True, use tracking error for η_a activation
        enable_equilibrium_gating: If True, add soft gating near equilibrium
        device, dtype: Torch device and dtype
    """
    
    def __init__(
        self,
        n_converters: int,
        hidden_dim: int = 32,
        share_weights: bool = True,
        theta: float = 0.5,
        use_tracking_error: bool = False,
        enable_equilibrium_gating: bool = False,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float64
    ):
        super().__init__()
        
        self.n_converters = n_converters
        self.share_weights = share_weights
        self.device = device
        self.dtype = dtype
        self.bypass_gain_scheduling = False  # Set True to use base gains only
        self.bypass_eta_scheduling = False   # Set True to use base eta only
        self.bypass_eta_a_adaptation = False # Set True to keep eta_a constant (eta_a_dot=0)

        if share_weights:
            # Single instance, shared across all converters
            self.gain_scheduler = LocalGainScheduler(hidden_dim=hidden_dim).to(device=device, dtype=dtype)
            self.eta_scheduler = LocalEtaScheduler(hidden_dim=hidden_dim).to(device=device, dtype=dtype)
            # Use EtaANeuralODE as default (simplified inputs, learned rate, soft constraints)
            self.eta_a_adapter = EtaANeuralODE(
                hidden_dim=hidden_dim,
                theta=theta,
                soft_constraint=True  # Gradient-friendly soft constraint
            ).to(device=device, dtype=dtype)
        else:
            # Per-converter instances
            self.gain_schedulers = nn.ModuleList([
                LocalGainScheduler(hidden_dim=hidden_dim).to(device=device, dtype=dtype) for _ in range(n_converters)
            ])
            self.eta_schedulers = nn.ModuleList([
                LocalEtaScheduler(hidden_dim=hidden_dim).to(device=device, dtype=dtype) for _ in range(n_converters)
            ])
            self.eta_a_adapters = nn.ModuleList([
                EtaANeuralODE(
                    hidden_dim=hidden_dim,
                    theta=theta,
                    soft_constraint=True
                ).to(device=device, dtype=dtype) for _ in range(n_converters)
            ])

        self.supervisor = SupervisoryLayer(device, dtype)
    
    def compute_local_phi(self, vhat_k: torch.Tensor, v_k: torch.Tensor) -> torch.Tensor:
        """Compute Φ_k = 1 - ||v̂_k||²/v_k*² for converter k."""
        vhat_norm_sq = (vhat_k ** 2).sum()
        v_k_sq = v_k ** 2 + 1e-12
        return 1.0 - vhat_norm_sq / v_k_sq
    
    def compute_local_sigma(self, phi_k: torch.Tensor, eta_k: torch.Tensor,
                            alpha1: torch.Tensor) -> torch.Tensor:
        """
        Compute CORRECT local σ_k contribution from Lyapunov function.
        
        From Eq. 27: σ_k = (1/2) η_k α₁ Φ_k²
        
        NOTE: This replaces the incorrect formula σ_k = (c_L / 10||K-L||²) Φ_k²
        """
        return 0.5 * eta_k * alpha1 * (phi_k ** 2)
    
    def compute_local_psi(self, vhat_k: torch.Tensor, phi_k: torch.Tensor,
                          eta_k: torch.Tensor, eta_a_k: torch.Tensor,
                          norm_KL: torch.Tensor, 
                          v_k: torch.Tensor) -> torch.Tensor:
        """
        Compute local ψ_k contribution.
        
        ψ_k ≈ η_k·(||K-L||·||v̂_k||/v_k* + η_{a,k}·|Φ_k|·||v̂_k||)
        """
        vhat_norm = torch.norm(vhat_k)
        sync_term = norm_KL * vhat_norm / (v_k + 1e-6)
        amp_term = eta_a_k * torch.abs(phi_k) * vhat_norm
        psi_k = eta_k * (sync_term + amp_term)
        return psi_k
    
    def schedule_all_converters(
        self,
        vhat: torch.Tensor,  # [2*Nc]
        eta_a_states: torch.Tensor,  # [Nc]
        setpoints,  # Setpoints dataclass
        rL: torch.Tensor,
        gains_base: Dict[str, torch.Tensor],
        cf: float,
        lf: float,
        network,  # For supervisor
        v_actual: torch.Tensor = None  # [2*Nc] actual output voltage (optional)
    ) -> Dict[str, torch.Tensor]:
        """
        Schedule parameters for ALL converters.

        Args:
            vhat: Reference voltages [2*Nc]
            eta_a_states: Current η_a values [Nc]
            setpoints: System setpoints
            rL: Load resistance
            gains_base: Dict with Kp_v, Ki_v, Kp_f, Ki_f base values
            cf, lf: Filter parameters
            network: PowerSystemNetwork for supervisor
            v_actual: Actual output voltage [2*Nc] (if None, uses vhat)

        Returns:
            Dict with:
            - 'Kp_v': [Nc] scheduled Kp_v for each converter
            - 'Ki_v': [Nc] scheduled Ki_v
            - 'Kp_f': [Nc] scheduled Kp_f
            - 'Ki_f': [Nc] scheduled Ki_f
            - 'eta': [Nc] scheduled η for each converter
            - 'eta_base': [Nc] conservative η_base
            - 'eta_a_dot': [Nc] η̇_a for each converter
        """
        Nc = self.n_converters
        
        # Get global bounds from supervisor
        bounds = self.supervisor.compute_bounds(network, setpoints)
        eta_upper = bounds['eta_upper']
        c_L = bounds['c_L']
        norm_KL = bounds['norm_KL']
        alpha1 = bounds['alpha1']
        
        # Extract per-unit rL value (must be scalar - per-batch loop handles batched rL)
        if isinstance(rL, torch.Tensor):
            if rL.dim() > 0:
                raise ValueError(f"schedule_all_converters expects scalar rL, got shape {rL.shape}. "
                                 "Use per-batch scheduling loop in simulation.py instead.")
            rL_val = rL
        else:
            rL_val = torch.tensor(rL, dtype=self.dtype, device=self.device)
        
        # === VECTORIZED COMPUTATION (replaces per-converter loop) ===

        # Extract setpoints as tensors [Nc]
        p_all = setpoints.p_star
        q_all = setpoints.q_star
        v_all = setpoints.v_star

        # Reshape vhat [2*Nc] -> [Nc, 2] for vectorized operations
        vhat_2d = vhat.view(Nc, 2)
        vhat_norm_all = torch.linalg.norm(vhat_2d, dim=1)  # [Nc]

        # Compute phi for all converters: Φ_k = 1 - ||v̂_k||²/v_k*²
        vhat_norm_sq = (vhat_2d ** 2).sum(dim=1)  # [Nc]
        v_sq = v_all ** 2 + 1e-12
        phi_all = 1.0 - vhat_norm_sq / v_sq  # [Nc]

        # Compute v_error for all converters (if v_actual available)
        if v_actual is not None:
            v_actual_2d = v_actual.view(Nc, 2)
            v_actual_mag = torch.linalg.norm(v_actual_2d, dim=1)  # [Nc]
            v_error_all = v_actual_mag - v_all  # [Nc]
        else:
            v_error_all = None

        # Get base eta parameter
        eta_base_param = gains_base.get('eta', torch.tensor(0.001, dtype=self.dtype, device=self.device))

        # === BATCHED GAIN SCHEDULING ===
        if self.bypass_gain_scheduling:
            Kp_v_all = gains_base['Kp_v'].expand(Nc)
            Ki_v_all = gains_base['Ki_v'].expand(Nc)
            Kp_f_all = gains_base['Kp_f'].expand(Nc)
            Ki_f_all = gains_base['Ki_f'].expand(Nc)
        else:
            if self.share_weights:
                Kp_v_all, Ki_v_all, Kp_f_all, Ki_f_all = self.gain_scheduler.forward_batch(
                    p_all, q_all, v_all, rL_val,
                    gains_base['Kp_v'], gains_base['Ki_v'],
                    gains_base['Kp_f'], gains_base['Ki_f'],
                    cf, lf, phi_all
                )
            else:
                # Per-converter schedulers: fall back to loop
                Kp_v_all = torch.zeros(Nc, dtype=self.dtype, device=self.device)
                Ki_v_all = torch.zeros(Nc, dtype=self.dtype, device=self.device)
                Kp_f_all = torch.zeros(Nc, dtype=self.dtype, device=self.device)
                Ki_f_all = torch.zeros(Nc, dtype=self.dtype, device=self.device)
                for k in range(Nc):
                    Kp_v_all[k], Ki_v_all[k], Kp_f_all[k], Ki_f_all[k] = self.gain_schedulers[k](
                        p_all[k], q_all[k], v_all[k], rL_val,
                        gains_base['Kp_v'], gains_base['Ki_v'],
                        gains_base['Kp_f'], gains_base['Ki_f'],
                        cf, lf, phi_all[k]
                    )

        # === BATCHED ETA SCHEDULING ===
        if self.bypass_eta_scheduling:
            eta_all = eta_base_param.expand(Nc)
            eta_base_all = eta_all
        else:
            if self.share_weights:
                eta_all, eta_base_all = self.eta_scheduler.forward_batch(
                    p_all, q_all, v_all, phi_all, vhat_norm_all,
                    eta_base_param, rL_val, eta_upper, eta_a_states  # Added rL_val
                )
            else:
                # Per-converter schedulers: fall back to loop
                eta_all = torch.zeros(Nc, dtype=self.dtype, device=self.device)
                eta_base_all = torch.zeros(Nc, dtype=self.dtype, device=self.device)
                for k in range(Nc):
                    eta_all[k], eta_base_all[k] = self.eta_schedulers[k](
                        p_all[k], q_all[k], v_all[k], phi_all[k], vhat_norm_all[k],
                        eta_base_param, rL_val, eta_upper, eta_a_states[k]  # Added rL_val
                    )

        # === BATCHED ETA_A ADAPTATION ===
        # EtaANeuralODE uses simplified interface: (eta_a, vhat, p*, q*, v*, rL, eta, alpha1, norm_KL)
        if self.bypass_eta_a_adaptation:
            eta_a_dot_all = torch.zeros(Nc, dtype=self.dtype, device=self.device)
        else:
            if self.share_weights:
                # EtaANeuralODE.forward() handles all converters at once
                eta_a_dot_all = self.eta_a_adapter(
                    eta_a_states,  # [Nc]
                    vhat,          # [2*Nc]
                    p_all,         # [Nc]
                    q_all,         # [Nc]
                    v_all,         # [Nc]
                    rL_val,        # scalar
                    eta_all,       # [Nc]
                    alpha1,        # scalar
                    norm_KL        # scalar
                )
            else:
                # Per-converter adapters: fall back to loop
                eta_a_dot_all = torch.zeros(Nc, dtype=self.dtype, device=self.device)
                for k in range(Nc):
                    vhat_k = vhat[2*k:2*(k+1)]
                    eta_a_dot_all[k] = self.eta_a_adapters[k](
                        eta_a_states[k].unsqueeze(0),  # [1]
                        vhat_k,                         # [2]
                        p_all[k].unsqueeze(0),          # [1]
                        q_all[k].unsqueeze(0),          # [1]
                        v_all[k].unsqueeze(0),          # [1]
                        rL_val,                         # scalar
                        eta_all[k].unsqueeze(0),        # [1]
                        alpha1,                         # scalar
                        norm_KL                         # scalar
                    ).squeeze(0)

        return {
            'Kp_v': Kp_v_all,
            'Ki_v': Ki_v_all,
            'Kp_f': Kp_f_all,
            'Ki_f': Ki_f_all,
            'eta': eta_all,
            'eta_base': eta_base_all,
            'eta_a_dot': eta_a_dot_all,
            'bounds': bounds
        }

    def schedule_all_converters_batched(
        self,
        vhat: torch.Tensor,  # [batch, 2*Nc]
        eta_a_states: torch.Tensor,  # [batch, Nc]
        batch_p_star: torch.Tensor,  # [batch, Nc]
        batch_q_star: torch.Tensor,  # [batch, Nc]
        batch_v_star: torch.Tensor,  # [batch, Nc]
        rL_batch: torch.Tensor,  # [batch]
        gains_base: Dict[str, torch.Tensor],
        cf: float,
        lf: float,
        network,
        v_actual: torch.Tensor = None  # [batch, 2*Nc]
    ) -> Dict[str, torch.Tensor]:
        """
        VECTORIZED scheduling for ALL converters across ALL batches.

        Avoids the per-batch loop by reshaping to [batch*Nc, features].

        Returns:
            Dict with [batch, Nc] shaped tensors for each parameter.
        """
        batch_size = vhat.shape[0]
        Nc = self.n_converters

        # Get bounds once (shared across batches - uses conservative values)
        # Create dummy setpoints for supervisor (it only needs network topology)
        dummy_setpoints = Setpoints(
            p_star=batch_p_star[0],
            q_star=batch_q_star[0],
            v_star=batch_v_star[0],
            theta_star=torch.zeros(Nc, dtype=self.dtype, device=self.device)
        )
        bounds = self.supervisor.compute_bounds(network, dummy_setpoints)
        eta_upper = bounds['eta_upper']
        c_L = bounds['c_L']
        norm_KL = bounds['norm_KL']
        alpha1 = bounds['alpha1']

        # Reshape vhat [batch, 2*Nc] -> [batch, Nc, 2]
        vhat_3d = vhat.view(batch_size, Nc, 2)
        vhat_norm = torch.linalg.norm(vhat_3d, dim=2)  # [batch, Nc]

        # Compute phi: [batch, Nc]
        vhat_norm_sq = (vhat_3d ** 2).sum(dim=2)
        v_sq = batch_v_star ** 2 + 1e-12
        phi = 1.0 - vhat_norm_sq / v_sq

        # v_error: [batch, Nc] if v_actual provided
        if v_actual is not None:
            v_actual_3d = v_actual.view(batch_size, Nc, 2)
            v_actual_mag = torch.linalg.norm(v_actual_3d, dim=2)
            v_error = v_actual_mag - batch_v_star
        else:
            v_error = None

        # Get base parameters
        eta_base_param = gains_base.get('eta', torch.tensor(0.001, dtype=self.dtype, device=self.device))

        # === VECTORIZED GAIN SCHEDULING ===
        if self.bypass_gain_scheduling:
            Kp_v = gains_base['Kp_v'].expand(batch_size, Nc)
            Ki_v = gains_base['Ki_v'].expand(batch_size, Nc)
            Kp_f = gains_base['Kp_f'].expand(batch_size, Nc)
            Ki_f = gains_base['Ki_f'].expand(batch_size, Nc)
        else:
            # Flatten to [batch*Nc] for network
            p_flat = batch_p_star.reshape(-1)  # [batch*Nc]
            q_flat = batch_q_star.reshape(-1)
            v_flat = batch_v_star.reshape(-1)
            phi_flat = phi.reshape(-1)

            # Expand rL to match: [batch] -> [batch, Nc] -> [batch*Nc]
            rL_expanded = rL_batch.unsqueeze(1).expand(batch_size, Nc).reshape(-1)

            # Call gain scheduler with flattened inputs
            if self.share_weights:
                Kp_v_flat, Ki_v_flat, Kp_f_flat, Ki_f_flat = self.gain_scheduler.forward_batch(
                    p_flat, q_flat, v_flat, rL_expanded,
                    gains_base['Kp_v'], gains_base['Ki_v'],
                    gains_base['Kp_f'], gains_base['Ki_f'],
                    cf, lf, phi_flat
                )
                Kp_v = Kp_v_flat.view(batch_size, Nc)
                Ki_v = Ki_v_flat.view(batch_size, Nc)
                Kp_f = Kp_f_flat.view(batch_size, Nc)
                Ki_f = Ki_f_flat.view(batch_size, Nc)
            else:
                # Fallback to loop for non-shared weights
                Kp_v = torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)
                Ki_v = torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)
                Kp_f = torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)
                Ki_f = torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)
                for b in range(batch_size):
                    for k in range(Nc):
                        Kp_v[b,k], Ki_v[b,k], Kp_f[b,k], Ki_f[b,k] = self.gain_schedulers[k](
                            batch_p_star[b,k], batch_q_star[b,k], batch_v_star[b,k], rL_batch[b],
                            gains_base['Kp_v'], gains_base['Ki_v'],
                            gains_base['Kp_f'], gains_base['Ki_f'],
                            cf, lf, phi[b,k]
                        )

        # === VECTORIZED ETA SCHEDULING ===
        if self.bypass_eta_scheduling:
            eta = eta_base_param.expand(batch_size, Nc)
            eta_base = eta
        else:
            p_flat = batch_p_star.reshape(-1)
            q_flat = batch_q_star.reshape(-1)
            v_flat = batch_v_star.reshape(-1)
            phi_flat = phi.reshape(-1)
            vhat_norm_flat = vhat_norm.reshape(-1)
            eta_a_flat = eta_a_states.reshape(-1)
            rL_expanded = rL_batch.unsqueeze(1).expand(batch_size, Nc).reshape(-1)

            if self.share_weights:
                eta_flat, eta_base_flat = self.eta_scheduler.forward_batch(
                    p_flat, q_flat, v_flat, phi_flat, vhat_norm_flat,
                    eta_base_param, rL_expanded, eta_upper, eta_a_flat
                )
                eta = eta_flat.view(batch_size, Nc)
                eta_base = eta_base_flat.view(batch_size, Nc)
            else:
                eta = torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)
                eta_base = torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)
                for b in range(batch_size):
                    for k in range(Nc):
                        eta[b,k], eta_base[b,k] = self.eta_schedulers[k](
                            batch_p_star[b,k], batch_q_star[b,k], batch_v_star[b,k],
                            phi[b,k], vhat_norm[b,k], eta_base_param, rL_batch[b],
                            eta_upper, eta_a_states[b,k]
                        )

        # === VECTORIZED ETA_A ADAPTATION ===
        # EtaANeuralODE uses simplified interface with forward_batch for [batch, Nc] inputs
        if self.bypass_eta_a_adaptation:
            eta_a_dot = torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)
        else:
            if self.share_weights:
                # Use EtaANeuralODE.forward_batch() for full batch vectorization
                eta_a_dot = self.eta_a_adapter.forward_batch(
                    eta_a_states,   # [batch, Nc]
                    vhat,           # [batch, 2*Nc]
                    batch_p_star,   # [batch, Nc]
                    batch_q_star,   # [batch, Nc]
                    batch_v_star,   # [batch, Nc]
                    rL_batch,       # [batch]
                    eta,            # [batch, Nc]
                    alpha1,         # scalar
                    norm_KL         # scalar
                )
            else:
                # Per-converter adapters: fall back to loop
                eta_a_dot = torch.zeros(batch_size, Nc, dtype=self.dtype, device=self.device)
                for b in range(batch_size):
                    for k in range(Nc):
                        vhat_k = vhat[b, 2*k:2*(k+1)]
                        eta_a_dot[b, k] = self.eta_a_adapters[k](
                            eta_a_states[b, k].unsqueeze(0),
                            vhat_k,
                            batch_p_star[b, k].unsqueeze(0),
                            batch_q_star[b, k].unsqueeze(0),
                            batch_v_star[b, k].unsqueeze(0),
                            rL_batch[b],
                            eta[b, k].unsqueeze(0),
                            alpha1,
                            norm_KL
                        ).squeeze(0)

        return {
            'Kp_v': Kp_v,  # [batch, Nc]
            'Ki_v': Ki_v,
            'Kp_f': Kp_f,
            'Ki_f': Ki_f,
            'eta': eta,
            'eta_base': eta_base,
            'eta_a_dot': eta_a_dot,
            'bounds': bounds
        }

    def invalidate_cache(self):
        """Call when operating point changes."""
        self.supervisor.invalidate_cache()
    
    def get_conservative_eta(self, setpoints, network, eta_base_param=None) -> torch.Tensor:
        """
        Get conservative η for constraint checking.

        Returns minimum of all η_base values.
        """
        bounds = self.supervisor.compute_bounds(network, setpoints)
        eta_upper = bounds['eta_upper']

        # Use provided base or fall back to eta_upper
        if eta_base_param is None:
            eta_base_param = eta_upper

        Nc = self.n_converters
        eta_base_min = eta_base_param

        for k in range(Nc):
            p_k = setpoints.p_star[k]
            q_k = setpoints.q_star[k]
            v_k = setpoints.v_star[k]

            if self.share_weights:
                eta_sched = self.eta_scheduler
            else:
                eta_sched = self.eta_schedulers[k]

            eta_base_k = eta_sched.compute_eta_base(p_k, q_k, v_k, eta_base_param)
            eta_base_min = torch.min(eta_base_min, eta_base_k)

        return eta_base_min


# =============================================================================
# Diagnostic utilities
# =============================================================================

def diagnose_eta_a_adaptation(
    eta_a_k: torch.Tensor,
    vhat_k: torch.Tensor,
    phi_k: torch.Tensor,
    eta_k: torch.Tensor,
    alpha1: torch.Tensor,
    norm_KL: torch.Tensor,
    c_L: torch.Tensor,
    v_k: torch.Tensor,
    adapter: LocalEtaAAdapter,
    v_error_k: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> Dict:
    """
    Diagnose why η_a might not be adapting.
    
    Returns detailed information about all quantities in the adaptation.
    
    Args:
        eta_a_k: Current η_a value
        vhat_k: Reference voltage [2]
        phi_k: Φ_k value
        eta_k: η value
        alpha1: Dissipation constant
        norm_KL: ||K-L|| norm
        c_L: Load margin
        v_k: Voltage setpoint
        adapter: LocalEtaAAdapter instance
        v_error_k: Voltage error (optional)
        verbose: Print results
    
    Returns:
        Dictionary with diagnostic information
    """
    results = {}
    
    # Basic quantities
    vhat_norm = torch.linalg.norm(vhat_k)
    results['vhat_norm'] = vhat_norm.item()
    results['phi_k'] = phi_k.item()
    results['eta_k'] = eta_k.item()
    results['eta_a_k'] = eta_a_k.item()
    results['alpha1'] = alpha1.item()
    results['norm_KL'] = norm_KL.item()
    results['c_L'] = c_L.item()
    
    # σ_k from CORRECT formula
    sigma_correct = adapter.compute_sigma_local(eta_k, alpha1, phi_k)
    results['sigma_k_correct'] = sigma_correct.item()
    
    # σ_k from original (incorrect) formula for comparison
    sigma_original = (c_L / (10.0 * norm_KL**2 + 1e-12)) * (phi_k ** 2)
    results['sigma_k_original'] = sigma_original.item()
    
    # ψ_k
    psi_k = adapter.compute_psi_local(eta_k, eta_a_k, vhat_k, phi_k, v_k, norm_KL)
    results['psi_k'] = psi_k.item()
    
    # ψ²/σ ratio
    eps = 1e-12
    if sigma_correct.abs() > eps:
        psi_sq_over_sigma = (psi_k ** 2) / sigma_correct
        results['psi_sq_over_sigma'] = psi_sq_over_sigma.item()
    else:
        results['psi_sq_over_sigma'] = float('inf')
        results['note'] = "σ≈0 (near equilibrium) → no constraint on η̇_a"
    
    # Max rate from passivity constraint
    max_rate = adapter.compute_passivity_max_rate(
        eta_k, eta_a_k, vhat_k, phi_k, v_k, alpha1, norm_KL
    )
    results['passivity_max_rate'] = max_rate.item()
    
    # Original err_mag that was causing the problem
    original_phi_scale = 0.05
    original_err_mag = torch.tanh(torch.abs(phi_k) / original_phi_scale).item()
    results['original_err_mag'] = original_err_mag
    results['original_err_mag_suppression'] = f"{100*(1-original_err_mag):.1f}%"
    
    # Analysis
    results['analysis'] = {}
    
    if original_err_mag < 0.1:
        results['analysis']['err_mag_issue'] = (
            f"Original err_mag = {original_err_mag:.4f} was very small. "
            f"This would have scaled η̇_a by {100*original_err_mag:.1f}% (suppressing {100*(1-original_err_mag):.1f}%) "
            "BEFORE applying passivity check. This double-counting is now REMOVED."
        )
    
    if sigma_correct.abs() < eps:
        results['analysis']['sigma_near_zero'] = (
            "σ_k ≈ 0 (Φ_k ≈ 0 means near equilibrium). "
            "This correctly means changing η_a has negligible effect on V, "
            "so passivity constraint is permissive (max_rate is large)."
        )
    
    if alpha1 < 0.01:
        results['analysis']['alpha1_small'] = (
            f"α₁ = {alpha1.item():.4f} is small. "
            f"This happens when η·||K-L||² is large. Check if η or ||K-L|| is too large."
        )
    
    if verbose:
        print("\n" + "=" * 70)
        print("η_a Adaptation Diagnostic (CORRECTED IMPLEMENTATION)")
        print("=" * 70)
        for key, val in results.items():
            if key != 'analysis':
                if isinstance(val, float):
                    print(f"  {key}: {val:.6f}")
                else:
                    print(f"  {key}: {val}")
        
        print("\n--- Analysis ---")
        for key, msg in results.get('analysis', {}).items():
            print(f"  [{key}]")
            print(f"    {msg}")
        print("=" * 70)
    
    return results


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Corrected LocalEtaAAdapter")
    print("=" * 70)
    
    device = torch.device('cpu')
    dtype = torch.float64
    
    # Create adapter with corrected implementation
    adapter = LocalEtaAAdapter(
        hidden_dim=32,
        theta=0.5,
        eta_a_min=0.1,
        eta_a_max=10.0,
        tau=0.01,
        use_tracking_error=False,
        enable_equilibrium_gating=False
    ).to(device=device, dtype=dtype)
    
    # Test case 1: Near equilibrium (small Φ_k) - where original would fail
    print("\nTest 1: Near equilibrium (Φ_k ≈ 0)")
    print("-" * 40)
    vhat_k = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
    phi_k = torch.tensor(0.001, dtype=dtype, device=device)  # Very small
    eta_a_k = torch.tensor(1.0, dtype=dtype, device=device)
    eta_k = torch.tensor(0.1, dtype=dtype, device=device)
    alpha1 = torch.tensor(0.01, dtype=dtype, device=device)
    norm_KL = torch.tensor(10.0, dtype=dtype, device=device)
    c_L = torch.tensor(0.1, dtype=dtype, device=device)
    v_k = torch.tensor(1.0, dtype=dtype, device=device)
    p_k = torch.tensor(0.05, dtype=dtype, device=device)
    q_k = torch.tensor(0.01, dtype=dtype, device=device)
    
    # Compute η̇_a with corrected adapter
    eta_a_dot = adapter.forward(
        eta_a_k=eta_a_k,
        vhat_k=vhat_k,
        phi_k=phi_k,
        p_k=p_k,
        q_k=q_k,
        v_k=v_k,
        eta_k=eta_k,
        c_L=c_L,
        norm_KL=norm_KL,
        alpha1=alpha1,
        v_error_k=None,  # Test with None (corrected handling)
        v_actual_k=None
    )
    
    print(f"  Φ_k = {phi_k.item():.6f}")
    print(f"  η̇_a (corrected) = {eta_a_dot.item():.6f}")
    
    # Compare with original behavior
    original_phi_scale = 0.05
    original_err_mag = torch.tanh(torch.abs(phi_k) / original_phi_scale)
    print(f"  Original err_mag would be = {original_err_mag.item():.6f}")
    print(f"  Original η̇_a would be suppressed by {100*(1-original_err_mag.item()):.1f}%")
    
    # Detailed diagnostic
    print("\nDetailed diagnostic:")
    diagnose_eta_a_adaptation(
        eta_a_k, vhat_k, phi_k, eta_k, alpha1, norm_KL, c_L, v_k, adapter
    )
    
    # Test case 2: During transient (moderate Φ_k)
    print("\n" + "=" * 70)
    print("\nTest 2: During transient (moderate Φ_k)")
    print("-" * 40)
    phi_k = torch.tensor(0.1, dtype=dtype, device=device)
    
    eta_a_dot = adapter.forward(
        eta_a_k=eta_a_k,
        vhat_k=vhat_k,
        phi_k=phi_k,
        p_k=p_k,
        q_k=q_k,
        v_k=v_k,
        eta_k=eta_k,
        c_L=c_L,
        norm_KL=norm_KL,
        alpha1=alpha1,
        v_error_k=torch.tensor(0.05, dtype=dtype, device=device),
        v_actual_k=None
    )
    
    print(f"  Φ_k = {phi_k.item():.6f}")
    print(f"  v_error_k = 0.05")
    print(f"  η̇_a (corrected) = {eta_a_dot.item():.6f}")
    
    diagnose_eta_a_adaptation(
        eta_a_k, vhat_k, phi_k, eta_k, alpha1, norm_KL, c_L, v_k, adapter,
        v_error_k=torch.tensor(0.05, dtype=dtype, device=device)
    )
    
    # Test case 3: With tracking error activation
    print("\n" + "=" * 70)
    print("\nTest 3: With tracking-error-based activation")
    print("-" * 40)
    
    adapter_tracking = LocalEtaAAdapter(
        hidden_dim=32,
        theta=0.5,
        use_tracking_error=True,
        tracking_error_scale=0.02
    ).to(device=device, dtype=dtype)
    
    # Near equilibrium but with tracking error
    phi_k = torch.tensor(0.001, dtype=dtype, device=device)  # Small Φ
    v_actual_k = torch.tensor([1.02, 0.01], dtype=dtype, device=device)  # Some tracking error
    
    eta_a_dot = adapter_tracking.forward(
        eta_a_k=eta_a_k,
        vhat_k=vhat_k,
        phi_k=phi_k,
        p_k=p_k,
        q_k=q_k,
        v_k=v_k,
        eta_k=eta_k,
        c_L=c_L,
        norm_KL=norm_KL,
        alpha1=alpha1,
        v_error_k=None,
        v_actual_k=v_actual_k
    )
    
    tracking_error = torch.linalg.norm(v_actual_k - vhat_k)
    print(f"  Φ_k = {phi_k.item():.6f} (small)")
    print(f"  ||v - v̂|| = {tracking_error.item():.6f} (tracking error)")
    print(f"  η̇_a (with tracking activation) = {eta_a_dot.item():.6f}")
    print("  Note: Adaptation possible despite small Φ because tracking error is nonzero")
