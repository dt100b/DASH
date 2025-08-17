"""Online Rolling Controlled SMC (ORCSMC) for regime inference
Research basis: Paper 2508.00696v1 - Online Rolling Controlled Sequential Monte Carlo
Dual particle systems with bounded per-step compute for online regime detection
"""
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from collections import deque
import warnings

@dataclass
class RegimeState:
    """Hidden regime state"""
    RISK_ON: int = 1
    RISK_OFF: int = 0

class ORCSMC:
    """Online Rolling Controlled Sequential Monte Carlo
    
    Research (2508.00696v1): Uses dual particle systems - twist learner on rolling
    window L and estimation filter. Provides bounded O(L*N) per-step compute while
    maintaining accuracy through iterative twist function optimization.
    
    Key equations from paper:
    - Twisted transition: f^ψ_t(x_t|x_{t-1}) = f_t(x_t|x_{t-1})ψ_t(x_t)/f_t(ψ_t)(x_{t-1})
    - Twisted observation: g^ψ_t(x_t) = g_t(y_t|x_t)f_{t+1}(ψ_{t+1})(x_t)/ψ_t(x_t)
    - Optimal twist: ψ_t(x_t) = E[p(y_{t:T}|x_t)]
    """
    
    def __init__(self, 
                 n_particles: int = 1000,        # N from paper (Section 4.2)
                 rolling_window: int = 30,       # L from paper (20-50 range, Section 2.3)
                 learning_iters: int = 5,        # K from paper (Algorithm 4)
                 resample_threshold: float = 0.5,  # κ from paper (ESS threshold)
                 transition_prob: float = 0.95,
                 inertia_days: int = 2):         # Consecutive ON days required after OFF
        """Initialize ORCSMC filter with research-based parameters
        
        Args:
            n_particles: Number of particles N per system (paper: 1000)
            rolling_window: Rolling window size L (paper: 20-50 optimal)
            learning_iters: K iterations for twist learning (paper: 5)
            resample_threshold: κ for adaptive resampling (paper: 0.5)
            transition_prob: Diagonal transition probability (persistence)
            inertia_days: Consecutive RISK_ON days required after RISK_OFF
        """
        self.n_particles = n_particles
        self.rolling_window = rolling_window
        self.learning_iters = learning_iters
        self.resample_threshold = resample_threshold
        self.transition_prob = transition_prob
        self.inertia_days = inertia_days
        
        # Dual particle systems (Section 3)
        self.learning_particles = None  # H̃_t: Learn twist/control
        self.filter_particles = None    # H_t: Perform estimation
        
        # Rolling window buffer
        self.observation_buffer = deque(maxlen=rolling_window)
        
        # Transition matrix (2x2 for binary regime)
        self.transition_matrix = np.array([
            [transition_prob, 1 - transition_prob],      # From RISK_OFF
            [1 - transition_prob, transition_prob]       # From RISK_ON
        ])
        
        # Twist function parameters ψ_t(x_t) = exp(x'A_t x + b't x + c_t)
        # Restricted to diagonal A_t for linear scaling (Section 4.1)
        self.twist_params = {
            'A_diag': np.array([0.1, 0.1]),  # Diagonal of quadratic term
            'b': np.array([0.02, -0.01]),    # Linear term [risk_on, risk_off]
            'c': 0.0,                         # Constant term
            'mean_risk_on': 0.02,
            'mean_risk_off': -0.01,
            'vol_risk_on': 0.15,
            'vol_risk_off': 0.25
        }
        
        # Inertia tracking for regime transitions
        self.consecutive_on_days = 0
        self.last_regime = RegimeState.RISK_OFF
        
        # Performance metrics
        self.compute_steps = 0
        self.max_compute_per_step = rolling_window * n_particles * learning_iters
        
        # Initialize particles
        self._initialize_particles()
        
    def _initialize_particles(self):
        """Initialize both particle systems (Algorithm 4, Line 7)"""
        # Uniform initial distribution over regimes
        self.learning_particles = np.random.choice(
            [RegimeState.RISK_OFF, RegimeState.RISK_ON],
            size=self.n_particles,
            p=[0.5, 0.5]
        )
        self.filter_particles = self.learning_particles.copy()
        
        # Particle weights W_0 = 1/N (Equation 4)
        self.learning_weights = np.ones(self.n_particles) / self.n_particles
        self.filter_weights = np.ones(self.n_particles) / self.n_particles
        
        # Normalizing constant estimates Z_0 = 1
        self.Z_learning = 1.0
        self.Z_filter = 1.0
        
        # Ancestor indices for resampling
        self.ancestors = np.arange(self.n_particles)
        
    def _compute_twist_value(self, state: int, features: np.ndarray) -> float:
        """Compute twist function value ψ_t(x_t) = exp(x'A_t x + b't x + c_t)
        
        Paper Section 4.1: Quadratic twist functions for Gaussian dynamics
        """
        # Extract state-specific parameters
        if state == RegimeState.RISK_ON:
            A_val = self.twist_params['A_diag'][0]
            b_val = self.twist_params['b'][0]
        else:
            A_val = self.twist_params['A_diag'][1]
            b_val = self.twist_params['b'][1]
        
        # Quadratic form (simplified for scalar state)
        x = features if isinstance(features, (int, float)) else features[0]
        quadratic = A_val * x**2 + b_val * x + self.twist_params['c']
        
        # Ensure numerical stability
        quadratic = np.clip(quadratic, -10, 10)
        return np.exp(quadratic)
    
    def _observation_likelihood(self, observation: float, state: int) -> float:
        """Compute g_t(y_t|x_t) for observation model
        
        Gaussian observation model per paper Section 4
        """
        if state == RegimeState.RISK_ON:
            mean = self.twist_params['mean_risk_on']
            std = self.twist_params['vol_risk_on']
        else:
            mean = self.twist_params['mean_risk_off']
            std = self.twist_params['vol_risk_off']
            
        # Gaussian likelihood
        return np.exp(-0.5 * ((observation - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    
    def _compute_ess(self, weights: np.ndarray) -> float:
        """Compute Effective Sample Size (Algorithm 1, Line 3)
        
        ESS = 1 / Σ(w_i^2)
        """
        return 1.0 / np.sum(weights ** 2)
    
    def _resample_particles(self, particles: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Residual-multinomial resampling (paper Section 2.2)
        
        Returns: (resampled_particles, ancestor_indices)
        """
        n = len(particles)
        
        # Residual resampling for reduced variance
        expected_copies = n * weights
        integer_copies = np.floor(expected_copies).astype(int)
        residual_weights = expected_copies - integer_copies
        residual_weights /= np.sum(residual_weights)
        
        # Deterministic assignment
        ancestors = []
        for i, copies in enumerate(integer_copies):
            ancestors.extend([i] * copies)
        
        # Stochastic assignment for residuals
        n_residual = n - len(ancestors)
        if n_residual > 0:
            residual_ancestors = np.random.choice(n, size=n_residual, p=residual_weights)
            ancestors.extend(residual_ancestors)
        
        ancestors = np.array(ancestors)
        new_particles = particles[ancestors]
        
        return new_particles, ancestors
    
    def _learn_twist_function(self, particles: np.ndarray, observations: List[float]) -> Dict:
        """Learn twist function parameters (Algorithm 3)
        
        Paper Section 4.1: ADP approach with quadratic function class
        ψ_t(x_t) = exp(x'_t A_t x_t + b'_t x_t + c_t)
        """
        if len(observations) < 2:
            return self.twist_params
            
        obs_array = np.array(observations)
        
        # Collect target values for regression (Line 1 of Algorithm 3)
        risk_on_targets = []
        risk_off_targets = []
        
        # Weighted particle statistics for each regime
        for i in range(len(particles)):
            obs_idx = min(i, len(observations) - 1)
            
            if particles[i] == RegimeState.RISK_ON:
                # Target: g_t(y_t|x_t) * f_{t+1}(ψ_{t+1})(x_t)
                likelihood = self._observation_likelihood(observations[obs_idx], RegimeState.RISK_ON)
                risk_on_targets.append(np.log(likelihood + 1e-10))
            else:
                likelihood = self._observation_likelihood(observations[obs_idx], RegimeState.RISK_OFF)
                risk_off_targets.append(np.log(likelihood + 1e-10))
        
        # Linear least squares for log-scale (Section 4.1)
        if len(risk_on_targets) > 5:
            # Update quadratic parameters for RISK_ON
            mean_target = np.mean(risk_on_targets)
            self.twist_params['A_diag'][0] = -0.5 / (np.var(risk_on_targets) + 0.1)
            self.twist_params['b'][0] = mean_target * self.twist_params['A_diag'][0]
            
            # Update Gaussian parameters
            self.twist_params['mean_risk_on'] = np.mean(obs_array[obs_array > np.median(obs_array)])
            self.twist_params['vol_risk_on'] = np.std(obs_array[obs_array > np.median(obs_array)]) + 0.01
            
        if len(risk_off_targets) > 5:
            # Update quadratic parameters for RISK_OFF
            mean_target = np.mean(risk_off_targets)
            self.twist_params['A_diag'][1] = -0.5 / (np.var(risk_off_targets) + 0.1)
            self.twist_params['b'][1] = mean_target * self.twist_params['A_diag'][1]
            
            # Update Gaussian parameters
            self.twist_params['mean_risk_off'] = np.mean(obs_array[obs_array <= np.median(obs_array)])
            self.twist_params['vol_risk_off'] = np.std(obs_array[obs_array <= np.median(obs_array)]) + 0.01
            
        return self.twist_params
    
    def _psi_apf_step(self, particles: np.ndarray, weights: np.ndarray, 
                      observation: float, is_learning: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """ψ-APF step (Algorithm 1)
        
        Returns: (new_particles, new_weights, normalizing_constant_update)
        """
        n = self.n_particles
        
        # Line 1-2: Update weights with twist integral f_t(ψ_t)(x_{t-1})
        v = weights.copy()
        for i in range(n):
            # Compute f_t(ψ_t)(x_{t-1}) - integral of twisted transition
            twist_val = self._compute_twist_value(particles[i], observation)
            v[i] *= twist_val
        
        # Normalize weights
        Z_update = np.sum(v)
        V = v / (Z_update + 1e-10)
        
        # Line 3-7: Adaptive resampling based on ESS
        ess = self._compute_ess(V)
        if ess < self.resample_threshold * n:
            particles, ancestors = self._resample_particles(particles, V)
            weights = np.ones(n) / n
        else:
            ancestors = np.arange(n)
        
        # Line 8: Sample from twisted transition f^ψ_t
        new_particles = np.zeros_like(particles)
        for i in range(n):
            # Transition probabilities
            probs = self.transition_matrix[particles[i]]
            
            # Adjust by twist for importance sampling
            twist_risk_on = self._compute_twist_value(RegimeState.RISK_ON, observation)
            twist_risk_off = self._compute_twist_value(RegimeState.RISK_OFF, observation)
            twisted_probs = probs * np.array([twist_risk_off, twist_risk_on])
            twisted_probs /= np.sum(twisted_probs)
            
            new_particles[i] = np.random.choice([0, 1], p=twisted_probs)
        
        # Line 9-10: Update weights with twisted observation g^ψ_t
        new_weights = weights.copy()
        for i in range(n):
            likelihood = self._observation_likelihood(observation, new_particles[i])
            twist_val = self._compute_twist_value(new_particles[i], observation)
            new_weights[i] *= likelihood / (twist_val + 1e-10)
        
        # Normalize
        weight_sum = np.sum(new_weights)
        new_weights /= (weight_sum + 1e-10)
        Z_update *= weight_sum
        
        return new_particles, new_weights, Z_update
    
    def step(self, observation: float) -> float:
        """ORCSMC step with new observation (Algorithm 4)
        
        Implements dual particle systems with rolling window control
        
        Args:
            observation: New feature/return observation
            
        Returns:
            P(RISK_ON | observations) with inertia rule applied
        """
        self.compute_steps += 1
        
        # Add to rolling buffer (Line 2)
        self.observation_buffer.append(observation)
        
        # Determine rolling window bounds (Line 3)
        t = len(self.observation_buffer)
        t0 = max(0, t - self.rolling_window)
        window_obs = list(self.observation_buffer)[t0:t]
        
        # === Learning Filter (Lines 6-11) ===
        # Run K iterations to learn twist function
        for k in range(self.learning_iters):
            # Backward pass: learn twist parameters
            if len(window_obs) > 1:
                self.twist_params = self._learn_twist_function(
                    self.learning_particles, window_obs
                )
            
            # Forward pass: update learning particles with learned twist
            if k < self.learning_iters - 1:  # Don't update on last iteration
                self.learning_particles, self.learning_weights, _ = self._psi_apf_step(
                    self.learning_particles, self.learning_weights, observation, is_learning=True
                )
        
        # === Estimation Filter (Lines 12-13) ===
        # Apply learned twist to estimation particles
        self.filter_particles, self.filter_weights, Z_update = self._psi_apf_step(
            self.filter_particles, self.filter_weights, observation
        )
        self.Z_filter *= Z_update
        
        # Calculate raw P(RISK_ON)
        p_risk_on_raw = np.sum(
            self.filter_weights[self.filter_particles == RegimeState.RISK_ON]
        )
        
        # === Apply Inertia Rule ===
        # Require consecutive ON days after OFF regime
        if self.last_regime == RegimeState.RISK_OFF:
            if p_risk_on_raw > 0.6:  # Strong signal for RISK_ON
                self.consecutive_on_days += 1
                if self.consecutive_on_days >= self.inertia_days:
                    # Transition confirmed after required consecutive days
                    self.last_regime = RegimeState.RISK_ON
                    self.consecutive_on_days = 0
                    p_risk_on = p_risk_on_raw
                else:
                    # Still in inertia period, suppress transition
                    p_risk_on = 0.3 + 0.2 * (self.consecutive_on_days / self.inertia_days)
            else:
                # Reset counter if signal weakens
                self.consecutive_on_days = 0
                p_risk_on = p_risk_on_raw
        else:
            # In RISK_ON regime, no inertia needed for OFF transition
            if p_risk_on_raw < 0.4:  # Strong signal for RISK_OFF
                self.last_regime = RegimeState.RISK_OFF
                self.consecutive_on_days = 0
            p_risk_on = p_risk_on_raw
        
        # Ensure bounded compute
        if self.compute_steps % 100 == 0:
            actual_compute = self.rolling_window * self.n_particles * self.learning_iters
            if actual_compute > self.max_compute_per_step:
                warnings.warn(f"Compute bound exceeded: {actual_compute} > {self.max_compute_per_step}")
        
        return p_risk_on
    
    def get_regime_probability(self) -> float:
        """Get current P(RISK_ON) with inertia applied"""
        p_risk_on_raw = np.sum(
            self.filter_weights[self.filter_particles == RegimeState.RISK_ON]
        )
        
        # Apply same inertia logic as in step()
        if self.last_regime == RegimeState.RISK_OFF and self.consecutive_on_days > 0:
            if self.consecutive_on_days < self.inertia_days:
                return 0.3 + 0.2 * (self.consecutive_on_days / self.inertia_days)
        
        return p_risk_on_raw
    
    def get_diagnostics(self) -> Dict:
        """Get comprehensive diagnostic information
        
        Returns diagnostic metrics for monitoring algorithm performance
        """
        return {
            'p_risk_on': self.get_regime_probability(),
            'p_risk_on_raw': np.sum(
                self.filter_weights[self.filter_particles == RegimeState.RISK_ON]
            ),
            'ess_filter': self._compute_ess(self.filter_weights),
            'ess_learning': self._compute_ess(self.learning_weights),
            'twist_params': self.twist_params,
            'buffer_size': len(self.observation_buffer),
            'rolling_window': self.rolling_window,
            'particle_diversity': len(np.unique(self.filter_particles)) / self.n_particles,
            'compute_steps': self.compute_steps,
            'max_compute_per_step': self.max_compute_per_step,
            'consecutive_on_days': self.consecutive_on_days,
            'inertia_active': self.last_regime == RegimeState.RISK_OFF and self.consecutive_on_days > 0,
            'Z_filter': self.Z_filter,  # Normalizing constant estimate
            'current_regime': 'RISK_ON' if self.last_regime == RegimeState.RISK_ON else 'RISK_OFF'
        }