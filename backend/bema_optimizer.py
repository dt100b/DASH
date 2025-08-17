"""Bias-Corrected Exponential Moving Average (BEMA) Optimizer
Implementation based on paper 2508.00180v1: EMA Without the Lag
"""
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class BEMAConfig:
    """Configuration for BEMA optimizer
    
    Based on empirical best practices from paper Section 5:
    - κ (ema_power): Controls EMA decay rate, 0.5 works best
    - η (bias_power): Controls bias correction strength, 0.2-0.4 optimal
    - γ (multiplier): Scaling factor for time-dependent weights
    - ρ (lag): Offset to prevent division by zero
    - τ (burn_in): Steps before applying stabilization
    - ϕ (update_freq): Update frequency to reduce computational cost
    """
    ema_power: float = 0.5  # κ in paper, EMA decay exponent
    bias_power: float = 0.4  # η in paper, bias correction strength (0.4 from Figure 1d)
    multiplier: float = 1.0  # γ in paper, time scaling
    lag: float = 1.0  # ρ in paper, offset term
    burn_in: int = 0  # τ in paper, initial steps without stabilization
    update_freq: int = 400  # ϕ in paper, update every N steps
    
    # Learning rate schedule integration
    lr_grid: List[float] = None  # Grid of learning rates to cycle through
    lr_schedule: str = "constant"  # "constant", "decay", "cyclic", "adaptive"
    lr_decay_factor: float = 0.95  # For decay schedule
    lr_cycle_steps: int = 1000  # Steps per cycle for cyclic schedule
    
    # Snapshot and restore for high-LR robustness
    snapshot_interval: int = 5000  # Save snapshot every N steps
    restore_threshold: float = 0.1  # Restore if loss increases by this fraction
    max_snapshots: int = 3  # Maximum snapshots to keep
    
    def __post_init__(self):
        """Initialize default LR grid if not provided"""
        if self.lr_grid is None:
            # Default grid spans typical training ranges
            self.lr_grid = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]

class BEMAOptimizer:
    """Bias-Corrected Exponential Moving Average optimizer
    
    Key innovations from paper:
    1. Eliminates lag through bias correction: α_t(θ_t - θ_0) + θ_EMA
    2. Maintains variance reduction benefits of EMA
    3. Enables higher learning rates through stabilization
    4. Provably accelerates convergence in quadratic landscapes
    """
    
    def __init__(self, initial_params: np.ndarray, config: Optional[BEMAConfig] = None):
        """Initialize BEMA optimizer
        
        Args:
            initial_params: Initial parameter values (θ_0 in paper)
            config: BEMA configuration
        """
        self.config = config or BEMAConfig()
        
        # Core parameters (Alg 1 initialization)
        self.theta_0 = initial_params.copy()  # Initial parameters for bias correction
        self.theta_ema = initial_params.copy()  # EMA of parameters
        self.theta_current = initial_params.copy()  # Current parameters
        self.theta_bema = initial_params.copy()  # BEMA corrected parameters
        
        # State tracking
        self.step_count = 0
        self.update_count = 0
        self.current_lr = self.config.lr_grid[0] if self.config.lr_grid else 1e-3
        self.lr_index = 0
        
        # Snapshots for high-LR recovery
        self.snapshots = []
        self.best_loss = float('inf')
        self.loss_history = []
        
        # Performance tracking
        self.ema_history = []
        self.bema_history = []
        self.bias_correction_history = []
        
    def update(self, new_params: np.ndarray, loss: Optional[float] = None) -> np.ndarray:
        """Apply BEMA update to parameters (Algorithm 1)
        
        Args:
            new_params: New parameter values after gradient step
            loss: Optional loss value for snapshot/restore logic
            
        Returns:
            BEMA corrected parameters
        """
        self.step_count += 1
        self.theta_current = new_params.copy()
        
        # Check burn-in period (line 5 in Algorithm 1)
        if self.step_count <= self.config.burn_in:
            self.theta_ema = new_params.copy()
            self.theta_bema = new_params.copy()
            self.theta_0 = new_params.copy()  # Reset baseline during burn-in
            return self.theta_bema
        
        # Check update frequency (line 7 in Algorithm 1)
        if (self.step_count - self.config.burn_in) % self.config.update_freq != 0:
            return self.theta_bema  # Return previous BEMA estimate
        
        self.update_count += 1
        
        # Compute time-dependent weights (line 11 in Algorithm 1)
        t = self.step_count - self.config.burn_in
        alpha_t = (self.config.lag + self.config.multiplier * t) ** (-self.config.bias_power)
        beta_t = (self.config.lag + self.config.multiplier * t) ** (-self.config.ema_power)
        
        # EMA update (line 12 in Algorithm 1)
        self.theta_ema = (1 - beta_t) * self.theta_ema + beta_t * new_params
        
        # BEMA bias correction (line 13 in Algorithm 1)
        bias_correction = alpha_t * (new_params - self.theta_0)
        self.theta_bema = self.theta_ema + bias_correction
        
        # Track correction magnitude for diagnostics
        self.bias_correction_history.append({
            'step': self.step_count,
            'alpha_t': alpha_t,
            'beta_t': beta_t,
            'correction_norm': np.linalg.norm(bias_correction),
            'ema_norm': np.linalg.norm(self.theta_ema),
            'bema_norm': np.linalg.norm(self.theta_bema)
        })
        
        # Handle snapshot/restore for high-LR schedules
        if loss is not None:
            self._handle_snapshot_restore(loss)
        
        # Update learning rate schedule
        self._update_learning_rate()
        
        return self.theta_bema
    
    def _handle_snapshot_restore(self, loss: float):
        """Manage snapshots for high-LR robustness
        
        Key idea: Higher LRs enable faster convergence but risk instability.
        Snapshots allow recovery from bad updates without losing progress.
        """
        self.loss_history.append(loss)
        
        # Check if we should restore from snapshot
        if len(self.loss_history) > 10:
            recent_avg = np.mean(self.loss_history[-10:])
            if loss > recent_avg * (1 + self.config.restore_threshold):
                if self.snapshots:
                    logger.info(f"Loss spike detected ({loss:.4f} vs {recent_avg:.4f}), restoring snapshot")
                    self._restore_best_snapshot()
                    return
        
        # Create snapshot at intervals or when performance improves
        if (self.step_count % self.config.snapshot_interval == 0 or 
            loss < self.best_loss * 0.98):  # 2% improvement threshold
            
            self._create_snapshot(loss)
            
            if loss < self.best_loss:
                self.best_loss = loss
    
    def _create_snapshot(self, loss: float):
        """Create parameter snapshot for potential restoration"""
        snapshot = {
            'step': self.step_count,
            'loss': loss,
            'theta_bema': self.theta_bema.copy(),
            'theta_ema': self.theta_ema.copy(),
            'theta_0': self.theta_0.copy(),
            'lr': self.current_lr
        }
        
        self.snapshots.append(snapshot)
        
        # Keep only best snapshots
        if len(self.snapshots) > self.config.max_snapshots:
            self.snapshots.sort(key=lambda x: x['loss'])
            self.snapshots = self.snapshots[:self.config.max_snapshots]
        
        logger.debug(f"Created snapshot at step {self.step_count}, loss={loss:.4f}")
    
    def _restore_best_snapshot(self):
        """Restore parameters from best snapshot"""
        if not self.snapshots:
            return
        
        best_snapshot = min(self.snapshots, key=lambda x: x['loss'])
        
        self.theta_bema = best_snapshot['theta_bema'].copy()
        self.theta_ema = best_snapshot['theta_ema'].copy()
        self.theta_0 = best_snapshot['theta_0'].copy()
        self.current_lr = best_snapshot['lr'] * 0.5  # Reduce LR after restore
        
        logger.info(f"Restored snapshot from step {best_snapshot['step']}, "
                   f"loss={best_snapshot['loss']:.4f}, new LR={self.current_lr:.6f}")
        
        # Clear loss history to prevent repeated restores
        self.loss_history = []
    
    def _update_learning_rate(self):
        """Update learning rate according to schedule
        
        BEMA enables higher LRs through stabilization, allowing:
        1. Faster initial convergence with high LR
        2. Fine-tuning with decayed LR
        3. Escape from local minima with cyclic LR
        """
        if self.config.lr_schedule == "decay":
            # Exponential decay
            decay_steps = self.step_count // 1000
            self.current_lr = self.config.lr_grid[0] * (self.config.lr_decay_factor ** decay_steps)
            
        elif self.config.lr_schedule == "cyclic":
            # Cycle through LR grid
            cycle_position = (self.step_count // self.config.lr_cycle_steps) % len(self.config.lr_grid)
            self.current_lr = self.config.lr_grid[cycle_position]
            
        elif self.config.lr_schedule == "adaptive":
            # Adaptive based on loss improvement
            if len(self.loss_history) > 100:
                recent_improvement = (self.loss_history[-100] - self.loss_history[-1]) / self.loss_history[-100]
                
                if recent_improvement < 0.001:  # Less than 0.1% improvement
                    # Move to next LR in grid
                    self.lr_index = min(self.lr_index + 1, len(self.config.lr_grid) - 1)
                    self.current_lr = self.config.lr_grid[self.lr_index]
                    logger.info(f"Adaptive LR change to {self.current_lr:.6f}")
    
    def get_learning_rate(self) -> float:
        """Get current learning rate for optimizer"""
        return self.current_lr
    
    def get_parameters(self, use_bema: bool = True) -> np.ndarray:
        """Get current parameter estimate
        
        Args:
            use_bema: If True, return BEMA corrected params; else return EMA
            
        Returns:
            Parameter array
        """
        return self.theta_bema if use_bema else self.theta_ema
    
    def reset_baseline(self):
        """Reset baseline θ_0 for bias correction
        
        Useful when entering new optimization phase or after major updates
        """
        self.theta_0 = self.theta_current.copy()
        logger.info(f"Reset baseline at step {self.step_count}")
    
    def get_diagnostics(self) -> Dict:
        """Get comprehensive diagnostics"""
        recent_corrections = self.bias_correction_history[-10:] if self.bias_correction_history else []
        
        diagnostics = {
            'step_count': self.step_count,
            'update_count': self.update_count,
            'current_lr': self.current_lr,
            'lr_schedule': self.config.lr_schedule,
            'num_snapshots': len(self.snapshots),
            'best_loss': self.best_loss,
            'config': {
                'ema_power': self.config.ema_power,
                'bias_power': self.config.bias_power,
                'update_freq': self.config.update_freq,
                'burn_in': self.config.burn_in
            }
        }
        
        if recent_corrections:
            diagnostics['recent_bias_corrections'] = {
                'mean_alpha': np.mean([c['alpha_t'] for c in recent_corrections]),
                'mean_beta': np.mean([c['beta_t'] for c in recent_corrections]),
                'mean_correction_norm': np.mean([c['correction_norm'] for c in recent_corrections])
            }
        
        return diagnostics
    
    def save_state(self, filepath: str):
        """Save optimizer state to file"""
        state = {
            'config': self.config.__dict__,
            'theta_0': self.theta_0.tolist(),
            'theta_ema': self.theta_ema.tolist(),
            'theta_bema': self.theta_bema.tolist(),
            'step_count': self.step_count,
            'update_count': self.update_count,
            'current_lr': self.current_lr,
            'lr_index': self.lr_index,
            'best_loss': self.best_loss,
            'snapshots': [
                {
                    'step': s['step'],
                    'loss': s['loss'],
                    'lr': s['lr'],
                    'theta_bema': s['theta_bema'].tolist()
                }
                for s in self.snapshots[:2]  # Save only best 2 snapshots
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved BEMA optimizer state to {filepath}")
    
    def load_state(self, filepath: str) -> bool:
        """Load optimizer state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore config
            for key, value in state['config'].items():
                setattr(self.config, key, value)
            
            # Restore parameters
            self.theta_0 = np.array(state['theta_0'])
            self.theta_ema = np.array(state['theta_ema'])
            self.theta_bema = np.array(state['theta_bema'])
            self.theta_current = self.theta_bema.copy()
            
            # Restore state
            self.step_count = state['step_count']
            self.update_count = state['update_count']
            self.current_lr = state['current_lr']
            self.lr_index = state.get('lr_index', 0)
            self.best_loss = state['best_loss']
            
            # Restore snapshots
            self.snapshots = [
                {
                    'step': s['step'],
                    'loss': s['loss'],
                    'lr': s['lr'],
                    'theta_bema': np.array(s['theta_bema']),
                    'theta_ema': np.array(s['theta_bema']),  # Approximate
                    'theta_0': self.theta_0.copy()
                }
                for s in state.get('snapshots', [])
            ]
            
            logger.info(f"Loaded BEMA optimizer state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BEMA state: {e}")
            return False