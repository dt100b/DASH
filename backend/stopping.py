"""Deep optimal stopping module for exit decisions
Research basis: Paper 1804.05394v4 - Deep optimal stopping
Decomposes stopping into recursive binary decisions f^θ_n with lower/upper bounds
Enhanced with BEMA (2508.00180v1) for training stability
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Dict, List
import os
import json
from datetime import datetime, timezone

# Import BEMA optimizer for training stability
from backend.bema_optimizer import BEMAOptimizer, BEMAConfig
# Import JAL for noise-robust training
from backend.jal_loss import JointAsymmetricLoss, JALConfig, create_jal_loss
# Import Large LR scheduler for robustness and compressibility
from backend.lr_scheduler import LargeLRScheduler, LargeLRConfig, create_large_lr_scheduler

class StoppingNet(nn.Module):
    """Deep optimal stopping network (Paper 1804.05394v4)
    
    Implements recursive binary stopping decisions f^θ_n : R^d → {0,1}
    via neural approximation F^θ : R^d → (0,1) with hard decision threshold.
    
    Key equations:
    - Soft decision: F^θ = ψ ∘ a^θ_I ∘ φ_{q_{I-1}} ∘ ... ∘ φ_{q_1} ∘ a^θ_1
    - Hard decision: f^θ = 1_{[0,∞)} ∘ a^θ_I ∘ φ_{q_{I-1}} ∘ ... ∘ φ_{q_1} ∘ a^θ_1
    - Stopping time: τ_n = Σ_{m=n}^N m·f_m(X_m)·Π_{j=n}^{m-1}(1-f_j(X_j))
    """
    
    def __init__(self, input_dim: int = 32, depth: int = 3):
        """Initialize stopping network
        
        Args:
            input_dim: Feature dimension d
            depth: Network depth I (paper recommends I≥2)
        """
        super().__init__()
        
        # Paper uses tiny MLPs: q_1 = q_2 = d + 40 for Bermudan option
        # We use smaller: q_1 = q_2 = 16 for efficiency
        q1 = min(16, input_dim + 8)  # First hidden layer
        q2 = min(16, input_dim + 8)  # Second hidden layer
        
        # F^θ network for soft stopping probability
        if depth == 2:
            self.F_theta = nn.Sequential(
                nn.Linear(input_dim, q1),   # a^θ_1
                nn.ReLU(),                   # φ_{q_1}
                nn.Linear(q1, 1),            # a^θ_2
                nn.Sigmoid()                 # ψ (logistic function)
            )
        else:  # depth >= 3
            self.F_theta = nn.Sequential(
                nn.Linear(input_dim, q1),   # a^θ_1
                nn.ReLU(),                   # φ_{q_1}
                nn.Linear(q1, q2),           # a^θ_2
                nn.ReLU(),                   # φ_{q_2}
                nn.Linear(q2, 1),            # a^θ_3
                nn.Sigmoid()                 # ψ (logistic function)
            )
        
        # Martingale network for dual upper bound (Section 3.2)
        self.martingale_net = nn.Sequential(
            nn.Linear(input_dim, q1),
            nn.ReLU(),
            nn.Linear(q1, 1)  # M^Θ_n increments
        )
        
        # Continuation value network for bounds
        self.continuation_net = nn.Sequential(
            nn.Linear(input_dim, q1),
            nn.ReLU(),
            nn.Linear(q1, 2)  # (lower_bound, upper_bound)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass computing stopping probability and bounds
        
        Args:
            x: Feature tensor of shape (batch_size, input_dim)
            
        Returns:
            F_theta: Soft stopping probability in (0,1)
            f_theta: Hard stopping decision in {0,1}
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
        """
        # Soft stopping probability F^θ(x) ∈ (0,1)
        F_theta = self.F_theta(x)
        
        # Hard stopping decision f^θ(x) ∈ {0,1}
        # Paper: f^θ = 1_{[0,∞)} if F^θ ≥ 0.5
        f_theta = (F_theta >= 0.5).float()
        
        # Continuation value bounds
        bounds = self.continuation_net(x)
        lower_bound = bounds[:, 0]
        upper_bound = bounds[:, 1]
        
        # Martingale increment for dual bound
        martingale_increment = self.martingale_net(x).squeeze()
        
        return F_theta.squeeze(), f_theta.squeeze(), lower_bound, upper_bound

class DeepOptimalStopping:
    """Deep optimal stopping with recursive binary decisions (Paper 1804.05394v4)
    
    Implements optimal stopping via backward induction with:
    - Recursive STOP/HOLD decisions: τ = Σ n·f_n(X_n)·Π(1-f_j(X_j))
    - Lower bound: L = E[g(τ^Θ, X_τ^Θ)]
    - Upper bound: U = E[max(g(n,X_n) - M^Θ_n - ε_n)] via dual method
    - 7-day hard cap on holding period
    - Training at 00:00 UTC to avoid look-ahead bias
    """
    
    def __init__(self, max_holding_days: int = 7, use_bema: bool = True, 
                 use_jal: bool = True, use_large_lr: bool = True):
        """Initialize deep optimal stopping
        
        Args:
            max_holding_days: Maximum holding period (paper: N=7)
            use_bema: Whether to use BEMA for training stability
            use_jal: Whether to use JAL for noise-robust training
            use_large_lr: Whether to use large LR scheduler for robustness
        """
        self.max_holding_days = max_holding_days  # N in paper
        self.model = StoppingNet(input_dim=32, depth=3)  # Paper uses I=3
        
        # Paper: Adam optimizer with Xavier init, batch norm
        # Initial LR will be overridden by large LR scheduler if enabled
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # JAL loss for noise-robust training (Paper 2507.17692v1)
        self.use_jal = use_jal
        if use_jal:
            # Binary classification for STOP/HOLD decision
            self.jal_loss = create_jal_loss(
                num_classes=2,
                noise_threshold=0.15,  # Enable when noise > 15%
                active_type="CE"  # Use CE as active loss
            )
            self.jal_epoch = 0
        
        # BEMA optimizer for training stability (Paper 2508.00180v1)
        self.use_bema = use_bema
        if use_bema:
            # Initialize BEMA with paper-recommended settings
            bema_config = BEMAConfig(
                ema_power=0.5,  # κ=0.5 optimal from paper
                bias_power=0.4,  # η=0.4 from paper Figure 1d
                update_freq=400,  # Update every 400 steps (paper recommendation)
                burn_in=0,  # No burn-in for finetuning
                lr_grid=[1e-4, 5e-4, 1e-3, 5e-3],  # Higher LRs enabled by BEMA
                lr_schedule="adaptive",  # Adapt based on loss improvement
                snapshot_interval=1000,  # Save snapshots for recovery
                restore_threshold=0.1  # Restore if loss spikes 10%
            )
            
            # Initialize BEMA for each parameter
            self.bema_optimizers = {}
            for name, param in self.model.named_parameters():
                self.bema_optimizers[name] = BEMAOptimizer(
                    initial_params=param.data.cpu().numpy(),
                    config=bema_config
                )
        
        # Large LR scheduler for robustness and compressibility (Paper 2507.17748v2)
        self.use_large_lr = use_large_lr
        if use_large_lr:
            # Create scheduler with paper-recommended settings
            self.lr_scheduler = create_large_lr_scheduler(
                self.optimizer,
                base_lr=0.3,  # Large LR for robustness
                total_epochs=100
            )
            self.compressibility_history = []
            self.robustness_metrics = []
        
        # Training state
        self.training_history = []
        self.last_training_utc = None
        self.monte_carlo_paths = 8192  # Paper uses 8,192 paths per batch
        self.training_step_count = 0
        self.current_epoch = 0
        
    def _prepare_features(self, position_features: Dict[str, float], 
                         holding_days: int) -> torch.Tensor:
        """Prepare features for stopping decision
        
        Args:
            position_features: Current position features (PnL, vol, etc.)
            holding_days: Days position has been held
            
        Returns:
            Feature tensor
        """
        # Extract relevant features
        features = []
        
        # Position P&L features
        features.append(position_features.get('unrealized_pnl', 0))
        features.append(position_features.get('pnl_volatility', 0))
        features.append(position_features.get('max_drawdown', 0))
        
        # Market features
        features.append(position_features.get('current_vol', 0))
        features.append(position_features.get('vol_change', 0))
        features.append(position_features.get('regime_prob', 0.5))
        
        # Time features
        features.append(holding_days / self.max_holding_days)  # Normalized
        features.append(1.0 if holding_days >= self.max_holding_days else 0.0)  # Hard limit
        
        # Pad to expected dimension
        while len(features) < 32:
            features.append(0.0)
        
        return torch.tensor(features[:32], dtype=torch.float32)
    
    def decide(self, position_features: Dict[str, float], 
              holding_days: int) -> Tuple[str, float, Tuple[float, float]]:
        """Make recursive stopping decision (Paper Section 2.1)
        
        Implements: τ_n = Σ_{m=n}^N m·f_m(X_m)·Π_{j=n}^{m-1}(1-f_j(X_j))
        
        Args:
            position_features: Current position features (X_n)
            holding_days: Days held (n)
            
        Returns:
            decision: "STOP" or "HOLD" based on f^θ_n(X_n)
            confidence: Stopping probability F^θ_n(X_n)
            bounds: (lower_bound, upper_bound) for value estimate
        """
        # Paper: Hard constraint at N (max holding period)
        if holding_days >= self.max_holding_days:
            return "STOP", 1.0, (0.0, 0.0)
        
        # Prepare features X_n
        features = self._prepare_features(position_features, holding_days)
        features_batch = features.unsqueeze(0)  # Add batch dimension
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            F_theta, f_theta, lower, upper = self.model(features_batch)
            
        # Extract scalar values
        F_theta_val = F_theta.item()  # Soft probability
        f_theta_val = f_theta.item()  # Hard decision {0,1}
        lower_val = lower.item()
        upper_val = upper.item()
        
        # Paper: f^θ = 1 means STOP, f^θ = 0 means HOLD
        if f_theta_val == 1.0:
            decision = "STOP"
            confidence = F_theta_val
        else:
            decision = "HOLD"  
            confidence = 1.0 - F_theta_val
        
        return decision, confidence, (lower_val, upper_val)
    
    def train_step(self, features_batch: torch.Tensor, 
                  rewards: torch.Tensor,
                  continuation_values: torch.Tensor,
                  holding_times: torch.Tensor):
        """Single training step using backward induction (Paper Section 2.3)
        
        Implements gradient ascent on:
        r_n(θ) = g(n,X_n)·F^θ(X_n) + g(τ_{n+1},X_{τ_{n+1}})·(1-F^θ(X_n))
        
        Args:
            features_batch: Batch of X_n features (K × d)
            rewards: Immediate rewards g(n,X_n) (K,)
            continuation_values: Continuation values g(τ_{n+1},X_{τ_{n+1}}) (K,)
            holding_times: Current holding times n (K,)
        """
        # Training only at 00:00 UTC to avoid look-ahead bias
        current_utc = datetime.now(timezone.utc)
        if current_utc.hour != 0:
            return  # Skip training outside 00:00 UTC window
        
        self.model.train()
        
        # Forward pass
        F_theta, f_theta, lower_bounds, upper_bounds = self.model(features_batch)
        
        # Paper Equation (15): Expected reward for soft stopping
        # r_n(θ) = g(n,X_n)·F^θ(X_n) + g(τ_{n+1},X_{τ_{n+1}})·(1-F^θ(X_n))
        expected_reward = rewards * F_theta + continuation_values * (1 - F_theta)
        
        # Compute loss based on JAL configuration
        if self.use_jal:
            # Create pseudo-labels for binary classification (STOP=1, HOLD=0)
            optimal_decisions = (rewards > continuation_values).long()
            
            # Convert soft probabilities to logits for JAL
            eps = 1e-8  # Prevent log(0)
            F_theta_clamped = torch.clamp(F_theta, eps, 1 - eps)
            logits = torch.stack([
                torch.log((1 - F_theta_clamped) / F_theta_clamped),  # HOLD logit
                torch.log(F_theta_clamped / (1 - F_theta_clamped))   # STOP logit
            ], dim=1)
            
            # Apply JAL loss for noise-robust training
            jal_loss, jal_metrics = self.jal_loss(logits, optimal_decisions, epoch=self.jal_epoch)
            
            # Combine with regularization losses
            bound_loss = torch.mean(
                torch.relu(lower_bounds - torch.where(f_theta == 1, rewards, continuation_values)) +
                torch.relu(torch.where(f_theta == 1, rewards, continuation_values) - upper_bounds)
            )
            width_loss = torch.mean(upper_bounds - lower_bounds)
            
            # Total loss with JAL
            total_loss = jal_loss + 0.01 * bound_loss + 0.001 * width_loss
            
            # Track JAL metrics
            if hasattr(self, 'jal_metrics_history'):
                self.jal_metrics_history.append(jal_metrics)
            else:
                self.jal_metrics_history = [jal_metrics]
        else:
            # Original loss computation
            # Maximize expected reward (gradient ascent -> minimize negative)
            reward_loss = -torch.mean(expected_reward)
            
            # Dual bound consistency (Paper Section 3.2)
            # Ensure lower ≤ g(τ,X_τ) ≤ upper
            actual_values = torch.where(f_theta == 1, rewards, continuation_values)
            bound_loss = torch.mean(
                torch.relu(lower_bounds - actual_values) +  # Lower bound violation
                torch.relu(actual_values - upper_bounds)    # Upper bound violation
            )
            
            # Martingale consistency loss (ensure bounds are tight)
            bound_width = upper_bounds - lower_bounds
            width_loss = torch.mean(bound_width)
            
            # Total loss with tiny weight on regularizers (paper uses small MLPs)
            total_loss = reward_loss + 0.01 * bound_loss + 0.001 * width_loss
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Apply gradient clipping (essential for large LRs)
        if self.use_large_lr:
            grad_norm = self.lr_scheduler.apply_gradient_clipping(self.model)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0).item()
        
        self.optimizer.step()
        
        # Update LR scheduler if enabled
        if self.use_large_lr:
            current_lr = self.lr_scheduler.step(epoch=self.current_epoch)
            
            # Track compressibility metrics periodically
            if self.training_step_count % 100 == 0:
                compressibility = self.lr_scheduler.estimate_compressibility(self.model)
                self.compressibility_history.append(compressibility)
                
                # Track robustness indicators with current features
                robustness = self.lr_scheduler.check_robustness_indicators(
                    features, targets
                )
                self.robustness_metrics.append(robustness)
        
        # Apply BEMA stabilization if enabled
        if self.use_bema:
            self.training_step_count += 1
        else:
            self.training_step_count += 1
            
            # Update BEMA with new parameters
            for name, param in self.model.named_parameters():
                if name in self.bema_optimizers:
                    # Get BEMA-corrected parameters
                    bema_params = self.bema_optimizers[name].update(
                        new_params=param.data.cpu().numpy(),
                        loss=total_loss.item()
                    )
                    
                    # Update model with BEMA parameters (bias-corrected)
                    if self.training_step_count % self.bema_optimizers[name].config.update_freq == 0:
                        param.data = torch.from_numpy(bema_params).to(param.device)
                    
                    # Adjust learning rate based on BEMA schedule
                    if self.training_step_count % 100 == 0:
                        new_lr = self.bema_optimizers[name].get_learning_rate()
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
        
        # Record training history
        history_entry = {
            'reward_loss': reward_loss.item() if hasattr(reward_loss, 'item') else 0,
            'bound_loss': bound_loss.item(),
            'width_loss': width_loss.item(),
            'total_loss': total_loss.item(),
            'timestamp': current_utc.isoformat(),
            'step': self.training_step_count,
            'gradient_norm': grad_norm if 'grad_norm' in locals() else 0
        }
        
        # Add Large LR diagnostics if enabled
        if self.use_large_lr:
            lr_diag = self.lr_scheduler.get_diagnostics()
            history_entry['large_lr'] = {
                'current_lr': lr_diag['current_lr'],
                'warmup_complete': lr_diag['warmup_complete'],
                'recent_sparsity': lr_diag['recent_sparsity'],
                'recent_confidence': lr_diag['recent_confidence']
            }
        
        # Add BEMA diagnostics if enabled
        if self.use_bema:
            bema_diag = list(self.bema_optimizers.values())[0].get_diagnostics()
            history_entry['bema'] = {
                'current_lr': bema_diag.get('current_lr', 0),
                'num_snapshots': bema_diag.get('num_snapshots', 0),
                'update_count': bema_diag.get('update_count', 0)
            }
        
        self.training_history.append(history_entry)
        self.last_training_utc = current_utc
    
    def compute_bounds(self, paths: np.ndarray, rewards: np.ndarray) -> Tuple[float, float]:
        """Compute lower and upper bounds (Paper Section 3)
        
        Lower bound: L = E[g(τ^Θ, X_τ^Θ)]
        Upper bound: U = E[max_{0≤n≤N}(g(n,X_n) - M^Θ_n - ε_n)]
        
        Args:
            paths: Monte Carlo paths (K_L × N × d)
            rewards: Rewards g(n,X_n) (K_L × N)
            
        Returns:
            lower_bound: L̂ estimate
            upper_bound: Û estimate
        """
        K_L, N, d = paths.shape
        
        # Convert to tensors
        paths_tensor = torch.tensor(paths, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            # Compute stopping times and lower bound
            stopped = torch.zeros(K_L, dtype=torch.bool)
            lower_values = torch.zeros(K_L)
            
            for n in range(N):
                if n >= self.max_holding_days:
                    # Hard stop at max holding
                    lower_values[~stopped] = rewards_tensor[~stopped, n]
                    break
                
                # Get stopping decisions
                features = paths_tensor[:, n, :]
                F_theta, f_theta, _, _ = self.model(features)
                
                # Apply stopping decisions
                stop_now = (f_theta == 1) & ~stopped
                lower_values[stop_now] = rewards_tensor[stop_now, n]
                stopped = stopped | stop_now
            
            # Lower bound estimate (Paper Section 3.1)
            lower_bound = torch.mean(lower_values).item()
            
            # Upper bound via dual method (Paper Section 3.2)
            martingale_sum = torch.zeros(K_L)
            upper_values = torch.zeros(K_L)
            
            for n in range(N):
                features = paths_tensor[:, n, :]
                # Get martingale increments
                M_n = self.model.martingale_net(features).squeeze()
                martingale_sum += M_n
                
                # Track maximum value
                current_value = rewards_tensor[:, n] - martingale_sum
                upper_values = torch.maximum(upper_values, current_value)
            
            # Upper bound estimate
            upper_bound = torch.mean(upper_values).item()
        
        return lower_bound, upper_bound
    
    def save_model(self, path: str):
        """Save model to disk including BEMA state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'max_holding_days': self.max_holding_days,
            'use_bema': self.use_bema,
            'training_step_count': self.training_step_count
        }
        
        torch.save(save_dict, path)
        
        # Save BEMA states if enabled
        if self.use_bema:
            bema_path = path.replace('.pt', '_bema')
            for name, bema_opt in self.bema_optimizers.items():
                bema_opt.save_state(f"{bema_path}_{name}.json")
        
        # Save metadata
        metadata_path = path.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'type': 'deep_optimal_stopping',
                'max_holding_days': self.max_holding_days,
                'training_samples': len(self.training_history),
                'use_bema': self.use_bema
            }, f)
    
    def load_model(self, path: str) -> bool:
        """Load model from disk including BEMA state"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_history = checkpoint.get('training_history', [])
            self.max_holding_days = checkpoint.get('max_holding_days', 7)
            self.use_bema = checkpoint.get('use_bema', False)
            self.training_step_count = checkpoint.get('training_step_count', 0)
            
            # Load BEMA states if enabled
            if self.use_bema:
                bema_path = path.replace('.pt', '_bema')
                for name, param in self.model.named_parameters():
                    bema_file = f"{bema_path}_{name}.json"
                    if os.path.exists(bema_file):
                        if name not in self.bema_optimizers:
                            self.bema_optimizers[name] = BEMAOptimizer(
                                initial_params=param.data.cpu().numpy(),
                                config=BEMAConfig()
                            )
                        self.bema_optimizers[name].load_state(bema_file)
            
            return True
        return False
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Get diagnostic metrics for deep optimal stopping"""
        if not self.training_history:
            return {
                'max_holding_days': self.max_holding_days,
                'monte_carlo_paths': self.monte_carlo_paths,
                'last_training_utc': None,
                'training_samples': 0,
                'use_bema': self.use_bema,
                'use_jal': self.use_jal
            }
        
        recent = self.training_history[-100:] if len(self.training_history) > 100 else self.training_history
        
        diagnostics = {
            'avg_reward_loss': np.mean([h['reward_loss'] for h in recent]),
            'avg_bound_loss': np.mean([h['bound_loss'] for h in recent]),
            'avg_width_loss': np.mean([h['width_loss'] for h in recent]),
            'avg_total_loss': np.mean([h['total_loss'] for h in recent]),
            'max_holding_days': self.max_holding_days,
            'monte_carlo_paths': self.monte_carlo_paths,
            'last_training_utc': self.last_training_utc.isoformat() if self.last_training_utc else None,
            'training_samples': len(self.training_history),
            'use_bema': self.use_bema,
            'use_jal': self.use_jal
        }
        
        # Add BEMA diagnostics if enabled
        if self.use_bema and self.bema_optimizers:
            bema_diag = list(self.bema_optimizers.values())[0].get_diagnostics()
            diagnostics['bema'] = {
                'current_lr': bema_diag.get('current_lr', 0),
                'lr_schedule': bema_diag.get('lr_schedule', 'constant'),
                'num_snapshots': bema_diag.get('num_snapshots', 0),
                'update_count': bema_diag.get('update_count', 0),
                'step_count': bema_diag.get('step_count', 0)
            }
            
        # Add JAL diagnostics if enabled
        if self.use_jal and hasattr(self, 'jal_loss'):
            jal_diag = self.jal_loss.get_diagnostics()
            diagnostics['jal'] = {
                'active': jal_diag.get('jal_active', False),
                'estimated_noise_rate': jal_diag.get('estimated_noise_rate', 0),
                'activation_rate': jal_diag.get('jal_activation_rate', 0),
                'alpha': jal_diag['config'].get('alpha', 0),
                'beta': jal_diag['config'].get('beta', 0),
                'asymmetry_param': jal_diag['config'].get('asymmetry_param', 0)
            }
            
        # Add Large LR diagnostics if enabled  
        if hasattr(self, 'use_large_lr') and self.use_large_lr and hasattr(self, 'lr_scheduler'):
            lr_diag = self.lr_scheduler.get_diagnostics()
            diagnostics['large_lr'] = {
                'current_lr': lr_diag['current_lr'],
                'base_lr': lr_diag['base_lr'],
                'warmup_complete': lr_diag['warmup_complete'],
                'schedule_type': lr_diag['schedule_type'],
                'recent_sparsity': lr_diag.get('recent_sparsity', 0),
                'recent_confidence': lr_diag.get('recent_confidence', 0),
                'recent_gradient_norm': lr_diag.get('recent_gradient_norm', 0)
            }
            
            # Add compressibility metrics if available
            if hasattr(self, 'compressibility_history') and self.compressibility_history:
                recent_compress = self.compressibility_history[-1]
                diagnostics['large_lr']['compressibility'] = recent_compress.get('compressible_ratio', 0)
                
            # Add robustness metrics if available
            if hasattr(self, 'robustness_metrics') and self.robustness_metrics:
                recent_robust = self.robustness_metrics[-1]
                diagnostics['large_lr']['class_separation'] = recent_robust.get('class_separation', 0)
                diagnostics['large_lr']['feature_utilization'] = recent_robust.get('feature_utilization', 0)
        
        return diagnostics