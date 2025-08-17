"""Large Learning Rate Scheduler for Robustness and Compressibility
Implementation of paper 2507.17748v2: Large Learning Rates Simultaneously Achieve
Robustness to Spurious Correlations and Compressibility

Key findings: Large LRs induce invariant feature learning, class separation,
activation sparsity, and model compressibility while improving robustness.
"""
import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)

@dataclass
class LargeLRConfig:
    """Configuration for large learning rate scheduling
    
    Paper findings:
    - Large LRs (>0.1) achieve robustness to spurious correlations
    - Warmup crucial for stability with large LRs
    - Cosine decay preserves benefits while improving convergence
    - Weight decay and gradient clipping essential for stability
    """
    # Base LR parameters (Paper Section 3)
    base_lr: float = 0.3  # Large LR from paper (0.1-1.0 range)
    min_lr: float = 1e-5  # Minimum LR for cosine decay
    
    # Warmup configuration (Paper Section 4.2)
    warmup_epochs: int = 5  # Critical for large LR stability
    warmup_start_lr: float = 0.01  # Start small, ramp up
    
    # Decay configuration
    decay_type: str = "cosine"  # "cosine", "step", "exponential"
    decay_epochs: List[int] = None  # For step decay
    decay_factor: float = 0.1  # For step/exponential decay
    
    # LR grid for robustness search (Paper Figure 1)
    lr_grid: List[float] = None  # Grid search values
    
    # Regularization for large LRs
    weight_decay: float = 5e-4  # L2 regularization
    gradient_clip: float = 1.0  # Gradient clipping threshold
    
    # Compressibility monitoring
    track_sparsity: bool = True  # Monitor activation sparsity
    sparsity_threshold: float = 0.1  # Threshold for sparse activations
    
    # Robustness monitoring
    track_confidence: bool = True  # Monitor prediction confidence
    track_feature_utilization: bool = True  # Core vs spurious features
    
    def __post_init__(self):
        """Initialize default LR grid if not provided"""
        if self.lr_grid is None:
            # Paper-recommended grid for small heads
            self.lr_grid = [0.01, 0.03, 0.1, 0.3, 0.5, 0.8, 1.0]
        if self.decay_epochs is None:
            self.decay_epochs = [30, 60, 90]  # Default milestones

class LargeLRScheduler:
    """Large learning rate scheduler for robustness and compressibility
    
    Key mechanisms from paper:
    1. Large LRs prevent memorization of spurious correlations
    2. Induce confident mispredictions on bias-conflicting samples
    3. Create sparse, compressible representations
    4. Improve invariant feature utilization
    """
    
    def __init__(self, optimizer: optim.Optimizer, 
                 config: Optional[LargeLRConfig] = None,
                 total_epochs: int = 100):
        """Initialize large LR scheduler
        
        Args:
            optimizer: PyTorch optimizer
            config: LR configuration
            total_epochs: Total training epochs for cosine schedule
        """
        self.optimizer = optimizer
        self.config = config or LargeLRConfig()
        self.total_epochs = total_epochs
        
        # Initialize state
        self.current_epoch = 0
        self.current_step = 0
        self.current_lr = self.config.warmup_start_lr
        
        # Metrics tracking
        self.lr_history = []
        self.sparsity_history = []
        self.confidence_history = []
        self.gradient_norm_history = []
        
        # Set initial LR
        self._set_lr(self.current_lr)
        
    def step(self, epoch: Optional[int] = None, 
             metrics: Optional[Dict[str, float]] = None) -> float:
        """Update learning rate based on schedule
        
        Args:
            epoch: Current epoch (if None, increments internally)
            metrics: Optional metrics for adaptive scheduling
            
        Returns:
            Current learning rate
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
            
        # Compute scheduled LR
        if self.current_epoch < self.config.warmup_epochs:
            # Linear warmup (Paper Section 4.2)
            lr = self._warmup_lr()
        else:
            # Main schedule (cosine, step, or exponential)
            lr = self._decay_lr()
            
        # Apply LR to optimizer
        self._set_lr(lr)
        self.current_lr = lr
        
        # Track history
        self.lr_history.append(lr)
        
        # Log if significant change
        if len(self.lr_history) > 1:
            prev_lr = self.lr_history[-2]
            if abs(lr - prev_lr) / (prev_lr + 1e-8) > 0.1:
                logger.info(f"LR changed: {prev_lr:.6f} -> {lr:.6f} at epoch {self.current_epoch}")
                
        return lr
    
    def _warmup_lr(self) -> float:
        """Linear warmup schedule (critical for large LRs)
        
        Paper insight: Warmup prevents early instability with large LRs
        """
        warmup_progress = self.current_epoch / self.config.warmup_epochs
        lr = self.config.warmup_start_lr + \
             (self.config.base_lr - self.config.warmup_start_lr) * warmup_progress
        return lr
    
    def _decay_lr(self) -> float:
        """Apply decay schedule after warmup"""
        adjusted_epoch = self.current_epoch - self.config.warmup_epochs
        adjusted_total = self.total_epochs - self.config.warmup_epochs
        
        if self.config.decay_type == "cosine":
            # Cosine annealing (Paper's preferred schedule)
            lr = self._cosine_decay(adjusted_epoch, adjusted_total)
        elif self.config.decay_type == "step":
            # Step decay at milestones
            lr = self._step_decay(self.current_epoch)
        elif self.config.decay_type == "exponential":
            # Exponential decay
            lr = self._exponential_decay(adjusted_epoch)
        else:
            # Constant LR after warmup
            lr = self.config.base_lr
            
        return max(lr, self.config.min_lr)
    
    def _cosine_decay(self, epoch: int, total: int) -> float:
        """Cosine annealing schedule
        
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * epoch / total))
        """
        lr = self.config.min_lr + 0.5 * (self.config.base_lr - self.config.min_lr) * \
             (1 + math.cos(math.pi * epoch / total))
        return lr
    
    def _step_decay(self, epoch: int) -> float:
        """Step decay at specified milestones"""
        lr = self.config.base_lr
        for milestone in self.config.decay_epochs:
            if epoch >= milestone:
                lr *= self.config.decay_factor
        return lr
    
    def _exponential_decay(self, epoch: int) -> float:
        """Exponential decay: lr = base_lr * (decay_factor ^ epoch)"""
        lr = self.config.base_lr * (self.config.decay_factor ** epoch)
        return lr
    
    def _set_lr(self, lr: float):
        """Set learning rate in optimizer"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def apply_gradient_clipping(self, model: torch.nn.Module) -> float:
        """Apply gradient clipping for large LR stability
        
        Paper: Gradient clipping essential for large LRs
        
        Returns:
            Gradient norm before clipping
        """
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.config.gradient_clip
        )
        self.gradient_norm_history.append(grad_norm.item())
        return grad_norm.item()
    
    def compute_activation_sparsity(self, activations: torch.Tensor) -> float:
        """Compute activation sparsity (Paper Section 3.3)
        
        Sparsity = fraction of activations below threshold
        High sparsity indicates compressibility
        """
        with torch.no_grad():
            sparsity = (torch.abs(activations) < self.config.sparsity_threshold).float().mean()
            self.sparsity_history.append(sparsity.item())
            return sparsity.item()
    
    def compute_prediction_confidence(self, logits: torch.Tensor) -> Dict[str, float]:
        """Compute prediction confidence metrics (Paper Section 4.3)
        
        Large LRs produce confident predictions even on conflicting samples
        """
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            
            metrics = {
                'mean_confidence': max_probs.mean().item(),
                'std_confidence': max_probs.std().item(),
                'high_confidence_ratio': (max_probs > 0.9).float().mean().item()
            }
            
            self.confidence_history.append(metrics['mean_confidence'])
            return metrics
    
    def estimate_compressibility(self, model: torch.nn.Module) -> Dict[str, float]:
        """Estimate model compressibility (Paper Section 3)
        
        Metrics indicating potential for pruning/quantization
        """
        total_params = 0
        sparse_params = 0
        small_params = 0
        
        with torch.no_grad():
            for param in model.parameters():
                total_params += param.numel()
                sparse_params += (torch.abs(param) < 0.01).sum().item()
                small_params += (torch.abs(param) < 0.1).sum().item()
                
        return {
            'sparsity_1e-2': sparse_params / total_params,
            'sparsity_1e-1': small_params / total_params,
            'total_parameters': total_params,
            'compressible_ratio': small_params / total_params
        }
    
    def check_robustness_indicators(self, features: torch.Tensor, 
                                   labels: torch.Tensor) -> Dict[str, float]:
        """Check indicators of robustness to spurious correlations
        
        Paper: Large LRs improve invariant feature utilization
        """
        with torch.no_grad():
            # Feature statistics
            feature_mean = features.mean(dim=0)
            feature_std = features.std(dim=0)
            
            # Class-wise feature utilization
            unique_labels = torch.unique(labels)
            class_separation = 0.0
            
            for label in unique_labels:
                mask = labels == label
                if mask.sum() > 1:
                    class_features = features[mask]
                    intra_class_var = class_features.var(dim=0).mean()
                    inter_class_dist = torch.norm(class_features.mean(dim=0) - feature_mean)
                    class_separation += inter_class_dist / (intra_class_var + 1e-8)
                    
            class_separation /= len(unique_labels)
            
            return {
                'feature_sparsity': (torch.abs(features) < 0.1).float().mean().item(),
                'feature_std': feature_std.mean().item(),
                'class_separation': class_separation.item(),
                'feature_utilization': (feature_std > 0.1).float().mean().item()
            }
    
    def get_lr_grid_search_config(self) -> List[float]:
        """Get LR grid for robustness search (Paper Figure 1)"""
        return self.config.lr_grid
    
    def get_current_regularization(self) -> Dict[str, float]:
        """Get current regularization parameters"""
        return {
            'learning_rate': self.current_lr,
            'weight_decay': self.config.weight_decay,
            'gradient_clip': self.config.gradient_clip
        }
    
    def get_diagnostics(self) -> Dict[str, any]:
        """Get comprehensive diagnostics for large LR training"""
        recent_sparsity = np.mean(self.sparsity_history[-100:]) if self.sparsity_history else 0
        recent_confidence = np.mean(self.confidence_history[-100:]) if self.confidence_history else 0
        recent_grad_norm = np.mean(self.gradient_norm_history[-100:]) if self.gradient_norm_history else 0
        
        return {
            'current_lr': self.current_lr,
            'current_epoch': self.current_epoch,
            'warmup_complete': self.current_epoch >= self.config.warmup_epochs,
            'schedule_type': self.config.decay_type,
            'base_lr': self.config.base_lr,
            'recent_sparsity': recent_sparsity,
            'recent_confidence': recent_confidence,
            'recent_gradient_norm': recent_grad_norm,
            'lr_history_len': len(self.lr_history),
            'config': {
                'warmup_epochs': self.config.warmup_epochs,
                'weight_decay': self.config.weight_decay,
                'gradient_clip': self.config.gradient_clip
            }
        }

# LR-forward recipe for small heads (Paper recommendations)
def get_small_head_lr_recipe() -> Dict[str, any]:
    """Get recommended LR recipe for small network heads
    
    Based on paper's findings for robust and compressible models
    """
    return {
        'lr_grid': [0.01, 0.03, 0.1, 0.3, 0.5, 0.8],  # Search grid
        'recommended_lr': 0.3,  # Paper's sweet spot
        'warmup': {
            'epochs': 5,
            'start_lr': 0.01,
            'type': 'linear'
        },
        'decay': {
            'type': 'cosine',
            'min_lr': 1e-5,
            'total_epochs': 100
        },
        'regularization': {
            'weight_decay': 5e-4,
            'gradient_clip': 1.0,
            'label_smoothing': 0.1
        },
        'expected_metrics': {
            'activation_sparsity': '>0.3',  # 30%+ sparse
            'compressibility': '>0.5',  # 50%+ compressible
            'confidence': '>0.85',  # High confidence predictions
            'class_separation': '>2.0',  # Clear class boundaries
            'robustness_gain': '10-20%'  # Expected improvement
        }
    }

def create_large_lr_scheduler(optimizer: optim.Optimizer,
                             base_lr: float = 0.3,
                             total_epochs: int = 100) -> LargeLRScheduler:
    """Factory function to create large LR scheduler"""
    config = LargeLRConfig(
        base_lr=base_lr,
        warmup_epochs=5,
        decay_type="cosine",
        weight_decay=5e-4,
        gradient_clip=1.0
    )
    return LargeLRScheduler(optimizer, config, total_epochs)