"""Joint Asymmetric Loss (JAL) for robust learning with noisy labels
Implementation of paper 2507.17692v1: Joint Asymmetric Loss for Learning with Noisy Labels

Key innovation: Combines active loss (CE/FL) with passive asymmetric loss (AMSE)
for enhanced robustness to label noise while maintaining fitting ability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class JALConfig:
    """Configuration for Joint Asymmetric Loss
    
    Paper recommendations:
    - alpha: Weight for active loss (0.5-0.8 typical)
    - beta: Weight for passive loss (0.2-0.5 typical)
    - noise_threshold: Switch JAL on when estimated noise > threshold
    - asymmetry_param: Controls asymmetry strength (2-4 optimal)
    """
    # Loss combination weights
    alpha: float = 0.7  # Weight for active loss (CE or FL)
    beta: float = 0.3   # Weight for passive loss (AMSE)
    
    # Active loss configuration
    active_loss_type: str = "CE"  # "CE" or "FL" (Focal Loss)
    focal_gamma: float = 2.0  # Gamma for Focal Loss
    
    # AMSE (Asymmetric MSE) parameters
    asymmetry_param: float = 3.0  # q parameter from paper (2-4 optimal)
    mse_reduction: str = "mean"  # Reduction for MSE
    
    # Noise detection
    noise_threshold: float = 0.1  # Enable JAL when noise > threshold
    noise_estimation_window: int = 100  # Samples for noise estimation
    warmup_epochs: int = 5  # Epochs before enabling JAL
    
    # Regularization
    label_smoothing: float = 0.1  # Label smoothing factor
    gradient_clip: float = 1.0  # Gradient clipping threshold
    
    # Monitoring
    track_metrics: bool = True  # Track noise and loss metrics
    log_interval: int = 100  # Log metrics every N batches

class AsymmetricMSE(nn.Module):
    """Asymmetric Mean Square Error (AMSE) - Passive Loss
    
    Key equation from paper (Section 4.2):
    AMSE(p, y) = (1/K) * Σ_{k≠y} |p_k|^q
    
    where q > 1 controls asymmetry strength
    
    Properties:
    - Satisfies asymmetric condition for q ∈ (1, ∞)
    - More robust to label noise than symmetric MSE
    - Passive loss: explicitly minimizes non-target probabilities
    """
    
    def __init__(self, num_classes: int, asymmetry_param: float = 3.0):
        super().__init__()
        self.num_classes = num_classes
        self.q = asymmetry_param  # Asymmetry parameter
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute AMSE loss
        
        Args:
            logits: Model outputs before softmax [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            AMSE loss value
        """
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Create one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Compute passive term: minimize non-target probabilities
        # AMSE = (1/K) * Σ_{k≠y} |p_k|^q
        non_target_mask = 1 - targets_one_hot
        non_target_probs = probs * non_target_mask
        
        # Apply asymmetric power
        amse_loss = torch.pow(non_target_probs, self.q)
        
        # Average over non-target classes
        loss = amse_loss.sum(dim=1) / (self.num_classes - 1)
        
        return loss.mean()

class FocalLoss(nn.Module):
    """Focal Loss - Alternative active loss for imbalanced data
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
    - p_t: Probability of correct class
    - γ: Focusing parameter (typically 2)
    - α_t: Class balancing weight
    """
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_term = (1 - p_t) ** self.gamma
        focal_loss = focal_term * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

class JointAsymmetricLoss(nn.Module):
    """Joint Asymmetric Loss (JAL) Framework
    
    Main equation (Section 4.3):
    JAL = α * L_active + β * L_passive
    
    where:
    - L_active: CE or FL (active loss)
    - L_passive: AMSE (passive asymmetric loss)
    - α, β: Combination weights
    
    Key innovation: Combines fitting ability of active loss
    with noise robustness of passive asymmetric loss
    """
    
    def __init__(self, num_classes: int, config: Optional[JALConfig] = None):
        super().__init__()
        
        self.config = config or JALConfig()
        self.num_classes = num_classes
        
        # Initialize component losses
        self.amse = AsymmetricMSE(num_classes, self.config.asymmetry_param)
        
        # Active loss selection
        if self.config.active_loss_type == "FL":
            self.active_loss = FocalLoss(gamma=self.config.focal_gamma)
        else:  # Default to CE
            self.active_loss = nn.CrossEntropyLoss()
            
        # Noise tracking
        self.noise_estimates = []
        self.epoch_count = 0
        self.is_active = False
        
        # Metrics tracking
        self.metrics_history = []
        
    def estimate_noise_rate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Estimate label noise rate using prediction disagreement
        
        Method: Track disagreement between model predictions and labels
        High disagreement suggests label noise
        """
        with torch.no_grad():
            pred_classes = predictions.argmax(dim=1)
            disagreement = (pred_classes != targets).float().mean()
            
            # Maintain rolling window
            self.noise_estimates.append(disagreement.item())
            if len(self.noise_estimates) > self.config.noise_estimation_window:
                self.noise_estimates.pop(0)
                
            # Return average estimate
            if self.noise_estimates:
                return np.mean(self.noise_estimates)
            return 0.0
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute JAL loss
        
        Args:
            logits: Model outputs [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            epoch: Current training epoch
            
        Returns:
            loss: Combined JAL loss
            metrics: Dictionary of metrics for monitoring
        """
        metrics = {}
        
        # Update epoch count
        if epoch is not None:
            self.epoch_count = epoch
            
        # Estimate noise rate
        noise_rate = self.estimate_noise_rate(logits, targets)
        metrics['noise_rate'] = noise_rate
        
        # Determine if JAL should be active
        warmup_complete = self.epoch_count >= self.config.warmup_epochs
        noise_detected = noise_rate > self.config.noise_threshold
        self.is_active = warmup_complete and noise_detected
        
        if self.is_active:
            # Compute active loss (CE or FL)
            active_loss = self.active_loss(logits, targets)
            metrics['active_loss'] = active_loss.item()
            
            # Compute passive loss (AMSE)
            passive_loss = self.amse(logits, targets)
            metrics['passive_loss'] = passive_loss.item()
            
            # Combine losses (Equation 4.3 from paper)
            loss = self.config.alpha * active_loss + self.config.beta * passive_loss
            metrics['jal_active'] = True
            
        else:
            # Use only active loss during warmup or low noise
            loss = self.active_loss(logits, targets)
            metrics['active_loss'] = loss.item()
            metrics['passive_loss'] = 0.0
            metrics['jal_active'] = False
            
        # Apply label smoothing if configured
        if self.config.label_smoothing > 0:
            smooth_loss = self._label_smoothing_loss(logits, targets)
            loss = (1 - self.config.label_smoothing) * loss + self.config.label_smoothing * smooth_loss
            
        metrics['total_loss'] = loss.item()
        
        # Track metrics
        if self.config.track_metrics:
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 1000:
                self.metrics_history.pop(0)
                
        return loss, metrics
    
    def _label_smoothing_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing regularization"""
        log_probs = F.log_softmax(logits, dim=1)
        
        # Smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.config.label_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 
                              1.0 - self.config.label_smoothing)
            
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))
    
    def get_diagnostics(self) -> Dict[str, any]:
        """Get diagnostic metrics for JAL"""
        if not self.metrics_history:
            return {
                'jal_active': self.is_active,
                'estimated_noise_rate': 0.0,
                'epoch_count': self.epoch_count,
                'config': {
                    'alpha': self.config.alpha,
                    'beta': self.config.beta,
                    'asymmetry_param': self.config.asymmetry_param,
                    'noise_threshold': self.config.noise_threshold
                }
            }
            
        recent = self.metrics_history[-100:] if len(self.metrics_history) > 100 else self.metrics_history
        
        return {
            'jal_active': self.is_active,
            'estimated_noise_rate': np.mean([m['noise_rate'] for m in recent]),
            'avg_active_loss': np.mean([m['active_loss'] for m in recent]),
            'avg_passive_loss': np.mean([m.get('passive_loss', 0) for m in recent]),
            'avg_total_loss': np.mean([m['total_loss'] for m in recent]),
            'jal_activation_rate': np.mean([m['jal_active'] for m in recent]),
            'epoch_count': self.epoch_count,
            'config': {
                'alpha': self.config.alpha,
                'beta': self.config.beta,
                'asymmetry_param': self.config.asymmetry_param,
                'noise_threshold': self.config.noise_threshold,
                'active_loss_type': self.config.active_loss_type
            }
        }
    
    def reset_noise_estimation(self):
        """Reset noise estimation statistics"""
        self.noise_estimates = []
        self.epoch_count = 0
        self.is_active = False
        
    def adjust_weights(self, alpha: Optional[float] = None, beta: Optional[float] = None):
        """Dynamically adjust loss combination weights"""
        if alpha is not None:
            self.config.alpha = max(0.0, min(1.0, alpha))
        if beta is not None:
            self.config.beta = max(0.0, min(1.0, beta))
            
        # Ensure weights sum to reasonable value
        total = self.config.alpha + self.config.beta
        if total > 1.5:
            self.config.alpha /= total
            self.config.beta /= total
            
        logger.info(f"JAL weights adjusted: α={self.config.alpha:.3f}, β={self.config.beta:.3f}")

# Utility functions for integration
def create_jal_loss(num_classes: int, noise_threshold: float = 0.1,
                    active_type: str = "CE") -> JointAsymmetricLoss:
    """Factory function to create JAL loss with custom configuration"""
    config = JALConfig(
        alpha=0.7,
        beta=0.3,
        active_loss_type=active_type,
        noise_threshold=noise_threshold,
        asymmetry_param=3.0,
        warmup_epochs=5
    )
    return JointAsymmetricLoss(num_classes, config)

def apply_jal_gradient_clipping(model: nn.Module, max_norm: float = 1.0):
    """Apply gradient clipping for stable JAL training"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)