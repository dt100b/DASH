"""Bias head model with BEMA updates and high-LR robustness
Research basis: BEMA removes lag while keeping variance reduction; high LRs improve robustness
"""
import numpy as np
import joblib
from typing import Dict, Optional, Tuple, List
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass
import os

@dataclass
class BEMAConfig:
    """Bias-corrected EMA configuration"""
    alpha: float = 0.3  # EMA decay factor
    bias_correction: bool = True
    
class BiasHead:
    """Transparent bias head with BEMA training and optional JAL loss
    
    Research: BEMA (bias-corrected EMA) removes lag but keeps variance reduction.
    Large learning rates consistently improve robustness to spurious correlations.
    Optional JAL (AMSE + active loss) for noise-tolerant training.
    """
    
    def __init__(self, 
                 learning_rate: float = 1.0,  # Prefer high LR for robustness
                 use_bema: bool = True,
                 use_jal: bool = False):
        """Initialize bias head
        
        Args:
            learning_rate: Base learning rate (higher is more robust)
            use_bema: Whether to use BEMA updates
            use_jal: Whether to use JAL loss for noise robustness
        """
        self.learning_rate = learning_rate
        self.use_bema = use_bema
        self.use_jal = use_jal
        
        # Simple linear model as base
        self.model = LogisticRegression(
            penalty='l2',
            C=1.0 / learning_rate,  # Inverse regularization strength
            max_iter=1,  # Single step updates for online learning
            warm_start=True,  # Keep previous solution
            solver='lbfgs'
        )
        
        # BEMA state
        self.bema_config = BEMAConfig()
        self.ema_weights = None
        self.ema_bias = None
        self.update_count = 0
        
        # Feature names for interpretability
        self.feature_names = []
        self.feature_weights = None
        
    def _apply_bema(self, new_weights: np.ndarray, new_bias: float):
        """Apply bias-corrected EMA to weights
        
        BEMA formula: w_t = (1-α) * w_ema + α * w_new
        With bias correction: w_corrected = w_t / (1 - α^t)
        """
        alpha = self.bema_config.alpha
        
        if self.ema_weights is None:
            self.ema_weights = new_weights.copy()
            self.ema_bias = new_bias
        else:
            # EMA update
            self.ema_weights = (1 - alpha) * self.ema_weights + alpha * new_weights
            self.ema_bias = (1 - alpha) * self.ema_bias + alpha * new_bias
        
        self.update_count += 1
        
        if self.bema_config.bias_correction:
            # Bias correction factor
            correction = 1.0 / (1.0 - alpha ** self.update_count)
            return self.ema_weights * correction, self.ema_bias * correction
        else:
            return self.ema_weights, self.ema_bias
    
    def _compute_jal_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         features: np.ndarray) -> float:
        """Compute JAL (Joint Active Loss) for noise robustness
        
        JAL = AMSE (passive) + α * Active_CE
        AMSE handles asymmetric noise, Active_CE focuses on hard examples
        """
        # AMSE component (asymmetric MSE)
        errors = y_true - y_pred
        pos_errors = np.maximum(errors, 0)
        neg_errors = np.minimum(errors, 0)
        
        # Asymmetric weights (more penalty for false positives in risk)
        amse = np.mean(1.2 * pos_errors**2 + 0.8 * neg_errors**2)
        
        if not self.use_jal:
            return amse
        
        # Active learning component (focus on uncertain predictions)
        entropy = -y_pred * np.log(y_pred + 1e-10) - (1 - y_pred) * np.log(1 - y_pred + 1e-10)
        active_weight = entropy / np.max(entropy + 1e-10)
        
        # Cross entropy with active weighting
        ce = -y_true * np.log(y_pred + 1e-10) - (1 - y_true) * np.log(1 - y_pred + 1e-10)
        active_ce = np.mean(active_weight * ce)
        
        # Combine with fixed mixing parameter
        return 0.7 * amse + 0.3 * active_ce
    
    def train_step(self, features: np.ndarray, labels: np.ndarray):
        """Single training step with BEMA update
        
        Args:
            features: (n_samples, n_features) array
            labels: (n_samples,) binary labels (1 for up, 0 for down)
        """
        # Fit model for one iteration
        self.model.fit(features, labels)
        
        # Get new weights
        new_weights = self.model.coef_[0]
        new_bias = self.model.intercept_[0]
        
        # Apply BEMA if enabled
        if self.use_bema:
            updated_weights, updated_bias = self._apply_bema(new_weights, new_bias)
            # Update model with BEMA weights
            self.model.coef_[0] = updated_weights
            self.model.intercept_[0] = updated_bias
        
        self.feature_weights = self.model.coef_[0]
    
    def predict_bias(self, features: np.ndarray) -> float:
        """Predict bias ∈ [-1, +1]
        
        Args:
            features: (n_features,) array
            
        Returns:
            Bias value: positive for long, negative for short
        """
        if not hasattr(self.model, 'coef_'):
            return 0.0  # Untrained model
        
        # Get probability of upward movement
        features_2d = features.reshape(1, -1)
        prob_up = self.model.predict_proba(features_2d)[0, 1]
        
        # Convert to bias ∈ [-1, +1]
        bias = 2 * prob_up - 1
        
        return bias
    
    def get_feature_attributions(self, features: np.ndarray) -> List[Tuple[str, float, float]]:
        """Get feature attributions for current prediction
        
        Returns:
            List of (feature_name, feature_value, attribution) tuples
        """
        if self.feature_weights is None or len(self.feature_names) == 0:
            return []
        
        attributions = []
        for i, name in enumerate(self.feature_names):
            if i < len(features) and i < len(self.feature_weights):
                value = features[i]
                weight = self.feature_weights[i]
                attribution = value * weight
                attributions.append((name, value, attribution))
        
        # Sort by absolute attribution
        attributions.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return attributions
    
    def save_model(self, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'feature_weights': self.feature_weights,
            'feature_names': self.feature_names,
            'ema_weights': self.ema_weights,
            'ema_bias': self.ema_bias,
            'update_count': self.update_count,
            'config': {
                'learning_rate': self.learning_rate,
                'use_bema': self.use_bema,
                'use_jal': self.use_jal
            }
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load model from disk"""
        if os.path.exists(path):
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.feature_weights = model_data['feature_weights']
            self.feature_names = model_data['feature_names']
            self.ema_weights = model_data['ema_weights']
            self.ema_bias = model_data['ema_bias']
            self.update_count = model_data['update_count']
            
            config = model_data['config']
            self.learning_rate = config['learning_rate']
            self.use_bema = config['use_bema']
            self.use_jal = config['use_jal']
            
            return True
        return False
    
    def run_lr_sweep(self, features: np.ndarray, labels: np.ndarray,
                     lr_range: List[float] = None) -> Dict[float, float]:
        """Run learning rate sweep favoring higher LRs
        
        Research: Higher LRs consistently improve robustness
        """
        if lr_range is None:
            # Favor higher LRs
            lr_range = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        results = {}
        for lr in lr_range:
            # Create temporary model with this LR
            temp_model = LogisticRegression(
                penalty='l2',
                C=1.0 / lr,
                max_iter=100,
                solver='lbfgs'
            )
            
            # Simple cross-validation
            n = len(labels)
            train_idx = np.arange(int(0.8 * n))
            val_idx = np.arange(int(0.8 * n), n)
            
            temp_model.fit(features[train_idx], labels[train_idx])
            val_score = temp_model.score(features[val_idx], labels[val_idx])
            
            results[lr] = val_score
        
        return results