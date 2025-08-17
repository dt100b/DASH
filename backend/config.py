"""Configuration for the quantitative trading system
Centralized configuration with feature flags for all modules
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TradingConfig:
    """Main configuration for trading system"""
    
    # Exchange and data
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    lookback_hours: int = 72
    
    # Risk management
    max_position_size: float = 0.1  # Max 10% of capital
    stop_loss: float = 0.02  # 2% stop loss
    daily_loss_limit: float = 0.05  # 5% daily limit
    max_leverage: float = 3.0
    
    # Module-specific configs
    use_signature_features: bool = True
    use_orcsmc_regime: bool = True
    use_deep_stopping: bool = True
    use_optimal_execution: bool = True
    use_bema_optimizer: bool = True
    use_jal_loss: bool = True  # Enable JAL for noise-robust training
    
    # JAL (Joint Asymmetric Loss) configuration
    jal_noise_threshold: float = 0.15  # Enable JAL when noise > 15%
    jal_alpha: float = 0.7  # Weight for active loss
    jal_beta: float = 0.3  # Weight for passive loss
    jal_asymmetry_param: float = 3.0  # AMSE asymmetry strength
    jal_warmup_epochs: int = 5  # Warmup before JAL activation
    jal_active_loss: str = "CE"  # CE or FL (Focal Loss)
    jal_monitor_metrics: bool = True  # Track noise metrics
    
    # Large LR scheduler settings (Paper 2507.17748v2)
    use_large_lr: bool = True
    large_lr_base: float = 0.3  # Paper's recommended for robustness
    large_lr_min: float = 1e-5
    large_lr_warmup_epochs: int = 5
    large_lr_warmup_start: float = 0.01
    large_lr_decay_type: str = "cosine"  # "cosine", "step", "exponential"
    large_lr_weight_decay: float = 5e-4
    large_lr_gradient_clip: float = 1.0
    large_lr_track_sparsity: bool = True
    large_lr_track_confidence: bool = True
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Execution
    execution_slippage: float = 0.001  # 0.1% slippage
    execution_delay_ms: int = 100
    order_timeout_seconds: int = 30
    
    # Monitoring
    log_level: str = "INFO"
    save_models: bool = True
    model_save_path: str = "./models"
    checkpoint_interval: int = 10  # Save every N epochs
    
    # Walk-forward validation
    walk_forward_window: int = 365  # Days for training
    walk_forward_step: int = 30  # Days for testing
    min_sharpe_ratio: float = 1.5  # Minimum acceptable Sharpe
    
    # API Keys (to be loaded from environment)
    api_key: str = ""
    api_secret: str = ""

@dataclass 
class SignatureConfig:
    """Configuration for signature features module"""
    order: int = 3  # Truncation order
    max_dimension: int = 32  # Maximum feature dimension
    use_time_augmentation: bool = True
    normalize: bool = True

@dataclass
class ORCSMCConfig:
    """Configuration for ORCSMC regime detection"""
    num_particles: int = 1000
    window_length: int = 30
    learning_iterations: int = 5
    ess_threshold: float = 0.5
    gate_threshold: float = 0.6  # P(RISK-ON) threshold

@dataclass
class DeepStoppingConfig:
    """Configuration for deep optimal stopping"""
    max_holding_days: int = 7
    monte_carlo_paths: int = 1000
    hidden_size: int = 16  # Tiny MLPs as per paper
    use_bema: bool = True  # Use BEMA optimizer
    use_jal: bool = True  # Use JAL loss for noise robustness

@dataclass
class ExecutionConfig:
    """Configuration for optimal execution"""
    impact_lambda: float = 0.01
    decay_beta: float = 0.1
    kappa_fast: float = 1.0
    kappa_slow: float = 0.5
    vol_threshold: float = 1.2  # Half-step when vol > 1.2μ

@dataclass
class BEMAConfig:
    """Configuration for BEMA optimizer"""
    ema_power: float = 0.5  # κ parameter
    bias_power: float = 0.4  # η parameter
    update_freq: int = 400
    snapshot_interval: int = 5000
    restore_threshold: float = 0.1

# Global configuration instance
TRADING_CONFIG = TradingConfig()
SIGNATURE_CONFIG = SignatureConfig()
ORCSMC_CONFIG = ORCSMCConfig()
STOPPING_CONFIG = DeepStoppingConfig()
EXECUTION_CONFIG = ExecutionConfig()
BEMA_CONFIG = BEMAConfig()

def load_config_from_env():
    """Load configuration from environment variables"""
    import os
    
    # Override with environment variables if present
    TRADING_CONFIG.api_key = os.getenv("TRADING_API_KEY", "")
    TRADING_CONFIG.api_secret = os.getenv("TRADING_API_SECRET", "")
    TRADING_CONFIG.exchange = os.getenv("EXCHANGE", TRADING_CONFIG.exchange)
    TRADING_CONFIG.symbol = os.getenv("SYMBOL", TRADING_CONFIG.symbol)
    
    # JAL configuration from environment
    if os.getenv("JAL_ENABLED"):
        TRADING_CONFIG.use_jal_loss = os.getenv("JAL_ENABLED", "true").lower() == "true"
    if os.getenv("JAL_NOISE_THRESHOLD"):
        TRADING_CONFIG.jal_noise_threshold = float(os.getenv("JAL_NOISE_THRESHOLD"))
    if os.getenv("JAL_ACTIVE_LOSS"):
        TRADING_CONFIG.jal_active_loss = os.getenv("JAL_ACTIVE_LOSS")
        
    return TRADING_CONFIG

def get_jal_config():
    """Get JAL configuration for noise-robust training"""
    from jal_loss import JALConfig
    
    return JALConfig(
        alpha=TRADING_CONFIG.jal_alpha,
        beta=TRADING_CONFIG.jal_beta,
        active_loss_type=TRADING_CONFIG.jal_active_loss,
        asymmetry_param=TRADING_CONFIG.jal_asymmetry_param,
        noise_threshold=TRADING_CONFIG.jal_noise_threshold,
        warmup_epochs=TRADING_CONFIG.jal_warmup_epochs,
        track_metrics=TRADING_CONFIG.jal_monitor_metrics
    )

def should_use_jal(estimated_noise: float = 0.0, epoch: int = 0) -> bool:
    """Determine if JAL should be activated based on noise level"""
    if not TRADING_CONFIG.use_jal_loss:
        return False
        
    # Check warmup period
    if epoch < TRADING_CONFIG.jal_warmup_epochs:
        return False
        
    # Check noise threshold
    return estimated_noise > TRADING_CONFIG.jal_noise_threshold