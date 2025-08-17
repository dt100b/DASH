"""Optimal Execution Module with Price Impact and Multiscale Stochastic Volatility
Based on paper 2507.17162v1: "Optimal Trading under Instantaneous and Persistent Price Impact"
"""
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class ExecutionConfig:
    """Configuration for optimal execution module"""
    # Transaction costs
    K: float = 1.0  # Instantaneous transaction cost
    lambda_impact: float = 0.1  # Price impact coefficient  
    beta_decay: float = 0.1  # Impact decay rate
    
    # Risk parameters
    gamma: float = 5.0  # Risk aversion
    rho: float = 0.2  # Discount factor
    
    # Signal dynamics
    kappa: float = 1.0  # Signal mean reversion
    eta: float = 1.0  # Signal volatility
    
    # Multiscale volatility
    epsilon: float = 0.1  # Fast volatility scale
    delta: float = 0.5  # Slow volatility scale
    chi_fast: float = 1.0  # Fast vol mean reversion
    mu_vol: float = 0.2  # Long-term vol level
    
    # Small impact approximation
    theta: float = 0.1  # Small price impact parameter
    
    # Calibration
    recalibrate_days: int = 7  # Weekly recalibration
    half_step_threshold: float = 1.2  # When vol > threshold * mu_vol

class OptimalExecution:
    """Optimal execution with multiscale stochastic volatility and price impact
    
    Key equations from paper 2507.17162v1:
    1. Optimal trading rate: u* = (1/K) * [(λ-Aqq+λAql)q + (Aql+λAll)l + (Aqx+λAxl)x]
    2. Target portfolio: aim = [(Aql+λAll)l + (Aqx+λAxl)x] / [Aqq-λ-λAql]
    3. Fast vol correction: -εγ(y-μ)q/(θK) 
    4. Slow vol correction: √δ(Bq(z)+λBl(z))/K
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.last_calibration = None
        self.calibration_history = []
        
        # Coefficient matrices (from algebraic system)
        self.A_coeffs = self._initialize_coefficients()
        self.B_coeffs = {}  # Slow-scale corrections
        
        # Slippage tracking
        self.realized_slippage = []
        self.expected_slippage = []
        
    def _initialize_coefficients(self) -> Dict[str, float]:
        """Initialize A coefficients from constant volatility solution
        
        Solves the algebraic system (13)-(19) from the paper
        Using small-impact approximation for tractability
        """
        K = self.config.K
        gamma = self.config.gamma
        rho = self.config.rho
        kappa = self.config.kappa
        eta = self.config.eta
        theta = self.config.theta
        lambda_eff = theta * self.config.lambda_impact
        beta_eff = theta * self.config.beta_decay
        
        # Zeroth order (no price impact)
        sigma_base = 0.2  # Base volatility
        Aqq_0 = K/2 * (np.sqrt(rho**2 + 4*gamma*sigma_base**2/K) - rho)
        Aqx_0 = 1 / (kappa + rho + Aqq_0/K)
        Axx_0 = Aqx_0**2 / (K*(2*kappa + rho))
        A0_0 = eta * Axx_0 / (2*rho)
        
        # First order corrections for small price impact
        Aqq_1 = 2*lambda_eff*Aqq_0 / (2*Aqq_0 + K*rho)
        Aql_1 = -beta_eff*K / (rho*K + Aqq_0)
        Aqx_1 = Aqx_0**2 * (lambda_eff - Aqq_1)
        Axl_1 = Aqx_0 * Aql_1 / (K*(kappa + rho))
        All_1 = 0.0  # First order (ensure float)
        Axx_1 = 2*Aqx_0*Aqx_1 / (K*(2*kappa + rho))
        
        return {
            'Aqq': Aqq_0 + Aqq_1,
            'Aql': Aql_1,
            'All': All_1,
            'Aqx': Aqx_0 + Aqx_1,
            'Axl': Axl_1,
            'Axx': Axx_0 + Axx_1,
            'A0': A0_0
        }
    
    def compute_target_exposure(self, bias: float, price_impact: float, 
                              current_vol: float, signal_strength: float = 1.0) -> float:
        """Compute optimal target exposure x* ∝ Bias
        
        Formula: x* = [(Aql + λAll)l + (Aqx + λAxl)x] / [Aqq - λ - λAql]
        
        Args:
            bias: Directional bias from predictions (-1 to 1)
            price_impact: Current accumulated price impact
            current_vol: Current volatility level
            signal_strength: Strength of signal (default 1.0)
            
        Returns:
            Optimal target exposure (fraction of capital)
        """
        A = self.A_coeffs
        lambda_eff = self.config.theta * self.config.lambda_impact
        
        # Numerator: signal contribution
        numerator = (A['Aql'] + lambda_eff * A['All']) * price_impact
        numerator += (A['Aqx'] + lambda_eff * A['Axl']) * bias * signal_strength
        
        # Denominator: tracking coefficient
        denominator = A['Aqq'] - lambda_eff - lambda_eff * A['Aql']
        
        if abs(denominator) < 1e-6:
            return 0.0
            
        target = numerator / denominator
        
        # Apply volatility scaling
        vol_ratio = current_vol / self.config.mu_vol
        if vol_ratio > 1.0:
            # Reduce exposure when vol is high
            target *= 1.0 / np.sqrt(vol_ratio)
            
        # Clip to reasonable bounds
        return np.clip(target, -1.0, 1.0)
    
    def compute_trading_rate(self, current_pos: float, target_pos: float,
                            current_vol: float, vol_factor_y: float,
                            vol_factor_z: float) -> Tuple[float, Dict[str, float]]:
        """Compute optimal trading rate with multiscale SV corrections
        
        Implements the κ step rule from fast/slow SV:
        - Fast correction: -εγ(y-μ)q/(θK)
        - Slow correction: √δ(Bq(z)+λBl(z))/K
        
        Args:
            current_pos: Current position
            target_pos: Target position from compute_target_exposure
            current_vol: Current volatility
            vol_factor_y: Fast volatility factor
            vol_factor_z: Slow volatility factor
            
        Returns:
            (trading_rate, diagnostics)
        """
        K = self.config.K
        A = self.A_coeffs
        lambda_eff = self.config.theta * self.config.lambda_impact
        
        # Base tracking speed rc(σ²)
        tracking_speed = (A['Aqq'] - lambda_eff - lambda_eff * A['Aql']) / K
        
        # Fast volatility correction (κ step rule)
        epsilon = self.config.epsilon
        gamma = self.config.gamma
        chi = self.config.chi_fast
        mu_vol = self.config.mu_vol
        
        fast_correction = -epsilon * gamma * (vol_factor_y - mu_vol) / (chi * K)
        
        # Slow volatility correction 
        delta = self.config.delta
        slow_correction = self._compute_slow_correction(vol_factor_z) * np.sqrt(delta) / K
        
        # Adjusted tracking speed
        adjusted_speed = tracking_speed + fast_correction * abs(current_pos)
        
        # Apply half-step rule when volatility exceeds threshold
        if current_vol > self.config.half_step_threshold * mu_vol:
            adjusted_speed *= 0.5  # Half-step when vol is too high
            
        # Trading rate u* = speed * (target - current)
        position_gap = target_pos - current_pos
        trading_rate = adjusted_speed * position_gap + slow_correction
        
        # Bound trading rate to prevent excessive trading
        max_rate = 2.0 * abs(position_gap) / K  # Max 200% turnover
        trading_rate = np.clip(trading_rate, -max_rate, max_rate)
        
        diagnostics = {
            'tracking_speed': tracking_speed,
            'fast_correction': fast_correction,
            'slow_correction': slow_correction,
            'adjusted_speed': adjusted_speed,
            'position_gap': position_gap,
            'half_stepped': current_vol > self.config.half_step_threshold * mu_vol
        }
        
        return trading_rate, diagnostics
    
    def _compute_slow_correction(self, z: float) -> float:
        """Compute slow-scale volatility correction Bq(z) + λBl(z)
        
        Based on formulas (28) from the paper
        """
        # Simplified computation for demonstration
        # In practice, this would solve the full system
        eta = self.config.eta
        rho2 = 0.3  # Return-vol correlation for slow factor
        
        # Approximate B coefficients
        Bq = -np.sqrt(eta) * rho2 * z * 0.1  # Simplified
        Bl = -np.sqrt(eta) * rho2 * z * 0.05
        
        lambda_eff = self.config.theta * self.config.lambda_impact
        return Bq + lambda_eff * Bl
    
    def update_price_impact(self, current_impact: float, trading_rate: float, 
                           dt: float = 1/252) -> float:
        """Update price impact based on trading
        
        Price impact dynamics: dl = (λu - βl)dt
        
        Args:
            current_impact: Current price impact level
            trading_rate: Current trading rate
            dt: Time step (default daily)
            
        Returns:
            Updated price impact
        """
        lambda_eff = self.config.theta * self.config.lambda_impact
        beta_eff = self.config.theta * self.config.beta_decay
        
        # Impact evolution
        d_impact = (lambda_eff * trading_rate - beta_eff * current_impact) * dt
        
        return current_impact + d_impact
    
    def compute_execution_cost(self, trading_rate: float, price_impact: float) -> float:
        """Compute total execution cost
        
        Cost = K/2 * u² + λ*q*u (instantaneous + persistent)
        
        Args:
            trading_rate: Trading rate u
            price_impact: Current price impact l
            
        Returns:
            Total execution cost
        """
        K = self.config.K
        lambda_eff = self.config.theta * self.config.lambda_impact
        
        instantaneous_cost = K/2 * trading_rate**2
        persistent_cost = lambda_eff * abs(trading_rate) * abs(price_impact)
        
        return instantaneous_cost + persistent_cost
    
    def calibrate_from_slippage(self, realized_trades: List[Dict]) -> Dict[str, float]:
        """Weekly recalibration based on realized slippage
        
        Adjusts λ and β parameters based on observed vs expected slippage
        
        Args:
            realized_trades: List of executed trades with slippage data
            
        Returns:
            Updated parameters
        """
        if not realized_trades:
            return {}
            
        # Calculate realized slippage
        total_slippage = sum(t.get('slippage', 0) for t in realized_trades)
        total_volume = sum(abs(t.get('size', 0)) for t in realized_trades)
        
        if total_volume == 0:
            return {}
            
        avg_slippage = total_slippage / total_volume
        
        # Expected slippage from model
        expected_slippage = self.config.lambda_impact * np.mean([
            abs(t.get('rate', 0)) for t in realized_trades
        ])
        
        # Adjustment factor
        if expected_slippage > 0:
            adjustment = avg_slippage / expected_slippage
            adjustment = np.clip(adjustment, 0.5, 2.0)  # Limit adjustments
            
            # Update parameters
            old_lambda = self.config.lambda_impact
            self.config.lambda_impact *= adjustment
            
            # Track calibration
            self.calibration_history.append({
                'timestamp': datetime.now(),
                'old_lambda': old_lambda,
                'new_lambda': self.config.lambda_impact,
                'adjustment': adjustment,
                'realized_slippage': avg_slippage,
                'expected_slippage': expected_slippage
            })
            
            self.last_calibration = datetime.now()
            
            return {
                'lambda_impact': self.config.lambda_impact,
                'adjustment_factor': adjustment
            }
        
        return {}
    
    def should_recalibrate(self) -> bool:
        """Check if recalibration is due (weekly)"""
        if self.last_calibration is None:
            return True
            
        days_since = (datetime.now() - self.last_calibration).days
        return days_since >= self.config.recalibrate_days
    
    def get_execution_summary(self, current_pos: float, target_pos: float,
                             bias: float, current_vol: float, 
                             vol_y: float, vol_z: float,
                             price_impact: float) -> Dict:
        """Get comprehensive execution summary with all formulas
        
        Returns dict with:
        - x*: Optimal target exposure (x* ∝ Bias)
        - u*: Optimal trading rate
        - κ rules: Fast/slow volatility adjustments
        - Half-step status
        - Expected costs
        """
        # Compute target exposure x* ∝ Bias
        target = self.compute_target_exposure(bias, price_impact, current_vol)
        
        # Compute trading rate with κ step rules
        rate, diagnostics = self.compute_trading_rate(
            current_pos, target, current_vol, vol_y, vol_z
        )
        
        # Compute costs
        exec_cost = self.compute_execution_cost(rate, price_impact)
        
        # Updated impact
        new_impact = self.update_price_impact(price_impact, rate)
        
        return {
            'optimal_target': target,
            'formula_x_star': f"x* = {target:.4f} (∝ Bias={bias:.2f})",
            'trading_rate': rate,
            'kappa_fast_rule': f"κ_fast = -{self.config.epsilon:.2f}γ(y-μ)/(θK) = {diagnostics['fast_correction']:.4f}",
            'kappa_slow_rule': f"κ_slow = √{self.config.delta:.2f}(Bq+λBl)/K = {diagnostics['slow_correction']:.4f}",
            'half_step_active': diagnostics['half_stepped'],
            'half_step_threshold': f"vol={current_vol:.3f} > {self.config.half_step_threshold}*μ={self.config.half_step_threshold*self.config.mu_vol:.3f}",
            'tracking_speed': diagnostics['tracking_speed'],
            'position_gap': diagnostics['position_gap'],
            'execution_cost': exec_cost,
            'new_price_impact': new_impact,
            'needs_recalibration': self.should_recalibrate()
        }
    
    def save_state(self, filepath: str):
        """Save execution module state"""
        state = {
            'config': self.config.__dict__,
            'A_coeffs': self.A_coeffs,
            'B_coeffs': self.B_coeffs,
            'last_calibration': self.last_calibration.isoformat() if self.last_calibration else None,
            'calibration_history': [
                {**c, 'timestamp': c['timestamp'].isoformat()} 
                for c in self.calibration_history[-100:]  # Keep last 100
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str) -> bool:
        """Load execution module state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore config
            for key, value in state['config'].items():
                setattr(self.config, key, value)
            
            # Restore coefficients
            self.A_coeffs = state['A_coeffs']
            self.B_coeffs = state.get('B_coeffs', {})
            
            # Restore calibration
            if state['last_calibration']:
                self.last_calibration = datetime.fromisoformat(state['last_calibration'])
            
            self.calibration_history = [
                {**c, 'timestamp': datetime.fromisoformat(c['timestamp'])}
                for c in state.get('calibration_history', [])
            ]
            
            return True
        except Exception:
            return False