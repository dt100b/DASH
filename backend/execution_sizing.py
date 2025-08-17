"""Impact-aware execution and sizing with multiscale stochastic volatility
Research basis: Small-impact approximation + fast/slow SV asymptotics for optimal execution
"""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ExecutionParams:
    """Execution parameters"""
    target_exposure: float  # Target x*
    current_exposure: float  # Current position
    base_kappa: float  # Base adjustment rate
    impact_epsilon: float  # Price impact parameter
    
class MultiscaleSVSizing:
    """Execution sizing with impact and multiscale stochastic volatility
    
    Research: Extends Gârleanu-Pedersen to realistic frictions using
    small-impact approximation and multiscale SV corrections for better PnL.
    """
    
    def __init__(self, 
                 base_kappa: float = 0.1,
                 impact_epsilon: float = 0.001):
        """Initialize sizing module
        
        Args:
            base_kappa: Base adjustment rate toward target
            impact_epsilon: Linear price impact parameter
        """
        self.base_kappa = base_kappa
        self.impact_epsilon = impact_epsilon
        
        # Multiscale vol parameters (fast and slow components)
        self.fast_vol_window = 24  # Hours
        self.slow_vol_window = 168  # 7 days
        
    def compute_target_exposure(self, bias: float, 
                               risk_scaling: float = 1.0) -> float:
        """Compute target exposure x* from bias
        
        Args:
            bias: Model bias ∈ [-1, +1]
            risk_scaling: Risk scaling factor (e.g., 0.5 for half size)
            
        Returns:
            Target exposure (normalized)
        """
        # Simple linear mapping with risk scaling
        target = bias * risk_scaling
        
        # Clip to position limits
        target = np.clip(target, -1.0, 1.0)
        
        return target
    
    def compute_adjustment_rate(self, 
                               fast_vol: float,
                               slow_vol: float,
                               vol_regime: str = "normal") -> float:
        """Compute adjustment rate κ using multiscale SV
        
        Research: κ adapts based on fast/slow volatility components.
        Slow down in rough/fast regimes, speed up in calm regimes.
        
        Args:
            fast_vol: Fast volatility component (24h)
            slow_vol: Slow volatility component (7d)
            vol_regime: Volatility regime ("calm", "normal", "rough")
            
        Returns:
            Adjusted κ for execution
        """
        # Vol ratio indicates regime
        if slow_vol > 0:
            vol_ratio = fast_vol / slow_vol
        else:
            vol_ratio = 1.0
        
        # Roughness indicator (vol of vol)
        roughness = abs(vol_ratio - 1.0)
        
        # Base adjustments
        if vol_regime == "rough" or roughness > 0.5:
            # High roughness: slow down execution
            kappa_mult = 0.3
        elif vol_regime == "calm" or roughness < 0.2:
            # Low roughness: can execute faster
            kappa_mult = 1.5
        else:
            # Normal regime
            kappa_mult = 1.0
        
        # Additional adjustment for extreme vol levels
        if fast_vol > slow_vol * 1.5:
            # Fast vol spike: slow down more
            kappa_mult *= 0.5
        elif fast_vol < slow_vol * 0.5:
            # Unusually calm: can be more aggressive
            kappa_mult *= 1.2
        
        # Apply multiplier to base kappa
        adjusted_kappa = self.base_kappa * kappa_mult
        
        # Ensure reasonable bounds
        adjusted_kappa = np.clip(adjusted_kappa, 0.01, 0.5)
        
        return adjusted_kappa
    
    def compute_execution_step(self,
                              target: float,
                              current: float,
                              kappa: float,
                              apply_impact: bool = True) -> float:
        """Compute single execution step toward target
        
        Args:
            target: Target exposure x*
            current: Current exposure
            kappa: Adjustment rate
            apply_impact: Whether to apply impact correction
            
        Returns:
            Next position after execution step
        """
        # Desired change
        desired_change = target - current
        
        # Step without impact
        step = kappa * desired_change
        
        if apply_impact and abs(step) > 0:
            # Small-impact approximation correction
            # Cost ≈ ε * step^2 → optimal step reduced by factor
            impact_factor = 1.0 / (1.0 + self.impact_epsilon * abs(step))
            step *= impact_factor
        
        # New position
        next_position = current + step
        
        # Ensure within bounds
        next_position = np.clip(next_position, -1.0, 1.0)
        
        return next_position
    
    def estimate_execution_cost(self, 
                               current: float,
                               target: float,
                               avg_vol: float) -> float:
        """Estimate total execution cost
        
        Args:
            current: Current exposure
            target: Target exposure
            avg_vol: Average volatility
            
        Returns:
            Estimated cost as fraction of position value
        """
        trade_size = abs(target - current)
        
        # Linear impact cost
        impact_cost = 0.5 * self.impact_epsilon * trade_size ** 2
        
        # Add vol-dependent slippage
        slippage = 0.0001 * trade_size * (1 + avg_vol / 0.2)  # Increases with vol
        
        total_cost = impact_cost + slippage
        
        return total_cost
    
    def apply_guards(self, 
                    target_exposure: float,
                    funding_guard: bool,
                    vol_drift_guard: bool) -> float:
        """Apply risk guards to target exposure
        
        Args:
            target_exposure: Original target
            funding_guard: Whether funding guard is active
            vol_drift_guard: Whether vol drift guard is active
            
        Returns:
            Adjusted target exposure
        """
        adjusted = target_exposure
        
        if funding_guard:
            # Halve size if funding too high
            adjusted *= 0.5
            
        if vol_drift_guard:
            # Halve size if vol has drifted significantly
            adjusted *= 0.5
        
        return adjusted
    
    def get_execution_plan(self,
                          bias: float,
                          current_exposure: float,
                          market_conditions: Dict[str, float],
                          risk_guards: Dict[str, bool]) -> Dict[str, float]:
        """Get complete execution plan
        
        Args:
            bias: Model bias
            current_exposure: Current position
            market_conditions: Dict with 'fast_vol', 'slow_vol', 'vol_regime'
            risk_guards: Dict with 'funding_guard', 'vol_drift_guard'
            
        Returns:
            Execution plan dictionary
        """
        # Compute target
        target = self.compute_target_exposure(bias)
        
        # Apply guards
        target = self.apply_guards(
            target,
            risk_guards.get('funding_guard', False),
            risk_guards.get('vol_drift_guard', False)
        )
        
        # Compute adjustment rate
        kappa = self.compute_adjustment_rate(
            market_conditions.get('fast_vol', 0.2),
            market_conditions.get('slow_vol', 0.2),
            market_conditions.get('vol_regime', 'normal')
        )
        
        # Compute next step
        next_exposure = self.compute_execution_step(
            target, current_exposure, kappa
        )
        
        # Estimate costs
        execution_cost = self.estimate_execution_cost(
            current_exposure, next_exposure,
            market_conditions.get('fast_vol', 0.2)
        )
        
        return {
            'target_exposure': target,
            'current_exposure': current_exposure,
            'next_exposure': next_exposure,
            'step_size': next_exposure - current_exposure,
            'kappa': kappa,
            'execution_cost': execution_cost,
            'funding_guard_active': risk_guards.get('funding_guard', False),
            'vol_drift_guard_active': risk_guards.get('vol_drift_guard', False)
        }