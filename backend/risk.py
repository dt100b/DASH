"""Risk management module - stops, limits, and guards"""
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    """Risk metrics for position"""
    position_pnl: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_vol: float = 0.0
    vol_since_entry: float = 0.0
    funding_cost: float = 0.0
    
class RiskManager:
    """Centralized risk management with strict limits"""
    
    def __init__(self,
                 stop_loss_pct: float = 0.05,
                 max_daily_loss_pct: float = 0.10,
                 vol_drift_threshold: float = 0.30,
                 funding_percentile_limit: float = 95.0):
        """Initialize risk manager
        
        Args:
            stop_loss_pct: Per-trade stop loss percentage
            max_daily_loss_pct: Maximum daily loss percentage
            vol_drift_threshold: Vol increase threshold for guard
            funding_percentile_limit: Funding percentile for guard
        """
        self.stop_loss_pct = stop_loss_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.vol_drift_threshold = vol_drift_threshold
        self.funding_percentile_limit = funding_percentile_limit
        
        # Track daily P&L
        self.daily_pnl_history = {}
        self.position_entry_vol = {}
        
    def check_stop_loss(self, 
                       entry_price: float,
                       current_price: float,
                       direction: str) -> Tuple[bool, float]:
        """Check if stop loss is hit
        
        Args:
            entry_price: Position entry price
            current_price: Current market price
            direction: "LONG" or "SHORT"
            
        Returns:
            (should_stop, pnl_pct)
        """
        if direction == "LONG":
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price
            
        should_stop = pnl_pct <= -self.stop_loss_pct
        
        return should_stop, pnl_pct
    
    def check_daily_loss_limit(self, 
                              symbol: str,
                              current_pnl: float) -> Tuple[bool, float]:
        """Check if daily loss limit is exceeded
        
        Args:
            symbol: Trading symbol
            current_pnl: Current unrealized P&L
            
        Returns:
            (limit_exceeded, total_daily_loss)
        """
        today = datetime.utcnow().date()
        
        # Get today's realized P&L
        daily_key = f"{symbol}_{today}"
        realized_pnl = self.daily_pnl_history.get(daily_key, 0.0)
        
        # Total daily loss (realized + unrealized)
        total_daily_loss = realized_pnl + min(current_pnl, 0)
        
        # Check against limit (as fraction of account)
        limit_exceeded = total_daily_loss <= -self.max_daily_loss_pct
        
        return limit_exceeded, total_daily_loss
    
    def update_daily_pnl(self, symbol: str, realized_pnl: float):
        """Update daily P&L tracking
        
        Args:
            symbol: Trading symbol
            realized_pnl: Realized P&L from closed position
        """
        today = datetime.utcnow().date()
        daily_key = f"{symbol}_{today}"
        
        if daily_key not in self.daily_pnl_history:
            self.daily_pnl_history[daily_key] = 0.0
            
        self.daily_pnl_history[daily_key] += realized_pnl
        
        # Clean up old entries (keep 30 days)
        cutoff_date = today - timedelta(days=30)
        keys_to_remove = [
            k for k in self.daily_pnl_history.keys()
            if k.split('_')[1] < str(cutoff_date)
        ]
        for k in keys_to_remove:
            del self.daily_pnl_history[k]
    
    def check_vol_drift_guard(self,
                             symbol: str,
                             entry_vol: float,
                             current_vol: float) -> bool:
        """Check if volatility has drifted significantly
        
        Args:
            symbol: Trading symbol
            entry_vol: Volatility at position entry
            current_vol: Current volatility
            
        Returns:
            True if guard should be active (vol increased > 30%)
        """
        if entry_vol <= 0:
            return False
            
        vol_increase = (current_vol - entry_vol) / entry_vol
        
        return vol_increase > self.vol_drift_threshold
    
    def check_funding_guard(self, funding_percentile: float) -> bool:
        """Check if funding guard should be active
        
        Args:
            funding_percentile: Current funding rate percentile
            
        Returns:
            True if in top 5% of historical distribution
        """
        return funding_percentile >= self.funding_percentile_limit
    
    def evaluate_risk_state(self,
                          position: Dict[str, float],
                          market_data: Dict[str, float]) -> Dict[str, any]:
        """Comprehensive risk evaluation
        
        Args:
            position: Position details (entry_price, current_price, direction, etc.)
            market_data: Market data (current_vol, funding_percentile, etc.)
            
        Returns:
            Risk evaluation results
        """
        results = {
            'should_stop': False,
            'stop_reason': None,
            'risk_flags': [],
            'metrics': RiskMetrics()
        }
        
        # Check stop loss
        if 'entry_price' in position and 'current_price' in position:
            stop_hit, pnl_pct = self.check_stop_loss(
                position['entry_price'],
                position['current_price'],
                position.get('direction', 'LONG')
            )
            
            if stop_hit:
                results['should_stop'] = True
                results['stop_reason'] = f"Stop loss hit: {pnl_pct:.2%}"
                results['risk_flags'].append('STOP_LOSS')
            
            results['metrics'].position_pnl = pnl_pct
        
        # Check daily loss limit
        if 'symbol' in position:
            limit_exceeded, daily_loss = self.check_daily_loss_limit(
                position['symbol'],
                position.get('unrealized_pnl', 0)
            )
            
            if limit_exceeded:
                results['should_stop'] = True
                results['stop_reason'] = f"Daily loss limit: {daily_loss:.2%}"
                results['risk_flags'].append('DAILY_LIMIT')
            
            results['metrics'].daily_pnl = daily_loss
        
        # Check vol drift guard
        if 'entry_vol' in position and 'current_vol' in market_data:
            vol_drift_active = self.check_vol_drift_guard(
                position.get('symbol', ''),
                position['entry_vol'],
                market_data['current_vol']
            )
            
            if vol_drift_active:
                results['risk_flags'].append('VOL_DRIFT')
            
            results['metrics'].current_vol = market_data['current_vol']
            results['metrics'].vol_since_entry = position.get('entry_vol', 0)
        
        # Check funding guard
        if 'funding_percentile' in market_data:
            funding_guard_active = self.check_funding_guard(
                market_data['funding_percentile']
            )
            
            if funding_guard_active:
                results['risk_flags'].append('HIGH_FUNDING')
            
            results['metrics'].funding_cost = market_data.get('funding_rate', 0)
        
        # Calculate max drawdown
        if 'high_water_mark' in position and 'current_value' in position:
            drawdown = (position['high_water_mark'] - position['current_value']) / position['high_water_mark']
            results['metrics'].max_drawdown = max(drawdown, 0)
            
            if drawdown > 0.10:  # 10% drawdown warning
                results['risk_flags'].append('HIGH_DRAWDOWN')
        
        return results
    
    def get_risk_summary(self) -> Dict[str, any]:
        """Get summary of current risk state"""
        today = datetime.utcnow().date()
        
        # Today's P&L across all symbols
        today_pnl = sum(
            v for k, v in self.daily_pnl_history.items()
            if k.endswith(str(today))
        )
        
        # 30-day statistics
        all_daily_pnls = list(self.daily_pnl_history.values())
        
        if all_daily_pnls:
            avg_daily_pnl = np.mean(all_daily_pnls)
            worst_day = min(all_daily_pnls)
            best_day = max(all_daily_pnls)
            win_rate = sum(1 for p in all_daily_pnls if p > 0) / len(all_daily_pnls)
        else:
            avg_daily_pnl = worst_day = best_day = win_rate = 0
        
        return {
            'today_pnl': today_pnl,
            'today_pnl_pct': today_pnl * 100,  # As percentage
            'daily_limit_remaining': self.max_daily_loss_pct + today_pnl,
            'avg_daily_pnl': avg_daily_pnl,
            'worst_day': worst_day,
            'best_day': best_day,
            'win_rate': win_rate,
            'active_limits': {
                'stop_loss': self.stop_loss_pct,
                'daily_loss': self.max_daily_loss_pct,
                'vol_drift': self.vol_drift_threshold,
                'funding_percentile': self.funding_percentile_limit
            }
        }