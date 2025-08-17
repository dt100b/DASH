"""Walk-forward validation with strict acceptance criteria"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from config import config
from data_io import get_data_adapter
from features_signature import SignatureFeatures
from features_positioning import PositioningFeatures
from regime_orcsmc import ORCSMC
from head_bias import BiasHead
from stopping import DeepOptimalStopping
from execution_sizing import MultiscaleSVSizing
from risk import RiskManager

@dataclass
class ValidationSlice:
    """Single validation window results"""
    period_start: str
    period_end: str
    n_trades: int = 0
    n_wins: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    gate_off_days: int = 0
    total_days: int = 0
    hit_rate: float = 0.0
    passed: bool = False
    
class WalkForwardValidator:
    """Rolling walk-forward validation with embargo"""
    
    def __init__(self,
                 train_window_days: int = 365,
                 test_window_days: int = 30,
                 embargo_days: int = 1):
        """Initialize validator
        
        Args:
            train_window_days: Training window size
            test_window_days: Test window size
            embargo_days: Embargo period between train/test
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.embargo_days = embargo_days
        
        # Initialize components
        self.data_adapter = get_data_adapter("csv")
        self.sig_features = SignatureFeatures()
        self.pos_features = PositioningFeatures(self.data_adapter)
        self.orcsmc = ORCSMC()
        self.bias_head = BiasHead()
        self.stopping = DeepOptimalStopping()
        self.sizing = MultiscaleSVSizing()
        self.risk_mgr = RiskManager()
        
        # Acceptance criteria
        self.min_hit_rate = config.MIN_HIT_RATE
        self.min_hit_rate_with_payoff = config.MIN_HIT_RATE_WITH_PAYOFF
        self.min_payoff_ratio = config.MIN_PAYOFF_RATIO
        self.min_sharpe = config.MIN_SHARPE
        self.max_drawdown = config.MAX_DRAWDOWN
        self.max_gate_off_rate = config.MAX_GATE_OFF_RATE
        
    def _apply_fees_and_slippage(self, 
                                pnl: float,
                                trade_size: float,
                                is_maker: bool = False) -> float:
        """Apply realistic fees and slippage
        
        Args:
            pnl: Raw P&L
            trade_size: Size of trade
            is_maker: Whether maker or taker fee applies
            
        Returns:
            P&L after fees and slippage
        """
        # Fee based on order type
        fee_rate = config.MAKER_FEE if is_maker else config.TAKER_FEE
        fee_cost = abs(trade_size) * fee_rate
        
        # Slippage (increases with size)
        slippage_rate = 0.0001 * (1 + abs(trade_size))  # 1-2 bps
        slippage_cost = abs(trade_size) * slippage_rate
        
        # Apply costs
        net_pnl = pnl - fee_cost - slippage_cost
        
        return net_pnl
    
    def _apply_funding_cost(self,
                           position_size: float,
                           holding_hours: int,
                           funding_rate: float) -> float:
        """Calculate funding cost for position
        
        Args:
            position_size: Size of position
            holding_hours: Hours held
            funding_rate: Funding rate (8h periods)
            
        Returns:
            Total funding cost (negative for cost)
        """
        # Funding charged every 8 hours
        funding_periods = holding_hours / 8
        total_funding = abs(position_size) * funding_rate * funding_periods
        
        return -total_funding  # Negative because it's a cost
    
    def _simulate_trading_day(self,
                             symbol: str,
                             date: datetime,
                             position: Optional[Dict] = None) -> Tuple[Dict, Optional[Dict]]:
        """Simulate single trading day
        
        Args:
            symbol: Trading symbol
            date: Current date
            position: Current position (if any)
            
        Returns:
            (day_results, updated_position)
        """
        results = {
            'date': date.isoformat(),
            'gate_prob': 0.0,
            'bias': 0.0,
            'decision': 'FLAT',
            'pnl': 0.0,
            'trade_executed': False
        }
        
        # Get market data
        bars = self.data_adapter.get_hourly_bars(symbol, 72)
        if len(bars) < 24:
            return results, position
        
        # Extract features
        sig_feats = self.sig_features.extract_features(bars)
        pos_feats = self.pos_features.extract_features(symbol)
        
        # Combine features
        all_features = {**sig_feats, **pos_feats}
        feature_array = np.array(list(all_features.values()))
        
        # Get signal from last bar
        last_return = np.log(bars['close'].iloc[-1] / bars['close'].iloc[-2])
        
        # ORCSMC regime check
        gate_prob = self.orcsmc.step(last_return)
        results['gate_prob'] = gate_prob
        
        # Check gate threshold
        if gate_prob < config.RISK_ON_THRESHOLD:
            results['decision'] = 'GATE_OFF'
            # Close position if gate off
            if position:
                close_price = bars['close'].iloc[-1]
                if position['direction'] == 'LONG':
                    raw_pnl = (close_price - position['entry_price']) / position['entry_price']
                else:
                    raw_pnl = (position['entry_price'] - close_price) / position['entry_price']
                
                # Apply costs
                results['pnl'] = self._apply_fees_and_slippage(
                    raw_pnl * position['size'],
                    position['size']
                )
                results['pnl'] += self._apply_funding_cost(
                    position['size'],
                    position['holding_hours'],
                    self.data_adapter.get_funding_rate(symbol, 1)
                )
                position = None
            return results, position
        
        # Get bias
        bias = self.bias_head.predict_bias(feature_array)
        results['bias'] = bias
        
        # Position management
        if position:
            # Check stop/hold decision
            position['holding_days'] += 1
            position['holding_hours'] += 24
            
            position_features = {
                'unrealized_pnl': 0.0,  # Would calculate from prices
                'current_vol': sig_feats['realized_vol_24h'],
                'regime_prob': gate_prob
            }
            
            stop_decision, _, _ = self.stopping.decide(
                position_features,
                position['holding_days']
            )
            
            if stop_decision == "STOP" or position['holding_days'] >= config.MAX_HOLDING_DAYS:
                # Close position
                close_price = bars['close'].iloc[-1]
                if position['direction'] == 'LONG':
                    raw_pnl = (close_price - position['entry_price']) / position['entry_price']
                else:
                    raw_pnl = (position['entry_price'] - close_price) / position['entry_price']
                
                # Apply costs
                results['pnl'] = self._apply_fees_and_slippage(
                    raw_pnl * position['size'],
                    position['size']
                )
                results['pnl'] += self._apply_funding_cost(
                    position['size'],
                    position['holding_hours'],
                    self.data_adapter.get_funding_rate(symbol, 1)
                )
                
                results['trade_executed'] = True
                position = None
        
        # Open new position if no position and bias strong enough
        if not position:
            if bias >= config.BIAS_LONG_THRESHOLD:
                direction = 'LONG'
                results['decision'] = 'LONG'
            elif bias <= config.BIAS_SHORT_THRESHOLD:
                direction = 'SHORT'
                results['decision'] = 'SHORT'
            else:
                return results, position
            
            # Get sizing
            execution_plan = self.sizing.get_execution_plan(
                bias,
                0.0,  # No current position
                {
                    'fast_vol': sig_feats['realized_vol_24h'],
                    'slow_vol': sig_feats['realized_vol_72h'],
                    'vol_regime': 'normal'
                },
                {
                    'funding_guard': pos_feats['funding_percentile'] >= 95,
                    'vol_drift_guard': False
                }
            )
            
            position = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': bars['close'].iloc[-1],
                'size': abs(execution_plan['target_exposure']),
                'holding_days': 0,
                'holding_hours': 0,
                'entry_vol': sig_feats['realized_vol_24h']
            }
            
            results['trade_executed'] = True
        
        return results, position
    
    def validate_slice(self, 
                      symbol: str,
                      train_start: datetime,
                      test_start: datetime,
                      test_end: datetime) -> ValidationSlice:
        """Validate single time slice
        
        Args:
            symbol: Trading symbol
            train_start: Training period start
            test_start: Test period start (after embargo)
            test_end: Test period end
            
        Returns:
            Validation results for slice
        """
        # Train models on training period
        # (In production, would actually train here)
        
        # Run test period simulation
        current_date = test_start
        position = None
        daily_pnls = []
        trades = []
        gate_off_count = 0
        
        while current_date <= test_end:
            day_results, position = self._simulate_trading_day(
                symbol, current_date, position
            )
            
            daily_pnls.append(day_results['pnl'])
            
            if day_results['decision'] == 'GATE_OFF':
                gate_off_count += 1
            
            if day_results['trade_executed'] and day_results['pnl'] != 0:
                trades.append(day_results['pnl'])
            
            current_date += timedelta(days=1)
        
        # Calculate metrics
        slice_result = ValidationSlice(
            period_start=test_start.isoformat(),
            period_end=test_end.isoformat(),
            total_days=(test_end - test_start).days + 1
        )
        
        if trades:
            slice_result.n_trades = len(trades)
            slice_result.n_wins = sum(1 for t in trades if t > 0)
            slice_result.hit_rate = slice_result.n_wins / slice_result.n_trades
            slice_result.total_pnl = sum(trades)
            
            # Sharpe ratio (annualized)
            if len(daily_pnls) > 1:
                daily_returns = [p for p in daily_pnls if p != 0]
                if daily_returns:
                    slice_result.sharpe_ratio = (
                        np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(365)
                    )
            
            # Max drawdown
            cumulative = np.cumsum(daily_pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (running_max - cumulative) / (np.abs(running_max) + 1e-8)
            slice_result.max_drawdown = np.max(drawdown)
        
        slice_result.gate_off_days = gate_off_count
        slice_result.gate_off_rate = gate_off_count / slice_result.total_days
        
        # Check acceptance criteria
        slice_result.passed = self._check_acceptance(slice_result)
        
        return slice_result
    
    def _check_acceptance(self, slice_result: ValidationSlice) -> bool:
        """Check if slice meets acceptance criteria"""
        
        # Hit rate check
        if slice_result.n_trades > 0:
            if slice_result.hit_rate >= self.min_hit_rate:
                hit_rate_ok = True
            else:
                # Check with payoff ratio
                avg_win = slice_result.total_pnl / slice_result.n_wins if slice_result.n_wins > 0 else 0
                avg_loss = -slice_result.total_pnl / (slice_result.n_trades - slice_result.n_wins) if slice_result.n_trades > slice_result.n_wins else 0
                payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                
                hit_rate_ok = (
                    slice_result.hit_rate >= self.min_hit_rate_with_payoff and
                    payoff_ratio >= self.min_payoff_ratio
                )
        else:
            hit_rate_ok = False
        
        # Other criteria
        sharpe_ok = slice_result.sharpe_ratio >= self.min_sharpe
        drawdown_ok = slice_result.max_drawdown <= self.max_drawdown
        gate_ok = slice_result.gate_off_rate <= self.max_gate_off_rate
        
        return hit_rate_ok and sharpe_ok and drawdown_ok and gate_ok
    
    def run_validation(self, symbol: str, 
                      start_date: datetime,
                      end_date: datetime) -> List[ValidationSlice]:
        """Run complete walk-forward validation
        
        Args:
            symbol: Trading symbol
            start_date: Overall start date
            end_date: Overall end date
            
        Returns:
            List of validation slices
        """
        results = []
        
        # Initial training period
        train_start = start_date
        train_end = train_start + timedelta(days=self.train_window_days)
        
        while train_end + timedelta(days=self.embargo_days + self.test_window_days) <= end_date:
            # Test period (after embargo)
            test_start = train_end + timedelta(days=self.embargo_days)
            test_end = test_start + timedelta(days=self.test_window_days)
            
            # Validate this slice
            slice_result = self.validate_slice(
                symbol, train_start, test_start, test_end
            )
            results.append(slice_result)
            
            # Roll forward
            train_start = test_start
            train_end = train_start + timedelta(days=self.train_window_days)
        
        return results
    
    def print_acceptance_report(self, results: List[ValidationSlice]):
        """Print detailed acceptance report"""
        
        print("\n" + "="*60)
        print("VALIDATION ACCEPTANCE REPORT")
        print("="*60)
        
        # Overall statistics
        n_slices = len(results)
        n_passed = sum(1 for r in results if r.passed)
        pass_rate = n_passed / n_slices if n_slices > 0 else 0
        
        print(f"\nSlices Tested: {n_slices}")
        print(f"Slices Passed: {n_passed} ({pass_rate:.1%})")
        
        # Aggregate metrics
        if results:
            avg_hit_rate = np.mean([r.hit_rate for r in results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in results])
            avg_drawdown = np.mean([r.max_drawdown for r in results])
            avg_gate_off = np.mean([r.gate_off_rate for r in results])
            
            print(f"\nAverage Metrics:")
            print(f"  Hit Rate: {avg_hit_rate:.1%} (bar: ≥{self.min_hit_rate:.1%})")
            print(f"  Sharpe: {avg_sharpe:.2f} (bar: ≥{self.min_sharpe:.1f})")
            print(f"  Max DD: {avg_drawdown:.1%} (bar: ≤{self.max_drawdown:.1%})")
            print(f"  Gate Off: {avg_gate_off:.1%} (bar: ≤{self.max_gate_off_rate:.1%})")
        
        # Failed slices analysis
        failed_slices = [r for r in results if not r.passed]
        if failed_slices:
            print(f"\n{len(failed_slices)} Failed Slices - Top Issues:")
            
            # Count failure reasons
            low_hit_rate = sum(1 for r in failed_slices if r.hit_rate < self.min_hit_rate)
            low_sharpe = sum(1 for r in failed_slices if r.sharpe_ratio < self.min_sharpe)
            high_dd = sum(1 for r in failed_slices if r.max_drawdown > self.max_drawdown)
            high_gate_off = sum(1 for r in failed_slices if r.gate_off_rate > self.max_gate_off_rate)
            
            if low_hit_rate > 0:
                print(f"  1. Low hit rate: {low_hit_rate} slices")
            if low_sharpe > 0:
                print(f"  2. Low Sharpe: {low_sharpe} slices")
            if high_dd > 0:
                print(f"  3. High drawdown: {high_dd} slices")
            if high_gate_off > 0:
                print(f"  4. Excessive gate-off: {high_gate_off} slices")
        
        # Final verdict
        print("\n" + "-"*60)
        if pass_rate >= 0.7:  # 70% of slices must pass
            print("✓ VALIDATION PASSED")
        else:
            print("✗ VALIDATION FAILED")
        print("="*60 + "\n")