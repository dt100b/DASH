"""Integrated Trading Pipeline
Orchestrates all research-based modules for quantitative trading
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging

# Import research modules
from backend.features_signature import SignatureFeatures
from backend.regime_orcsmc import ORCSMCRegime
from backend.stopping import DeepOptimalStopping
from backend.execution import OptimalExecution, ExecutionConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data container"""
    timestamp: datetime
    price: float
    volume: float
    bid: float
    ask: float
    open_interest: Optional[float] = None

@dataclass
class Position:
    """Current position state"""
    size: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    holding_days: int = 0
    price_impact: float = 0.0
    last_update: Optional[datetime] = None

class TradingPipeline:
    """Integrated trading pipeline with all research modules
    
    Pipeline steps:
    1. Feature extraction (Signature features)
    2. Regime detection (ORCSMC)
    3. Bias computation (if regime ON)
    4. Target sizing (Optimal execution x* ∝ Bias)
    5. Execution with impact (κ step rules)
    6. Exit decisions (Deep optimal stopping)
    """
    
    def __init__(self):
        # Initialize all modules
        self.signature = SignatureFeatures(order=3, time_augment=True)
        self.regime = ORCSMCRegime(
            n_particles=1000,
            window_length=30,
            learning_iterations=5
        )
        self.stopping = DeepOptimalStopping(max_holding_days=7)
        self.execution = OptimalExecution()
        
        # Trading state
        self.position = Position()
        self.market_history = []
        self.trade_history = []
        self.last_pipeline_run = None
        
        # Risk parameters
        self.max_position_size = 1.0
        self.daily_loss_limit = -0.05
        self.max_drawdown = -0.10
        
        # Volatility tracking
        self.current_vol = 0.2
        self.vol_factor_fast = 0.2
        self.vol_factor_slow = 0.0
        
    def process_market_data(self, data: List[MarketData]) -> Dict:
        """Process market data through full pipeline
        
        Args:
            data: List of recent market data points (72h recommended)
            
        Returns:
            Pipeline execution summary with decisions
        """
        logger.info(f"Processing {len(data)} market data points")
        
        # Step 1: Extract signature features
        prices = np.array([d.price for d in data])
        volumes = np.array([d.volume for d in data])
        
        sig_features = self.signature.compute_features({
            'prices': prices,
            'volumes': volumes,
            'timestamp': data[-1].timestamp
        })
        
        # Update volatility estimates
        self._update_volatility(prices)
        
        # Step 2: ORCSMC regime detection
        regime_input = {
            'features': sig_features['signature_features'],
            'volatility': self.current_vol,
            'volume_profile': np.mean(volumes[-10:]) / np.mean(volumes)
        }
        
        regime_prob, regime_state = self.regime.update(
            observation=regime_input['features'][:10],  # Use subset
            timestamp=data[-1].timestamp
        )
        
        logger.info(f"Regime probability: {regime_prob:.3f}, State: {regime_state}")
        
        # Step 3: Check regime gate (≥0.6 threshold)
        if regime_prob < 0.6:
            logger.info("Regime gate closed, no trading")
            return self._create_summary(
                action="HOLD",
                reason="Regime OFF",
                regime_prob=regime_prob,
                features=sig_features
            )
        
        # Step 4: Compute directional bias
        bias = self._compute_bias(sig_features, prices)
        logger.info(f"Directional bias: {bias:.3f}")
        
        # Step 5: Compute optimal target exposure (x* ∝ Bias)
        target_exposure = self.execution.compute_target_exposure(
            bias=bias,
            price_impact=self.position.price_impact,
            current_vol=self.current_vol,
            signal_strength=regime_prob  # Use regime prob as signal strength
        )
        
        # Step 6: Check if we should exit existing position
        if self.position.size != 0:
            exit_decision = self._check_exit_decision(
                sig_features, prices[-1]
            )
            
            if exit_decision['should_exit']:
                logger.info(f"Exiting position: {exit_decision['reason']}")
                return self._execute_exit(prices[-1], exit_decision)
        
        # Step 7: Compute optimal trading rate with κ corrections
        trading_rate, exec_diagnostics = self.execution.compute_trading_rate(
            current_pos=self.position.size,
            target_pos=target_exposure,
            current_vol=self.current_vol,
            vol_factor_y=self.vol_factor_fast,
            vol_factor_z=self.vol_factor_slow
        )
        
        # Step 8: Apply risk checks
        risk_check = self._apply_risk_checks(trading_rate, target_exposure)
        if not risk_check['passed']:
            logger.warning(f"Risk check failed: {risk_check['reason']}")
            return self._create_summary(
                action="BLOCKED",
                reason=risk_check['reason'],
                regime_prob=regime_prob,
                bias=bias,
                target=target_exposure
            )
        
        # Step 9: Execute trade
        if abs(trading_rate) > 0.001:  # Min trade threshold
            execution_result = self._execute_trade(
                trading_rate=trading_rate,
                target_exposure=target_exposure,
                current_price=prices[-1],
                exec_diagnostics=exec_diagnostics
            )
            
            # Step 10: Update price impact
            self.position.price_impact = self.execution.update_price_impact(
                current_impact=self.position.price_impact,
                trading_rate=trading_rate
            )
            
            return execution_result
        
        return self._create_summary(
            action="HOLD",
            reason="Below minimum trade size",
            regime_prob=regime_prob,
            bias=bias,
            target=target_exposure,
            features=sig_features
        )
    
    def _update_volatility(self, prices: np.ndarray):
        """Update volatility estimates"""
        if len(prices) < 20:
            return
            
        # Simple volatility estimate
        returns = np.diff(np.log(prices))
        self.current_vol = np.std(returns) * np.sqrt(252)
        
        # Fast factor (short window)
        if len(returns) >= 5:
            self.vol_factor_fast = np.std(returns[-5:]) * np.sqrt(252)
        
        # Slow factor (long window)
        if len(returns) >= 20:
            self.vol_factor_slow = np.std(returns[-20:]) * np.sqrt(252) - self.current_vol
    
    def _compute_bias(self, features: Dict, prices: np.ndarray) -> float:
        """Compute directional bias from features
        
        Combines multiple signals:
        - Momentum from signatures
        - Mean reversion signals
        - Volume patterns
        """
        bias = 0.0
        
        # Momentum from signature features
        if 'momentum_score' in features:
            bias += features['momentum_score'] * 0.4
        
        # Mean reversion component
        if len(prices) >= 20:
            ma_20 = np.mean(prices[-20:])
            deviation = (prices[-1] - ma_20) / ma_20
            bias -= deviation * 0.3  # Contrarian
        
        # Volume signal
        if 'volume_signal' in features:
            bias += features['volume_signal'] * 0.3
        
        # Normalize to [-1, 1]
        return np.clip(bias, -1.0, 1.0)
    
    def _check_exit_decision(self, features: Dict, current_price: float) -> Dict:
        """Check if position should be exited"""
        position_features = {
            'unrealized_pnl': self.position.unrealized_pnl,
            'pnl_volatility': 0.02,  # Simplified
            'max_drawdown': min(0, self.position.unrealized_pnl),
            'current_vol': self.current_vol,
            'vol_change': self.vol_factor_fast - self.current_vol,
            'regime_prob': self.regime.get_current_probability()
        }
        
        # Deep optimal stopping decision
        decision, confidence, bounds = self.stopping.decide(
            position_features=position_features,
            holding_days=self.position.holding_days
        )
        
        should_exit = decision == "STOP"
        
        # Override with hard stops
        if self.position.unrealized_pnl < -0.02:  # 2% stop loss
            should_exit = True
            reason = "Stop loss triggered"
        elif self.position.holding_days >= 7:  # Max holding
            should_exit = True
            reason = "Max holding period"
        else:
            reason = f"Optimal stopping (confidence: {confidence:.2f})"
        
        return {
            'should_exit': should_exit,
            'reason': reason,
            'confidence': confidence,
            'bounds': bounds
        }
    
    def _apply_risk_checks(self, trading_rate: float, target_exposure: float) -> Dict:
        """Apply risk management checks"""
        # Check position limits
        new_position = self.position.size + trading_rate
        if abs(new_position) > self.max_position_size:
            return {'passed': False, 'reason': 'Position limit exceeded'}
        
        # Check daily loss limit
        if self.position.realized_pnl < self.daily_loss_limit:
            return {'passed': False, 'reason': 'Daily loss limit reached'}
        
        # Check volatility regime
        if self.current_vol > 0.5:  # 50% annualized vol
            return {'passed': False, 'reason': 'Volatility too high'}
        
        # Check drawdown
        if self.position.unrealized_pnl < self.max_drawdown:
            return {'passed': False, 'reason': 'Max drawdown reached'}
        
        return {'passed': True, 'reason': 'All checks passed'}
    
    def _execute_trade(self, trading_rate: float, target_exposure: float,
                      current_price: float, exec_diagnostics: Dict) -> Dict:
        """Execute trade and update position"""
        # Calculate trade size
        trade_size = trading_rate
        
        # Update position
        old_size = self.position.size
        self.position.size += trade_size
        
        # Calculate costs
        exec_cost = self.execution.compute_execution_cost(
            trading_rate=trading_rate,
            price_impact=self.position.price_impact
        )
        
        # Update P&L
        if old_size != 0:
            # Realize P&L on closed portion
            closed_pnl = -trade_size * (current_price - self.position.entry_price)
            self.position.realized_pnl += closed_pnl - exec_cost
        
        # Update entry price (weighted average)
        if self.position.size != 0:
            if old_size * self.position.size > 0:  # Same direction
                total_value = old_size * self.position.entry_price + trade_size * current_price
                self.position.entry_price = total_value / self.position.size
            else:  # New position
                self.position.entry_price = current_price
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now(),
            'size': trade_size,
            'price': current_price,
            'target': target_exposure,
            'cost': exec_cost,
            'impact': self.position.price_impact,
            'diagnostics': exec_diagnostics
        }
        self.trade_history.append(trade_record)
        
        logger.info(f"Executed trade: size={trade_size:.4f}, target={target_exposure:.4f}")
        
        return self._create_summary(
            action="TRADE",
            trade_size=trade_size,
            new_position=self.position.size,
            target=target_exposure,
            execution_cost=exec_cost,
            diagnostics=exec_diagnostics
        )
    
    def _execute_exit(self, current_price: float, exit_decision: Dict) -> Dict:
        """Execute position exit"""
        exit_size = -self.position.size
        
        # Calculate final P&L
        final_pnl = self.position.size * (current_price - self.position.entry_price)
        self.position.realized_pnl += final_pnl
        
        # Reset position
        self.position.size = 0.0
        self.position.entry_price = 0.0
        self.position.unrealized_pnl = 0.0
        self.position.holding_days = 0
        self.position.price_impact = 0.0
        
        logger.info(f"Position exited: P&L={final_pnl:.4f}, Reason={exit_decision['reason']}")
        
        return self._create_summary(
            action="EXIT",
            exit_size=exit_size,
            realized_pnl=final_pnl,
            total_pnl=self.position.realized_pnl,
            reason=exit_decision['reason']
        )
    
    def _create_summary(self, **kwargs) -> Dict:
        """Create pipeline execution summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'position': {
                'size': self.position.size,
                'entry_price': self.position.entry_price,
                'unrealized_pnl': self.position.unrealized_pnl,
                'realized_pnl': self.position.realized_pnl,
                'holding_days': self.position.holding_days,
                'price_impact': self.position.price_impact
            },
            'volatility': {
                'current': self.current_vol,
                'fast_factor': self.vol_factor_fast,
                'slow_factor': self.vol_factor_slow
            },
            'regime': {
                'probability': self.regime.get_current_probability(),
                'state': 'ON' if self.regime.get_current_probability() >= 0.6 else 'OFF'
            }
        }
        summary.update(kwargs)
        return summary
    
    def run_daily_pipeline(self, market_data: List[MarketData]) -> Dict:
        """Run complete daily trading pipeline at 00:00 UTC
        
        This is the main entry point for scheduled trading
        """
        logger.info("=" * 50)
        logger.info("Starting daily trading pipeline")
        logger.info(f"Time: {datetime.now()} UTC")
        
        # Update holding days
        if self.position.size != 0:
            self.position.holding_days += 1
        
        # Run pipeline
        result = self.process_market_data(market_data)
        
        # Check for recalibration
        if self.execution.should_recalibrate() and self.trade_history:
            recent_trades = self.trade_history[-20:]  # Last 20 trades
            calib_result = self.execution.calibrate_from_slippage(recent_trades)
            if calib_result:
                logger.info(f"Recalibrated: {calib_result}")
        
        # Save state
        self.save_state('pipeline_state.json')
        
        self.last_pipeline_run = datetime.now()
        
        logger.info("Daily pipeline complete")
        logger.info("=" * 50)
        
        return result
    
    def get_diagnostics(self) -> Dict:
        """Get comprehensive diagnostics from all modules"""
        return {
            'pipeline': {
                'last_run': self.last_pipeline_run.isoformat() if self.last_pipeline_run else None,
                'total_trades': len(self.trade_history),
                'position': self.position.__dict__
            },
            'signature': self.signature.get_diagnostics(),
            'regime': self.regime.get_diagnostics(),
            'stopping': self.stopping.get_diagnostics(),
            'execution': self.execution.get_execution_summary(
                current_pos=self.position.size,
                target_pos=0,  # Placeholder
                bias=0,
                current_vol=self.current_vol,
                vol_y=self.vol_factor_fast,
                vol_z=self.vol_factor_slow,
                price_impact=self.position.price_impact
            )
        }
    
    def save_state(self, filepath: str):
        """Save complete pipeline state"""
        state = {
            'position': self.position.__dict__,
            'market_history': [
                {**m.__dict__, 'timestamp': m.timestamp.isoformat()}
                for m in self.market_history[-100:]
            ],
            'trade_history': [
                {**t, 'timestamp': t['timestamp'].isoformat()}
                for t in self.trade_history[-100:]
            ],
            'last_pipeline_run': self.last_pipeline_run.isoformat() if self.last_pipeline_run else None,
            'volatility': {
                'current': self.current_vol,
                'fast': self.vol_factor_fast,
                'slow': self.vol_factor_slow
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save module states
        self.signature.save_calibration(filepath.replace('.json', '_signature.json'))
        self.regime.save_state(filepath.replace('.json', '_regime.pt'))
        self.stopping.save_model(filepath.replace('.json', '_stopping.pt'))
        self.execution.save_state(filepath.replace('.json', '_execution.json'))
    
    def load_state(self, filepath: str) -> bool:
        """Load complete pipeline state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore position
            for key, value in state['position'].items():
                if key == 'last_update' and value:
                    setattr(self.position, key, datetime.fromisoformat(value))
                else:
                    setattr(self.position, key, value)
            
            # Restore volatility
            self.current_vol = state['volatility']['current']
            self.vol_factor_fast = state['volatility']['fast']
            self.vol_factor_slow = state['volatility']['slow']
            
            # Restore history (limited)
            self.trade_history = [
                {**t, 'timestamp': datetime.fromisoformat(t['timestamp'])}
                for t in state.get('trade_history', [])
            ]
            
            if state['last_pipeline_run']:
                self.last_pipeline_run = datetime.fromisoformat(state['last_pipeline_run'])
            
            # Load module states
            self.signature.load_calibration(filepath.replace('.json', '_signature.json'))
            self.regime.load_state(filepath.replace('.json', '_regime.pt'))
            self.stopping.load_model(filepath.replace('.json', '_stopping.pt'))
            self.execution.load_state(filepath.replace('.json', '_execution.json'))
            
            logger.info("Pipeline state loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False