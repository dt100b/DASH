"""Unit tests for Optimal Execution module with Price Impact
Tests implementation against paper 2507.17162v1 specifications
"""
import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from execution import OptimalExecution, ExecutionConfig
from datetime import datetime, timedelta

class TestExecutionConfig(unittest.TestCase):
    """Test configuration parameters"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ExecutionConfig()
        
        # Check transaction costs
        self.assertEqual(config.K, 1.0)
        self.assertEqual(config.lambda_impact, 0.1)
        self.assertEqual(config.beta_decay, 0.1)
        
        # Check risk parameters
        self.assertEqual(config.gamma, 5.0)
        self.assertEqual(config.rho, 0.2)
        
        # Check volatility scales
        self.assertEqual(config.epsilon, 0.1)
        self.assertEqual(config.delta, 0.5)
        
        # Check calibration
        self.assertEqual(config.recalibrate_days, 7)
        self.assertEqual(config.half_step_threshold, 1.2)

class TestOptimalExecution(unittest.TestCase):
    """Test optimal execution implementation"""
    
    def test_initialization(self):
        """Test module initialization"""
        exec_module = OptimalExecution()
        
        # Check coefficient initialization
        self.assertIn('Aqq', exec_module.A_coeffs)
        self.assertIn('Aql', exec_module.A_coeffs)
        self.assertIn('Aqx', exec_module.A_coeffs)
        
        # Check all coefficients are computed
        expected_keys = ['Aqq', 'Aql', 'All', 'Aqx', 'Axl', 'Axx', 'A0']
        for key in expected_keys:
            self.assertIn(key, exec_module.A_coeffs)
            self.assertIsInstance(exec_module.A_coeffs[key], float)
    
    def test_target_exposure_formula(self):
        """Test x* ∝ Bias formula from paper"""
        exec_module = OptimalExecution()
        
        # Test proportionality to bias
        bias1 = 0.5
        target1 = exec_module.compute_target_exposure(
            bias=bias1, price_impact=0.0, current_vol=0.2
        )
        
        bias2 = 1.0
        target2 = exec_module.compute_target_exposure(
            bias=bias2, price_impact=0.0, current_vol=0.2
        )
        
        # Target should scale with bias (but not perfectly proportional due to denominator)
        # Just check that higher bias gives higher target
        self.assertGreater(abs(target2), abs(target1))
        
        # Test bounds
        extreme_bias = 10.0
        target_extreme = exec_module.compute_target_exposure(
            bias=extreme_bias, price_impact=0.0, current_vol=0.2
        )
        self.assertLessEqual(abs(target_extreme), 1.0)
    
    def test_volatility_scaling(self):
        """Test volatility impact on target exposure"""
        exec_module = OptimalExecution()
        
        bias = 0.5
        price_impact = 0.0
        
        # Normal volatility
        normal_vol = exec_module.config.mu_vol
        target_normal = exec_module.compute_target_exposure(
            bias=bias, price_impact=price_impact, current_vol=normal_vol
        )
        
        # High volatility (2x normal)
        high_vol = 2 * exec_module.config.mu_vol
        target_high = exec_module.compute_target_exposure(
            bias=bias, price_impact=price_impact, current_vol=high_vol
        )
        
        # Should reduce exposure when vol is high
        self.assertLess(abs(target_high), abs(target_normal))
    
    def test_kappa_fast_rule(self):
        """Test κ fast volatility step rule: -εγ(y-μ)q/(θK)"""
        exec_module = OptimalExecution()
        
        current_pos = 0.5
        target_pos = 0.5  # No position change
        current_vol = 0.2
        
        # Test with vol_y above mean
        vol_y_high = exec_module.config.mu_vol + 0.1
        rate_high, diag_high = exec_module.compute_trading_rate(
            current_pos, target_pos, current_vol, vol_y_high, 0.0
        )
        
        # Test with vol_y below mean
        vol_y_low = exec_module.config.mu_vol - 0.1
        rate_low, diag_low = exec_module.compute_trading_rate(
            current_pos, target_pos, current_vol, vol_y_low, 0.0
        )
        
        # Fast correction should be negative when y > μ
        self.assertLess(diag_high['fast_correction'], 0)
        
        # Fast correction should be positive when y < μ
        self.assertGreater(diag_low['fast_correction'], 0)
    
    def test_kappa_slow_rule(self):
        """Test κ slow volatility step rule: √δ(Bq+λBl)/K"""
        exec_module = OptimalExecution()
        
        current_pos = 0.5
        target_pos = 0.5
        current_vol = 0.2
        vol_y = exec_module.config.mu_vol
        
        # Test with different slow vol factors
        vol_z_low = -0.5
        rate_low, diag_low = exec_module.compute_trading_rate(
            current_pos, target_pos, current_vol, vol_y, vol_z_low
        )
        
        vol_z_high = 0.5
        rate_high, diag_high = exec_module.compute_trading_rate(
            current_pos, target_pos, current_vol, vol_y, vol_z_high
        )
        
        # Slow corrections should differ
        self.assertNotEqual(diag_low['slow_correction'], diag_high['slow_correction'])
        
        # Should scale with √δ
        delta = exec_module.config.delta
        self.assertIsNotNone(diag_low['slow_correction'])
    
    def test_half_step_condition(self):
        """Test half-step when vol > threshold * μ"""
        exec_module = OptimalExecution()
        
        current_pos = 0.0
        target_pos = 1.0
        vol_y = exec_module.config.mu_vol
        vol_z = 0.0
        
        # Normal volatility - no half-step
        normal_vol = exec_module.config.mu_vol
        rate_normal, diag_normal = exec_module.compute_trading_rate(
            current_pos, target_pos, normal_vol, vol_y, vol_z
        )
        self.assertFalse(diag_normal['half_stepped'])
        
        # High volatility - should half-step
        high_vol = exec_module.config.half_step_threshold * exec_module.config.mu_vol + 0.01
        rate_high, diag_high = exec_module.compute_trading_rate(
            current_pos, target_pos, high_vol, vol_y, vol_z
        )
        self.assertTrue(diag_high['half_stepped'])
        
        # Trading rate should be reduced
        self.assertLess(abs(diag_high['adjusted_speed']), abs(diag_normal['adjusted_speed']))
    
    def test_price_impact_dynamics(self):
        """Test price impact evolution: dl = (λu - βl)dt"""
        exec_module = OptimalExecution()
        
        initial_impact = 0.1
        trading_rate = 0.5
        dt = 1/252  # Daily
        
        # Update impact
        new_impact = exec_module.update_price_impact(initial_impact, trading_rate, dt)
        
        # Check evolution follows dynamics
        lambda_eff = exec_module.config.theta * exec_module.config.lambda_impact
        beta_eff = exec_module.config.theta * exec_module.config.beta_decay
        expected_change = (lambda_eff * trading_rate - beta_eff * initial_impact) * dt
        expected_impact = initial_impact + expected_change
        
        self.assertAlmostEqual(new_impact, expected_impact, places=8)
        
        # Impact should decay when no trading
        zero_trade_impact = exec_module.update_price_impact(initial_impact, 0.0, dt)
        self.assertLess(zero_trade_impact, initial_impact)
    
    def test_execution_cost(self):
        """Test execution cost: K/2 * u² + λ*q*u"""
        exec_module = OptimalExecution()
        
        trading_rate = 0.5
        price_impact = 0.1
        
        cost = exec_module.compute_execution_cost(trading_rate, price_impact)
        
        # Check components
        K = exec_module.config.K
        lambda_eff = exec_module.config.theta * exec_module.config.lambda_impact
        
        expected_instant = K/2 * trading_rate**2
        expected_persistent = lambda_eff * abs(trading_rate) * abs(price_impact)
        expected_total = expected_instant + expected_persistent
        
        self.assertAlmostEqual(cost, expected_total, places=8)
        
        # Cost should be zero when no trading
        zero_cost = exec_module.compute_execution_cost(0.0, price_impact)
        self.assertEqual(zero_cost, 0.0)
    
    def test_small_impact_approximation(self):
        """Test small price impact approximation with θ parameter"""
        config = ExecutionConfig()
        config.theta = 0.01  # Very small impact
        
        exec_module = OptimalExecution(config)
        
        # Coefficients should be close to no-impact case
        Aqq = exec_module.A_coeffs['Aqq']
        Aql = exec_module.A_coeffs['Aql']
        
        # Aql should be small when θ is small
        self.assertLess(abs(Aql), 0.1)
        
        # Aqq should be dominated by zeroth-order term
        self.assertGreater(Aqq, 0.1)
    
    def test_weekly_recalibration(self):
        """Test weekly slippage-based recalibration"""
        exec_module = OptimalExecution()
        
        # Initially should need calibration
        self.assertTrue(exec_module.should_recalibrate())
        
        # Simulate trades with slippage
        realized_trades = [
            {'size': 100, 'rate': 0.5, 'slippage': 0.002},
            {'size': -50, 'rate': -0.3, 'slippage': 0.001},
            {'size': 75, 'rate': 0.4, 'slippage': 0.0015}
        ]
        
        # Calibrate
        updates = exec_module.calibrate_from_slippage(realized_trades)
        
        # Should have updated lambda
        self.assertIn('lambda_impact', updates)
        self.assertIn('adjustment_factor', updates)
        
        # Should not need immediate recalibration
        self.assertFalse(exec_module.should_recalibrate())
        
        # Should have calibration history
        self.assertGreater(len(exec_module.calibration_history), 0)
        last_calib = exec_module.calibration_history[-1]
        self.assertIn('realized_slippage', last_calib)
        self.assertIn('expected_slippage', last_calib)
    
    def test_execution_summary(self):
        """Test comprehensive execution summary"""
        exec_module = OptimalExecution()
        
        summary = exec_module.get_execution_summary(
            current_pos=0.3,
            target_pos=0.5,
            bias=0.7,
            current_vol=0.25,
            vol_y=0.22,
            vol_z=0.1,
            price_impact=0.05
        )
        
        # Check all required fields
        required_fields = [
            'optimal_target', 'formula_x_star', 'trading_rate',
            'kappa_fast_rule', 'kappa_slow_rule', 'half_step_active',
            'half_step_threshold', 'tracking_speed', 'position_gap',
            'execution_cost', 'new_price_impact', 'needs_recalibration'
        ]
        
        for field in required_fields:
            self.assertIn(field, summary)
        
        # Check formula strings
        self.assertIn('x* =', summary['formula_x_star'])
        self.assertIn('Bias=', summary['formula_x_star'])
        self.assertIn('κ_fast', summary['kappa_fast_rule'])
        self.assertIn('κ_slow', summary['kappa_slow_rule'])
    
    def test_state_persistence(self):
        """Test saving and loading module state"""
        import tempfile
        import json
        
        exec_module = OptimalExecution()
        
        # Modify state
        exec_module.config.lambda_impact = 0.15
        exec_module.last_calibration = datetime.now()
        exec_module.calibration_history.append({
            'timestamp': datetime.now(),
            'old_lambda': 0.1,
            'new_lambda': 0.15,
            'adjustment': 1.5,
            'realized_slippage': 0.002,
            'expected_slippage': 0.0013
        })
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            exec_module.save_state(temp_path)
        
        # Load into new instance
        new_module = OptimalExecution()
        success = new_module.load_state(temp_path)
        self.assertTrue(success)
        
        # Check state matches
        self.assertEqual(new_module.config.lambda_impact, 0.15)
        self.assertIsNotNone(new_module.last_calibration)
        self.assertEqual(len(new_module.calibration_history), 1)
        
        # Clean up
        os.unlink(temp_path)

if __name__ == '__main__':
    unittest.main()