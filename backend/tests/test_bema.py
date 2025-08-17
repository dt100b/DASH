"""Unit tests for BEMA (Bias-Corrected Exponential Moving Average) optimizer
Tests implementation of paper 2508.00180v1: EMA Without the Lag
"""
import unittest
import numpy as np
import tempfile
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bema_optimizer import BEMAOptimizer, BEMAConfig

class TestBEMAConfig(unittest.TestCase):
    """Test BEMA configuration"""
    
    def test_default_config(self):
        """Test default configuration values from paper"""
        config = BEMAConfig()
        
        # Paper recommended defaults
        self.assertEqual(config.ema_power, 0.5)  # κ=0.5 optimal
        self.assertEqual(config.bias_power, 0.4)  # η=0.4 from Figure 1d
        self.assertEqual(config.multiplier, 1.0)  # γ=1.0
        self.assertEqual(config.lag, 1.0)  # ρ=1.0
        self.assertEqual(config.burn_in, 0)  # τ=0 for finetuning
        self.assertEqual(config.update_freq, 400)  # ϕ=400 steps
        
        # LR schedule defaults
        self.assertEqual(config.lr_schedule, "constant")
        self.assertIsNotNone(config.lr_grid)
        self.assertIn(1e-3, config.lr_grid)
        
        # Snapshot defaults
        self.assertEqual(config.snapshot_interval, 5000)
        self.assertEqual(config.restore_threshold, 0.1)
        self.assertEqual(config.max_snapshots, 3)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = BEMAConfig(
            ema_power=0.2,
            bias_power=0.3,
            lr_grid=[1e-4, 1e-3],
            lr_schedule="decay"
        )
        
        self.assertEqual(config.ema_power, 0.2)
        self.assertEqual(config.bias_power, 0.3)
        self.assertEqual(config.lr_schedule, "decay")
        self.assertEqual(len(config.lr_grid), 2)

class TestBEMAOptimizer(unittest.TestCase):
    """Test BEMA optimizer implementation"""
    
    def setUp(self):
        """Set up test parameters"""
        self.dim = 10
        self.initial_params = np.random.randn(self.dim)
        self.config = BEMAConfig(
            ema_power=0.5,
            bias_power=0.4,
            update_freq=10,  # More frequent updates for testing
            burn_in=5
        )
    
    def test_initialization(self):
        """Test BEMA initialization"""
        optimizer = BEMAOptimizer(self.initial_params, self.config)
        
        # Check initial state
        np.testing.assert_array_equal(optimizer.theta_0, self.initial_params)
        np.testing.assert_array_equal(optimizer.theta_ema, self.initial_params)
        np.testing.assert_array_equal(optimizer.theta_bema, self.initial_params)
        
        self.assertEqual(optimizer.step_count, 0)
        self.assertEqual(optimizer.update_count, 0)
        self.assertEqual(optimizer.current_lr, self.config.lr_grid[0])
    
    def test_burn_in_period(self):
        """Test burn-in period behavior (Algorithm 1, lines 5-6)"""
        optimizer = BEMAOptimizer(self.initial_params, self.config)
        
        # During burn-in, parameters should reset to current
        for step in range(self.config.burn_in):
            new_params = self.initial_params + 0.1 * step
            result = optimizer.update(new_params, loss=1.0)
            
            # During burn-in, BEMA should equal current params
            np.testing.assert_array_equal(result, new_params)
            np.testing.assert_array_equal(optimizer.theta_0, new_params)
    
    def test_ema_update(self):
        """Test EMA update equation (Algorithm 1, line 12)"""
        optimizer = BEMAOptimizer(self.initial_params, self.config)
        
        # Skip burn-in
        for _ in range(self.config.burn_in + 1):
            optimizer.update(self.initial_params)
        
        # Test EMA update
        new_params = self.initial_params + 1.0
        t = optimizer.step_count - self.config.burn_in
        
        # Update at correct frequency
        for _ in range(self.config.update_freq - 1):
            optimizer.update(new_params)
        
        # This update should trigger EMA
        result = optimizer.update(new_params)
        
        # Compute expected beta_t
        t = optimizer.update_count
        beta_t = (self.config.lag + self.config.multiplier * t * self.config.update_freq) ** (-self.config.ema_power)
        
        # Check EMA update
        expected_ema = (1 - beta_t) * self.initial_params + beta_t * new_params
        np.testing.assert_allclose(optimizer.theta_ema, expected_ema, rtol=1e-5)
    
    def test_bias_correction(self):
        """Test BEMA bias correction (Algorithm 1, line 13)"""
        optimizer = BEMAOptimizer(self.initial_params, self.config)
        
        # Skip burn-in
        for _ in range(self.config.burn_in + 1):
            optimizer.update(self.initial_params)
        
        # Apply updates
        new_params = self.initial_params + 1.0
        
        # Update at correct frequency
        for _ in range(self.config.update_freq - 1):
            optimizer.update(new_params)
        
        result = optimizer.update(new_params)
        
        # Compute expected alpha_t
        t = optimizer.update_count
        alpha_t = (self.config.lag + self.config.multiplier * t * self.config.update_freq) ** (-self.config.bias_power)
        
        # BEMA = EMA + α_t(θ_t - θ_0)
        bias_correction = alpha_t * (new_params - optimizer.theta_0)
        expected_bema = optimizer.theta_ema + bias_correction
        
        np.testing.assert_allclose(result, expected_bema, rtol=1e-5)
    
    def test_update_frequency(self):
        """Test update frequency control (Algorithm 1, line 7)"""
        optimizer = BEMAOptimizer(self.initial_params, self.config)
        
        # Skip burn-in
        for _ in range(self.config.burn_in + 1):
            optimizer.update(self.initial_params)
        
        initial_bema = optimizer.theta_bema.copy()
        
        # Updates before frequency threshold should not change BEMA
        for i in range(1, self.config.update_freq):
            new_params = self.initial_params + 0.1 * i
            result = optimizer.update(new_params)
            np.testing.assert_array_equal(result, initial_bema)
        
        # Update at frequency should change BEMA
        new_params = self.initial_params + 1.0
        result = optimizer.update(new_params)
        self.assertFalse(np.array_equal(result, initial_bema))
    
    def test_variance_reduction(self):
        """Test that BEMA reduces variance compared to vanilla params"""
        optimizer = BEMAOptimizer(self.initial_params, self.config)
        
        # Skip burn-in
        for _ in range(self.config.burn_in + 1):
            optimizer.update(self.initial_params)
        
        # Apply noisy updates
        vanilla_params = []
        bema_params = []
        
        for i in range(100):
            noise = np.random.randn(self.dim) * 0.5
            new_params = self.initial_params + 0.01 * i + noise
            
            vanilla_params.append(new_params)
            bema_result = optimizer.update(new_params)
            bema_params.append(bema_result)
        
        # BEMA should have lower variance
        vanilla_var = np.var(vanilla_params, axis=0).mean()
        bema_var = np.var(bema_params, axis=0).mean()
        
        self.assertLess(bema_var, vanilla_var * 0.8)  # At least 20% reduction
    
    def test_snapshot_restore(self):
        """Test snapshot and restore functionality for high-LR robustness"""
        config = BEMAConfig(
            update_freq=1,
            burn_in=0,
            snapshot_interval=10,
            restore_threshold=0.1
        )
        optimizer = BEMAOptimizer(self.initial_params, config)
        
        # Good updates with decreasing loss
        for i in range(20):
            new_params = self.initial_params + 0.1 * i
            loss = 10.0 - 0.4 * i  # Decreasing loss
            optimizer.update(new_params, loss=loss)
        
        # Check snapshots were created
        self.assertGreater(len(optimizer.snapshots), 0)
        best_loss = optimizer.best_loss
        
        # Simulate loss spike (bad update)
        bad_params = self.initial_params + 100.0  # Large deviation
        spike_loss = best_loss * 1.5  # 50% increase
        
        # Multiple updates to trigger restore
        for _ in range(15):
            optimizer.update(bad_params, loss=spike_loss)
        
        # Should have restored from snapshot
        self.assertLess(optimizer.current_lr, config.lr_grid[0])  # LR reduced
    
    def test_learning_rate_schedules(self):
        """Test different learning rate schedules"""
        # Test decay schedule
        config_decay = BEMAConfig(
            lr_schedule="decay",
            lr_grid=[1e-3],
            lr_decay_factor=0.9,
            update_freq=1,
            burn_in=0
        )
        opt_decay = BEMAOptimizer(self.initial_params, config_decay)
        
        initial_lr = opt_decay.get_learning_rate()
        
        # Advance many steps
        for _ in range(2000):
            opt_decay.update(self.initial_params)
        
        final_lr = opt_decay.get_learning_rate()
        self.assertLess(final_lr, initial_lr)
        
        # Test cyclic schedule
        config_cyclic = BEMAConfig(
            lr_schedule="cyclic",
            lr_grid=[1e-4, 1e-3, 1e-2],
            lr_cycle_steps=10,
            update_freq=1,
            burn_in=0
        )
        opt_cyclic = BEMAOptimizer(self.initial_params, config_cyclic)
        
        lr_history = []
        for _ in range(35):
            opt_cyclic.update(self.initial_params)
            lr_history.append(opt_cyclic.get_learning_rate())
        
        # Should cycle through grid
        self.assertEqual(lr_history[0], 1e-4)
        self.assertEqual(lr_history[10], 1e-3)
        self.assertEqual(lr_history[20], 1e-2)
        self.assertEqual(lr_history[30], 1e-4)  # Cycle repeats
    
    def test_adaptive_lr(self):
        """Test adaptive learning rate based on loss improvement"""
        config = BEMAConfig(
            lr_schedule="adaptive",
            lr_grid=[1e-4, 1e-3, 1e-2],
            update_freq=1,
            burn_in=0
        )
        optimizer = BEMAOptimizer(self.initial_params, config)
        
        # Simulate stagnant loss (no improvement)
        stagnant_loss = 5.0
        for _ in range(150):
            optimizer.update(self.initial_params, loss=stagnant_loss)
        
        # Should move to next LR in grid
        self.assertGreater(optimizer.lr_index, 0)
    
    def test_reset_baseline(self):
        """Test baseline reset functionality"""
        optimizer = BEMAOptimizer(self.initial_params, self.config)
        
        # Skip burn-in and apply updates
        for _ in range(20):
            new_params = self.initial_params + np.random.randn(self.dim) * 0.1
            optimizer.update(new_params)
        
        # Reset baseline
        current_params = self.initial_params + 1.0
        optimizer.theta_current = current_params
        optimizer.reset_baseline()
        
        np.testing.assert_array_equal(optimizer.theta_0, current_params)
    
    def test_diagnostics(self):
        """Test diagnostic output"""
        optimizer = BEMAOptimizer(self.initial_params, self.config)
        
        # Apply some updates
        for i in range(50):
            new_params = self.initial_params + 0.1 * i
            optimizer.update(new_params, loss=10.0 - 0.1 * i)
        
        diagnostics = optimizer.get_diagnostics()
        
        # Check diagnostic keys
        self.assertIn('step_count', diagnostics)
        self.assertIn('update_count', diagnostics)
        self.assertIn('current_lr', diagnostics)
        self.assertIn('lr_schedule', diagnostics)
        self.assertIn('num_snapshots', diagnostics)
        self.assertIn('best_loss', diagnostics)
        self.assertIn('config', diagnostics)
        
        self.assertEqual(diagnostics['step_count'], 50)
        self.assertEqual(diagnostics['lr_schedule'], 'constant')
    
    def test_save_load_state(self):
        """Test saving and loading optimizer state"""
        optimizer = BEMAOptimizer(self.initial_params, self.config)
        
        # Apply updates
        for i in range(30):
            new_params = self.initial_params + 0.1 * i
            optimizer.update(new_params, loss=10.0 - 0.1 * i)
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        optimizer.save_state(filepath)
        self.assertTrue(os.path.exists(filepath))
        
        # Create new optimizer and load state
        new_optimizer = BEMAOptimizer(self.initial_params, self.config)
        success = new_optimizer.load_state(filepath)
        
        self.assertTrue(success)
        self.assertEqual(new_optimizer.step_count, optimizer.step_count)
        self.assertEqual(new_optimizer.update_count, optimizer.update_count)
        self.assertEqual(new_optimizer.current_lr, optimizer.current_lr)
        self.assertEqual(new_optimizer.best_loss, optimizer.best_loss)
        np.testing.assert_array_equal(new_optimizer.theta_bema, optimizer.theta_bema)
        
        # Clean up
        os.unlink(filepath)
    
    def test_comparison_with_standard_ema(self):
        """Test that BEMA converges faster than standard EMA"""
        # Standard EMA (no bias correction)
        ema_params = self.initial_params.copy()
        
        # BEMA optimizer
        bema_optimizer = BEMAOptimizer(self.initial_params, self.config)
        
        # Target parameters (what we're converging to)
        target_params = self.initial_params + 5.0
        
        # Skip burn-in
        for _ in range(self.config.burn_in + 1):
            bema_optimizer.update(self.initial_params)
        
        ema_distances = []
        bema_distances = []
        
        # Apply updates towards target
        for i in range(100):
            # Noisy gradient towards target
            noise = np.random.randn(self.dim) * 0.1
            step = 0.1 * (target_params - ema_params) + noise
            new_params = ema_params + step
            
            # Standard EMA update
            t = i + 1
            beta_t = (self.config.lag + self.config.multiplier * t) ** (-self.config.ema_power)
            ema_params = (1 - beta_t) * ema_params + beta_t * new_params
            
            # BEMA update
            bema_result = bema_optimizer.update(new_params)
            
            # Track distances to target
            ema_distances.append(np.linalg.norm(ema_params - target_params))
            bema_distances.append(np.linalg.norm(bema_result - target_params))
        
        # BEMA should converge faster (lower average distance)
        avg_ema_dist = np.mean(ema_distances[50:])  # After initial convergence
        avg_bema_dist = np.mean(bema_distances[50:])
        
        self.assertLess(avg_bema_dist, avg_ema_dist * 0.9)  # At least 10% improvement

if __name__ == '__main__':
    unittest.main()