"""Unit tests for Deep Optimal Stopping module
Tests implementation against paper 1804.05394v4 specifications
"""
import unittest
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stopping import DeepOptimalStopping, StoppingNet
from datetime import datetime, timezone

class TestStoppingNet(unittest.TestCase):
    """Test neural network architecture"""
    
    def test_network_structure(self):
        """Test network follows paper architecture"""
        net = StoppingNet(input_dim=32, depth=3)
        
        # Check F_theta network exists
        self.assertIsNotNone(net.F_theta)
        
        # Check martingale network exists
        self.assertIsNotNone(net.martingale_net)
        
        # Check continuation network exists
        self.assertIsNotNone(net.continuation_net)
    
    def test_forward_pass(self):
        """Test forward pass dimensions"""
        net = StoppingNet(input_dim=32, depth=3)
        batch_size = 64
        x = torch.randn(batch_size, 32)
        
        F_theta, f_theta, lower, upper = net(x)
        
        # Check output shapes
        self.assertEqual(F_theta.shape, (batch_size,))
        self.assertEqual(f_theta.shape, (batch_size,))
        self.assertEqual(lower.shape, (batch_size,))
        self.assertEqual(upper.shape, (batch_size,))
        
        # Check F_theta is in (0,1)
        self.assertTrue(torch.all(F_theta >= 0))
        self.assertTrue(torch.all(F_theta <= 1))
        
        # Check f_theta is in {0,1}
        self.assertTrue(torch.all((f_theta == 0) | (f_theta == 1)))
    
    def test_hard_decision_threshold(self):
        """Test f^θ = 1_{[0,∞)} if F^θ ≥ 0.5"""
        net = StoppingNet(input_dim=32, depth=3)
        
        # Create test input
        x = torch.randn(100, 32)
        F_theta, f_theta, _, _ = net(x)
        
        # Check hard decision follows soft probability
        stop_mask = F_theta >= 0.5
        self.assertTrue(torch.all(f_theta[stop_mask] == 1))
        self.assertTrue(torch.all(f_theta[~stop_mask] == 0))

class TestDeepOptimalStopping(unittest.TestCase):
    """Test Deep Optimal Stopping implementation"""
    
    def test_initialization(self):
        """Test initialization with paper parameters"""
        dos = DeepOptimalStopping(max_holding_days=7)
        
        # Check parameters
        self.assertEqual(dos.max_holding_days, 7)
        self.assertEqual(dos.monte_carlo_paths, 8192)
        self.assertIsNone(dos.last_training_utc)
        
        # Check model structure
        self.assertIsInstance(dos.model, StoppingNet)
        self.assertIsNotNone(dos.optimizer)
    
    def test_hard_cap_constraint(self):
        """Test 7-day hard cap on holding period"""
        dos = DeepOptimalStopping(max_holding_days=7)
        
        # Test at max holding days
        position_features = {
            'unrealized_pnl': 0.05,
            'pnl_volatility': 0.02,
            'max_drawdown': -0.01,
            'current_vol': 0.15,
            'vol_change': 0.01,
            'regime_prob': 0.7
        }
        
        decision, confidence, bounds = dos.decide(position_features, holding_days=7)
        
        # Must stop at max holding
        self.assertEqual(decision, "STOP")
        self.assertEqual(confidence, 1.0)
        self.assertEqual(bounds, (0.0, 0.0))
    
    def test_recursive_decision(self):
        """Test recursive stopping decision τ_n"""
        dos = DeepOptimalStopping(max_holding_days=7)
        
        position_features = {
            'unrealized_pnl': 0.03,
            'pnl_volatility': 0.02,
            'max_drawdown': -0.005,
            'current_vol': 0.12,
            'vol_change': -0.01,
            'regime_prob': 0.6
        }
        
        # Test before max holding
        decision, confidence, bounds = dos.decide(position_features, holding_days=3)
        
        # Check decision is STOP or HOLD
        self.assertIn(decision, ["STOP", "HOLD"])
        
        # Check confidence is in [0,1]
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Check bounds are returned
        self.assertEqual(len(bounds), 2)
        lower, upper = bounds
        self.assertIsInstance(lower, float)
        self.assertIsInstance(upper, float)
    
    def test_training_at_utc_midnight(self):
        """Test training only occurs at 00:00 UTC"""
        dos = DeepOptimalStopping(max_holding_days=7)
        
        # Prepare training data
        batch_size = 64
        features = torch.randn(batch_size, 32)
        rewards = torch.randn(batch_size)
        continuation = torch.randn(batch_size)
        holding_times = torch.randint(0, 7, (batch_size,))
        
        # Get current UTC time
        current_utc = datetime.now(timezone.utc)
        
        # Training should only happen at 00:00 UTC
        initial_history_len = len(dos.training_history)
        dos.train_step(features, rewards, continuation, holding_times)
        
        if current_utc.hour != 0:
            # Should not train outside 00:00 UTC
            self.assertEqual(len(dos.training_history), initial_history_len)
        else:
            # Should train at 00:00 UTC
            self.assertEqual(len(dos.training_history), initial_history_len + 1)
    
    def test_bounds_computation(self):
        """Test lower and upper bounds calculation"""
        dos = DeepOptimalStopping(max_holding_days=7)
        
        # Create synthetic paths
        K_L = 1000  # Number of paths
        N = 7  # Time steps
        d = 32  # Feature dimension
        
        paths = np.random.randn(K_L, N, d).astype(np.float32)
        rewards = np.random.randn(K_L, N).astype(np.float32)
        
        # Compute bounds
        lower_bound, upper_bound = dos.compute_bounds(paths, rewards)
        
        # Check bounds are computed
        self.assertIsInstance(lower_bound, float)
        self.assertIsInstance(upper_bound, float)
        
        # Paper property: lower ≤ upper (in expectation)
        # Due to randomness, we allow some tolerance
        self.assertLess(lower_bound - upper_bound, 1.0)
    
    def test_feature_preparation(self):
        """Test feature preparation for stopping decision"""
        dos = DeepOptimalStopping(max_holding_days=7)
        
        position_features = {
            'unrealized_pnl': 0.02,
            'pnl_volatility': 0.015,
            'max_drawdown': -0.008,
            'current_vol': 0.18,
            'vol_change': 0.02,
            'regime_prob': 0.8
        }
        
        features = dos._prepare_features(position_features, holding_days=4)
        
        # Check feature tensor shape
        self.assertEqual(features.shape, (32,))
        
        # Check feature tensor type
        self.assertEqual(features.dtype, torch.float32)
        
        # Check time normalization
        time_feature = features[6].item()
        self.assertAlmostEqual(time_feature, 4/7, places=5)
    
    def test_model_save_load(self):
        """Test model persistence"""
        import tempfile
        import os
        
        dos1 = DeepOptimalStopping(max_holding_days=7)
        
        # Train for one step (if at 00:00 UTC)
        features = torch.randn(64, 32)
        rewards = torch.randn(64)
        continuation = torch.randn(64)
        holding_times = torch.randint(0, 7, (64,))
        dos1.train_step(features, rewards, continuation, holding_times)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pt')
            dos1.save_model(model_path)
            
            # Check files exist
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(model_path.replace('.pt', '_metadata.json')))
            
            # Load into new instance
            dos2 = DeepOptimalStopping(max_holding_days=7)
            success = dos2.load_model(model_path)
            self.assertTrue(success)
            
            # Check parameters match
            self.assertEqual(dos2.max_holding_days, dos1.max_holding_days)
            self.assertEqual(len(dos2.training_history), len(dos1.training_history))
    
    def test_diagnostics(self):
        """Test diagnostic metrics"""
        dos = DeepOptimalStopping(max_holding_days=7)
        
        # Get initial diagnostics
        diag = dos.get_diagnostics()
        
        # Check required fields
        self.assertIn('max_holding_days', diag)
        self.assertIn('monte_carlo_paths', diag)
        self.assertIn('last_training_utc', diag)
        self.assertIn('training_samples', diag)
        
        # Check values
        self.assertEqual(diag['max_holding_days'], 7)
        self.assertEqual(diag['monte_carlo_paths'], 8192)
        self.assertEqual(diag['training_samples'], 0)
        self.assertIsNone(diag['last_training_utc'])

if __name__ == '__main__':
    unittest.main()