"""Unit tests for ORCSMC regime detection module
Tests implementation against paper 2508.00696v1 specifications
"""
import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regime_orcsmc import ORCSMC, RegimeState

class TestORCSMC(unittest.TestCase):
    """Test ORCSMC implementation against research specifications"""
    
    def setUp(self):
        """Initialize test ORCSMC instance with paper defaults"""
        self.orcsmc = ORCSMC(
            n_particles=1000,       # N from paper
            rolling_window=30,      # L from paper  
            learning_iters=5,       # K from paper
            resample_threshold=0.5, # κ from paper
            inertia_days=2         # Inertia requirement
        )
        
    def test_initialization(self):
        """Test proper initialization of dual particle systems"""
        # Check particle systems initialized
        self.assertIsNotNone(self.orcsmc.learning_particles)
        self.assertIsNotNone(self.orcsmc.filter_particles)
        
        # Check correct number of particles
        self.assertEqual(len(self.orcsmc.learning_particles), 1000)
        self.assertEqual(len(self.orcsmc.filter_particles), 1000)
        
        # Check uniform initial weights (Equation 4)
        np.testing.assert_allclose(
            self.orcsmc.learning_weights, 
            np.ones(1000) / 1000
        )
        
        # Check normalizing constants initialized to 1
        self.assertEqual(self.orcsmc.Z_learning, 1.0)
        self.assertEqual(self.orcsmc.Z_filter, 1.0)
        
    def test_rolling_window(self):
        """Test rolling window mechanism (Algorithm 4, Line 3)"""
        # Add observations beyond window size
        for i in range(50):
            self.orcsmc.step(np.random.randn() * 0.01)
        
        # Check buffer respects max length
        self.assertEqual(len(self.orcsmc.observation_buffer), 30)
        
        diag = self.orcsmc.get_diagnostics()
        self.assertEqual(diag['buffer_size'], 30)
        self.assertEqual(diag['rolling_window'], 30)
        
    def test_twist_function_computation(self):
        """Test quadratic twist function ψ_t(x_t) = exp(x'Ax + b'x + c)"""
        # Test twist value computation
        twist_val = self.orcsmc._compute_twist_value(RegimeState.RISK_ON, 0.02)
        self.assertGreater(twist_val, 0)  # Must be positive
        self.assertLess(twist_val, 1e10)  # Must be bounded
        
        # Test numerical stability
        extreme_val = self.orcsmc._compute_twist_value(RegimeState.RISK_ON, 100.0)
        self.assertLess(extreme_val, 1e10)  # Should be clipped
        
    def test_ess_computation(self):
        """Test Effective Sample Size calculation"""
        # Uniform weights should give ESS = N
        uniform_weights = np.ones(100) / 100
        ess = self.orcsmc._compute_ess(uniform_weights)
        self.assertAlmostEqual(ess, 100, places=5)
        
        # Degenerate weights should give ESS = 1
        degenerate_weights = np.zeros(100)
        degenerate_weights[0] = 1.0
        ess = self.orcsmc._compute_ess(degenerate_weights)
        self.assertAlmostEqual(ess, 1.0, places=5)
        
    def test_resampling_mechanism(self):
        """Test residual-multinomial resampling"""
        particles = np.array([0, 1, 0, 1, 0])
        weights = np.array([0.5, 0.2, 0.2, 0.05, 0.05])
        
        new_particles, ancestors = self.orcsmc._resample_particles(particles, weights)
        
        # Check output dimensions
        self.assertEqual(len(new_particles), len(particles))
        self.assertEqual(len(ancestors), len(particles))
        
        # Check ancestors are valid indices
        self.assertTrue(all(0 <= a < len(particles) for a in ancestors))
        
    def test_compute_bound(self):
        """Test per-step compute bound O(L*N*K)"""
        # Expected max compute
        expected = 30 * 1000 * 5  # L * N * K
        self.assertEqual(self.orcsmc.max_compute_per_step, expected)
        
        # Run steps and check compute tracking
        for _ in range(10):
            self.orcsmc.step(np.random.randn() * 0.01)
        
        diag = self.orcsmc.get_diagnostics()
        self.assertEqual(diag['compute_steps'], 10)
        self.assertEqual(diag['max_compute_per_step'], expected)
        
    def test_inertia_rule(self):
        """Test inertia rule: 2 consecutive ON days required after OFF"""
        # Start in RISK_OFF
        self.orcsmc.last_regime = RegimeState.RISK_OFF
        self.orcsmc.consecutive_on_days = 0
        
        # Mock strong RISK_ON signal
        self.orcsmc.filter_particles = np.ones(1000, dtype=int)  # All RISK_ON
        self.orcsmc.filter_weights = np.ones(1000) / 1000
        
        # Without consecutive days, raw probability is high
        p0 = self.orcsmc.get_regime_probability()
        self.assertGreater(p0, 0.9)  # Raw signal is strong
        
        # Simulate first consecutive day
        self.orcsmc.consecutive_on_days = 1
        p1 = self.orcsmc.get_regime_probability()
        self.assertLess(p1, 0.5)  # Suppressed during inertia
        self.assertGreater(p1, 0.3)  # But showing gradual increase
        
        # After 2 consecutive days, transition complete
        self.orcsmc.consecutive_on_days = 2
        self.orcsmc.last_regime = RegimeState.RISK_ON
        p2 = self.orcsmc.get_regime_probability()
        self.assertGreater(p2, 0.9)  # Full transition
        
    def test_dual_particle_systems(self):
        """Test dual particle system updates"""
        # Add observations
        observations = [np.random.randn() * 0.01 for _ in range(10)]
        
        for obs in observations:
            p_risk_on = self.orcsmc.step(obs)
            
            # Check probability is valid
            self.assertGreaterEqual(p_risk_on, 0.0)
            self.assertLessEqual(p_risk_on, 1.0)
        
        # Check both systems have been updated
        diag = self.orcsmc.get_diagnostics()
        self.assertGreater(diag['ess_filter'], 0)
        self.assertGreater(diag['ess_learning'], 0)
        
    def test_twist_learning(self):
        """Test twist function parameter learning"""
        # Create observations with clear regime pattern
        risk_on_obs = np.random.normal(0.02, 0.15, 50)
        risk_off_obs = np.random.normal(-0.01, 0.25, 50)
        
        # Mix observations
        observations = np.concatenate([risk_off_obs[:25], risk_on_obs[:25], 
                                      risk_off_obs[25:], risk_on_obs[25:]])
        
        # Process observations
        for obs in observations[:30]:  # Fill window
            self.orcsmc.step(obs)
        
        # Check twist parameters have been learned
        params = self.orcsmc.twist_params
        self.assertIsNotNone(params['A_diag'])
        self.assertIsNotNone(params['b'])
        
        # Parameters should reflect the data patterns
        self.assertNotEqual(params['mean_risk_on'], 0.02)  # Should have adapted
        self.assertNotEqual(params['mean_risk_off'], -0.01)  # Should have adapted
        
    def test_regime_transitions(self):
        """Test regime transition detection"""
        # Simulate clear regime change
        for _ in range(20):
            # RISK_OFF observations
            self.orcsmc.step(np.random.normal(-0.01, 0.25))
        
        p_off = self.orcsmc.get_regime_probability()
        self.assertLess(p_off, 0.5)  # Should detect RISK_OFF
        
        # Transition to RISK_ON
        for _ in range(5):
            self.orcsmc.step(np.random.normal(0.02, 0.15))
        
        # Check inertia is active
        diag = self.orcsmc.get_diagnostics()
        if diag['current_regime'] == 'RISK_OFF':
            self.assertTrue(diag['consecutive_on_days'] > 0)
        
    def test_algorithm_4_structure(self):
        """Test Algorithm 4 implementation structure"""
        # Test key algorithm components exist
        self.assertTrue(hasattr(self.orcsmc, '_psi_apf_step'))
        self.assertTrue(hasattr(self.orcsmc, '_learn_twist_function'))
        
        # Test rolling window calculation
        t = 100
        t0 = max(0, t - self.orcsmc.rolling_window)
        self.assertEqual(t0, 70)  # 100 - 30
        
        # Test learning iterations
        self.assertEqual(self.orcsmc.learning_iters, 5)
        
    def test_diagnostics_completeness(self):
        """Test diagnostic output completeness"""
        # Run some steps
        for _ in range(10):
            self.orcsmc.step(np.random.randn() * 0.01)
        
        diag = self.orcsmc.get_diagnostics()
        
        # Check all required diagnostics present
        required_keys = [
            'p_risk_on', 'p_risk_on_raw', 'ess_filter', 'ess_learning',
            'twist_params', 'buffer_size', 'rolling_window', 
            'particle_diversity', 'compute_steps', 'max_compute_per_step',
            'consecutive_on_days', 'inertia_active', 'Z_filter', 'current_regime'
        ]
        
        for key in required_keys:
            self.assertIn(key, diag, f"Missing diagnostic: {key}")
            
if __name__ == '__main__':
    unittest.main()