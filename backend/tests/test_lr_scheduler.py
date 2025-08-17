"""Unit tests for Large Learning Rate Scheduler
Tests implementation of paper 2507.17748v2: Large Learning Rates Simultaneously
Achieve Robustness to Spurious Correlations and Compressibility
"""
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lr_scheduler import (
    LargeLRConfig, LargeLRScheduler, create_large_lr_scheduler,
    get_small_head_lr_recipe
)

class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class TestLargeLRConfig(unittest.TestCase):
    """Test configuration for large LR scheduler"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = LargeLRConfig()
        
        # Check key parameters from paper
        self.assertEqual(config.base_lr, 0.3)  # Paper recommended
        self.assertEqual(config.warmup_epochs, 5)
        self.assertEqual(config.decay_type, "cosine")
        self.assertEqual(config.weight_decay, 5e-4)
        self.assertEqual(config.gradient_clip, 1.0)
        
    def test_lr_grid(self):
        """Test LR grid initialization"""
        config = LargeLRConfig()
        
        # Should have default grid
        self.assertIsNotNone(config.lr_grid)
        self.assertIn(0.3, config.lr_grid)  # Paper's sweet spot
        self.assertIn(0.01, config.lr_grid)  # Small LR
        self.assertIn(0.8, config.lr_grid)  # Large LR
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = LargeLRConfig(
            base_lr=0.5,
            warmup_epochs=10,
            decay_type="step",
            lr_grid=[0.1, 0.5, 1.0]
        )
        
        self.assertEqual(config.base_lr, 0.5)
        self.assertEqual(config.warmup_epochs, 10)
        self.assertEqual(config.decay_type, "step")
        self.assertEqual(config.lr_grid, [0.1, 0.5, 1.0])

class TestLargeLRScheduler(unittest.TestCase):
    """Test Large LR Scheduler functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.model = SimpleModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.config = LargeLRConfig(
            base_lr=0.3,
            warmup_epochs=5,
            warmup_start_lr=0.01
        )
        self.scheduler = LargeLRScheduler(
            self.optimizer,
            self.config,
            total_epochs=100
        )
        
    def test_initialization(self):
        """Test scheduler initialization"""
        self.assertEqual(self.scheduler.current_epoch, 0)
        self.assertEqual(self.scheduler.current_lr, self.config.warmup_start_lr)
        self.assertEqual(self.scheduler.total_epochs, 100)
        
    def test_warmup_schedule(self):
        """Test linear warmup (Paper Section 4.2)"""
        # Test warmup progression
        lrs = []
        for epoch in range(self.config.warmup_epochs):
            lr = self.scheduler.step(epoch=epoch)
            lrs.append(lr)
            
        # Should start at warmup_start_lr
        self.assertAlmostEqual(lrs[0], self.config.warmup_start_lr, places=4)
        
        # Should increase linearly
        for i in range(1, len(lrs)):
            self.assertGreater(lrs[i], lrs[i-1])
            
        # Should reach base_lr at end of warmup
        final_warmup_lr = self.scheduler.step(epoch=self.config.warmup_epochs - 1)
        expected_lr = self.config.warmup_start_lr + \
                     (self.config.base_lr - self.config.warmup_start_lr) * \
                     ((self.config.warmup_epochs - 1) / self.config.warmup_epochs)
        self.assertAlmostEqual(final_warmup_lr, expected_lr, places=4)
        
    def test_cosine_decay(self):
        """Test cosine annealing after warmup"""
        # Skip warmup
        for epoch in range(self.config.warmup_epochs):
            self.scheduler.step(epoch=epoch)
            
        # Test cosine decay
        lrs = []
        for epoch in range(self.config.warmup_epochs, 20):
            lr = self.scheduler.step(epoch=epoch)
            lrs.append(lr)
            
        # Should decrease following cosine schedule
        for i in range(1, len(lrs)):
            self.assertLessEqual(lrs[i], lrs[i-1])
            
        # Verify cosine formula
        test_epoch = 10
        lr = self.scheduler.step(epoch=test_epoch)
        adjusted_epoch = test_epoch - self.config.warmup_epochs
        adjusted_total = self.scheduler.total_epochs - self.config.warmup_epochs
        expected_lr = self.config.min_lr + 0.5 * (self.config.base_lr - self.config.min_lr) * \
                     (1 + math.cos(math.pi * adjusted_epoch / adjusted_total))
        self.assertAlmostEqual(lr, expected_lr, places=4)
        
    def test_step_decay(self):
        """Test step decay schedule"""
        config = LargeLRConfig(
            base_lr=0.3,
            decay_type="step",
            decay_epochs=[10, 20, 30],
            decay_factor=0.5,
            warmup_epochs=0  # No warmup for simplicity
        )
        scheduler = LargeLRScheduler(self.optimizer, config, total_epochs=50)
        
        # Before first milestone
        lr_5 = scheduler.step(epoch=5)
        self.assertAlmostEqual(lr_5, config.base_lr, places=4)
        
        # After first milestone
        lr_15 = scheduler.step(epoch=15)
        self.assertAlmostEqual(lr_15, config.base_lr * config.decay_factor, places=4)
        
        # After second milestone
        lr_25 = scheduler.step(epoch=25)
        self.assertAlmostEqual(lr_25, config.base_lr * (config.decay_factor ** 2), places=4)
        
    def test_gradient_clipping(self):
        """Test gradient clipping for large LR stability"""
        # Create dummy gradients
        for param in self.model.parameters():
            param.grad = torch.randn_like(param) * 10  # Large gradients
            
        # Apply gradient clipping
        grad_norm = self.scheduler.apply_gradient_clipping(self.model)
        
        # Should return gradient norm
        self.assertIsInstance(grad_norm, float)
        self.assertGreater(grad_norm, 0)
        
        # Check gradients are clipped
        total_norm = 0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.assertLessEqual(total_norm, self.config.gradient_clip + 1e-6)
        
    def test_activation_sparsity(self):
        """Test activation sparsity computation (Paper Section 3.3)"""
        # Create test activations
        activations = torch.randn(100, 50)
        # Make some activations small
        activations[activations.abs() < 0.5] = 0.01
        
        sparsity = self.scheduler.compute_activation_sparsity(activations)
        
        # Should return sparsity ratio
        self.assertIsInstance(sparsity, float)
        self.assertGreaterEqual(sparsity, 0.0)
        self.assertLessEqual(sparsity, 1.0)
        
        # Should track history
        self.assertEqual(len(self.scheduler.sparsity_history), 1)
        
    def test_prediction_confidence(self):
        """Test confidence metrics (Paper Section 4.3)"""
        # Create test logits
        batch_size = 32
        num_classes = 10
        logits = torch.randn(batch_size, num_classes)
        
        metrics = self.scheduler.compute_prediction_confidence(logits)
        
        # Should return confidence metrics
        self.assertIn('mean_confidence', metrics)
        self.assertIn('std_confidence', metrics)
        self.assertIn('high_confidence_ratio', metrics)
        
        # Values should be in valid range
        self.assertGreaterEqual(metrics['mean_confidence'], 0.0)
        self.assertLessEqual(metrics['mean_confidence'], 1.0)
        self.assertGreaterEqual(metrics['high_confidence_ratio'], 0.0)
        self.assertLessEqual(metrics['high_confidence_ratio'], 1.0)
        
    def test_compressibility_estimation(self):
        """Test model compressibility estimation"""
        compressibility = self.scheduler.estimate_compressibility(self.model)
        
        # Should return compressibility metrics
        self.assertIn('sparsity_1e-2', compressibility)
        self.assertIn('sparsity_1e-1', compressibility)
        self.assertIn('total_parameters', compressibility)
        self.assertIn('compressible_ratio', compressibility)
        
        # Values should be in valid range
        self.assertGreaterEqual(compressibility['sparsity_1e-2'], 0.0)
        self.assertLessEqual(compressibility['sparsity_1e-2'], 1.0)
        self.assertGreater(compressibility['total_parameters'], 0)
        
    def test_robustness_indicators(self):
        """Test robustness indicators computation"""
        # Create test features and labels
        batch_size = 32
        feature_dim = 64
        features = torch.randn(batch_size, feature_dim)
        labels = torch.randint(0, 10, (batch_size,))
        
        indicators = self.scheduler.check_robustness_indicators(features, labels)
        
        # Should return robustness metrics
        self.assertIn('feature_sparsity', indicators)
        self.assertIn('feature_std', indicators)
        self.assertIn('class_separation', indicators)
        self.assertIn('feature_utilization', indicators)
        
        # Values should be valid
        self.assertGreaterEqual(indicators['feature_sparsity'], 0.0)
        self.assertLessEqual(indicators['feature_sparsity'], 1.0)
        self.assertGreater(indicators['feature_std'], 0.0)
        
    def test_lr_grid_search(self):
        """Test LR grid search configuration"""
        grid = self.scheduler.get_lr_grid_search_config()
        
        # Should return LR grid
        self.assertIsInstance(grid, list)
        self.assertGreater(len(grid), 0)
        
        # Should contain both small and large LRs
        self.assertLess(min(grid), 0.1)  # Small LR
        self.assertGreater(max(grid), 0.5)  # Large LR
        
    def test_regularization_params(self):
        """Test regularization parameter retrieval"""
        reg_params = self.scheduler.get_current_regularization()
        
        # Should return regularization parameters
        self.assertIn('learning_rate', reg_params)
        self.assertIn('weight_decay', reg_params)
        self.assertIn('gradient_clip', reg_params)
        
        # Values should match config
        self.assertEqual(reg_params['weight_decay'], self.config.weight_decay)
        self.assertEqual(reg_params['gradient_clip'], self.config.gradient_clip)
        
    def test_diagnostics(self):
        """Test comprehensive diagnostics"""
        # Run some steps to generate history
        for epoch in range(10):
            self.scheduler.step(epoch=epoch)
            
            # Add some fake metrics
            if epoch > 5:
                activations = torch.randn(100, 50)
                self.scheduler.compute_activation_sparsity(activations)
                
                logits = torch.randn(32, 10)
                self.scheduler.compute_prediction_confidence(logits)
                
                # Fake gradient norm
                self.scheduler.gradient_norm_history.append(np.random.rand())
                
        diagnostics = self.scheduler.get_diagnostics()
        
        # Should return comprehensive diagnostics
        self.assertIn('current_lr', diagnostics)
        self.assertIn('current_epoch', diagnostics)
        self.assertIn('warmup_complete', diagnostics)
        self.assertIn('schedule_type', diagnostics)
        self.assertIn('base_lr', diagnostics)
        self.assertIn('recent_sparsity', diagnostics)
        self.assertIn('recent_confidence', diagnostics)
        self.assertIn('recent_gradient_norm', diagnostics)
        
        # Warmup should be complete
        self.assertTrue(diagnostics['warmup_complete'])
        
        # Should have valid metrics
        self.assertGreater(diagnostics['recent_sparsity'], 0)
        self.assertGreater(diagnostics['recent_confidence'], 0)
        
    def test_optimizer_lr_update(self):
        """Test that optimizer LR is actually updated"""
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # Step scheduler
        new_lr = self.scheduler.step(epoch=0)
        
        # Optimizer LR should be updated
        self.assertEqual(self.optimizer.param_groups[0]['lr'], new_lr)
        self.assertNotEqual(self.optimizer.param_groups[0]['lr'], initial_lr)

class TestLRRecipe(unittest.TestCase):
    """Test LR recipe for small heads"""
    
    def test_small_head_recipe(self):
        """Test recommended recipe for small network heads"""
        recipe = get_small_head_lr_recipe()
        
        # Should have all required components
        self.assertIn('lr_grid', recipe)
        self.assertIn('recommended_lr', recipe)
        self.assertIn('warmup', recipe)
        self.assertIn('decay', recipe)
        self.assertIn('regularization', recipe)
        self.assertIn('expected_metrics', recipe)
        
        # Check recommended values from paper
        self.assertEqual(recipe['recommended_lr'], 0.3)
        self.assertEqual(recipe['warmup']['epochs'], 5)
        self.assertEqual(recipe['decay']['type'], 'cosine')
        self.assertEqual(recipe['regularization']['weight_decay'], 5e-4)
        self.assertEqual(recipe['regularization']['gradient_clip'], 1.0)
        
        # Check expected metrics
        metrics = recipe['expected_metrics']
        self.assertIn('activation_sparsity', metrics)
        self.assertIn('compressibility', metrics)
        self.assertIn('confidence', metrics)
        self.assertIn('class_separation', metrics)
        self.assertIn('robustness_gain', metrics)

class TestFactoryFunction(unittest.TestCase):
    """Test factory function for creating scheduler"""
    
    def test_create_scheduler(self):
        """Test scheduler creation via factory"""
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        
        scheduler = create_large_lr_scheduler(
            optimizer,
            base_lr=0.5,
            total_epochs=50
        )
        
        # Should create valid scheduler
        self.assertIsInstance(scheduler, LargeLRScheduler)
        self.assertEqual(scheduler.config.base_lr, 0.5)
        self.assertEqual(scheduler.total_epochs, 50)
        
        # Should have default values for unspecified params
        self.assertEqual(scheduler.config.warmup_epochs, 5)
        self.assertEqual(scheduler.config.decay_type, "cosine")

if __name__ == '__main__':
    unittest.main()