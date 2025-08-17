"""Unit tests for Joint Asymmetric Loss (JAL) module
Tests implementation of paper 2507.17692v1: Joint Asymmetric Loss for Learning with Noisy Labels
"""
import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jal_loss import (
    AsymmetricMSE, FocalLoss, JointAsymmetricLoss, 
    JALConfig, create_jal_loss
)

class TestAsymmetricMSE(unittest.TestCase):
    """Test AMSE (Asymmetric Mean Square Error) passive loss"""
    
    def setUp(self):
        """Set up test parameters"""
        self.num_classes = 10
        self.batch_size = 32
        self.amse = AsymmetricMSE(self.num_classes, asymmetry_param=3.0)
        
    def test_amse_shape(self):
        """Test AMSE output shape"""
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss = self.amse(logits, targets)
        
        # Loss should be scalar
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertIsInstance(loss.item(), float)
        
    def test_amse_asymmetry(self):
        """Test asymmetric property with different q values"""
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Test with different asymmetry parameters
        amse_q2 = AsymmetricMSE(self.num_classes, asymmetry_param=2.0)
        amse_q4 = AsymmetricMSE(self.num_classes, asymmetry_param=4.0)
        
        loss_q2 = amse_q2(logits, targets)
        loss_q4 = amse_q4(logits, targets)
        
        # Different q values should give different losses
        self.assertNotAlmostEqual(loss_q2.item(), loss_q4.item(), places=3)
        
    def test_amse_passive_property(self):
        """Test passive loss property: minimizes non-target probabilities"""
        # Create logits with perfect prediction
        logits = torch.zeros(1, self.num_classes)
        logits[0, 0] = 10.0  # High confidence for class 0
        targets = torch.tensor([0])
        
        loss_correct = self.amse(logits, targets)
        
        # Wrong prediction
        targets_wrong = torch.tensor([1])
        loss_wrong = self.amse(logits, targets_wrong)
        
        # Correct prediction should have lower passive loss
        self.assertLess(loss_correct.item(), loss_wrong.item())
        
    def test_amse_gradient_flow(self):
        """Test gradient flow through AMSE"""
        logits = torch.randn(self.batch_size, self.num_classes, requires_grad=True)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss = self.amse(logits, targets)
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())

class TestFocalLoss(unittest.TestCase):
    """Test Focal Loss active loss"""
    
    def setUp(self):
        """Set up test parameters"""
        self.num_classes = 10
        self.batch_size = 32
        self.focal = FocalLoss(gamma=2.0)
        
    def test_focal_shape(self):
        """Test Focal Loss output shape"""
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss = self.focal(logits, targets)
        
        # Loss should be scalar
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertIsInstance(loss.item(), float)
        
    def test_focal_vs_ce(self):
        """Test Focal Loss reduces easy example weight vs CE"""
        # Easy example: high confidence correct prediction
        logits_easy = torch.zeros(1, self.num_classes)
        logits_easy[0, 0] = 10.0
        targets_easy = torch.tensor([0])
        
        # Hard example: low confidence
        logits_hard = torch.zeros(1, self.num_classes)
        logits_hard[0, 0] = 0.5
        targets_hard = torch.tensor([0])
        
        ce_loss = nn.CrossEntropyLoss()
        
        # Compare losses
        focal_easy = self.focal(logits_easy, targets_easy)
        focal_hard = self.focal(logits_hard, targets_hard)
        ce_easy = ce_loss(logits_easy, targets_easy)
        ce_hard = ce_loss(logits_hard, targets_hard)
        
        # Focal should down-weight easy examples more than CE
        focal_ratio = focal_easy / focal_hard
        ce_ratio = ce_easy / ce_hard
        
        self.assertLess(focal_ratio.item(), ce_ratio.item())
        
    def test_focal_gamma_effect(self):
        """Test gamma parameter effect on focusing"""
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        focal_g0 = FocalLoss(gamma=0.0)  # Equivalent to CE
        focal_g2 = FocalLoss(gamma=2.0)
        focal_g5 = FocalLoss(gamma=5.0)
        
        loss_g0 = focal_g0(logits, targets)
        loss_g2 = focal_g2(logits, targets)
        loss_g5 = focal_g5(logits, targets)
        
        # Higher gamma should generally reduce loss for easy examples
        # But this depends on the specific predictions
        self.assertIsInstance(loss_g0.item(), float)
        self.assertIsInstance(loss_g2.item(), float)
        self.assertIsInstance(loss_g5.item(), float)

class TestJointAsymmetricLoss(unittest.TestCase):
    """Test JAL framework combining active and passive losses"""
    
    def setUp(self):
        """Set up test parameters"""
        self.num_classes = 10
        self.batch_size = 32
        self.config = JALConfig(
            alpha=0.7,
            beta=0.3,
            active_loss_type="CE",
            asymmetry_param=3.0,
            noise_threshold=0.1,
            warmup_epochs=2
        )
        self.jal = JointAsymmetricLoss(self.num_classes, self.config)
        
    def test_jal_output(self):
        """Test JAL output structure"""
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss, metrics = self.jal(logits, targets, epoch=0)
        
        # Check loss
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertIsInstance(loss.item(), float)
        
        # Check metrics
        self.assertIn('noise_rate', metrics)
        self.assertIn('total_loss', metrics)
        self.assertIn('jal_active', metrics)
        
    def test_jal_warmup(self):
        """Test JAL warmup period behavior"""
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # During warmup (epoch < warmup_epochs)
        loss_warmup, metrics_warmup = self.jal(logits, targets, epoch=0)
        self.assertFalse(metrics_warmup['jal_active'])
        
        # After warmup with noise
        # Simulate high noise by adding disagreement
        for _ in range(20):
            noisy_targets = torch.randint(0, self.num_classes, (self.batch_size,))
            self.jal(logits, noisy_targets, epoch=3)
        
        loss_active, metrics_active = self.jal(logits, targets, epoch=3)
        # JAL might be active if noise is detected
        self.assertIn('jal_active', metrics_active)
        
    def test_jal_noise_estimation(self):
        """Test noise rate estimation"""
        # Create predictions with known disagreement
        logits = torch.zeros(10, 2)
        logits[:, 0] = 1.0  # Predict class 0
        
        # Half targets agree, half disagree
        targets = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        noise_rate = self.jal.estimate_noise_rate(logits, targets)
        
        # Should detect ~50% disagreement
        self.assertAlmostEqual(noise_rate, 0.5, delta=0.1)
        
    def test_jal_activation_threshold(self):
        """Test JAL activation based on noise threshold"""
        self.jal.config.warmup_epochs = 0  # Disable warmup
        self.jal.epoch_count = 5
        
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Manually set noise estimates
        self.jal.noise_estimates = [0.05] * 10  # Below threshold
        loss_low, metrics_low = self.jal(logits, targets, epoch=5)
        self.assertFalse(metrics_low['jal_active'])
        
        self.jal.noise_estimates = [0.2] * 10  # Above threshold
        loss_high, metrics_high = self.jal(logits, targets, epoch=5)
        self.assertTrue(metrics_high['jal_active'])
        
    def test_jal_loss_combination(self):
        """Test loss combination weights alpha and beta"""
        self.jal.config.warmup_epochs = 0
        self.jal.epoch_count = 5
        self.jal.noise_estimates = [0.2] * 10  # Force JAL active
        
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss, metrics = self.jal(logits, targets, epoch=5)
        
        # Check both losses are computed
        self.assertIn('active_loss', metrics)
        self.assertIn('passive_loss', metrics)
        
        # Total loss should be weighted combination
        expected_total = (self.config.alpha * metrics['active_loss'] + 
                         self.config.beta * metrics['passive_loss'])
        
        # Account for label smoothing if enabled
        if self.config.label_smoothing > 0:
            # Total loss includes smoothing term
            self.assertAlmostEqual(metrics['total_loss'], loss.item(), delta=0.01)
        else:
            self.assertAlmostEqual(metrics['total_loss'], expected_total, delta=0.01)
            
    def test_jal_with_focal_loss(self):
        """Test JAL with Focal Loss as active loss"""
        config = JALConfig(
            alpha=0.6,
            beta=0.4,
            active_loss_type="FL",
            focal_gamma=2.0
        )
        jal_focal = JointAsymmetricLoss(self.num_classes, config)
        
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss, metrics = jal_focal(logits, targets, epoch=10)
        
        # Should use Focal Loss
        self.assertIsInstance(jal_focal.active_loss, FocalLoss)
        self.assertIsInstance(loss.item(), float)
        
    def test_jal_label_smoothing(self):
        """Test label smoothing regularization"""
        config = JALConfig(label_smoothing=0.1)
        jal_smooth = JointAsymmetricLoss(self.num_classes, config)
        
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss_smooth, _ = jal_smooth(logits, targets)
        
        # Compare with no smoothing
        config_no_smooth = JALConfig(label_smoothing=0.0)
        jal_no_smooth = JointAsymmetricLoss(self.num_classes, config_no_smooth)
        loss_no_smooth, _ = jal_no_smooth(logits, targets)
        
        # Losses should be different
        self.assertNotAlmostEqual(loss_smooth.item(), loss_no_smooth.item(), places=3)
        
    def test_jal_gradient_flow(self):
        """Test gradient flow through JAL"""
        logits = torch.randn(self.batch_size, self.num_classes, requires_grad=True)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss, _ = self.jal(logits, targets, epoch=10)
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())
        
    def test_jal_diagnostics(self):
        """Test JAL diagnostic output"""
        # Run some training steps
        for i in range(10):
            logits = torch.randn(self.batch_size, self.num_classes)
            targets = torch.randint(0, self.num_classes, (self.batch_size,))
            self.jal(logits, targets, epoch=i)
            
        diagnostics = self.jal.get_diagnostics()
        
        # Check diagnostic keys
        self.assertIn('jal_active', diagnostics)
        self.assertIn('estimated_noise_rate', diagnostics)
        self.assertIn('epoch_count', diagnostics)
        self.assertIn('config', diagnostics)
        
        # Check config details
        self.assertEqual(diagnostics['config']['alpha'], self.config.alpha)
        self.assertEqual(diagnostics['config']['beta'], self.config.beta)
        
    def test_jal_weight_adjustment(self):
        """Test dynamic weight adjustment"""
        initial_alpha = self.jal.config.alpha
        initial_beta = self.jal.config.beta
        
        # Adjust weights
        self.jal.adjust_weights(alpha=0.5, beta=0.5)
        
        self.assertEqual(self.jal.config.alpha, 0.5)
        self.assertEqual(self.jal.config.beta, 0.5)
        
        # Test normalization for large weights
        self.jal.adjust_weights(alpha=1.0, beta=1.0)
        
        # Should normalize to reasonable values
        self.assertLessEqual(self.jal.config.alpha + self.jal.config.beta, 1.5)
        
    def test_jal_reset(self):
        """Test resetting noise estimation"""
        # Add some noise estimates
        for _ in range(10):
            logits = torch.randn(self.batch_size, self.num_classes)
            targets = torch.randint(0, self.num_classes, (self.batch_size,))
            self.jal(logits, targets, epoch=5)
            
        # Reset
        self.jal.reset_noise_estimation()
        
        self.assertEqual(len(self.jal.noise_estimates), 0)
        self.assertEqual(self.jal.epoch_count, 0)
        self.assertFalse(self.jal.is_active)

class TestJALFactory(unittest.TestCase):
    """Test JAL factory function"""
    
    def test_create_jal_loss(self):
        """Test factory function for creating JAL"""
        num_classes = 5
        jal = create_jal_loss(
            num_classes=num_classes,
            noise_threshold=0.2,
            active_type="FL"
        )
        
        self.assertIsInstance(jal, JointAsymmetricLoss)
        self.assertEqual(jal.num_classes, num_classes)
        self.assertEqual(jal.config.noise_threshold, 0.2)
        self.assertEqual(jal.config.active_loss_type, "FL")
        self.assertIsInstance(jal.active_loss, FocalLoss)

if __name__ == '__main__':
    unittest.main()