"""
Test suite for signature features and positioning calculations
"""

import unittest
import numpy as np
from features_signature import SignatureFeatures
from features_positioning import PositioningFeatures

class TestSignatureFeatures(unittest.TestCase):
    """Test signature-based volatility features"""
    
    def setUp(self):
        self.sf = SignatureFeatures()
        # Generate synthetic price path
        np.random.seed(42)
        t = np.linspace(0, 1, 100)
        prices = 100 * np.exp(0.2 * t + 0.3 * np.random.randn(100).cumsum() / 10)
        self.path = np.column_stack([t, prices])
    
    def test_compute_signatures(self):
        """Test signature computation"""
        sigs = self.sf.compute_signatures(self.path)
        
        # Check signature shape
        self.assertEqual(len(sigs), 6)
        
        # Check signature bounds
        for sig in sigs.values():
            self.assertIsInstance(sig, float)
            self.assertGreaterEqual(sig, -10)
            self.assertLessEqual(sig, 10)
    
    def test_normalize_features(self):
        """Test feature normalization"""
        sigs = self.sf.compute_signatures(self.path)
        normalized = self.sf.normalize_features(sigs)
        
        # Check normalization bounds
        for val in normalized.values():
            self.assertGreaterEqual(val, -1)
            self.assertLessEqual(val, 1)

class TestPositioningFeatures(unittest.TestCase):
    """Test positioning and market microstructure features"""
    
    def setUp(self):
        self.pf = PositioningFeatures()
    
    def test_funding_features(self):
        """Test funding rate features"""
        features = self.pf.compute_funding_features(
            funding_rate=0.0001,
            historical_funding=[0.0001, 0.0002, 0.0001, -0.0001, 0.0000]
        )
        
        self.assertIn('funding_ma_7d', features)
        self.assertIn('funding_momentum', features)
        self.assertIn('funding_percentile', features)
    
    def test_volume_features(self):
        """Test volume profile features"""
        features = self.pf.compute_volume_features(
            volumes=[100, 150, 200, 180, 220, 190, 210],
            prices=[100, 101, 102, 101.5, 103, 102.5, 103.5]
        )
        
        self.assertIn('volume_ratio', features)
        self.assertIn('volume_trend', features)
        self.assertIn('vwap_deviation', features)
    
    def test_oi_features(self):
        """Test open interest features"""
        features = self.pf.compute_oi_features(
            oi=1000000,
            oi_history=[900000, 920000, 950000, 980000, 1000000]
        )
        
        self.assertIn('oi_change_rate', features)
        self.assertIn('oi_acceleration', features)
        self.assertIn('oi_percentile', features)

if __name__ == '__main__':
    unittest.main()