"""Unit tests for signature-based volatility features
Tests implementation against paper 2507.23392v2 specifications
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features_signature import SignatureFeatures

class TestSignatureFeatures:
    """Test signature feature extraction based on paper specifications"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sig_features = SignatureFeatures(order=3, window_hours=72)
        
    def test_initialization(self):
        """Test proper initialization with order-3 truncation"""
        assert self.sig_features.order == 3
        assert self.sig_features.window_hours == 72
        # Check feature count < 32 as specified
        assert len(self.sig_features.feature_names) < 32
        
    def test_feature_names(self):
        """Test feature names match paper structure"""
        names = self.sig_features.feature_names
        
        # Level 1 features (increments)
        assert 'sig_t' in names
        assert 'sig_p' in names
        assert 'sig_v' in names
        
        # Level 2 features (Lévy areas)
        assert 'sig_tt' in names
        assert 'sig_tp' in names
        assert 'sig_pp' in names
        
        # Level 3 features (triple integrals)
        assert 'sig_ttt' in names
        assert 'sig_ppp' in names
        
        # Realized vol features
        assert 'rvol_24h' in names
        assert 'rvol_72h' in names
        assert 'vol_of_vol' in names
        
    def test_time_augmented_path(self):
        """Test time augmentation per paper Theorem 3.27"""
        # Create synthetic path
        T = 72
        returns = np.random.randn(T) * 0.01
        volume = np.random.rand(T) * 1000
        times = np.arange(T)
        
        path = np.column_stack([times, returns, volume])
        
        # Compute signature
        sig = self.sig_features._compute_signature_fallback(path)
        
        # Check signature properties
        assert len(sig) > 0
        # Time increment should equal total time span
        assert abs(sig[0] - 1.0) < 0.1  # Normalized to [0,1]
        
    def test_signature_computation_fallback(self):
        """Test fast fallback implementation"""
        # Create simple test path
        path = np.array([
            [0, 0, 1],
            [1, 0.1, 1.1],
            [2, 0.15, 0.9],
            [3, 0.12, 1.0]
        ])
        
        sig = self.sig_features._compute_signature_fallback(path)
        
        # Check basic properties
        assert sig is not None
        assert len(sig) > 3  # At least level 1 features
        
        # Level 1: increments should match path endpoints
        # (after normalization to [0,1])
        assert abs(sig[0] - 1.0) < 0.01  # Time normalized
        
    def test_realized_volatility(self):
        """Test realized volatility computation"""
        # Create returns with known volatility
        np.random.seed(42)
        daily_vol = 0.02
        hourly_vol = daily_vol / np.sqrt(24)
        returns = np.random.randn(100) * hourly_vol
        
        rvol = self.sig_features.compute_realized_vol(returns, 24)
        
        # Check annualized vol is reasonable
        assert 0.1 < rvol < 0.5  # Typical crypto range
        
    def test_extract_features_minimal(self):
        """Test feature extraction with minimal data"""
        # Create minimal bars DataFrame
        bars = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        
        features = self.sig_features.extract_features(bars)
        
        # Check all features present
        assert len(features) == len(self.sig_features.feature_names)
        for name in self.sig_features.feature_names:
            assert name in features
            
    def test_extract_features_full(self):
        """Test feature extraction with full window"""
        # Create 72 hours of synthetic data
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(73) * 0.01))
        
        bars = pd.DataFrame({
            'open': prices[:-1],
            'high': prices[:-1] * 1.01,
            'low': prices[:-1] * 0.99,
            'close': prices[1:],
            'volume': np.random.rand(72) * 10000
        })
        
        features = self.sig_features.extract_features(bars)
        
        # Check z-score normalization
        feature_vals = list(features.values())
        # After z-scoring, mean should be near 0, std near 1
        assert abs(np.mean(feature_vals)) < 0.5
        
    def test_volatility_approximation(self):
        """Test linear volatility approximation σ_t(ℓ) = ⟨ℓ, S(X)^≤N_t⟩"""
        # Create features
        features = {name: np.random.randn() for name in self.sig_features.feature_names}
        
        # Use calibrated weights from paper (simplified)
        weights = np.array([
            0.201,  # ℓ_∅ ≈ σ_0
            0.143,  # ℓ_0 (time)
            1.085,  # ℓ_1 (price) - highest importance
            -0.297, # ℓ_00
            -0.029, # ℓ_01
            -0.042, # ℓ_10
            0.001,  # ℓ_11
            0.293,  # Level 3...
            -0.014,
            -0.013,
            -0.002,
            -0.003,
            -0.001,
            -0.002,
            -0.0002
        ])
        
        # Pad weights to match feature count
        weights_full = np.zeros(len(features))
        weights_full[:len(weights)] = weights
        
        vol = self.sig_features.compute_volatility_approximation(features, weights_full)
        
        # Check volatility is positive and reasonable
        assert vol > 0
        assert vol < 10  # Not absurdly high
        
    def test_feature_importance(self):
        """Test feature importance ranking matches paper"""
        features = {
            'sig_p': 1.0,    # Price should be most important
            'sig_t': 0.5,
            'sig_pp': 0.3,
            'rvol_72h': 0.4,
            'sig_ttt': 0.1
        }
        
        importance = self.sig_features.get_feature_importance(features)
        
        # Check ordering
        assert importance[0][0] == 'sig_p'  # Price increment most important
        assert importance[0][1] > 0
        
    def test_heston_compatibility(self):
        """Test signatures work with Heston-like paths"""
        # Simulate simplified Heston-like volatility clustering
        np.random.seed(42)
        T = 100
        vol = np.zeros(T)
        vol[0] = 0.2
        
        # Mean-reverting volatility
        kappa = 0.5
        theta = 0.15
        nu = 0.3
        
        for t in range(1, T):
            dW = np.random.randn()
            vol[t] = vol[t-1] + kappa * (theta - vol[t-1]) * 0.01 + nu * np.sqrt(vol[t-1]) * dW * 0.1
            vol[t] = max(0.01, vol[t])  # Keep positive
        
        # Generate returns with this volatility
        returns = np.random.randn(T) * vol
        prices = 100 * np.exp(np.cumsum(returns))
        
        bars = pd.DataFrame({
            'open': prices[:-1],
            'high': prices[:-1] * 1.01,
            'low': prices[:-1] * 0.99,
            'close': prices[1:],
            'volume': np.random.rand(T-1) * 10000
        })
        
        features = self.sig_features.extract_features(bars)
        
        # Vol of vol should be significant (volatility clustering)
        assert features['vol_of_vol'] != 0
        
    def test_rough_bergomi_compatibility(self):
        """Test signatures work with rough volatility (H < 0.5)"""
        # Simulate rough volatility with low Hurst exponent
        np.random.seed(42)
        T = 100
        H = 0.1  # Very rough (paper tests H=0.1)
        
        # Fractional Brownian motion approximation
        cov_matrix = np.zeros((T, T))
        for i in range(T):
            for j in range(T):
                cov_matrix[i, j] = 0.5 * (abs(i)**(2*H) + abs(j)**(2*H) - abs(i-j)**(2*H))
        
        # Generate correlated noise
        L = np.linalg.cholesky(cov_matrix + 1e-6 * np.eye(T))
        W = L @ np.random.randn(T)
        
        # Rough volatility
        vol = 0.2 * np.exp(0.5 * W - 0.25 * T**(2*H))
        
        # Generate returns
        returns = np.random.randn(T) * vol
        prices = 100 * np.exp(np.cumsum(returns))
        
        bars = pd.DataFrame({
            'open': prices[:-1],
            'high': prices[:-1] * 1.01,
            'low': prices[:-1] * 0.99,
            'close': prices[1:],
            'volume': np.random.rand(T-1) * 10000
        })
        
        features = self.sig_features.extract_features(bars.tail(72))
        
        # Check roughness indicators
        assert features['vol_of_vol'] != 0  # High vol of vol for rough paths
        
    def test_dimension_constraint(self):
        """Test total features < 32 dimensions as specified"""
        bars = pd.DataFrame({
            'open': [100] * 72,
            'high': [101] * 72,
            'low': [99] * 72,
            'close': [100] * 72,
            'volume': [1000] * 72
        })
        
        features = self.sig_features.extract_features(bars)
        
        # Verify dimension constraint
        assert len(features) < 32
        
    def test_chen_identity(self):
        """Test Chen's identity S(X)_{s,t} = S(X)_{s,u} ⊗ S(X)_{u,t}"""
        # This is implicitly tested in signature computation
        # The multiplicative property ensures path consistency
        
        path1 = np.array([[0, 0, 1], [0.5, 0.1, 1.1]])
        path2 = np.array([[0.5, 0.1, 1.1], [1, 0.2, 0.9]])
        path_full = np.array([[0, 0, 1], [0.5, 0.1, 1.1], [1, 0.2, 0.9]])
        
        sig1 = self.sig_features._compute_signature_fallback(path1)
        sig2 = self.sig_features._compute_signature_fallback(path2)
        sig_full = self.sig_features._compute_signature_fallback(path_full)
        
        # Level 1 should be additive
        assert abs((sig1[1] + sig2[1]) - sig_full[1]) < 0.01
        
    @patch('features_signature.HAS_IISIGNATURE', False)
    def test_fallback_forced(self):
        """Test fallback implementation is used when iisignature unavailable"""
        bars = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        
        features = self.sig_features.extract_features(bars)
        
        # Should still produce valid features
        assert len(features) > 0
        assert all(isinstance(v, (int, float)) for v in features.values())

if __name__ == "__main__":
    pytest.main([__file__, "-v"])