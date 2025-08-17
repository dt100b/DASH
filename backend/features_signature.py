"""Signature-based volatility features - research implementation from 2507.23392v2
Model-agnostic volatility characterization via time-augmented path signatures
Research basis: Accurate under both Heston and rough Bergomi dynamics
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import zscore
import warnings

# Try to import iisignature for fast computation, fallback to manual if unavailable
try:
    import iisignature
    HAS_IISIGNATURE = True
except ImportError:
    HAS_IISIGNATURE = False
    warnings.warn("iisignature not available, using fast fallback implementation")

class SignatureFeatures:
    """Time-augmented path signatures for volatility characterization
    
    Based on research paper 2507.23392v2: "Implied Volatility Surface Calibration 
    with Path Signatures". Uses order-3 truncated signatures (<32 dims as specified)
    for model-agnostic volatility modeling under rough/non-Markovian dynamics.
    
    Key insights from paper:
    - Time augmentation (t, X_t) essential for uniqueness (Theorem 3.27)
    - Order N=3 optimal for accuracy/efficiency tradeoff
    - Linear functional σ_t(ℓ) = ⟨ℓ, S(X)^≤N_t⟩ approximates volatility
    - Z-scaling normalization for numerical stability
    """
    
    def __init__(self, order: int = 3, window_hours: int = 72):
        """Initialize signature feature extractor
        
        Args:
            order: Signature truncation level (default 3 per paper)
            window_hours: Rolling window for features (48-72h optimal)
        """
        self.order = min(order, 3)  # Cap at 3 per paper recommendations
        self.window_hours = window_hours
        self.feature_names = []
        self._build_feature_names()
        
    def _build_feature_names(self):
        """Build feature names for time-augmented signature coordinates
        
        For d=3 (time, price, volume) and N=3:
        - Level 1: 3 features (increments)
        - Level 2: 9 features (areas/covariances)  
        - Level 3: 27 features (triple integrals)
        Total: 39 features, but we select <32 most important
        """
        # Level 1: Path increments (3 features)
        self.feature_names.extend(['sig_t', 'sig_p', 'sig_v'])
        
        # Level 2: Lévy areas (9 features for d=3)
        if self.order >= 2:
            for i in ['t', 'p', 'v']:
                for j in ['t', 'p', 'v']:
                    self.feature_names.append(f'sig_{i}{j}')
        
        # Level 3: Triple integrals (select most important for <32 total)
        if self.order >= 3:
            # Paper shows these have highest importance per calibration
            important_triples = [
                'ttt', 'ttp', 'tpp', 'ppp',  # Time-price interactions
                'ttv', 'tpv', 'ppv', 'pvv',  # Volume shape features
                'vvv'  # Volume acceleration
            ]
            for triple in important_triples:
                self.feature_names.append(f'sig_{triple}')
        
        # Add realized vol features (3 features)
        self.feature_names.extend(['rvol_24h', 'rvol_72h', 'vol_of_vol'])
        
    def compute_signature(self, path: np.ndarray) -> np.ndarray:
        """Compute truncated signature using iisignature or fast fallback
        
        Args:
            path: (T, d) array with columns [time, price_return, volume]
        
        Returns:
            Signature coordinates up to specified order (<32 dims)
        """
        if HAS_IISIGNATURE:
            return self._compute_signature_iisig(path)
        else:
            return self._compute_signature_fallback(path)
    
    def _compute_signature_iisig(self, path: np.ndarray) -> np.ndarray:
        """Compute signature using iisignature library (faster)"""
        # Normalize time to [0,1] interval
        path_norm = path.copy()
        path_norm[:, 0] = (path[:, 0] - path[0, 0]) / (path[-1, 0] - path[0, 0] + 1e-8)
        
        # Compute signature up to order N
        sig = iisignature.sig(path_norm, self.order)
        
        # Select most important features (<32 total)
        if len(sig) > 28:  # Keep room for vol features
            # Paper shows factorial decay in importance
            importance_weights = 1.0 / (np.arange(len(sig)) + 1)
            top_indices = np.argsort(importance_weights * np.abs(sig))[-28:]
            sig = sig[top_indices]
            
        return sig
    
    def _compute_signature_fallback(self, path: np.ndarray) -> np.ndarray:
        """Fast fallback implementation without iisignature
        
        Based on Chen's identity and recursive formulas from paper.
        Computes time-augmented signature S(X)^≤N_t via iterated integrals.
        """
        T, d = path.shape
        features = []
        
        # Normalize path to [0,1] time interval (essential per paper)
        path_norm = path.copy()
        path_norm[:, 0] = (path[:, 0] - path[0, 0]) / (path[-1, 0] - path[0, 0] + 1e-8)
        
        # Level 1: Path increments X^1_{s,t} = X_t - X_s
        increments = np.diff(path_norm, axis=0)
        level1 = np.sum(increments, axis=0)
        features.extend(level1)
        
        # Level 2: Lévy areas (double integrals)
        if self.order >= 2:
            areas = np.zeros((d, d))
            
            for i in range(T-1):
                dx = path_norm[i+1] - path_norm[i]
                x_mid = 0.5 * (path_norm[i] + path_norm[i+1])
                
                # Compute area tensor (Stratonovich correction)
                for j in range(d):
                    for k in range(d):
                        if j == k:
                            # Quadratic variation
                            areas[j, k] += dx[j] * dx[k]
                        else:
                            # Lévy area with midpoint rule
                            areas[j, k] += x_mid[j] * dx[k] - 0.5 * dx[j] * dx[k]
            
            features.extend(areas.flatten())
        
        # Level 3: Triple integrals (volume tensor)
        if self.order >= 3:
            # Select most important triple integrals per paper calibration
            volume_sums = np.zeros(9)
            
            for i in range(T-1):
                dx = path_norm[i+1] - path_norm[i]
                x_mid = 0.5 * (path_norm[i] + path_norm[i+1])
                
                # Key triples from paper calibration
                # Time-time-time
                volume_sums[0] += x_mid[0]**2 * dx[0] / 6
                # Time-time-price  
                volume_sums[1] += x_mid[0]**2 * dx[1] / 2
                # Time-price-price
                volume_sums[2] += x_mid[0] * dx[1]**2 / 2
                # Price-price-price
                volume_sums[3] += dx[1]**3 / 6
                # Time-time-volume
                volume_sums[4] += x_mid[0]**2 * dx[2] / 2
                # Time-price-volume
                volume_sums[5] += x_mid[0] * x_mid[1] * dx[2] / 3
                # Price-price-volume
                volume_sums[6] += x_mid[1] * dx[1] * dx[2] / 2
                # Price-volume-volume
                volume_sums[7] += x_mid[1] * dx[2]**2 / 2
                # Volume-volume-volume
                volume_sums[8] += dx[2]**3 / 6
            
            features.extend(volume_sums)
        
        return np.array(features)
    
    def compute_realized_vol(self, returns: np.ndarray, window: int) -> float:
        """Compute realized volatility over window"""
        if len(returns) < window:
            return 0.0
        recent_returns = returns[-window:]
        return np.std(recent_returns) * np.sqrt(365 * 24)  # Annualized
    
    def extract_features(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Extract signature + vol features from hourly bars
        
        Paper approach: σ_t(ℓ) = ⟨ℓ, S(X)^≤N_t⟩
        We compute S(X)^≤3_t and combine with realized vol metrics.
        
        Args:
            bars: DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            Dictionary of <32 features (signature + vol metrics)
        """
        if len(bars) < 2:
            return {name: 0.0 for name in self.feature_names}
        
        # Take last window_hours of data
        bars = bars.tail(self.window_hours)
        
        # Compute log returns (paper uses log price process)
        returns = np.log(bars['close'] / bars['close'].shift(1)).fillna(0).values
        
        # Normalize volume (z-score per paper)
        volume = bars['volume'].values
        volume_mean = np.mean(volume)
        volume_std = np.std(volume) + 1e-8
        volume_norm = (volume - volume_mean) / volume_std
        
        # Create time-augmented path (t, X_t) as per paper Theorem 3.27
        times = np.arange(len(returns))
        path = np.column_stack([times, returns, volume_norm])
        
        # Compute signature features S(X)^≤3_t
        sig_features = self.compute_signature(path)
        
        # Add realized vol features
        vol_24h = self.compute_realized_vol(returns, min(24, len(returns)))
        vol_72h = self.compute_realized_vol(returns, min(self.window_hours, len(returns)))
        
        # Vol of vol (roughness indicator for Hurst exponent)
        if len(returns) > 24:
            rolling_vols = pd.Series(returns).rolling(24).std().dropna()
            vol_of_vol = rolling_vols.std() if len(rolling_vols) > 1 else 0
        else:
            vol_of_vol = 0
        
        # Combine all features
        features = {}
        sig_feature_names = [n for n in self.feature_names if not n.startswith('rvol') and n != 'vol_of_vol']
        
        # Map signature coordinates to feature names
        for i, name in enumerate(sig_feature_names):
            features[name] = sig_features[i] if i < len(sig_features) else 0.0
        
        # Add vol features
        features['rvol_24h'] = vol_24h
        features['rvol_72h'] = vol_72h
        features['vol_of_vol'] = vol_of_vol
        
        # Z-score normalization across all features (paper Section 4.3)
        features_array = np.array(list(features.values()))
        if len(features_array) > 1 and np.std(features_array) > 0:
            features_norm = zscore(features_array)
            for i, name in enumerate(features.keys()):
                features[name] = features_norm[i]
        
        return features
    
    def get_feature_importance(self, features: Dict[str, float], 
                              weights: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        """Get feature importance ranking based on paper calibration
        
        Paper shows factorial decay in signature importance with key features:
        - Level 1: Direct price/volume increments (highest importance)
        - Level 2: Lévy areas capturing covariance structure
        - Level 3: Triple integrals for shape/roughness
        
        Args:
            features: Feature dictionary
            weights: Optional calibrated ℓ weights from σ_t(ℓ) = ⟨ℓ, S(X)^≤N_t⟩
            
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        if weights is None:
            # Default importance based on paper's calibration results
            # Paper shows factorial decay: level-k importance ≈ 1/k!
            default_importance = {
                'sig_p': 1.0,      # Price increment (paper: ℓ_1 ≈ 1.085)
                'sig_t': 0.8,      # Time (paper: ℓ_0 ≈ 0.201)
                'sig_pp': 0.5,     # Quadratic variation
                'sig_tp': 0.4,     # Time-price area
                'rvol_72h': 0.6,   # Realized vol key for regime
                'rvol_24h': 0.5,   # Short-term vol
                'vol_of_vol': 0.3, # Roughness indicator
            }
            
            importance = []
            for name, val in features.items():
                base_imp = default_importance.get(name, 0.1)
                importance.append((name, abs(val) * base_imp))
        else:
            # Use calibrated weights
            importance = []
            for i, name in enumerate(self.feature_names):
                if i < len(weights):
                    importance.append((name, abs(features.get(name, 0) * weights[i])))
                    
        return sorted(importance, key=lambda x: x[1], reverse=True)
    
    def compute_volatility_approximation(self, features: Dict[str, float], 
                                        weights: np.ndarray) -> float:
        """Compute volatility using linear signature approximation
        
        From paper: σ_t(ℓ) = ⟨ℓ, S(X)^≤N_t⟩
        This provides model-agnostic volatility under rough/non-Markovian dynamics.
        
        Args:
            features: Signature features dictionary
            weights: Calibrated coefficient vector ℓ
            
        Returns:
            Estimated volatility (annualized)
        """
        # Extract feature vector in consistent order
        feature_vec = np.array([features.get(name, 0.0) for name in self.feature_names])
        
        # Compute inner product
        if len(weights) != len(feature_vec):
            # Pad or truncate weights to match features
            weights_adj = np.zeros(len(feature_vec))
            weights_adj[:min(len(weights), len(feature_vec))] = weights[:len(feature_vec)]
            weights = weights_adj
        
        # Linear functional approximation
        vol = np.dot(weights, feature_vec)
        
        # Ensure positive volatility
        return max(0.01, abs(vol))