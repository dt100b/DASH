"""Positioning and options features - funding, OI, optional Greeks deltas"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from data_io import DataAdapter

class PositioningFeatures:
    """Extract positioning and flow features from market data"""
    
    def __init__(self, data_adapter: DataAdapter):
        self.data_adapter = data_adapter
        self.feature_names = [
            'delta_funding_24h',
            'delta_oi_pct_24h',
            'funding_percentile',
            'oi_magnitude'
        ]
        # Optional options features
        self.options_feature_names = [
            'delta_25d_rr',  # 25-delta risk reversal change
            'delta_term_slope'  # Term structure slope change (7D vs 30D)
        ]
        
    def extract_features(self, symbol: str, 
                        include_options: bool = False) -> Dict[str, float]:
        """Extract positioning features
        
        Args:
            symbol: Trading symbol
            include_options: Whether to include options Greeks
            
        Returns:
            Dictionary of positioning features
        """
        features = {}
        
        # Get 24h funding rate change
        current_funding = self.data_adapter.get_funding_rate(symbol, hours=1)
        prev_funding = self.data_adapter.get_funding_rate(symbol, hours=25)
        features['delta_funding_24h'] = current_funding - prev_funding
        
        # Get 24h OI change
        current_oi, oi_change_pct = self.data_adapter.get_open_interest(symbol, hours=24)
        features['delta_oi_pct_24h'] = oi_change_pct
        features['oi_magnitude'] = np.log10(current_oi + 1)  # Log scale for magnitude
        
        # Calculate funding percentile (for risk guard)
        features['funding_percentile'] = self._calculate_funding_percentile(
            symbol, current_funding
        )
        
        # Optional: Options features (placeholder for now)
        if include_options:
            features.update(self._get_options_features(symbol))
            
        return features
    
    def _calculate_funding_percentile(self, symbol: str, current_rate: float) -> float:
        """Calculate historical percentile of funding rate
        
        This is used for the funding guard - if in top 5% (95th percentile),
        we halve position size.
        """
        # In production, would query 1 year of funding history
        # For now, use synthetic distribution
        np.random.seed(hash(symbol) % 1000)  # Deterministic per symbol
        historical_rates = np.random.normal(0.0001, 0.0002, 365 * 8)  # 8h funding periods
        percentile = np.percentile(historical_rates, 
                                  np.searchsorted(np.sort(historical_rates), current_rate) * 100 / len(historical_rates))
        return percentile
    
    def _get_options_features(self, symbol: str) -> Dict[str, float]:
        """Get options-based features (placeholder)
        
        In production, these would come from Deribit or similar
        """
        features = {}
        
        # 25-delta risk reversal change (call IV - put IV)
        # Positive = calls more expensive = bullish flow
        features['delta_25d_rr'] = 0.0  # Placeholder
        
        # Term structure slope change (30D IV - 7D IV)  
        # Positive = longer-term vol higher = uncertainty
        features['delta_term_slope'] = 0.0  # Placeholder
        
        return features
    
    def get_funding_guard_active(self, symbol: str) -> bool:
        """Check if funding guard should be active
        
        Returns True if current funding is in top 5% of historical distribution
        """
        current_funding = self.data_adapter.get_funding_rate(symbol, hours=1)
        percentile = self._calculate_funding_percentile(symbol, current_funding)
        return percentile >= 95.0
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get human-readable descriptions of features"""
        return {
            'delta_funding_24h': '24h change in funding rate',
            'delta_oi_pct_24h': '24h % change in open interest',
            'funding_percentile': 'Historical percentile of current funding',
            'oi_magnitude': 'Log10 of total open interest',
            'delta_25d_rr': '25-delta risk reversal change (options)',
            'delta_term_slope': 'Term structure slope change 7D→30D (options)'
        }