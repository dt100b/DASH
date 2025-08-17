"""Data IO adapters for CSV and exchange data - handles 1h bars and daily close"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
from abc import ABC, abstractmethod

class DataAdapter(ABC):
    """Abstract base for data adapters"""
    
    @abstractmethod
    def get_hourly_bars(self, symbol: str, hours: int) -> pd.DataFrame:
        """Get hourly OHLCV bars"""
        pass
    
    @abstractmethod
    def get_daily_close(self, symbol: str, days: int) -> pd.Series:
        """Get daily close prices"""
        pass
    
    @abstractmethod
    def get_funding_rate(self, symbol: str, hours: int = 24) -> float:
        """Get funding rate over period"""
        pass
    
    @abstractmethod
    def get_open_interest(self, symbol: str, hours: int = 24) -> Tuple[float, float]:
        """Get open interest change (current, change_pct)"""
        pass

class CSVDataAdapter(DataAdapter):
    """CSV file-based data adapter for backtesting"""
    
    def __init__(self, data_path: str = "backend/data"):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        self._cache = {}
        
    def _load_data(self, symbol: str) -> pd.DataFrame:
        """Load and cache CSV data"""
        if symbol not in self._cache:
            file_path = os.path.join(self.data_path, f"{symbol.lower()}_1h.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
                self._cache[symbol] = df
            else:
                # Generate synthetic data for testing if no CSV exists
                self._cache[symbol] = self._generate_synthetic_data(symbol)
        return self._cache[symbol]
    
    def _generate_synthetic_data(self, symbol: str) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing (research basis: rough Bergomi dynamics)"""
        np.random.seed(42)  # Deterministic
        n_hours = 24 * 365  # 1 year of hourly data
        
        # Base price (BTC-like or ETH-like)
        base_price = 40000 if symbol == "BTC" else 2500
        
        # Generate rough volatility path (H ≈ 0.1 for rough dynamics)
        dt = 1/24  # Daily units
        H = 0.1  # Hurst parameter for rough volatility
        
        # Simple fractional Brownian motion approximation
        times = np.arange(n_hours) * dt
        cov_matrix = np.abs(np.subtract.outer(times, times)) ** (2 * H)
        vol_path = np.exp(0.2 * np.random.multivariate_normal(np.zeros(n_hours), cov_matrix))
        
        # Price path with stochastic volatility
        returns = np.random.normal(0, vol_path * np.sqrt(dt))
        returns[0] = 0
        log_prices = np.cumsum(returns)
        prices = base_price * np.exp(log_prices - 0.5 * np.var(log_prices))
        
        # Create OHLCV data
        df = pd.DataFrame()
        df['open'] = prices * (1 + np.random.normal(0, 0.001, n_hours))
        df['high'] = prices * (1 + np.abs(np.random.normal(0, 0.002, n_hours)))
        df['low'] = prices * (1 - np.abs(np.random.normal(0, 0.002, n_hours)))
        df['close'] = prices
        df['volume'] = np.abs(np.random.lognormal(15, 1, n_hours))  # Log-normal volume
        
        # Add timestamp index
        end_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        timestamps = pd.date_range(end=end_time, periods=n_hours, freq='1H')
        df.index = timestamps
        df.index.name = 'timestamp'
        
        # Add funding and OI columns
        df['funding_rate'] = np.random.normal(0.0001, 0.0002, n_hours)
        df['open_interest'] = np.abs(np.random.lognormal(20, 0.5, n_hours))
        
        return df
    
    def get_hourly_bars(self, symbol: str, hours: int) -> pd.DataFrame:
        """Get last N hours of OHLCV data"""
        df = self._load_data(symbol)
        return df[['open', 'high', 'low', 'close', 'volume']].tail(hours)
    
    def get_daily_close(self, symbol: str, days: int) -> pd.Series:
        """Get daily close prices"""
        df = self._load_data(symbol)
        # Resample to daily
        daily = df['close'].resample('1D').last()
        return daily.tail(days)
    
    def get_funding_rate(self, symbol: str, hours: int = 24) -> float:
        """Get average funding rate over period"""
        df = self._load_data(symbol)
        if 'funding_rate' in df.columns:
            return df['funding_rate'].tail(hours).mean()
        return 0.0001  # Default funding rate
    
    def get_open_interest(self, symbol: str, hours: int = 24) -> Tuple[float, float]:
        """Get open interest and % change"""
        df = self._load_data(symbol)
        if 'open_interest' in df.columns:
            oi_series = df['open_interest'].tail(hours + 1)
            current = oi_series.iloc[-1]
            previous = oi_series.iloc[0]
            change_pct = (current - previous) / previous if previous > 0 else 0
            return current, change_pct
        return 1e9, 0.0  # Default 1B OI, no change

class ExchangeDataAdapter(DataAdapter):
    """Placeholder for real exchange data adapter"""
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        # In production, initialize exchange client here
        
    def get_hourly_bars(self, symbol: str, hours: int) -> pd.DataFrame:
        """Fetch from exchange API"""
        # Placeholder - would call exchange API
        # For now, fallback to CSV adapter
        csv_adapter = CSVDataAdapter()
        return csv_adapter.get_hourly_bars(symbol, hours)
    
    def get_daily_close(self, symbol: str, days: int) -> pd.Series:
        """Fetch daily close from exchange"""
        csv_adapter = CSVDataAdapter()
        return csv_adapter.get_daily_close(symbol, days)
    
    def get_funding_rate(self, symbol: str, hours: int = 24) -> float:
        """Fetch funding rate from exchange"""
        csv_adapter = CSVDataAdapter()
        return csv_adapter.get_funding_rate(symbol, hours)
    
    def get_open_interest(self, symbol: str, hours: int = 24) -> Tuple[float, float]:
        """Fetch OI from exchange"""
        csv_adapter = CSVDataAdapter()
        return csv_adapter.get_open_interest(symbol, hours)

# Factory function
def get_data_adapter(source: str = "csv", **kwargs) -> DataAdapter:
    """Get appropriate data adapter"""
    if source == "csv":
        return CSVDataAdapter(**kwargs)
    elif source == "exchange":
        return ExchangeDataAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown data source: {source}")