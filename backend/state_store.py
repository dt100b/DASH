"""State storage using SQLite for runs, positions, metrics, and model weights"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import os

from config import config

@dataclass
class TradingRun:
    """Record of a trading run"""
    id: Optional[int] = None
    symbol: str = ""
    timestamp: str = ""
    gate_prob: float = 0.0
    bias: float = 0.0
    decision: str = "FLAT"
    target_exposure: float = 0.0
    current_exposure: float = 0.0
    kappa: float = 0.0
    features: str = "{}"  # JSON
    risk_flags: str = "{}"  # JSON
    
@dataclass 
class Position:
    """Active position record"""
    id: Optional[int] = None
    symbol: str = ""
    entry_time: str = ""
    exit_time: Optional[str] = None
    direction: str = ""  # LONG/SHORT
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    size: float = 0.0
    pnl: Optional[float] = None
    status: str = "OPEN"  # OPEN/CLOSED/STOPPED
    holding_days: int = 0
    
@dataclass
class ValidationMetric:
    """Validation metrics record"""
    id: Optional[int] = None
    symbol: str = ""
    period_start: str = ""
    period_end: str = ""
    hit_rate: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    gate_off_rate: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    passed: bool = False

class StateStore:
    """SQLite-based state management"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_PATH
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Trading runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    gate_prob REAL,
                    bias REAL,
                    decision TEXT,
                    target_exposure REAL,
                    current_exposure REAL,
                    kappa REAL,
                    features TEXT,
                    risk_flags TEXT
                )
            """)
            
            # Positions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL,
                    pnl REAL,
                    status TEXT,
                    holding_days INTEGER
                )
            """)
            
            # Validation metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    period_start TEXT,
                    period_end TEXT,
                    hit_rate REAL,
                    sharpe REAL,
                    max_drawdown REAL,
                    gate_off_rate REAL,
                    total_trades INTEGER,
                    total_pnl REAL,
                    passed BOOLEAN
                )
            """)
            
            # Model metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    symbol TEXT,
                    version TEXT,
                    path TEXT,
                    timestamp TEXT,
                    metadata TEXT
                )
            """)
            
            conn.commit()
    
    def save_run(self, run: TradingRun) -> int:
        """Save trading run record"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO runs (symbol, timestamp, gate_prob, bias, decision,
                                target_exposure, current_exposure, kappa, features, risk_flags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run.symbol, run.timestamp, run.gate_prob, run.bias, run.decision,
                  run.target_exposure, run.current_exposure, run.kappa, 
                  run.features, run.risk_flags))
            conn.commit()
            return cursor.lastrowid
    
    def get_latest_run(self, symbol: str) -> Optional[TradingRun]:
        """Get latest run for symbol"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM runs 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            if row:
                return TradingRun(**dict(row))
            return None
    
    def save_position(self, position: Position) -> int:
        """Save position record"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO positions (symbol, entry_time, exit_time, direction,
                                      entry_price, exit_price, size, pnl, status, holding_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (position.symbol, position.entry_time, position.exit_time,
                  position.direction, position.entry_price, position.exit_price,
                  position.size, position.pnl, position.status, position.holding_days))
            conn.commit()
            return cursor.lastrowid
    
    def get_open_position(self, symbol: str) -> Optional[Position]:
        """Get open position for symbol"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM positions 
                WHERE symbol = ? AND status = 'OPEN'
                ORDER BY entry_time DESC 
                LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            if row:
                return Position(**dict(row))
            return None
    
    def update_position(self, position_id: int, **kwargs):
        """Update position fields"""
        with sqlite3.connect(self.db_path) as conn:
            set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
            values = list(kwargs.values()) + [position_id]
            conn.execute(f"""
                UPDATE positions 
                SET {set_clause}
                WHERE id = ?
            """, values)
            conn.commit()
    
    def save_metrics(self, metrics: ValidationMetric) -> int:
        """Save validation metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (symbol, period_start, period_end, hit_rate,
                                   sharpe, max_drawdown, gate_off_rate, total_trades,
                                   total_pnl, passed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (metrics.symbol, metrics.period_start, metrics.period_end,
                  metrics.hit_rate, metrics.sharpe, metrics.max_drawdown,
                  metrics.gate_off_rate, metrics.total_trades, metrics.total_pnl,
                  metrics.passed))
            conn.commit()
            return cursor.lastrowid
    
    def get_recent_metrics(self, symbol: str, limit: int = 10) -> List[ValidationMetric]:
        """Get recent validation metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM metrics 
                WHERE symbol = ? 
                ORDER BY period_end DESC 
                LIMIT ?
            """, (symbol, limit))
            return [ValidationMetric(**dict(row)) for row in cursor.fetchall()]
    
    def save_model_metadata(self, model_type: str, symbol: str, version: str, 
                           path: str, metadata: Dict[str, Any]):
        """Save model metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO model_metadata (model_type, symbol, version, path, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (model_type, symbol, version, path, 
                  datetime.utcnow().isoformat(), json.dumps(metadata)))
            conn.commit()
    
    def get_latest_model_path(self, model_type: str, symbol: str) -> Optional[str]:
        """Get path to latest model"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT path FROM model_metadata 
                WHERE model_type = ? AND symbol = ?
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (model_type, symbol))
            row = cursor.fetchone()
            return row[0] if row else None

# Global store instance
store = StateStore()