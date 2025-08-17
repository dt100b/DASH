"""FastAPI application for trading dashboard"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import json
import os
import numpy as np

# Import all modules
from config import config
from data_io import get_data_adapter
from features_signature import SignatureFeatures
from features_positioning import PositioningFeatures
from regime_orcsmc import ORCSMC
from head_bias import BiasHead
from stopping import DeepOptimalStopping
from execution_sizing import MultiscaleSVSizing
from risk import RiskManager
from state_store import store, TradingRun, Position, ValidationMetric
from scheduler import trading_scheduler
from validate_walkforward import WalkForwardValidator

# Create FastAPI app
app = FastAPI(title="Trading Dashboard API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_adapter = get_data_adapter("csv")
sig_features = SignatureFeatures()
pos_features = PositioningFeatures(data_adapter)
orcsmc = ORCSMC()
bias_head = BiasHead()
stopping = DeepOptimalStopping()
sizing = MultiscaleSVSizing()
risk_mgr = RiskManager()

# Request/Response models
class RunRequest(BaseModel):
    symbol: str = "BTC"

class BacktestRequest(BaseModel):
    symbol: str = "BTC"
    days: int = 30

class ImportWeightsRequest(BaseModel):
    model_type: str  # "bias_head" or "stopping"
    
class TelemetryResponse(BaseModel):
    symbol: str
    gate_prob: float
    bias: float
    target_exposure: float
    current_exposure: float
    kappa: float
    guards: Dict[str, bool]
    latest_decision: str
    recent_pnl: float
    features: Dict[str, float]
    attributions: List[Dict[str, float]]

# Main trading pipeline
async def run_trading_pipeline(symbol: str) -> Dict[str, Any]:
    """Run the complete daily trading pipeline
    
    Research implementation of daily runbook at 00:00 UTC
    """
    try:
        # 1. Pull data and compute features
        bars = data_adapter.get_hourly_bars(symbol, 72)
        if len(bars) < 24:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Extract signature features
        sig_feats = sig_features.extract_features(bars)
        
        # Extract positioning features
        pos_feats = pos_features.extract_features(symbol)
        
        # Combine all features
        all_features = {**sig_feats, **pos_feats}
        feature_array = np.array(list(all_features.values()))
        
        # Get last return for regime detection
        last_return = np.log(bars['close'].iloc[-1] / bars['close'].iloc[-2])
        
        # 2. ORCSMC regime check
        gate_prob = orcsmc.step(last_return)
        
        # Check gate threshold
        if gate_prob < config.RISK_ON_THRESHOLD:
            decision = "GATE_OFF"
            target_exposure = 0.0
        else:
            # 3. Compute bias
            bias = bias_head.predict_bias(feature_array)
            
            # Determine direction
            if bias >= config.BIAS_LONG_THRESHOLD:
                decision = "LONG"
            elif bias <= config.BIAS_SHORT_THRESHOLD:
                decision = "SHORT"
            else:
                decision = "FLAT"
                
            # 4. Get execution plan
            market_conditions = {
                'fast_vol': sig_feats['realized_vol_24h'],
                'slow_vol': sig_feats['realized_vol_72h'],
                'vol_regime': 'normal'
            }
            
            risk_guards = {
                'funding_guard': pos_feats['funding_percentile'] >= 95,
                'vol_drift_guard': False
            }
            
            execution_plan = sizing.get_execution_plan(
                bias if decision != "FLAT" else 0.0,
                0.0,  # Assume no current position for simplicity
                market_conditions,
                risk_guards
            )
            
            target_exposure = execution_plan['target_exposure']
            kappa = execution_plan['kappa']
        
        # 5. Check existing position for stop/hold
        open_position = store.get_open_position(symbol)
        if open_position:
            position_features = {
                'unrealized_pnl': 0.0,  # Would calculate from current price
                'current_vol': sig_feats['realized_vol_24h'],
                'regime_prob': gate_prob
            }
            
            stop_decision, confidence, bounds = stopping.decide(
                position_features,
                open_position.holding_days
            )
            
            if stop_decision == "STOP":
                # Close position
                store.update_position(
                    open_position.id,
                    status="CLOSED",
                    exit_time=datetime.utcnow().isoformat(),
                    exit_price=bars['close'].iloc[-1]
                )
                decision = f"STOP_{open_position.direction}"
        
        # 6. Save run to database
        run = TradingRun(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            gate_prob=gate_prob,
            bias=bias if gate_prob >= config.RISK_ON_THRESHOLD else 0.0,
            decision=decision,
            target_exposure=target_exposure if decision not in ["FLAT", "GATE_OFF"] else 0.0,
            current_exposure=0.0,
            kappa=kappa if 'kappa' in locals() else 0.0,
            features=json.dumps(all_features),
            risk_flags=json.dumps(risk_guards if 'risk_guards' in locals() else {})
        )
        store.save_run(run)
        
        # Get feature attributions
        attributions = []
        if gate_prob >= config.RISK_ON_THRESHOLD:
            feature_attrs = bias_head.get_feature_attributions(feature_array)
            for name, value, attr in feature_attrs[:8]:  # Top 8
                attributions.append({
                    'name': name,
                    'value': float(value),
                    'attribution': float(attr)
                })
        
        return {
            'status': 'success',
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'gate_prob': float(gate_prob),
            'bias': float(bias) if gate_prob >= config.RISK_ON_THRESHOLD else 0.0,
            'decision': decision,
            'target_exposure': float(target_exposure) if decision not in ["FLAT", "GATE_OFF"] else 0.0,
            'kappa': float(kappa) if 'kappa' in locals() else 0.0,
            'attributions': attributions
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Trading Dashboard API", "version": "1.0.0"}

@app.get("/status")
async def get_status():
    """Get system status and versions"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "scheduler": trading_scheduler.get_status(),
        "symbols": config.SYMBOLS,
        "models": {
            "bias_head": os.path.exists(f"{config.MODELS_PATH}/bias_head.joblib"),
            "stopping": os.path.exists(f"{config.MODELS_PATH}/stopping.pt")
        }
    }

@app.post("/run-now")
async def run_now(request: RunRequest):
    """Manually trigger trading pipeline"""
    result = await run_trading_pipeline(request.symbol)
    return result

@app.post("/import-weights")
async def import_weights(
    model_type: str,
    file: UploadFile = File(...)
):
    """Import model weights from file"""
    try:
        # Save uploaded file
        file_path = f"{config.MODELS_PATH}/{model_type}_{datetime.now().timestamp()}"
        
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        # Load into appropriate model
        if model_type == "bias_head":
            success = bias_head.load_model(file_path)
        elif model_type == "stopping":
            success = stopping.load_model(file_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if success:
            # Save metadata
            store.save_model_metadata(
                model_type=model_type,
                symbol="ALL",
                version="imported",
                path=file_path,
                metadata={"source": "user_upload"}
            )
            
            return {"status": "success", "model_type": model_type}
        else:
            raise ValueError("Failed to load model")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """Run backtest for specified period"""
    try:
        validator = WalkForwardValidator()
        
        # Run validation for recent period
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=request.days + 365)  # Need training data
        
        results = validator.run_validation(
            request.symbol,
            start_date,
            end_date
        )
        
        # Calculate aggregate metrics
        if results:
            avg_metrics = {
                'hit_rate': np.mean([r.hit_rate for r in results]),
                'sharpe': np.mean([r.sharpe_ratio for r in results]),
                'max_drawdown': np.mean([r.max_drawdown for r in results]),
                'gate_off_rate': np.mean([r.gate_off_rate for r in results]),
                'pass_rate': sum(1 for r in results if r.passed) / len(results)
            }
            
            # Check overall pass/fail
            passed = (
                avg_metrics['hit_rate'] >= config.MIN_HIT_RATE and
                avg_metrics['sharpe'] >= config.MIN_SHARPE and
                avg_metrics['max_drawdown'] <= config.MAX_DRAWDOWN and
                avg_metrics['gate_off_rate'] <= config.MAX_GATE_OFF_RATE
            )
            
            # Save to database
            metric = ValidationMetric(
                symbol=request.symbol,
                period_start=start_date.isoformat(),
                period_end=end_date.isoformat(),
                hit_rate=avg_metrics['hit_rate'],
                sharpe=avg_metrics['sharpe'],
                max_drawdown=avg_metrics['max_drawdown'],
                gate_off_rate=avg_metrics['gate_off_rate'],
                total_trades=sum(r.n_trades for r in results),
                total_pnl=sum(r.total_pnl for r in results),
                passed=passed
            )
            store.save_metrics(metric)
            
            return {
                'status': 'success',
                'symbol': request.symbol,
                'days': request.days,
                'metrics': avg_metrics,
                'passed': passed,
                'slices': len(results)
            }
        else:
            raise ValueError("No validation results generated")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/telemetry")
async def get_telemetry(symbol: str = "BTC"):
    """Get current telemetry data for dashboard"""
    try:
        # Get latest run
        latest_run = store.get_latest_run(symbol)
        
        if not latest_run:
            # Run pipeline to get fresh data
            result = await run_trading_pipeline(symbol)
            gate_prob = result.get('gate_prob', 0.0)
            bias = result.get('bias', 0.0)
            decision = result.get('decision', 'FLAT')
            target_exposure = result.get('target_exposure', 0.0)
            kappa = result.get('kappa', 0.1)
            features = json.loads(result.get('features', '{}')) if 'features' in result else {}
            attributions = result.get('attributions', [])
        else:
            gate_prob = latest_run.gate_prob
            bias = latest_run.bias
            decision = latest_run.decision
            target_exposure = latest_run.target_exposure
            kappa = latest_run.kappa
            features = json.loads(latest_run.features)
            
            # Get attributions
            feature_array = np.array(list(features.values()))
            feature_attrs = bias_head.get_feature_attributions(feature_array)
            attributions = [
                {'name': name, 'value': float(value), 'attribution': float(attr)}
                for name, value, attr in feature_attrs[:8]
            ]
        
        # Get open position if any
        open_position = store.get_open_position(symbol)
        current_exposure = open_position.size if open_position else 0.0
        
        # Get recent metrics
        recent_metrics = store.get_recent_metrics(symbol, limit=1)
        recent_pnl = recent_metrics[0].total_pnl if recent_metrics else 0.0
        
        # Check guards
        guards = {
            'funding_guard': features.get('funding_percentile', 0) >= 95,
            'vol_drift_guard': False  # Would check from position data
        }
        
        return TelemetryResponse(
            symbol=symbol,
            gate_prob=gate_prob,
            bias=bias,
            target_exposure=target_exposure,
            current_exposure=current_exposure,
            kappa=kappa,
            guards=guards,
            latest_decision=decision,
            recent_pnl=recent_pnl,
            features=features,
            attributions=attributions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Startup/Shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    # Set up scheduler callback
    trading_scheduler.set_daily_run_callback(run_trading_pipeline)
    trading_scheduler.start()
    
    # Load existing models if available
    bias_model_path = store.get_latest_model_path("bias_head", "ALL")
    if bias_model_path:
        bias_head.load_model(bias_model_path)
    
    stopping_model_path = store.get_latest_model_path("stopping", "ALL")
    if stopping_model_path:
        stopping.load_model(stopping_model_path)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    trading_scheduler.stop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)