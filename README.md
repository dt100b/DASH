# Quantitative Trading Dashboard

A one-screen trading dashboard with a Python/FastAPI backend implementing signature-based volatility features, ORCSMC regime detection, and deep optimal stopping.

## Architecture

### Backend (Python/FastAPI)
- **Signature Features**: Time-augmented path signatures for model-agnostic volatility characterization
- **ORCSMC**: Online Rolling Controlled SMC for regime inference (RISK-ON/RISK-OFF)
- **Bias Head**: Transparent logistic head with BEMA updates and high-LR robustness
- **Deep Optimal Stopping**: Recursive binary decision networks with bounds
- **Multiscale SV Sizing**: Impact-aware execution with volatility adaptation
- **Risk Management**: Hard stops, daily limits, funding/vol drift guards
- **Walk-Forward Validation**: 365/30 rolling windows with strict acceptance criteria

### Frontend (React)
- Ultra-dark monochrome UI with thin fonts
- One-screen dashboard with real-time telemetry
- Custom radial gauges, sparklines, and status pills
- Zero visual noise, immaculate spacing

## Quick Start

### Running in Replit

The project is configured to run both backend and frontend in Replit:

1. **Install Python dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Start the backend**:
```bash
cd backend
python app.py
```

3. **In a new terminal, install frontend dependencies**:
```bash
cd frontend
npm install
```

4. **Start the frontend**:
```bash
cd frontend
npm run dev
```

The dashboard will be available at http://localhost:3000

## GPU Training (Local RTX 3080)

For training models on your local GPU:

### 1. Export training data from Replit:
```bash
cd backend
python train_local/export_training_data.py
```

### 2. On your local machine with RTX 3080:
```bash
# Train bias head with BEMA and high-LR sweep
python train_local/train_bias_head_gpu.py

# Train deep optimal stopping
python train_local/train_stopping_gpu.py
```

### 3. Import trained weights back to Replit:
- Use the "Import Weights" button in the dashboard
- Or POST to `/api/import-weights` endpoint

## Daily Trading Pipeline

The system runs automatically at 00:00 UTC daily:

1. **Data Pull**: Fetch 72h of hourly bars
2. **Feature Extraction**: Compute signature and positioning features
3. **Regime Gate**: ORCSMC determines P(RISK-ON)
4. **Bias Computation**: If gate passes (≥0.6), compute directional bias
5. **Execution**: Impact-aware sizing with multiscale SV adjustments
6. **Risk Checks**: Apply stops, limits, and guards
7. **Position Management**: Deep optimal stopping for exits

## Acceptance Criteria

The system enforces strict validation bars:
- Hit Rate ≥ 56% (or ≥54% with payoff >1.2x)
- Sharpe Ratio ≥ 1.0
- Max Drawdown ≤ 15%
- Gate Off Rate ≤ 60%

## API Endpoints

- `GET /status` - System health and scheduler status
- `POST /run-now` - Manually trigger trading pipeline
- `POST /backtest` - Run 30-day backtesting
- `POST /import-weights` - Import trained model weights
- `GET /telemetry` - Get current trading telemetry

## Research Foundations

The implementation reflects cutting-edge quantitative research:
- **Signatures**: Model-agnostic under Heston and rough Bergomi dynamics
- **ORCSMC**: Dual particle systems with bounded per-step compute
- **BEMA**: Bias-corrected EMA removing lag while keeping variance reduction
- **High LRs**: Improved robustness to spurious correlations
- **JAL/AMSE**: Noise-tolerant loss for weak labels
- **Multiscale SV**: Small-impact approximation with fast/slow vol asymptotics

## Configuration

Edit `backend/config.py` to adjust:
- Trading symbols (BTC, ETH)
- Risk thresholds and limits
- Validation acceptance bars
- Fee structure

## Theme

Ultra-dark monochrome palette:
- Backgrounds: #0A0A0B → #1B1B1E
- Strokes: #2A2A2F → #4A4A55
- Text: #D7D7DC → #7A7A82
- Accents: #5A5A66 → #70707B

Font: Inter (100-400 weights) for thin, modern typography