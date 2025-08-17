# Trading Dashboard Backend

A quantitative trading system implementing signature-based volatility features, ORCSMC regime detection, and deep optimal stopping.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Backend

### Development
```bash
cd backend
python app.py
```

The API will be available at `http://localhost:8000`

### Production
```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

## GPU Training (Local RTX 3080)

For training the bias head and stopping models on your local GPU:

### 1. Export Training Data
```bash
python train_local/export_training_data.py
```

### 2. Train Models Locally
```bash
# Train bias head with BEMA and high-LR sweep
python train_local/train_bias_head_gpu.py

# Train deep optimal stopping
python train_local/train_stopping_gpu.py
```

### 3. Import Weights
After training, upload the generated model files through the dashboard UI or API:
- POST `/import-weights` with model file

## API Endpoints

- `GET /status` - System health and status
- `POST /run-now` - Manually trigger trading pipeline
- `POST /import-weights` - Import trained model weights
- `POST /backtest` - Run backtesting for specified period
- `GET /telemetry` - Get current trading telemetry

## Research Components

- **Signature Features**: Model-agnostic volatility characterization under rough dynamics
- **ORCSMC**: Online rolling controlled SMC for regime detection
- **BEMA**: Bias-corrected EMA for stable training
- **Deep Optimal Stopping**: Recursive binary decision networks with bounds
- **Multiscale SV Sizing**: Impact-aware execution with volatility adaptation

## Configuration

Edit `config.py` to adjust:
- Trading thresholds
- Risk limits
- Validation criteria
- API settings