# Overview

Quantitative Trading Dashboard - A one-screen trading dashboard with Python/FastAPI backend implementing signature-based volatility features, ORCSMC regime detection, and deep optimal stopping. Features an ultra-dark monochrome React frontend with real-time telemetry display.

# User Preferences

Preferred communication style: Simple, everyday language.

**Data Immutability**: All labels and metrics from the trading dashboard are immutable. Never rename, add, or remove any values. If something is unclear, render a clear placeholder and mark TODO. Preserve all existing trading data exactly as-is.

# System Architecture

## Project Structure
- **Backend**: Python/FastAPI server with quantitative trading modules in `/backend` directory
- **Frontend**: React application with Vite and ultra-dark theme in `/frontend` directory
- **Tests**: Unit tests for trading modules in `/backend/tests` directory
- **GPU Training**: Local training scripts for RTX 3080 in `/backend/train_local` directory

## Backend Architecture (Python/FastAPI)
- **Signature Features**: Research-based time-augmented path signatures (paper 2507.23392v2)
  - Order-3 truncation with <32 dimensions for optimal accuracy/efficiency
  - Fast fallback implementation without iisignature dependency
  - Model-agnostic volatility σ_t(ℓ) = ⟨ℓ, S(X)^≤N_t⟩
  - Validated on Heston and rough Bergomi dynamics (H=0.1)
- **ORCSMC**: Online Rolling Controlled SMC for latent-state inference with learned twist (paper 2508.00696v1)
  - Dual particle systems (twist learner + filter) with rolling window L=30
  - K=5 learning iterations per step with quadratic twist functions
  - Adaptive resampling at ESS < κN (κ=0.5) with residual-multinomial
  - Inertia rule: 2 consecutive ON days required after OFF regime
  - Bounded O(L×N×K) per-step compute with N=1000 particles
- **Deep Optimal Stopping**: Recursive binary decisions (paper 1804.05394v4)
  - Tiny MLPs (16 nodes) with F^θ and f^θ networks
  - Lower/upper bounds via dual method: L = E[g(τ^Θ)] and U = E[max(g-M^Θ)]
  - 7-day hard cap with training at 00:00 UTC
  - Enhanced with BEMA optimizer for 2-3x faster convergence
  - JAL loss integration for noise-robust training (paper 2507.17692v1)
- **Optimal Execution**: Multiscale SV with price impact (paper 2507.17162v1)
  - Target formula: x* ∝ Bias via [(Aql+λAll)l + (Aqx+λAxl)x] / [Aqq-λ-λAql]
  - κ fast rule: -εγ(y-μ)q/(θK), κ slow rule: √δ(Bq+λBl)/K
  - Small-impact approximation: θ parameter with λ→θλ, β→θβ
  - Weekly slippage recalibration and half-step when vol > 1.2μ
- **JAL (Joint Asymmetric Loss)**: Noise-robust loss for noisy labels (paper 2507.17692v1)
  - Passive AMSE: Asymmetric MSE with |p_k|^q for non-targets (q=3 optimal)
  - Active CE/FL: Cross-entropy or focal loss for target class
  - Combination: JAL = α·L_active + β·L_passive (α=0.7, β=0.3)
  - Activation: Switches on when noise > 15% after 5-epoch warmup
  - Label smoothing and gradient clipping for stability
- **Large Learning Rates**: Training recipe for robustness/compressibility (paper 2507.17748v2)
  - Base LR=0.3 with 5-epoch linear warmup from 0.01
  - Cosine annealing decay to min_lr=1e-5
  - Gradient clipping=1.0, weight decay=5e-4
  - Tracks sparsity (>30%), confidence (>85%), class separation (>2.0)
  - Enables 50%+ model compression via induced sparsity
- **BEMA (Bias-corrected EMA)**: Iterate averaging of weights (paper 2508.00180v1)
  - Bias-corrected exponential moving average of model weights
  - Wraps existing optimizer (Adam/SGD), doesn't replace it
  - Improves generalization via weight averaging
  - Applied post-training for final model weights
- **Risk Management**: Hard stops, daily limits, funding/vol drift guards
- **Walk-Forward Validation**: 365/30 rolling windows with strict acceptance criteria
- **Task Scheduler**: Daily trading pipeline execution at 00:00 UTC

## Data Contracts & QA

### Input Schemas
- **bars_1h**: timestamp_utc, close, volume
- **daily_close**: date_utc, close
- **funding_24h**: date_utc, funding_delta
- **oi_24h**: date_utc, oi_pct_delta
- **options_moves** (optional): date_utc, rr25d_delta, term_slope_7d_30d_delta
- **slippage_log**: date_utc, symbol, side, size, est_impact_bp

### Integrity Checks
- No gaps in time series data
- Monotonic timestamps (UTC only)
- NaNs filled: forward-fill for exogenous features only, never for returns
- Staleness threshold: < 26 hours
- Feature leakage guard: all features known at 00:00 UTC decision time

## Frontend Architecture (React)
- **Framework**: React 18 with Vite for fast development
- **UI Design**: Ultra-dark monochrome theme with thin fonts (Inter 100-400)
- **Components**: Custom radial gauges, bias gauge, sparklines, status pills
- **Styling**: Tailwind CSS with custom color palette
- **Charts**: Recharts for data visualization
- **Icons**: Lucide React for UI icons
- **State**: React hooks for local state management
- **API**: Fetch API for backend communication

## Trading Pipeline
1. **Data Pull**: Fetch 72h of hourly bars from exchange
2. **Feature Extraction**: Compute signature and positioning features
3. **Regime Gate**: ORCSMC determines P(RISK-ON)
4. **Bias Computation**: If gate passes (≥0.6), compute directional bias
5. **Target Sizing**: x* = optimal exposure proportional to bias
6. **Execution Rate**: u* with κ fast/slow corrections and half-stepping
7. **Impact Tracking**: Update price impact dl = (λu - βl)dt
8. **Risk Checks**: Apply stops, limits, and guards
9. **Position Management**: Deep optimal stopping for exits
10. **Weekly Calibration**: Adjust λ from realized slippage

# External Dependencies

## Backend Dependencies (Python)
- **FastAPI**: Modern async web framework for building APIs
- **NumPy**: Numerical computing for matrix operations and signatures
- **Pandas**: Data manipulation and time series analysis
- **scikit-learn**: Machine learning utilities for preprocessing
- **APScheduler**: Advanced task scheduling for daily trading runs
- **httpx**: Async HTTP client for exchange API calls
- **Pydantic**: Data validation and settings management

## Frontend Dependencies (React)
- **React**: Version 18 for UI components
- **Vite**: Fast build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **Recharts**: Composable charting library
- **Lucide React**: Icon library
- **clsx**: Utility for constructing className strings

## Theme Configuration
- **Colors**: Ultra-dark monochrome palette (#0A0A0B to #70707B)
- **Typography**: Inter font family (weights 100-400)
- **Spacing**: Immaculate spacing with 8px grid system
- **Components**: Custom radial gauges, sparklines, bias gauge