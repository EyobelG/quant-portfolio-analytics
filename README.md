# Quant Portfolio Analytics

A full-stack web application for quantitative portfolio analysis. Enter a set of holdings and it pulls live market data, computes risk-adjusted performance metrics, runs mean-variance optimization to trace the efficient frontier, and backtests the optimized allocation against both your original weights and the S&P 500.

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![React](https://img.shields.io/badge/React-18-61dafb)
![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178c6)

---

## What it does

**Risk metrics** — annualized return and volatility, Sharpe and Sortino ratios, maximum drawdown, 95% Value at Risk and Conditional VaR, plus beta and Jensen's alpha measured against the benchmark.

**Mean-variance optimization** — solves for the max-Sharpe and minimum-volatility portfolios using [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt), then sweeps target returns to trace the full efficient frontier. Your current portfolio is plotted alongside it, so the gap between where you are and the frontier is visible directly.

**Backtesting** — cumulative growth of $1 for your portfolio, the max-Sharpe reweighting, and the benchmark over the same window.

**Correlation matrix** — a color-scaled heatmap of pairwise return correlations, since highly correlated holdings limit real diversification.

---

## Architecture

```
backend/          FastAPI service
  app/
    main.py       API routes and request orchestration
    data.py       yfinance price fetching, TTL-cached
    metrics.py    risk metrics via empyrical
    optimize.py   efficient frontier via PyPortfolioOpt
    backtest.py   cumulative growth series
    schemas.py    Pydantic request/response models

frontend/         React + TypeScript + Vite
  src/
    App.tsx       layout and API integration
    components/   builder, metrics grid, charts, tables
```

A single `POST /api/analyze` call returns metrics, optimization results, and the backtest together, so the UI renders in one round trip. Price downloads are cached for an hour to stay well within Yahoo Finance's rate limits.

### Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI, Pydantic v2, Uvicorn |
| Quant | PyPortfolioOpt, empyrical, pandas, NumPy |
| Data | yfinance (Yahoo Finance, no API key required) |
| Frontend | React 18, TypeScript, Vite, Recharts |

---

## Running locally

Requires Python 3.12 and Node 18+.

**Backend**

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend** (in a second terminal)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api` to the backend on port 8000, so no CORS setup is needed in development.

---

## API

### `POST /api/analyze`

```json
{
  "holdings": [
    { "ticker": "AAPL", "weight": 0.4 },
    { "ticker": "MSFT", "weight": 0.3 },
    { "ticker": "JNJ",  "weight": 0.3 }
  ],
  "benchmark": "^GSPC",
  "period": "3y"
}
```

Weights must sum to 1.0. `period` accepts any yfinance period string (`1y`, `2y`, `3y`, `5y`, `10y`).

Returns `metrics`, `optimization` (frontier points, optimal weights, correlation matrix), and `backtest` (aligned date and growth series). Interactive docs are served at `/docs`.

---

## Deployment

The frontend is a static bundle (`npm run build` → `dist/`) deployable to Vercel, Netlify, or any static host. Set `VITE_API_BASE` to the deployed backend URL at build time.

The backend runs anywhere that supports a Python web service — Render, Railway, Fly.io:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

---

## Notes and limitations

Mean-variance optimization is highly sensitive to its return estimates. Because expected returns here come from historical means, the "optimal" weights describe what would have worked over the lookback window, not what will work going forward — the classic critique of naive Markowitz optimization. Treating the frontier as a diagnostic of the current allocation is more defensible than treating it as a recommendation.

Metrics use a 0% risk-free rate. Long-only, fully-invested constraints apply (no shorting or leverage). Educational project — not investment advice.
