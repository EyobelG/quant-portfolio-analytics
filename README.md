# Quant Portfolio Analytics

A full-stack web application for quantitative portfolio analysis. Enter a set of holdings and it pulls live market data, computes risk-adjusted performance metrics, runs mean-variance optimization to trace the efficient frontier, and backtests the optimized allocation against both your original weights and the S&P 500.

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![React](https://img.shields.io/badge/React-18-61dafb)
![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178c6)
![Tests](https://img.shields.io/badge/tests-176%20passing-2fbf71)

---

## What it does

The tool is built around one argument: **mean-variance optimization looks impressive in-sample and mostly fails out-of-sample.** Everything below either demonstrates that or quantifies how much to trust any given number.

### The headline result

A walk-forward test re-fits each allocation rule at every rebalance using only data available at that time, holds the weights forward through returns the optimizer has never seen, lets them drift with the market in between, and charges transaction costs on the turnover this generates.

On a typical five-stock portfolio over five years, max-Sharpe optimization scores about **1.39 in-sample and 0.87 out-of-sample**, and loses to naive equal weighting — reproducing DeMiguel, Garlappi & Uppal (2009). Estimation error in expected returns swamps the benefit of optimizing over them, and the optimizer's 0.82x annual turnover pays for the privilege. Risk parity and hierarchical risk parity, neither of which needs a return forecast, come out on top.

The in-sample backtest is still shown alongside it, precisely so the gap is visible.

### The principled repair

If historical means are the problem, the fix is to stop estimating them. **Black-Litterman** reverse-optimizes the market's own capitalization weights into the returns the market must already be assuming — no sample means anywhere — and then moves off that prior only as far as an explicit view at an explicit confidence justifies.

Views are entered as sentences, absolute ("AAPL returns 12% a year") or relative ("AAPL beats XOM by 9%"), each with a confidence slider mapped through Idzorek's method onto the view-uncertainty matrix. With no views the posterior is exactly the prior and the optimum is the market portfolio.

The claim is then tested rather than asserted: the equilibrium prior runs as its own row in the walk-forward table, with historical cap weights reconstructed as `shares × price` so there is no lookahead. On a five-year book it lands around Sharpe 1.01 against 0.74 for both max-Sharpe variants — better than optimizing over historical means, still short of equal weight.

### Correlation under stress

A single correlation matrix averaged over the whole sample is the most misleading number in portfolio analysis. Splitting by the benchmark's return decile shows why: average pairwise correlation on a typical five-stock book runs **0.03 in calm markets and 0.23 in the worst decile**, so the diversification the average implies is largely absent on the days it is supposed to matter. Rendered as calm and stressed matrices side by side with a delta view, plus per-asset downside and upside betas.

The split is by benchmark return, not portfolio return — conditioning on the portfolio's own losses would be circular.

### Statistical inference

Every headline metric is a point estimate from one finite, non-normal, autocorrelated sample:

- **Probabilistic Sharpe ratio** (Bailey & López de Prado) — probability the true Sharpe exceeds zero, with the estimator's standard error widened for skew and kurtosis.
- **Deflated Sharpe ratio** — the same test against the Sharpe a *lucky* search would produce. Sweeping 25 frontier portfolios and reporting the best is multiple hypothesis testing, and the expected maximum under a null of no skill is the honest benchmark.
- **Stationary bootstrap** (Politis & Romano) — 2,000 block resamples for a confidence interval on the Sharpe that preserves volatility clustering, rather than an IID bootstrap that would come out artificially narrow.
- **Jarque-Bera** normality test on the return distribution.

### Risk decomposition

- **Ledoit-Wolf shrinkage**, implemented from the closed-form optimal intensity rather than imported — no hyperparameter, no cross-validation, and guaranteed well-conditioned.
- **Marchenko-Pastur denoising** — random-matrix theory gives the eigenvalue distribution of a correlation matrix built from pure noise, so anything inside the bulk can be flattened before the matrix is inverted.
- **Euler risk contributions** — weight is not risk; a 25% position in AAPL routinely carries 35% of portfolio volatility.
- **Meucci's effective number of bets** — exponential entropy of variance shares across principal components. Five correlated holdings frequently score below 1.5.
- **Equal-risk-contribution (risk parity)** weights, solved as a constrained least-squares problem.

### Conditional volatility

- **GARCH(1,1)** fitted by maximum likelihood, written from the log-likelihood rather than imported from `arch`, with a mean-reverting forecast and shock half-life.
- **EWMA** (RiskMetrics, λ = 0.94).
- **Engle's ARCH-LM test**, which is the diagnostic that justifies fitting a conditional model at all.

### Tail risk, validated

- **Cornish-Fisher modified VaR**, correcting the normal quantile for skew and fat tails.
- **Rolling out-of-sample VaR backtest** with **Kupiec** proportion-of-failures and **Christoffersen** independence tests, plus the joint conditional-coverage test. Each day's VaR is estimated from prior data only, so this measures how the model would actually have performed.

### Factor attribution

Multi-factor regression against market, size, value, momentum, quality and low-volatility spreads built from liquid ETFs, with **Newey-West** HAC standard errors implemented by hand. Single-index Jensen's alpha credits the manager for every style tilt in the book; this separates the tilts from what is left over.

### Also included

Risk metrics (Sharpe, Sortino, max drawdown, VaR/CVaR, beta, alpha), the efficient frontier with the current portfolio plotted against it, a correlation heatmap, an underwater plot, sector composition, shareable URLs, and **PDF export** via a print stylesheet — charts stay vector, not screenshots.

---

## Architecture

```
backend/          FastAPI service
  app/
    main.py         API routes and request orchestration
    data.py         yfinance price fetching, TTL-cached
    metrics.py      risk metrics via empyrical
    optimize.py     efficient frontier via PyPortfolioOpt
    backtest.py     in-sample cumulative growth
    schemas.py      Pydantic request/response models
    advanced.py     orchestrates the blocks below, each degrading independently
    risk_stats.py   PSR, DSR, stationary bootstrap, Cornish-Fisher,
                    Kupiec, Christoffersen, Newey-West
    risk_model.py   Ledoit-Wolf, Marchenko-Pastur, Euler risk
                    contributions, effective bets, risk parity
    volatility.py   GARCH(1,1) MLE, EWMA, ARCH-LM
    walkforward.py  out-of-sample rolling backtest with turnover costs
    factors.py      multi-factor regression on ETF spreads
    regimes.py      correlation and beta split by market regime
    blacklitterman.py  equilibrium prior, views, posterior frontier
  tests/            108 tests, seeded

frontend/         React + TypeScript + Vite
  src/
    App.tsx       layout and API integration
    components/   builder, metrics grid, charts, analytics panels
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

Returns `metrics`, `optimization` (frontier points, optimal weights, correlation matrix), and `backtest` (aligned date and growth series). Typically under a second against warm price data.

### `POST /api/advanced`

Same request body. Returns five independently-degradable blocks — `inference`, `risk_structure`, `volatility`, `walk_forward`, `factors` — each either carrying its payload or `{"available": false, "reason": "..."}`.

Split from `/api/analyze` deliberately: this path fits a GARCH model by maximum likelihood, bootstraps 2,000 resamples, and re-runs six optimizers at every rebalance, so it takes roughly 3-8 seconds. The frontend renders the headline metrics first and streams these in behind them rather than making the fast path wait.

### `POST /api/black-litterman`

Takes the portfolio plus a `views` array — each view `absolute` or `relative`, with a target return and a 0-1 confidence. Returns equilibrium and posterior expected returns, market and posterior-optimal weights, and both frontiers.

Interactive docs are served at `/docs`.

---

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

176 tests, all seeded, run in about 20 seconds. They are written against closed forms and known-parameter simulations rather than recorded outputs, so a failure is a real regression:

- GARCH is fitted to simulated paths with known ω, α, β and checked for parameter recovery.
- Risk contributions are asserted to sum exactly to portfolio volatility (Euler's theorem), and denoising to preserve the trace of the correlation matrix.
- Newey-West standard errors are checked to collapse to White's HC0 at zero lags, and to widen under injected autocorrelation.
- Kupiec's statistic is asserted against its closed form at the zero-exception and all-exception boundaries.
- Effective bets is checked to equal *n* for uncorrelated assets and 1 for perfectly correlated ones.
- The walk-forward engine is tested for **no lookahead**: rewriting the tail of the price history must leave the earliest out-of-sample block bit-identical.
- Black-Litterman is checked against its own boundaries: no views leaves the posterior exactly equal to the prior, full confidence binds the view exactly, and zero confidence leaves the prior untouched.
- Regime correlation is checked on synthetic data with a deliberately elevated tail correlation, and downside/upside capture is checked to recover a known ratio.

CI runs the suite plus a frontend typecheck and build on every push.

---

## Deployment

`render.yaml` is a Render Blueprint that provisions both services and wires them together — the frontend receives the API's hostname automatically via `VITE_API_BASE`, so there is no URL to hardcode.

1. Sign in at [render.com](https://render.com) with GitHub.
2. **New → Blueprint**, select this repository, and apply.

Render builds `quant-portfolio-api` (Python web service) and `quant-portfolio-web` (static site). On the free tier the API sleeps after ~15 minutes idle, so the first request after a pause takes roughly 50 seconds to wake it.

To deploy elsewhere, the frontend is a plain static bundle (`npm run build` → `dist/`) for Vercel, Netlify, or any static host — set `VITE_API_BASE` to the API's URL at build time. The backend runs on anything that hosts a Python web service, including the included `Dockerfile`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

---

## Notes and limitations

Mean-variance optimization is highly sensitive to its return estimates. Because expected returns come from historical means, the "optimal" weights describe what would have worked over the lookback window, not what will work going forward. Rather than leave that as a caveat, the walk-forward test measures it: the gap between in-sample and out-of-sample Sharpe *is* the cost of the assumption, and it is usually large enough that equal weighting wins.

The frontier is therefore presented as a diagnostic of the current allocation, not a recommendation.

Known limitations of the analysis itself:

- **Factor proxies are ETF spreads, not the Fama-French research factors.** They are tradeable, which makes the residual alpha more meaningful, but they are noisier and the ETFs carry their own fees and tracking error.
- **Marchenko-Pastur denoising assumes many assets and many observations.** With a handful of holdings the bulk edge is indicative rather than reliable, and the UI says so.
- **A single walk-forward path is one sample.** It shows what happened over one history, not a distribution of outcomes; combinatorially-purged cross-validation would be the stronger design.
- **GARCH(1,1) assumes Gaussian innovations,** which understates the tails it is being used to model. A Student-t or skewed-t likelihood would fit daily equity returns better.
- **Transaction costs are a flat 10bps on turnover** — no market impact, no bid-ask modelling, no borrow costs.
- **The market-implied risk aversion is inflated by the 0% risk-free rate.** δ = (market return − rf) / variance, so with rf pinned at zero a strong bull window pushes δ into the 4-9 range and the equilibrium returns scale with it. They are meaningful as *relative* magnitudes and as a prior to be updated, not as forecasts.
- **Idzorek confidence is calibrated for absolute views.** A relative view at 50% confidence moves the posterior far less than an absolute one, because omega for a spread scales with the spread's variance. Both endpoints still behave correctly; the panel says so rather than hiding it.
- Metrics use a 0% risk-free rate. Long-only, fully-invested constraints apply (no shorting or leverage).

Educational project — not investment advice.
