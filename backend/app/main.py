from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import sectors
from app.analytics import drawdown_series, return_distribution, sector_weights
from app.backtest import run_backtest
from app.data import fetch_meta, fetch_prices, fetch_returns
from app.metrics import compute_metrics, portfolio_returns
from app.optimize import run_optimization
from app.schemas import AnalyzeResponse, PortfolioRequest

app = FastAPI(
    title="Quant Portfolio Analytics API",
    description="Computes risk metrics, an efficient frontier, and a backtest for a user-supplied portfolio.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: PortfolioRequest):
    tickers = tuple(h.ticker.upper() for h in req.holdings)
    weights = {h.ticker.upper(): h.weight for h in req.holdings}

    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-3:
        raise HTTPException(status_code=422, detail=f"Weights must sum to 1.0 (got {total_weight:.3f})")

    try:
        prices = fetch_prices(tickers, req.period)
        returns = fetch_returns(tickers, req.period)
        bench_prices = fetch_prices((req.benchmark,), req.period)
        bench_returns = bench_prices[req.benchmark].pct_change().dropna()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    periods = sectors.periods_per_year(tickers)
    port_ret = portfolio_returns(returns, weights)
    metrics = compute_metrics(port_ret, bench_returns, periods)
    optimization = run_optimization(prices, weights, periods)
    backtest = run_backtest(returns, bench_returns, weights, optimization["max_sharpe_weights"])

    # Composition depends on a slower, less reliable upstream endpoint than the
    # price data, so it must never take the rest of the analysis down with it.
    try:
        holdings_meta = [{**fetch_meta(t), "weight": weights[t]} for t in tickers]
        composition = {
            "holdings": holdings_meta,
            "sector_weights": sector_weights(holdings_meta),
            "available": any(h.get("sector") for h in holdings_meta),
        }
    except Exception:
        composition = {"holdings": [], "sector_weights": {}, "available": False}

    crypto = [t for t in tickers if sectors.is_crypto(t)]

    return {
        "calendar": {
            "periods_per_year": periods,
            "has_crypto": bool(crypto),
            "mixed": bool(crypto) and len(crypto) < len(tickers),
        },
        "metrics": metrics,
        "optimization": optimization,
        "backtest": backtest,
        "drawdown": drawdown_series(port_ret),
        "distribution": return_distribution(port_ret),
        "composition": composition,
    }
