from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.backtest import run_backtest
from app.data import fetch_prices, fetch_returns
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

    port_ret = portfolio_returns(returns, weights)
    metrics = compute_metrics(port_ret, bench_returns)
    optimization = run_optimization(prices, weights)
    backtest = run_backtest(returns, bench_returns, weights, optimization["max_sharpe_weights"])

    return {
        "metrics": metrics,
        "optimization": optimization,
        "backtest": backtest,
    }
