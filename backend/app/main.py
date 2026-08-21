import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import sectors
from app.advanced import run_advanced
from app.analytics import drawdown_series, return_distribution, sector_weights
from app.backtest import run_backtest
from app.blacklitterman import run_black_litterman
from app.data import fetch_meta, fetch_prices, fetch_returns
from app.metrics import compute_metrics, portfolio_returns
from app.optimize import run_optimization
from app.schemas import AnalyzeResponse, BlackLittermanRequest, PortfolioRequest

logging.basicConfig(level=logging.INFO)

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


def _resolve_holdings(req: PortfolioRequest) -> tuple[tuple[str, ...], dict[str, float]]:
    """Normalize tickers and validate the weight vector.

    Duplicate tickers used to collapse silently in the dict comprehension and
    then trip the sum check with a baffling message ("weights must sum to 1.0,
    got 0.500" for two half-weighted copies of the same symbol), so they are
    now rejected by name.
    """
    seen: list[str] = []
    duplicates: set[str] = set()
    for h in req.holdings:
        t = h.ticker.strip().upper()
        if not t:
            raise HTTPException(status_code=422, detail="Ticker cannot be blank")
        if t in seen:
            duplicates.add(t)
        seen.append(t)

    if duplicates:
        raise HTTPException(
            status_code=422,
            detail=f"Duplicate ticker(s): {', '.join(sorted(duplicates))}. Combine them into one holding.",
        )

    weights = {t: h.weight for t, h in zip(seen, req.holdings)}
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-3:
        raise HTTPException(
            status_code=422, detail=f"Weights must sum to 1.0 (got {total:.3f})"
        )

    return tuple(seen), weights


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: PortfolioRequest):
    tickers, weights = _resolve_holdings(req)

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


@app.post("/api/advanced")
def advanced(req: PortfolioRequest):
    """Statistical inference, risk decomposition, GARCH, walk-forward, factors.

    Deliberately not typed with a `response_model`. Each block is either its
    full payload or `{available: false, reason: ...}` when its dependencies fail,
    and expressing that union for five independently-degradable blocks costs
    a few hundred lines of Pydantic to describe a shape the frontend already
    guards on. The fast `/api/analyze` path stays strictly typed.

    Runs in roughly 3-8 seconds against warm price data.
    """
    tickers, weights = _resolve_holdings(req)

    try:
        prices = fetch_prices(tickers, req.period)
        returns = fetch_returns(tickers, req.period)
        bench_prices = fetch_prices((req.benchmark,), req.period)
        bench_returns = bench_prices[req.benchmark].pct_change().dropna()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    periods = sectors.periods_per_year(tickers)
    port_ret = portfolio_returns(returns, weights)

    # The frontier sweep is the search whose selection bias the deflated Sharpe
    # ratio corrects for, so its points are the trial population.
    optimization = run_optimization(prices, weights, periods)
    trial_sharpes = [p["sharpe"] for p in optimization["frontier"]]

    return run_advanced(
        returns=returns,
        bench_returns=bench_returns,
        port_ret=port_ret,
        weights=weights,
        periods=periods,
        period=req.period,
        trial_sharpes=trial_sharpes,
        prices=prices,
    )


@app.post("/api/black-litterman")
def black_litterman(req: BlackLittermanRequest):
    """Equilibrium implied returns, optionally blended with user views.

    Fast enough to call on every slider change: the covariance and the prior are
    the only real work, and both come from cached prices. Views only add a small
    linear solve on top.
    """
    tickers, _ = _resolve_holdings(req)

    try:
        prices = fetch_prices(tickers, req.period)
        returns = fetch_returns(tickers, req.period)
        bench_prices = fetch_prices((req.benchmark,), req.period)[req.benchmark]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if len(tickers) < 2:
        raise HTTPException(
            status_code=422, detail="Black-Litterman needs at least two holdings"
        )

    try:
        return run_black_litterman(
            prices=prices,
            returns=returns,
            bench_prices=bench_prices,
            views=[v.model_dump() for v in req.views],
            periods_per_year=sectors.periods_per_year(tickers),
        )
    except Exception as e:
        logging.getLogger(__name__).warning("Black-Litterman failed: %s", e, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Could not solve the model: {e}")
