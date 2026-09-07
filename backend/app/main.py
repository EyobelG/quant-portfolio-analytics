import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import advisor, broker, sectors
from app.advanced import cap_weight_panel, run_advanced
from app.analytics import drawdown_series, return_distribution, sector_weights
from app.backtest import run_backtest
from app.blacklitterman import run_black_litterman
from app.data import fetch_meta, fetch_prices, fetch_returns
from app.metrics import compute_metrics, portfolio_returns
from app.optimize import run_optimization
from app.ticker_metrics import build_ticker_metrics
from app.schemas import (
    AnalyzeResponse,
    BlackLittermanRequest,
    PortfolioRequest,
    RebalanceSubmitRequest,
)
from app.walkforward import select_best_strategy

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


app.include_router(advisor.router)


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


@app.get("/api/ticker/{ticker}/metrics")
def ticker_metrics(ticker: str, period: str = "3y"):
    """Single-ticker snapshot for the financial-advisor agent's data analyst.

    Not portfolio-shaped like `/api/analyze` — one symbol, one weight (1.0).
    """
    try:
        return build_ticker_metrics(ticker, period)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


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


def _broker_call(fn):
    """Run a broker.py call, mapping missing keys and Alpaca failures to clean HTTP errors."""
    try:
        return fn()
    except broker.BrokerNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logging.getLogger(__name__).warning("Alpaca request failed: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Alpaca request failed: {e}")


@app.get("/api/broker/account")
def broker_account():
    return _broker_call(broker.get_account)


@app.get("/api/broker/positions")
def broker_positions():
    return _broker_call(broker.get_positions)


@app.get("/api/broker/orders")
def broker_orders():
    return {"orders": _broker_call(broker.get_recent_orders)}


@app.post("/api/broker/rebalance/preview")
def broker_rebalance_preview(req: PortfolioRequest):
    """Pick whichever strategy currently backtests best out-of-sample and diff
    its target weights against the live paper account, without submitting
    anything. `/api/broker/rebalance/submit` takes this response's `orders`
    list unchanged."""
    tickers, weights = _resolve_holdings(req)
    if len(tickers) < 2:
        raise HTTPException(status_code=422, detail="Rebalancing needs at least two holdings")

    try:
        prices = fetch_prices(tickers, req.period)
        returns = fetch_returns(tickers, req.period)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    periods = sectors.periods_per_year(tickers)
    cap_weights = cap_weight_panel(prices)

    try:
        choice = select_best_strategy(
            returns, weights, periods_per_year=periods, cap_weights=cap_weights
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    account = _broker_call(broker.get_account)
    positions = _broker_call(broker.get_positions)
    orders = broker.build_rebalance_orders(choice["weights"], account["equity"], positions)

    return {
        "strategy": choice["key"],
        "strategy_label": choice["label"],
        "strategy_sharpe": choice["sharpe_ratio"],
        "target_weights": choice["weights"],
        "account": account,
        "orders": [broker.order_preview_dict(o) for o in orders],
    }


@app.post("/api/broker/rebalance/submit")
def broker_rebalance_submit(req: RebalanceSubmitRequest):
    """Submit exactly the orders the caller previewed and confirmed."""
    orders = [
        broker.OrderPreview(
            ticker=o.ticker, side=o.side, notional=o.notional, current_value=0.0, target_value=0.0
        )
        for o in req.orders
    ]
    results = _broker_call(lambda: broker.submit_orders(orders))
    return {"results": [broker.order_result_dict(r) for r in results]}
