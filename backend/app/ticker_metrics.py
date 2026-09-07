"""Single-ticker quantitative snapshot for the financial-advisor agent.

The agent's data_analyst persona needs one ticker's numbers, not a weighted
portfolio, so this reuses the existing price/volatility/factor/regime modules
directly rather than routing through the portfolio-shaped `/api/analyze` and
`/api/advanced` endpoints.
"""

from __future__ import annotations

import yfinance as yf

from app.data import fetch_meta, fetch_prices, fetch_returns
from app.factors import run_factor_analysis
from app.regime_classifier import compute_features, fit_regimes
from app.sectors import periods_per_year


def build_ticker_metrics(ticker: str, period: str = "3y") -> dict:
    ticker = ticker.upper().strip()
    prices = fetch_prices((ticker,), period)
    returns = fetch_returns((ticker,), period)
    periods = periods_per_year((ticker,))

    port_ret = returns[ticker]
    last_price = float(prices[ticker].iloc[-1])
    ann_vol = float(port_ret.std(ddof=1) * (periods**0.5))
    ann_return = float(port_ret.mean() * periods)
    max_drawdown = float(((prices[ticker] / prices[ticker].cummax()) - 1.0).min())

    try:
        pe_ratio = yf.Ticker(ticker).info.get("trailingPE")
    except Exception:
        pe_ratio = None

    regime: dict = {"available": False}
    try:
        features = compute_features(returns)
        fit = fit_regimes(features)
        idx = -1
        label_idx = int(fit["labels"].iloc[idx])
        regime = {
            "available": True,
            "label": fit["level_names"][label_idx],
            "confidence": float(fit["confidence"].iloc[idx]),
        }
    except ValueError as e:
        regime = {"available": False, "reason": str(e)}

    bench = fetch_returns(("SPY",), period)["SPY"].reindex(port_ret.index).dropna()
    aligned = port_ret.reindex(bench.index)
    factor_analysis = run_factor_analysis(aligned, bench, period, periods)

    meta = fetch_meta(ticker)

    return {
        "ticker": ticker,
        "period": period,
        "name": meta.get("name"),
        "sector": meta.get("sector"),
        "last_price": round(last_price, 2),
        "annualized_return": round(ann_return, 4),
        "annualized_volatility": round(ann_vol, 4),
        "max_drawdown": round(max_drawdown, 4),
        "dividend_yield": meta.get("dividend_yield"),
        "trailing_pe": round(pe_ratio, 2) if isinstance(pe_ratio, (int, float)) else None,
        "regime": regime,
        "factors": factor_analysis,
    }
