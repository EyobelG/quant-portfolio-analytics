import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models


def _sharpe(ret: float, vol: float, rf: float = 0.0) -> float:
    return (ret - rf) / vol if vol > 0 else 0.0


def _point(ret: float, vol: float) -> dict:
    return {"return": ret, "volatility": vol, "sharpe": _sharpe(ret, vol)}


def run_optimization(
    prices: pd.DataFrame, current_weights: dict[str, float], periods_per_year: int = 252
) -> dict:
    # frequency must match the trading calendar, or the frontier is drawn in
    # different units than the metrics beside it.
    mu = expected_returns.mean_historical_return(prices, frequency=periods_per_year)
    S = risk_models.sample_cov(prices, frequency=periods_per_year)

    # Max Sharpe portfolio
    ef_sharpe = EfficientFrontier(mu, S)
    ef_sharpe.max_sharpe()
    max_sharpe_weights = ef_sharpe.clean_weights()
    ms_ret, ms_vol, _ = ef_sharpe.portfolio_performance()

    # Min volatility portfolio
    ef_minvol = EfficientFrontier(mu, S)
    ef_minvol.min_volatility()
    min_vol_weights = ef_minvol.clean_weights()
    mv_ret, mv_vol, _ = ef_minvol.portfolio_performance()

    # Sweep target returns between min-vol and max achievable return to trace the frontier
    target_returns = np.linspace(mv_ret, mu.max() * 0.98, 25)
    frontier_points = []
    for target in target_returns:
        try:
            ef = EfficientFrontier(mu, S)
            ef.efficient_return(target_return=target)
            ret, vol, _ = ef.portfolio_performance()
            frontier_points.append(_point(ret, vol))
        except Exception:
            continue

    # Current portfolio's own risk/return, using the same mu/S
    weight_vector = np.array([current_weights.get(t, 0.0) for t in prices.columns])
    current_ret = float(np.dot(weight_vector, mu))
    current_vol = float(np.sqrt(weight_vector.T @ S.values @ weight_vector))

    corr = prices.pct_change().dropna().corr()

    return {
        "frontier": frontier_points,
        "max_sharpe_weights": max_sharpe_weights,
        "max_sharpe_point": _point(ms_ret, ms_vol),
        "min_vol_weights": min_vol_weights,
        "min_vol_point": _point(mv_ret, mv_vol),
        "current_point": _point(current_ret, current_vol),
        "correlation_matrix": corr.round(4).to_dict(),
    }
