"""Orchestration for the advanced analytics endpoint.

Split from `/api/analyze` deliberately. The headline metrics need one covariance
matrix and return in well under a second; the work here fits a GARCH model by
maximum likelihood, bootstraps two thousand resamples, and re-runs six
optimizers at every rebalance across the sample. Bundling them would make the
fast path wait on the slow one for no reason.

Every block degrades independently: a failure in the factor download must not
cost the user their walk-forward results.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from app import regimes, risk_model, risk_stats, volatility, walkforward
from app.blacklitterman import fetch_shares_outstanding, historical_cap_weights
from app.factors import run_factor_analysis

logger = logging.getLogger(__name__)


def _json_safe(obj):
    """Replace non-finite floats with None, recursively.

    NaN and infinity are not valid JSON, and Starlette's encoder raises rather
    than emitting them — so a single undefined ratio anywhere in this response
    turns the whole request into a 500. Several statistics here are legitimately
    undefined on some inputs (a Calmar ratio with no drawdown, a beta ratio
    whose legs straddle zero), so they are mapped to null and the frontend
    renders them as a dash.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _block(name: str, fn):
    """Run one analysis block, degrading to an error payload on failure.

    Unlike the bare `except Exception` elsewhere in this codebase, this logs
    what went wrong — a silently-empty panel in production is not debuggable.
    """
    try:
        return {"available": True, **fn()}
    except Exception as exc:
        logger.warning("advanced analytics block %r failed: %s", name, exc, exc_info=True)
        return {"available": False, "reason": str(exc)[:200]}


def inference_block(port_ret: pd.Series, periods: int, trial_sharpes: list[float]) -> dict:
    """Is the track record distinguishable from luck?"""
    out = {
        "moments": risk_stats.moments(port_ret),
        "psr": risk_stats.probabilistic_sharpe_ratio(port_ret, 0.0, periods),
        "bootstrap": risk_stats.bootstrap_sharpe_ci(port_ret, periods, n_boot=2000),
        "tail": risk_stats.cornish_fisher_var(port_ret, alpha=0.05),
    }

    # The frontier sweep is a search over candidate portfolios, so the reported
    # best is subject to selection bias and has to be deflated for it.
    if len(trial_sharpes) >= 2:
        try:
            out["dsr"] = risk_stats.deflated_sharpe_ratio(port_ret, trial_sharpes, periods)
        except ValueError as exc:
            logger.info("deflated Sharpe unavailable: %s", exc)

    # Needs a year of history before the rolling window has anything to test.
    try:
        window = min(250, max(60, port_ret.size // 3))
        out["var_backtest"] = risk_stats.var_backtest(port_ret, alpha=0.05, window=window)
    except ValueError as exc:
        logger.info("VaR backtest unavailable: %s", exc)

    return out


def risk_structure_block(
    returns: pd.DataFrame, weights: dict[str, float], periods: int
) -> dict:
    """Where does the risk actually sit, and is the covariance trustworthy?"""
    tickers = list(returns.columns)
    w = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)
    total = w.sum()
    if total > 0:
        w = w / total

    shrunk = risk_model.ledoit_wolf_shrinkage(returns, frequency=periods)
    cov = shrunk["covariance"].to_numpy()
    denoise = risk_model.marchenko_pastur_denoise(returns, frequency=periods)

    contrib = risk_model.risk_contributions(w, cov)
    bets = risk_model.effective_number_of_bets(w, cov)
    erc = risk_model.equal_risk_contribution(cov)

    return {
        "tickers": tickers,
        "weights": [float(v) for v in w],
        "shrinkage": {
            "intensity": shrunk["shrinkage_intensity"],
            "condition_number_sample": shrunk["condition_number_sample"],
            "condition_number_shrunk": shrunk["condition_number_shrunk"],
            "n_observations": shrunk["n_observations"],
            "n_assets": shrunk["n_assets"],
        },
        "eigen": {
            "eigenvalues": denoise["eigenvalues"],
            "variance_explained": denoise["variance_explained"],
            "cumulative_variance_explained": denoise["cumulative_variance_explained"],
            "lambda_plus": denoise["lambda_plus"],
            "n_signal_factors": denoise["n_signal_factors"],
            "market_factor_share": denoise["market_factor_share"],
            "reliable": denoise["reliable"],
        },
        "contributions": {
            "portfolio_volatility": contrib["portfolio_volatility"],
            "percent": contrib["percent_contributions"],
            "marginal": contrib["marginal_contributions"],
            "diversification_ratio": contrib["diversification_ratio"],
            "effective_positions": contrib["effective_positions"],
        },
        "effective_bets": {
            "value": bets["effective_bets"],
            "n_assets": bets["n_assets"],
            "concentration": bets["concentration"],
        },
        "risk_parity": {
            "weights": erc["weights"],
            "volatility": erc["volatility"],
            "converged": erc["converged"],
        },
    }


def volatility_block(port_ret: pd.Series, periods: int) -> dict:
    """Is volatility constant, and if not, where is it heading?"""
    arch = volatility.arch_lm_test(port_ret, lags=5)
    ewma = volatility.ewma_volatility(port_ret, periods_per_year=periods)

    out = {
        "arch_test": arch,
        "ewma": {
            "current": ewma["current"],
            "unconditional": ewma["unconditional"],
            "half_life_days": ewma["half_life_days"],
            # Thinned for transport; the shape is what matters, not every point.
            "series": [round(v, 5) for v in ewma["volatility"]],
        },
    }

    try:
        garch = volatility.fit_garch(port_ret, periods_per_year=periods, horizon=21)
        out["garch"] = {
            k: garch[k]
            for k in (
                "omega",
                "alpha",
                "beta",
                "persistence",
                "converged",
                "log_likelihood",
                "aic",
                "bic",
                "current_volatility",
                "long_run_volatility",
                "half_life_days",
                "vol_ratio",
                "forecast",
                "forecast_horizon",
            )
        }
        out["garch"]["conditional_volatility"] = [
            round(v, 5) for v in garch["conditional_volatility"]
        ]
    except ValueError as exc:
        logger.info("GARCH unavailable: %s", exc)

    return out


def _cap_weight_panel(prices: pd.DataFrame | None) -> pd.DataFrame | None:
    """Market-cap weights through time, or None if any share count is missing.

    Adds the Black-Litterman equilibrium prior to the walk-forward comparison.
    Share counts are cached for a day, so only the first request of a session
    pays for the lookups, and a failure here must cost nothing but that one row.
    """
    if prices is None:
        return None
    try:
        shares = {t: fetch_shares_outstanding(t) for t in prices.columns}
        return historical_cap_weights(prices, shares)
    except Exception as exc:
        logger.info("market-cap weights unavailable for the walk-forward: %s", exc)
        return None


def run_advanced(
    returns: pd.DataFrame,
    bench_returns: pd.Series,
    port_ret: pd.Series,
    weights: dict[str, float],
    periods: int,
    period: str,
    trial_sharpes: list[float],
    prices: pd.DataFrame | None = None,
) -> dict:
    """Assemble every advanced block, each independently degradable."""
    cap_weights = _cap_weight_panel(prices)

    return _json_safe({
        "inference": _block("inference", lambda: inference_block(port_ret, periods, trial_sharpes)),
        "risk_structure": _block(
            "risk_structure", lambda: risk_structure_block(returns, weights, periods)
        ),
        "volatility": _block("volatility", lambda: volatility_block(port_ret, periods)),
        # A groupby and a few corr() calls — milliseconds, so it rides along
        # here rather than costing the frontend another round trip.
        "regimes": _block(
            "regimes",
            lambda: regimes.regime_analysis(returns, bench_returns, port_ret=port_ret),
        ),
        "walk_forward": _block(
            "walk_forward",
            lambda: walkforward.walk_forward(
                returns, weights, periods_per_year=periods, cap_weights=cap_weights
            ),
        ),
        # Already returns its own availability flag rather than raising.
        "factors": run_factor_analysis(port_ret, bench_returns, period, periods),
    })
