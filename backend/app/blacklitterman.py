"""Black-Litterman: equilibrium returns blended with explicit views.

The walk-forward test in `walkforward.py` documents the failure. This module is
the repair.

Mean-variance optimization collapses out-of-sample because the historical mean
is a hopeless estimator of expected return — its standard error shrinks only
with the square root of sample length, so a decade of data still leaves the
estimate wide enough to swamp the covariance structure the optimizer is trying
to exploit. Black-Litterman never estimates a mean at all. It starts from the
returns the market must already be assuming, obtained by reverse-optimizing
observed market-cap weights, and moves away from them only as far as an
explicit view — weighted by an explicit confidence — justifies.

With no views the posterior is exactly the prior, and the "optimal" portfolio is
the market portfolio. That is the honest default, and it is why the equilibrium
prior earns its own row in the walk-forward comparison: it forecasts nothing.

References
----------
Black & Litterman (1992), "Global Portfolio Optimization",
    Financial Analysts Journal 48(5).
Idzorek (2007), "A Step-by-Step Guide to the Black-Litterman Model",
    in Forecasting Expected Returns in the Financial Markets — the
    percentage-confidence specification of omega used here.
Walters (2014), "The Black-Litterman Model in Detail" — the closed form
    PyPortfolioOpt implements for Idzorek's method.
He & Litterman (1999), "The Intuition Behind Black-Litterman Model Portfolios".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf
from cachetools import TTLCache, cached
from cachetools.keys import hashkey
from pypfopt import EfficientFrontier
from pypfopt.black_litterman import (
    BlackLittermanModel,
    market_implied_prior_returns,
    market_implied_risk_aversion,
)

from app.risk_model import ledoit_wolf_shrinkage

logger = logging.getLogger(__name__)

# The scalar on the prior's uncertainty. 0.05 is PyPortfolioOpt's default and
# the value most of the literature uses; the posterior is famously insensitive
# to it once view confidences are specified directly, as Idzorek's method does.
TAU = 0.05

# Risk aversion is derived from the benchmark, but a window where the benchmark
# lost money produces a negative delta, which flips the sign of every implied
# return and makes the prior nonsense. Clamp into a defensible band — 2.5 is the
# textbook long-run equity value and the midpoint here.
_MIN_DELTA, _MAX_DELTA, _DEFAULT_DELTA = 0.5, 10.0, 2.5

# Share counts change only on buybacks and issuance, so they can be cached for
# far longer than prices.
_shares_cache: TTLCache = TTLCache(maxsize=512, ttl=86400)


@cached(cache=_shares_cache, key=lambda ticker: hashkey(ticker))
def fetch_shares_outstanding(ticker: str) -> float | None:
    """Shares outstanding, or None when Yahoo will not supply them.

    `fast_info.shares` is the primary source because it survives in production
    where `.info` does not — `.info` is blocked from datacenter IPs, the same
    constraint documented in `data.py`. The `.info` path is kept only as a
    local-development fallback, and indices and most ETFs have no share count
    at all, which is a legitimate answer rather than an error.
    """
    try:
        fast = yf.Ticker(ticker).fast_info
        shares = getattr(fast, "shares", None)
        if shares and float(shares) > 0:
            return float(shares)
    except Exception as exc:
        logger.info("fast_info.shares unavailable for %s: %s", ticker, exc)

    # Fallback: back out the share count from a market cap and a price.
    try:
        info = yf.Ticker(ticker).info or {}
        cap, price = info.get("marketCap"), info.get("currentPrice") or info.get("previousClose")
        if cap and price and float(price) > 0:
            return float(cap) / float(price)
    except Exception as exc:
        logger.info("info-derived shares unavailable for %s: %s", ticker, exc)

    return None


def market_cap_weights(tickers: tuple[str, ...], prices: pd.DataFrame) -> dict:
    """Market-capitalization weights for the prior.

    Caps are reconstructed as shares x price rather than read directly, because
    the direct `marketCap` field comes from the endpoint that fails in
    production. If any holding's share count is missing the whole prior degrades
    to equal weight: mixing real caps with fabricated ones would produce a prior
    that looks authoritative and is quietly wrong, which is worse than a stated
    fallback.
    """
    shares = {t: fetch_shares_outstanding(t) for t in tickers}
    missing = [t for t, s in shares.items() if not s]

    latest = prices.iloc[-1]
    if not missing:
        caps = pd.Series({t: shares[t] * float(latest[t]) for t in tickers})
        total = caps.sum()
        if total > 0:
            return {
                "weights": (caps / total).reindex(list(tickers)),
                "caps": caps.reindex(list(tickers)),
                "available": True,
                "missing": [],
            }

    equal = pd.Series(1.0 / len(tickers), index=list(tickers))
    return {
        "weights": equal,
        "caps": None,
        "available": False,
        "missing": missing,
    }


def build_view_matrices(
    views: list[dict], tickers: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Turn user views into the picking matrix P, target vector Q, and confidences.

    Both view types are expressed in one matrix rather than routing absolute
    views through PyPortfolioOpt's `absolute_views` shortcut, because mixing the
    two APIs in one model is not supported — and a uniform P also makes the
    per-view diagnostics uniform, since `P @ returns` is the quantity the view
    constrains regardless of its type.

    An absolute view is a single +1. A relative view is +1 on the long leg and
    -1 on the short leg, so its row sums to zero and it constrains a spread
    while saying nothing about the level.
    """
    index = {t: i for i, t in enumerate(tickers)}
    P = np.zeros((len(views), len(tickers)))
    Q = np.zeros((len(views), 1))
    confidences: list[float] = []

    for row, view in enumerate(views):
        asset = view["asset"]
        P[row, index[asset]] = 1.0
        if view["type"] == "relative":
            P[row, index[view["versus"]]] = -1.0
        Q[row, 0] = float(view["value"])
        confidences.append(float(view["confidence"]))

    return P, Q, confidences


def _frontier(mu: pd.Series, S: pd.DataFrame, n_points: int = 25) -> dict:
    """Trace an efficient frontier and locate its max-Sharpe portfolio.

    Long-only and fully invested, matching every other optimizer in the project.
    `BlackLittermanModel.bl_weights()` would be the shorter route but it solves
    the unconstrained problem and routinely returns large negative weights, so
    its output is not comparable with anything else the app displays.
    """
    ef_min = EfficientFrontier(mu, S)
    ef_min.min_volatility()
    mv_ret, mv_vol, _ = ef_min.portfolio_performance(risk_free_rate=0.0)

    try:
        ef_max = EfficientFrontier(mu, S)
        ef_max.max_sharpe(risk_free_rate=0.0)
        weights = ef_max.clean_weights()
        ms_ret, ms_vol, _ = ef_max.portfolio_performance(risk_free_rate=0.0)
        available = True
    except Exception:
        # Every asset's posterior return is non-positive; min-volatility is
        # always solvable and stands in, as it does in `optimize.py`.
        weights = ef_min.clean_weights()
        ms_ret, ms_vol = mv_ret, mv_vol
        available = False

    points = []
    for target in np.linspace(mv_ret, float(mu.max()) * 0.98, n_points):
        try:
            ef = EfficientFrontier(mu, S)
            ef.efficient_return(target_return=target)
            ret, vol, _ = ef.portfolio_performance(risk_free_rate=0.0)
            points.append(
                {
                    "return": float(ret),
                    "volatility": float(vol),
                    "sharpe": float(ret / vol) if vol > 0 else 0.0,
                }
            )
        except Exception:
            continue

    return {
        "frontier": points,
        "weights": {k: float(v) for k, v in weights.items()},
        "point": {
            "return": float(ms_ret),
            "volatility": float(ms_vol),
            "sharpe": float(ms_ret / ms_vol) if ms_vol > 0 else 0.0,
        },
        "max_sharpe_available": available,
    }


def implied_risk_aversion(bench_prices: pd.Series, periods_per_year: int) -> tuple[float, bool]:
    """Market-implied risk aversion, clamped to a usable range.

    delta = (R_market - R_f) / sigma^2. Over a losing window the numerator is
    negative and so is delta, which would invert every implied return. The
    second element of the return says whether the raw value was usable, so the
    UI can say that the prior fell back to a long-run assumption.
    """
    try:
        raw = float(
            market_implied_risk_aversion(
                bench_prices, frequency=periods_per_year, risk_free_rate=0.0
            )
        )
    except Exception as exc:
        logger.info("implied risk aversion unavailable: %s", exc)
        return _DEFAULT_DELTA, False

    if not np.isfinite(raw) or raw < _MIN_DELTA or raw > _MAX_DELTA:
        return _DEFAULT_DELTA, False
    return raw, True


def run_black_litterman(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    bench_prices: pd.Series,
    views: list[dict],
    periods_per_year: int = 252,
) -> dict:
    """Blend equilibrium returns with views and re-optimize.

    Returns both the prior and posterior frontiers so the caller can show how
    far the views moved the opportunity set, not just where it ended up.
    """
    tickers = tuple(returns.columns)

    # Shrunk covariance, consistent with the rest of the project. The sample
    # matrix is the input the whole app argues against trusting.
    S = ledoit_wolf_shrinkage(returns, frequency=periods_per_year)["covariance"]

    delta, delta_from_market = implied_risk_aversion(bench_prices, periods_per_year)
    caps = market_cap_weights(tickers, prices)

    # Pi = delta * S * w_mkt. Risk-free rate is 0 to match the Sharpe ratios
    # reported everywhere else in the app.
    pi = market_implied_prior_returns(caps["weights"], delta, S, risk_free_rate=0.0)
    pi = pi.reindex(list(tickers))

    P, Q, confidences = build_view_matrices(views, tickers)

    if views:
        bl = BlackLittermanModel(
            S,
            pi=pi,
            Q=Q,
            P=P,
            omega="idzorek",
            view_confidences=confidences,
            tau=TAU,
            risk_aversion=delta,
        )
        posterior = bl.bl_returns().reindex(list(tickers))
        posterior_cov = bl.bl_cov()
    else:
        # With no views the posterior is identically the prior. Constructing the
        # model with empty matrices gives the same answer, but skipping it keeps
        # the no-view case free of any dependence on solver behaviour.
        posterior = pi.copy()
        posterior_cov = S

    prior_side = _frontier(pi, S)
    posterior_side = _frontier(posterior, S)

    # P @ returns is exactly what each view constrains, for both view types, so
    # the same three numbers describe an absolute and a relative view.
    #
    # `adoption` is worth reading carefully. For an absolute view it tracks the
    # confidence slider almost exactly — 0.8 confidence moves the posterior 80%
    # of the way to the view — because that is the case Idzorek's closed form is
    # derived for. For a relative view it does not: omega scales with the
    # variance of the *spread*, which can dwarf the prior's uncertainty about
    # it, so a half-confidence spread view often barely moves. Both endpoints
    # still behave (0 changes nothing, 1 binds exactly); only the middle is
    # non-linear. Showing adoption rather than just echoing the slider is the
    # point — it tells the user what the model actually did with their view.
    diagnostics = []
    for row, view in enumerate(views):
        p = P[row]
        prior_implied = float(p @ pi.to_numpy())
        post_implied = float(p @ posterior.to_numpy())
        target = float(Q[row, 0])
        gap = target - prior_implied
        diagnostics.append(
            {
                "type": view["type"],
                "asset": view["asset"],
                "versus": view.get("versus"),
                "value": target,
                "confidence": float(view["confidence"]),
                "prior_implied": prior_implied,
                "posterior_implied": post_implied,
                # Share of the distance from the prior to the view that the
                # posterior actually travelled. 1.0 means the view fully bound.
                "adoption": float((post_implied - prior_implied) / gap) if abs(gap) > 1e-12 else 1.0,
            }
        )

    return {
        "tickers": list(tickers),
        "prior_returns": [float(v) for v in pi],
        "posterior_returns": [float(v) for v in posterior],
        "market_weights": [float(v) for v in caps["weights"]],
        "prior": prior_side,
        "posterior": posterior_side,
        "views": diagnostics,
        "risk_aversion": float(delta),
        "risk_aversion_from_market": bool(delta_from_market),
        "tau": TAU,
        "market_caps_available": bool(caps["available"]),
        "missing_caps": caps["missing"],
        "posterior_volatility": [
            float(np.sqrt(posterior_cov.to_numpy()[i, i])) for i in range(len(tickers))
        ],
    }


def historical_cap_weights(
    prices: pd.DataFrame, shares: dict[str, float]
) -> pd.DataFrame | None:
    """Market-cap weights through time, for the walk-forward comparison.

    Caps are rebuilt as shares x price_t so the weights move with the market
    rather than being pinned to today's snapshot. The share count itself is
    today's and is held constant, which ignores buybacks and issuance — a real
    approximation, but a far smaller one than applying current cap weights to a
    five-year-old market, and it introduces no dependence on future *prices*,
    which is the lookahead that would actually invalidate the backtest.

    Returns None when any share count is missing, so the caller can drop the
    strategy rather than silently substituting a different one.
    """
    if any(not shares.get(t) for t in prices.columns):
        return None

    caps = prices.mul(pd.Series(shares).reindex(prices.columns), axis=1)
    totals = caps.sum(axis=1)
    if (totals <= 0).any():
        return None
    return caps.div(totals, axis=0)
