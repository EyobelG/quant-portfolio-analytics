"""Walk-forward out-of-sample evaluation.

The backtest in `backtest.py` fits optimal weights on the whole sample and then
scores them on that same sample, so the optimized line beats the original by
construction. It is a description of the past, not a test of anything.

This module does the honest version: at each rebalance the optimizer sees only
data that existed at the time, the resulting weights are held forward through
returns it has never seen, and the weights drift with the market in between.
Transaction costs are charged on the turnover that rebalancing actually
generates.

The expected result is that mean-variance optimization loses to equal weight.
That is not a bug in the implementation — it is the DeMiguel, Garlappi & Uppal
(2009) finding, and reproducing it is the point.

References
----------
DeMiguel, Garlappi & Uppal (2009), "Optimal Versus Naive Diversification: How
    Inefficient is the 1/N Portfolio Strategy?", Review of Financial Studies 22(5).
López de Prado (2016), "Building Diversified Portfolios that Outperform Out of
    Sample", Journal of Portfolio Management 42(4) — hierarchical risk parity.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.hierarchical_portfolio import HRPOpt

from app.regime_classifier import compute_features, fit_regimes
from app.risk_model import equal_risk_contribution, ledoit_wolf_shrinkage

# Minimum out-of-sample days before the comparison means anything. Below roughly
# one trading year the Sharpe standard error swamps any difference between
# strategies.
_MIN_OOS_DAYS = 200


def _max_sharpe_weights(train: pd.DataFrame, periods: int, shrink: bool) -> np.ndarray:
    """Max-Sharpe weights, optionally on a shrunk covariance matrix."""
    prices = (1 + train).cumprod()
    mu = expected_returns.mean_historical_return(prices, frequency=periods)
    if shrink:
        S = ledoit_wolf_shrinkage(train, frequency=periods)["covariance"]
    else:
        S = risk_models.sample_cov(prices, frequency=periods)

    ef = EfficientFrontier(mu, S)
    try:
        ef.max_sharpe(risk_free_rate=0.0)
    except Exception:
        # No allocation beats the risk-free rate over this window; minimum
        # volatility is always solvable and is the natural stand-in.
        ef = EfficientFrontier(mu, S)
        ef.min_volatility()
    return np.array([ef.clean_weights()[c] for c in train.columns], dtype=float)


def _min_vol_weights(train: pd.DataFrame, periods: int) -> np.ndarray:
    prices = (1 + train).cumprod()
    mu = expected_returns.mean_historical_return(prices, frequency=periods)
    S = risk_models.sample_cov(prices, frequency=periods)
    ef = EfficientFrontier(mu, S)
    ef.min_volatility()
    return np.array([ef.clean_weights()[c] for c in train.columns], dtype=float)


def _risk_parity_weights(train: pd.DataFrame, periods: int) -> np.ndarray:
    S = ledoit_wolf_shrinkage(train, frequency=periods)["covariance"].to_numpy()
    return np.array(equal_risk_contribution(S)["weights"], dtype=float)


def _hrp_weights(train: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        # PyPortfolioOpt 1.5.5 trips a pandas dtype FutureWarning inside its own
        # cluster-weight recursion; nothing we can fix from here.
        warnings.simplefilter("ignore", FutureWarning)
        w = HRPOpt(train).optimize()
    return np.array([w[c] for c in train.columns], dtype=float)


_STRATEGIES = {
    "max_sharpe": lambda tr, p: _max_sharpe_weights(tr, p, shrink=False),
    "max_sharpe_shrunk": lambda tr, p: _max_sharpe_weights(tr, p, shrink=True),
    "min_volatility": lambda tr, p: _min_vol_weights(tr, p),
    "risk_parity": lambda tr, p: _risk_parity_weights(tr, p),
    "hrp": lambda tr, p: _hrp_weights(tr),
    "equal_weight": lambda tr, p: np.full(tr.shape[1], 1.0 / tr.shape[1]),
}

# Snapshot of every non-ML candidate, captured before `regime_ml` is added
# below -- the regime solver picks among these, and must not be able to pick
# itself.
_BASE_STRATEGIES = dict(_STRATEGIES)

# Below this many in-sample days for a regime, its Sharpe is too noisy to
# trust as "the strategy that wins in this regime" -- the days are pooled
# into whichever adjacent regime the current day actually falls in instead of
# ranking on a handful of observations.
_MIN_DAYS_PER_REGIME = 20


def _regime_ml_weights(train: pd.DataFrame, periods: int) -> np.ndarray:
    """Weights of whichever strategy currently backtests best in today's regime.

    Fits the regime classifier on `train` only (never on data the walk-forward
    loop has not yet reached), then scores every other candidate strategy's
    full-`train` weights against the daily returns realized on the subset of
    `train` days sharing the same regime as the most recent one. This is a
    real, testable claim -- not a hand-tuned rule -- and it is judged by
    exactly the same out-of-sample walk-forward test as every other row in
    this table.
    """
    try:
        features = compute_features(train)
        regimes = fit_regimes(features)
    except Exception:
        return np.full(train.shape[1], 1.0 / train.shape[1])

    labels = regimes["labels"]
    current_regime = labels.iloc[-1]
    aligned = train.loc[labels.index]

    best_key, best_sharpe = None, -np.inf
    for key, solver in _BASE_STRATEGIES.items():
        try:
            w = np.clip(np.asarray(solver(train, periods), dtype=float), 0, None)
            total = w.sum()
            w = w / total if total > 0 else np.full(train.shape[1], 1.0 / train.shape[1])
        except Exception:
            continue

        in_regime = aligned.loc[labels == current_regime]
        if len(in_regime) < _MIN_DAYS_PER_REGIME:
            in_regime = aligned  # too few regime-specific days to rank on; fall back to the whole window
        r = in_regime.to_numpy(dtype=float) @ w
        sd = r.std(ddof=1)
        sharpe = float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0

        if sharpe > best_sharpe:
            best_key, best_sharpe = key, sharpe
            best_weights = w

    if best_key is None:
        return np.full(train.shape[1], 1.0 / train.shape[1])
    return best_weights


_STRATEGIES["regime_ml"] = _regime_ml_weights

STRATEGY_LABELS = {
    "max_sharpe": "Max Sharpe",
    "max_sharpe_shrunk": "Max Sharpe (Ledoit-Wolf)",
    "min_volatility": "Min Volatility",
    "risk_parity": "Risk Parity (ERC)",
    "hrp": "Hierarchical Risk Parity",
    "equal_weight": "Equal Weight (1/N)",
    "market_equilibrium": "Market Equilibrium (BL prior)",
    "buy_and_hold": "Your Portfolio (buy & hold)",
    "regime_ml": "ML Regime Classifier",
}


def _equilibrium_solver(cap_weights: pd.DataFrame):
    """Market-cap weights as of the last day the optimizer is allowed to see.

    This is the Black-Litterman prior with no views attached: the posterior
    collapses to the prior, and the optimal portfolio is the market portfolio.
    It is the only rule in this comparison that estimates no expected return at
    all, which is the entire reason it is worth running alongside the others.

    `train` is always `returns.iloc[:start]`, so its length is the index of the
    first out-of-sample day and `len(train) - 1` is the last in-sample one.
    """

    def solve(train: pd.DataFrame, periods: int) -> np.ndarray:
        row = min(len(train) - 1, len(cap_weights) - 1)
        return cap_weights.iloc[row].to_numpy(dtype=float)

    return solve


def _run_period(
    weights: np.ndarray, block: pd.DataFrame
) -> tuple[list[float], np.ndarray]:
    """Hold `weights` through `block`, letting them drift with returns.

    Returns the realized daily portfolio returns and the drifted end weights.
    Without the drift the backtest would silently assume free daily rebalancing.
    """
    w = weights.copy()
    out: list[float] = []
    for _, row in block.iterrows():
        r = row.to_numpy(dtype=float)
        rp = float(w @ r)
        out.append(rp)
        denom = 1.0 + rp
        if abs(denom) > 1e-12:
            w = w * (1.0 + r) / denom
    return out, w


def _summarize(rets: pd.Series, periods: int, turnover: float, n_rebalances: int) -> dict:
    r = rets.to_numpy(dtype=float)
    vol = float(r.std(ddof=1) * np.sqrt(periods))
    growth = float(np.prod(1 + r))
    years = r.size / periods
    cagr = float(growth ** (1 / years) - 1) if years > 0 and growth > 0 else float("nan")

    curve = np.cumprod(1 + r)
    dd = curve / np.maximum.accumulate(curve) - 1.0

    downside = r[r < 0]
    sortino = (
        float(r.mean() / downside.std(ddof=1) * np.sqrt(periods))
        if downside.size > 1 and downside.std(ddof=1) > 0
        else float("nan")
    )

    return {
        "annual_return": cagr,
        "annual_volatility": vol,
        "sharpe_ratio": float(r.mean() / r.std(ddof=1) * np.sqrt(periods)) if r.std(ddof=1) > 0 else 0.0,
        "sortino_ratio": sortino,
        "max_drawdown": float(dd.min()),
        "calmar_ratio": float(cagr / abs(dd.min())) if dd.min() < 0 else float("nan"),
        "total_growth": growth,
        # Annualized one-way turnover: 1.0 means the book is replaced once a year.
        "annual_turnover": float(turnover / years) if years > 0 else 0.0,
        "n_rebalances": int(n_rebalances),
    }


def walk_forward(
    returns: pd.DataFrame,
    initial_weights: dict[str, float],
    periods_per_year: int = 252,
    train_window: int | None = None,
    rebalance_every: int = 63,
    cost_bps: float = 10.0,
    cap_weights: pd.DataFrame | None = None,
) -> dict:
    """Roll a set of allocation rules forward through unseen data.

    `cost_bps` is charged on one-way turnover at each rebalance — 10bps is a
    reasonable retail all-in estimate for liquid US equities. Strategies that
    trade more pay more, which is most of why the aggressive optimizers lose
    their paper advantage.

    `cap_weights` is an optional date-by-ticker panel of market-cap weights. When
    supplied it adds the Black-Litterman equilibrium prior to the comparison;
    when the share counts behind it are unavailable the caller passes None and
    the row is dropped, rather than a different rule being substituted under the
    same name.
    """
    n_obs, n_assets = returns.shape

    if train_window is None:
        # Prefer two years of training data, but never spend more than half the
        # sample on it, or there is nothing left to test on.
        train_window = int(min(2 * periods_per_year, n_obs // 2))

    oos_days = n_obs - train_window
    if oos_days < _MIN_OOS_DAYS or n_assets < 2:
        return {
            "available": False,
            "reason": (
                f"Needs at least {_MIN_OOS_DAYS} out-of-sample days after a "
                f"{train_window}-day training window; this window has {max(oos_days, 0)}. "
                "Try a longer lookback period."
            ),
            "strategies": [],
            "dates": [],
        }

    cost_rate = cost_bps / 10_000.0
    results: dict[str, dict] = {}

    strategies = dict(_STRATEGIES)
    if cap_weights is not None and list(cap_weights.columns) == list(returns.columns):
        strategies["market_equilibrium"] = _equilibrium_solver(cap_weights)

    # Every strategy is evaluated on the identical date range so the Sharpe
    # ratios are directly comparable.
    for key, solver in strategies.items():
        daily: list[float] = []
        turnover_total = 0.0
        n_rebal = 0
        held = np.zeros(n_assets)

        start = train_window
        while start < n_obs:
            stop = min(start + rebalance_every, n_obs)
            train = returns.iloc[:start]
            block = returns.iloc[start:stop]
            if block.empty:
                break

            try:
                target = solver(train, periods_per_year)
            except Exception:
                # A failed solve holds the previous book rather than going flat,
                # which is what a desk would actually do.
                target = held if n_rebal else np.full(n_assets, 1.0 / n_assets)

            target = np.clip(np.asarray(target, dtype=float), 0, None)
            total = target.sum()
            target = target / total if total > 0 else np.full(n_assets, 1.0 / n_assets)

            # One-way turnover against the weights we drifted into.
            turnover = float(np.abs(target - held).sum() / 2.0) if n_rebal else 1.0
            turnover_total += turnover
            n_rebal += 1

            block_rets, held = _run_period(target, block)
            # Charge the rebalance on its first day.
            block_rets[0] -= turnover * cost_rate
            daily.extend(block_rets)

            start = stop

        series = pd.Series(daily, index=returns.index[train_window : train_window + len(daily)])
        results[key] = {"returns": series, "turnover": turnover_total, "n_rebalances": n_rebal}

    # Buy and hold the user's own weights across the same out-of-sample window —
    # the baseline that costs nothing and requires no model at all.
    w0 = np.array([initial_weights.get(c, 0.0) for c in returns.columns], dtype=float)
    if w0.sum() > 0:
        w0 = w0 / w0.sum()
        bh_rets, _ = _run_period(w0, returns.iloc[train_window:])
        results["buy_and_hold"] = {
            "returns": pd.Series(bh_rets, index=returns.index[train_window:]),
            "turnover": 1.0,
            "n_rebalances": 1,
        }

    oos_index = returns.index[train_window:]
    strategies = []
    for key, data in results.items():
        summary = _summarize(
            data["returns"], periods_per_year, data["turnover"], data["n_rebalances"]
        )
        strategies.append(
            {
                "key": key,
                "label": STRATEGY_LABELS.get(key, key),
                "growth": [round(float(v), 5) for v in (1 + data["returns"]).cumprod()],
                **summary,
            }
        )

    # In-sample versus out-of-sample for the headline optimizer. The gap between
    # these two numbers is the entire argument against trusting the frontier.
    in_sample = _in_sample_sharpe(returns, periods_per_year)
    oos_lookup = {s["key"]: s["sharpe_ratio"] for s in strategies}

    strategies.sort(key=lambda s: (np.isnan(s["sharpe_ratio"]), -s["sharpe_ratio"]))

    return {
        "available": True,
        "dates": [d.strftime("%Y-%m-%d") for d in oos_index],
        "strategies": strategies,
        "train_window": int(train_window),
        "rebalance_every": int(rebalance_every),
        "oos_days": int(len(oos_index)),
        "cost_bps": float(cost_bps),
        "in_sample_sharpe": in_sample,
        "out_of_sample_sharpe": float(oos_lookup.get("max_sharpe", float("nan"))),
        "sharpe_decay": float(in_sample - oos_lookup.get("max_sharpe", float("nan"))),
        "equal_weight_wins": bool(
            oos_lookup.get("equal_weight", -np.inf) > oos_lookup.get("max_sharpe", -np.inf)
        ),
        "best_strategy": strategies[0]["label"] if strategies else None,
    }


def select_best_strategy(
    returns: pd.DataFrame,
    initial_weights: dict[str, float],
    periods_per_year: int = 252,
    cap_weights: pd.DataFrame | None = None,
    **walk_forward_kwargs,
) -> dict:
    """Pick the strategy with the best out-of-sample Sharpe and its current weights.

    `walk_forward` only keeps each strategy's realized return path, not the
    target weights it held at any point, since a full history of intermediate
    allocations is never displayed. So once the winner is known, its current
    weights are recomputed on the *whole* sample (the same "train on
    everything available today" call every other current-allocation endpoint
    in this app makes) rather than pulled from the backtest.
    """
    result = walk_forward(
        returns, initial_weights, periods_per_year, cap_weights=cap_weights, **walk_forward_kwargs
    )
    if not result["available"] or not result["strategies"]:
        raise ValueError(result.get("reason", "Walk-forward comparison is unavailable"))

    best = result["strategies"][0]
    key = best["key"]
    tickers = list(returns.columns)

    if key == "buy_and_hold":
        total = sum(initial_weights.get(t, 0.0) for t in tickers)
        weights = {t: initial_weights.get(t, 0.0) / total for t in tickers} if total > 0 else {}
    elif key == "market_equilibrium" and cap_weights is not None:
        w = cap_weights.iloc[-1].to_numpy(dtype=float)
        total = w.sum()
        w = w / total if total > 0 else np.full(len(tickers), 1.0 / len(tickers))
        weights = dict(zip(tickers, (float(v) for v in w)))
    else:
        w = np.clip(np.asarray(_STRATEGIES[key](returns, periods_per_year), dtype=float), 0, None)
        total = w.sum()
        w = w / total if total > 0 else np.full(len(tickers), 1.0 / len(tickers))
        weights = dict(zip(tickers, (float(v) for v in w)))

    return {
        "key": key,
        "label": best["label"],
        "sharpe_ratio": best["sharpe_ratio"],
        "weights": weights,
        "walk_forward": result,
    }


def regime_overview(returns: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """Current regime, its history, and which strategy actually wins in each.

    A full-sample fit for display -- distinct from `_regime_ml_weights`, which
    refits per-rebalance on `train` only inside the walk-forward loop. This
    answers "what regime are we in right now and why does that matter", not
    "what would a trading rule have done"; the trading question is answered by
    the `regime_ml` row `walk_forward` already produces.
    """
    features = compute_features(returns)
    regimes = fit_regimes(features)
    labels = regimes["labels"]
    aligned = returns.loc[labels.index]

    leaderboard = []
    for level, name in enumerate(regimes["level_names"]):
        in_regime = aligned.loc[labels == level]
        feat_in_regime = features.loc[labels == level]
        if in_regime.empty:
            continue

        best_key, best_sharpe = None, -np.inf
        for key, solver in _BASE_STRATEGIES.items():
            try:
                w = np.clip(np.asarray(solver(returns, periods_per_year), dtype=float), 0, None)
                total = w.sum()
                w = w / total if total > 0 else np.full(returns.shape[1], 1.0 / returns.shape[1])
            except Exception:
                continue
            r = in_regime.to_numpy(dtype=float) @ w
            sd = r.std(ddof=1)
            sharpe = float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
            if sharpe > best_sharpe:
                best_key, best_sharpe = key, sharpe

        leaderboard.append(
            {
                "regime": name,
                "level": level,
                "days": int(len(in_regime)),
                "best_strategy": best_key,
                "best_strategy_label": STRATEGY_LABELS.get(best_key, best_key),
                "sharpe": round(best_sharpe, 4) if best_key else None,
                "avg_volatility": round(float(feat_in_regime["volatility"].mean()), 4),
                "avg_correlation": round(float(feat_in_regime["correlation"].mean()), 4),
            }
        )

    current_level = int(labels.iloc[-1])
    return {
        "current_regime": {
            "label": regimes["level_names"][current_level],
            "level": current_level,
            "confidence": round(float(regimes["confidence"].iloc[-1]), 4),
        },
        "n_regimes": len(regimes["level_names"]),
        "converged": regimes["converged"],
        "dates": [d.strftime("%Y-%m-%d") for d in labels.index],
        "regime_levels": [int(v) for v in labels.to_numpy()],
        "level_names": regimes["level_names"],
        "leaderboard": leaderboard,
    }


def _in_sample_sharpe(returns: pd.DataFrame, periods: int) -> float:
    """Sharpe of max-Sharpe weights fitted and scored on the whole sample."""
    try:
        w = _max_sharpe_weights(returns, periods, shrink=False)
        r = returns.to_numpy(dtype=float) @ w
        sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0
    except Exception:
        return float("nan")
