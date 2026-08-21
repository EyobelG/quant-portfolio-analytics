"""Correlation and beta conditioned on the market regime.

A single correlation matrix averaged over the whole sample is the most
misleading number on a portfolio page. Correlations are not stable: they rise
sharply in selloffs, so the diversification the average implies is largely
absent on the days it is supposed to protect you. The same asymmetry shows up in
beta — most equities fall with the market more readily than they rise with it,
which a single OLS beta averages away entirely.

This module splits the sample by the *benchmark's* return rather than the
portfolio's. Conditioning on the portfolio's own returns would select the days
the portfolio happened to fall, which is circular; conditioning on the market
asks the question a risk desk actually cares about, namely what this book does
when the market breaks.

A caveat this module cannot remove, and should not pretend to: conditioning on
the size of the market's move biases the measured correlation in the direction
of the conditioning. Selecting the largest moves inflates it, and selecting the
smallest deflates it, even when the true correlation is constant — Forbes &
Rigobon (2002). So the calm figure reported here is biased low and the stressed
figure biased high, and the gap between them overstates the true change.

The effect is nonetheless real rather than purely mechanical: Longin & Solnik
(2001) found genuine excess correlation in the lower tail after correcting for
the bias with extreme value theory. Treat the direction as sound and the
magnitude as an upper bound.

References
----------
Longin & Solnik (2001), "Extreme Correlation of International Equity Markets",
    Journal of Finance 56(2) — correlation rises in bear markets beyond what
    the conditioning bias alone would produce.
Forbes & Rigobon (2002), "No Contagion, Only Interdependence: Measuring Stock
    Market Comovements", Journal of Finance 57(5) — the conditioning bias.
Ang & Chen (2002), "Asymmetric Correlations of Equity Portfolios",
    Journal of Financial Economics 63(3).
Bawa & Lindenberg (1977), "Capital Market Equilibrium in a Mean-Lower Partial
    Moment Framework", Journal of Financial Economics 5(2) — downside beta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A correlation matrix estimated from very few observations is mostly noise, and
# the stressed regime is by construction the smallest. Require both an absolute
# floor and a multiple of the asset count before calling the split trustworthy.
_MIN_REGIME_OBS = 30
_OBS_PER_ASSET = 5


def _average_pairwise_correlation(corr: pd.DataFrame) -> float:
    """Mean of the off-diagonal entries.

    Taken over the upper triangle only: the matrix is symmetric, so including
    both halves changes nothing but including the unit diagonal would drag the
    average toward 1 and hide exactly the effect being measured.
    """
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, k=1)
    return float(np.asarray(corr)[iu].mean())


def _beta(asset: np.ndarray, market: np.ndarray) -> float:
    """OLS beta of one asset against the market."""
    var = market.var(ddof=1)
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return float(np.cov(asset, market, ddof=1)[0, 1] / var)


def _capture_ratio(port: np.ndarray, bench: np.ndarray) -> float:
    """Mean portfolio return over mean benchmark return, within one regime.

    Morningstar's convention compounds the two legs and divides the growth
    rates, but that definition assumes monthly data. Applied to the ~1,000
    same-signed daily observations in this sample it degenerates: both legs
    compound toward -100%, so their ratio saturates near 1.0 and a portfolio
    with genuinely half the market's downside still scores 0.99. The ratio of
    means has no such pathology and reads the same way — below 1.0 on down days
    means the portfolio fell less than the market.
    """
    if port.size == 0 or bench.size == 0:
        return float("nan")
    bench_mean = float(bench.mean())
    if abs(bench_mean) < 1e-12:
        return float("nan")
    return float(port.mean() / bench_mean)


def asymmetric_betas(
    returns: pd.DataFrame, bench_returns: pd.Series, threshold: float = 0.0
) -> list[dict]:
    """Beta on days the market fell versus days it rose.

    A ratio above 1 means the holding participates more in the market's losses
    than in its gains — the payoff profile nobody wants and every backward-
    looking single beta conceals.
    """
    aligned = pd.concat([returns, bench_returns.rename("__bench__")], axis=1, join="inner").dropna()
    bench = aligned["__bench__"].to_numpy(dtype=float)
    assets = aligned.drop(columns="__bench__")

    down_mask = bench < threshold
    up_mask = bench > threshold

    out = []
    for ticker in assets.columns:
        series = assets[ticker].to_numpy(dtype=float)
        down = _beta(series[down_mask], bench[down_mask]) if down_mask.sum() > 2 else float("nan")
        up = _beta(series[up_mask], bench[up_mask]) if up_mask.sum() > 2 else float("nan")
        full = _beta(series, bench)

        # The ratio only reads as "downside participation" when both legs are
        # positive. If they straddle zero the asset rises with one direction of
        # the market and falls with the other, and the quotient becomes a
        # negative number whose interpretation silently inverts — better to
        # report it as undefined than to print something meaningless.
        comparable = np.isfinite(down) and np.isfinite(up) and down > 0 and up > 1e-9
        ratio = down / up if comparable else float("nan")
        out.append(
            {
                "ticker": str(ticker),
                "beta": full,
                "downside_beta": down,
                "upside_beta": up,
                # Above 1.0 is the unattractive asymmetry: more downside
                # participation than upside.
                "ratio": ratio,
                "asymmetric": bool(np.isfinite(ratio) and ratio > 1.1),
            }
        )

    return out


def regime_analysis(
    returns: pd.DataFrame,
    bench_returns: pd.Series,
    port_ret: pd.Series | None = None,
    tail_quantile: float = 0.10,
) -> dict:
    """Correlation structure and capture, split by how the market behaved.

    `tail_quantile` sets the size of each tail: 0.10 puts the worst 10% of
    benchmark days in the stressed regime, the best 10% in the rally regime, and
    the remaining 80% in calm. Widening it trades a cleaner definition of stress
    for a better-estimated matrix.
    """
    if not 0 < tail_quantile < 0.5:
        raise ValueError("tail_quantile must be in (0, 0.5)")

    frame = pd.concat(
        [returns, bench_returns.rename("__bench__")], axis=1, join="inner"
    ).dropna()
    if frame.shape[0] < 3 * _MIN_REGIME_OBS:
        raise ValueError(
            f"Need at least {3 * _MIN_REGIME_OBS} overlapping observations to split by regime; "
            f"got {frame.shape[0]}"
        )

    assets = frame.drop(columns="__bench__")
    bench = frame["__bench__"]
    n_assets = assets.shape[1]
    if n_assets < 2:
        raise ValueError("Need at least 2 holdings to compare correlations")

    # Rank-based cuts rather than value-based, so the regimes hold the intended
    # share of days even when returns repeat.
    ranks = bench.rank(method="first", pct=True)
    masks = {
        "stressed": ranks <= tail_quantile,
        "calm": (ranks > tail_quantile) & (ranks <= 1.0 - tail_quantile),
        "rally": ranks > 1.0 - tail_quantile,
    }

    regimes = {}
    rounded: dict[str, pd.DataFrame] = {}
    for name, mask in masks.items():
        corr = assets[mask].corr()
        # Round once and reuse. Differencing the unrounded matrices instead
        # would let the delta disagree with the two panels the UI puts it
        # beside, by up to 1e-4 of double rounding.
        rounded[name] = corr.round(4)
        regimes[name] = {
            "observations": int(mask.sum()),
            "correlation_matrix": rounded[name].to_dict(),
            "average_correlation": _average_pairwise_correlation(corr),
            "benchmark_mean": float(bench[mask].mean()),
            "benchmark_worst": float(bench[mask].min()),
            "benchmark_best": float(bench[mask].max()),
        }

    delta = (rounded["stressed"] - rounded["calm"]).round(4)

    tickers = [str(c) for c in assets.columns]
    smallest = min(r["observations"] for r in regimes.values())

    out = {
        "tickers": tickers,
        "tail_quantile": float(tail_quantile),
        "observations": int(frame.shape[0]),
        "regimes": regimes,
        "delta_matrix": delta.to_dict(),
        "correlation_increase": (
            regimes["stressed"]["average_correlation"] - regimes["calm"]["average_correlation"]
        ),
        "betas": asymmetric_betas(assets, bench),
        # The tails are the small regimes by construction, so the honesty flag
        # keys off the smallest of them.
        "min_regime_observations": int(smallest),
        "reliable": bool(smallest >= max(_MIN_REGIME_OBS, _OBS_PER_ASSET * n_assets)),
    }

    if port_ret is not None:
        aligned_port = port_ret.reindex(frame.index).dropna()
        if aligned_port.shape[0] == frame.shape[0]:
            p = aligned_port.to_numpy(dtype=float)
            b = bench.to_numpy(dtype=float)
            down, up = b < 0.0, b > 0.0
            out["capture"] = {
                # Below 1.0 on the downside and above 1.0 on the upside is the
                # combination worth paying for; most portfolios manage neither.
                "downside": _capture_ratio(p[down], b[down]),
                "upside": _capture_ratio(p[up], b[up]),
                "down_days": int(down.sum()),
                "up_days": int(up.sum()),
            }

    return out
