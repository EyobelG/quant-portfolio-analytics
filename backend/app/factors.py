"""Factor attribution.

Jensen's alpha against a single index answers "did this portfolio beat the
market?" — which is the wrong question when the portfolio is simply tilted
toward small caps, or value, or momentum. Those tilts are compensated risk
premia available cheaply through an index fund, not skill, and a single-factor
alpha attributes all of them to the manager.

This runs the multi-factor version. Factors are built from liquid ETF spreads
rather than the Fama-French research files: the academic factors are cleaner but
require a second data source with its own release lag, and the ETF proxies are
tradeable, which makes the resulting alpha the more honest number — it is the
return left over after everything an investor could actually have replicated.

References
----------
Fama & French (1993), "Common risk factors in the returns on stocks and bonds",
    Journal of Financial Economics 33(1).
Carhart (1997), "On Persistence in Mutual Fund Performance",
    Journal of Finance 52(1) — the momentum factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.data import fetch_prices
from app.risk_stats import newey_west_tstats

# Each factor is a long-short spread between two ETFs. The market factor is the
# benchmark's own excess return, so it is assembled separately.
_FACTOR_SPECS: dict[str, tuple[str, str, str]] = {
    "size": ("IWM", "SPY", "Small caps minus large caps"),
    "value": ("IWD", "IWF", "Value minus growth"),
    "momentum": ("MTUM", "SPY", "Momentum stocks minus the market"),
    "quality": ("QUAL", "SPY", "Quality stocks minus the market"),
    "low_volatility": ("USMV", "SPY", "Low-volatility stocks minus the market"),
}

_FACTOR_TICKERS = tuple(sorted({t for pair in _FACTOR_SPECS.values() for t in pair[:2]}))


def fetch_factor_returns(period: str = "3y") -> pd.DataFrame:
    """Build the factor return panel from ETF spreads.

    Shares `fetch_prices`, so the ETF basket is downloaded once an hour and
    reused across every request regardless of which portfolio asked for it.
    """
    prices = fetch_prices(_FACTOR_TICKERS, period)
    rets = prices.pct_change().dropna()

    factors = {}
    for name, (long_leg, short_leg, _) in _FACTOR_SPECS.items():
        if long_leg in rets.columns and short_leg in rets.columns:
            factors[name] = rets[long_leg] - rets[short_leg]

    if not factors:
        raise ValueError("No factor spreads could be constructed")
    return pd.DataFrame(factors).dropna()


def factor_regression(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    factor_rets: pd.DataFrame,
    periods_per_year: int = 252,
) -> dict:
    """Regress portfolio returns on the market plus style factors.

    Standard errors are Newey-West, because monthly-overlapping style exposures
    and volatility clustering both induce serial correlation that would
    otherwise inflate every t-statistic.
    """
    design = pd.concat(
        [port_ret.rename("portfolio"), bench_ret.rename("market"), factor_rets],
        axis=1,
        join="inner",
    ).dropna()

    if design.shape[0] < 60:
        raise ValueError("Need at least 60 overlapping observations")

    y = design["portfolio"].to_numpy()
    factor_names = ["market"] + list(factor_rets.columns)
    X = np.column_stack([np.ones(len(design))] + [design[f].to_numpy() for f in factor_names])

    fit = newey_west_tstats(y, X)

    # Index 0 is the intercept; annualize it to compare against the metrics grid.
    alpha_daily = fit["coefficients"][0]
    loadings = []
    for i, name in enumerate(factor_names, start=1):
        spec = _FACTOR_SPECS.get(name)
        loadings.append(
            {
                "factor": name,
                "label": name.replace("_", " ").title(),
                "description": spec[2] if spec else "Benchmark excess return",
                "beta": fit["coefficients"][i],
                "t_statistic": fit["t_statistics"][i],
                "p_value": fit["p_values"][i],
                "significant": bool(fit["p_values"][i] < 0.05),
            }
        )

    return {
        "available": True,
        "alpha_annualized": float(alpha_daily * periods_per_year),
        "alpha_t_statistic": float(fit["t_statistics"][0]),
        "alpha_p_value": float(fit["p_values"][0]),
        # The conventional bar for a real effect after accounting for the fact
        # that many strategies get tested.
        "alpha_significant": bool(fit["p_values"][0] < 0.05),
        "loadings": loadings,
        "r_squared": fit["r_squared"],
        "adj_r_squared": fit["adj_r_squared"],
        "newey_west_lags": fit["lags"],
        "observations": fit["observations"],
        # How much of the portfolio's variation the factors already explain. A
        # high number means there is little left for stock selection to claim.
        "unexplained_share": float(max(0.0, 1.0 - fit["r_squared"])),
    }


def run_factor_analysis(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    period: str = "3y",
    periods_per_year: int = 252,
) -> dict:
    """Factor attribution, degrading to unavailable rather than failing.

    This depends on six extra ETF downloads, so it is the part of the analysis
    most likely to break on an upstream hiccup — and the least essential.
    """
    try:
        factors = fetch_factor_returns(period)
        return factor_regression(port_ret, bench_ret, factors, periods_per_year)
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc)[:200],
            "loadings": [],
        }
