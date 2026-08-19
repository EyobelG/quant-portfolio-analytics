import numpy as np
import pandas as pd


def drawdown_series(port_ret: pd.Series) -> dict:
    """Underwater curve: how far below the running peak the portfolio sits.

    A single max-drawdown figure hides duration, so this also reports how long
    the portfolio stayed below its previous peak.
    """
    growth = (1 + port_ret).cumprod()
    running_max = growth.cummax()
    dd = growth / running_max - 1.0

    trough_idx = dd.idxmin()

    # Longest stretch spent below a previous peak, in calendar days.
    underwater = dd < -1e-9
    longest = 0
    current_start = None
    recovery_days = None
    for date, is_under in underwater.items():
        if is_under and current_start is None:
            current_start = date
        elif not is_under and current_start is not None:
            span = (date - current_start).days
            longest = max(longest, span)
            if current_start <= trough_idx <= date:
                recovery_days = span
            current_start = None
    if current_start is not None:
        # Still underwater at the end of the window.
        span = (underwater.index[-1] - current_start).days
        longest = max(longest, span)

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dd.index],
        "drawdown": dd.round(5).tolist(),
        "max_drawdown": float(dd.min()),
        "max_drawdown_date": trough_idx.strftime("%Y-%m-%d"),
        "recovery_days": recovery_days,
        "longest_underwater_days": longest,
    }


def return_distribution(port_ret: pd.Series, bins: int = 41) -> dict:
    """Histogram of daily returns, with the VaR/CVaR thresholds to overlay."""
    counts, edges = np.histogram(port_ret.values, bins=bins)
    var_95 = float(np.percentile(port_ret, 5))
    tail = port_ret[port_ret <= var_95]

    return {
        "bin_edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
        "var_95": var_95,
        "cvar_95": float(tail.mean()) if len(tail) else var_95,
        "mean": float(port_ret.mean()),
    }


def sector_weights(holdings: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for h in holdings:
        key = h.get("sector") or "Unclassified"
        totals[key] = totals.get(key, 0.0) + h["weight"]
    return {k: round(v, 6) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])}
