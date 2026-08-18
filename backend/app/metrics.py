import empyrical as ep
import numpy as np
import pandas as pd


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    aligned_weights = np.array([weights[t] for t in returns.columns])
    return (returns * aligned_weights).sum(axis=1)


def compute_metrics(port_ret: pd.Series, bench_ret: pd.Series) -> dict:
    aligned = pd.concat([port_ret, bench_ret], axis=1, join="inner").dropna()
    port, bench = aligned.iloc[:, 0], aligned.iloc[:, 1]

    return {
        "annual_return": float(ep.annual_return(port)),
        "annual_volatility": float(ep.annual_volatility(port)),
        "sharpe_ratio": float(ep.sharpe_ratio(port)),
        "sortino_ratio": float(ep.sortino_ratio(port)),
        "max_drawdown": float(ep.max_drawdown(port)),
        "var_95": float(np.percentile(port, 5)),
        "cvar_95": float(port[port <= np.percentile(port, 5)].mean()),
        "beta": float(ep.beta(port, bench)),
        "alpha": float(ep.alpha(port, bench)),
    }
