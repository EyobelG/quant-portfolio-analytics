import pandas as pd

from app.metrics import portfolio_returns


def cumulative_growth(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def run_backtest(
    returns: pd.DataFrame,
    bench_returns: pd.Series,
    current_weights: dict[str, float],
    optimized_weights: dict[str, float],
) -> dict:
    port_ret = portfolio_returns(returns, current_weights)
    opt_ret = portfolio_returns(returns, optimized_weights)

    aligned = pd.concat(
        [port_ret.rename("portfolio"), opt_ret.rename("optimized"), bench_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()

    growth = aligned.apply(cumulative_growth)

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in growth.index],
        "portfolio": growth["portfolio"].round(4).tolist(),
        "optimized": growth["optimized"].round(4).tolist(),
        "benchmark": growth["benchmark"].round(4).tolist(),
    }
