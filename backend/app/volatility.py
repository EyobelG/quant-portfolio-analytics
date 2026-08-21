"""Conditional volatility models.

Annualized volatility is a single number describing a quantity that is not
constant. Volatility clusters: calm follows calm, and a large move today raises
the probability of a large move tomorrow. Reporting one unconditional sigma
throws that away, and a VaR built on it is wrong in both directions — too loose
in quiet regimes, far too tight in a crisis.

Implemented from the likelihood rather than imported so the estimation is
visible: `arch` would be one import, but the maximum-likelihood machinery is
what the model actually is.

References
----------
Bollerslev (1986), "Generalized Autoregressive Conditional Heteroskedasticity",
    Journal of Econometrics 31(3).
Engle (1982), "Autoregressive Conditional Heteroscedasticity with Estimates of
    the Variance of United Kingdom Inflation", Econometrica 50(4).
J.P. Morgan (1996), RiskMetrics Technical Document, 4th ed. — EWMA with
    lambda = 0.94 for daily data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# Returns are scaled by this before estimation. Daily returns are order 1e-2, so
# omega lands near 1e-6 and the optimizer's convergence tolerances — which are
# absolute — stop being meaningful. Working in percent fixes the conditioning.
_SCALE = 100.0


def ewma_volatility(
    returns: pd.Series, lam: float = 0.94, periods_per_year: int = 252
) -> dict:
    """RiskMetrics exponentially-weighted volatility.

    A one-parameter model with no estimation at all: each squared return decays
    at a fixed rate. It reacts to a shock immediately, which is its advantage
    over a rolling window, but it has no mean reversion, so it cannot forecast
    beyond tomorrow.
    """
    r = np.asarray(returns, dtype=float)
    if not 0 < lam < 1:
        raise ValueError("lambda must be in (0, 1)")

    var = np.empty(r.size)
    var[0] = r.var(ddof=1)
    for t in range(1, r.size):
        var[t] = lam * var[t - 1] + (1.0 - lam) * r[t - 1] ** 2

    ann = np.sqrt(var * periods_per_year)
    return {
        "lambda": float(lam),
        "volatility": [float(v) for v in ann],
        "current": float(ann[-1]),
        "unconditional": float(r.std(ddof=1) * np.sqrt(periods_per_year)),
        # Weight decays as lam^k, so this is where half the mass sits.
        "half_life_days": float(np.log(0.5) / np.log(lam)),
    }


def _garch_negative_loglik(params: np.ndarray, r: np.ndarray, backcast: float) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
        return 1e10

    n = r.size
    var = np.empty(n)
    var[0] = backcast
    for t in range(1, n):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
        if var[t] <= 0:
            return 1e10

    # Gaussian log-likelihood for a zero-mean series with time-varying variance.
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(var) + r**2 / var)
    return float(-ll) if np.isfinite(ll) else 1e10


def fit_garch(
    returns: pd.Series, periods_per_year: int = 252, horizon: int = 21
) -> dict:
    """Fit GARCH(1,1) by maximum likelihood and forecast forward.

    alpha measures how sharply volatility reacts to news; beta how long it
    remembers. Their sum is the persistence — close to 1 for most financial
    series, meaning shocks decay slowly and today's turbulence is still visible
    weeks out.
    """
    r = np.asarray(returns, dtype=float) * _SCALE
    n = r.size
    if n < 100:
        raise ValueError("Need at least 100 observations to fit GARCH(1,1)")

    r = r - r.mean()  # model the variance of the demeaned series
    backcast = float(np.mean(r**2))

    # Typical equity starting values; the persistence constraint keeps the
    # search inside the stationary region.
    best = None
    for x0 in ([backcast * 0.05, 0.08, 0.90], [backcast * 0.10, 0.05, 0.85]):
        result = minimize(
            _garch_negative_loglik,
            x0=np.array(x0, dtype=float),
            args=(r, backcast),
            method="SLSQP",
            bounds=[(1e-8, None), (1e-8, 0.999), (1e-8, 0.999)],
            constraints=[
                # Stationarity: alpha + beta < 1, held just inside the boundary.
                {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}
            ],
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if best is None or result.fun < best.fun:
            best = result

    omega, alpha, beta = (float(v) for v in best.x)
    persistence = alpha + beta

    # Rebuild the fitted variance path at the optimum.
    var = np.empty(n)
    var[0] = backcast
    for t in range(1, n):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]

    # Forecast. The first step uses the last observation; beyond that the
    # process decays geometrically toward its unconditional level.
    long_run_var = omega / (1.0 - persistence) if persistence < 1 else var[-1]
    forecasts = np.empty(horizon)
    forecasts[0] = omega + alpha * r[-1] ** 2 + beta * var[-1]
    for h in range(1, horizon):
        forecasts[h] = long_run_var + persistence * (forecasts[h - 1] - long_run_var)

    # Undo the percent scaling, then annualize.
    ann = np.sqrt(periods_per_year) / _SCALE
    k = 3  # omega, alpha, beta
    loglik = -float(best.fun)

    return {
        # Unscaled back to raw return units so every number in this dict lives
        # in the same space; alpha and beta are ratios and need no conversion.
        "omega": omega / _SCALE**2,
        "alpha": alpha,
        "beta": beta,
        "persistence": float(persistence),
        "converged": bool(best.success),
        "log_likelihood": loglik,
        "aic": float(2 * k - 2 * loglik),
        "bic": float(k * np.log(n) - 2 * loglik),
        "conditional_volatility": [float(v) for v in np.sqrt(var) * ann],
        "forecast": [float(v) for v in np.sqrt(forecasts) * ann],
        "forecast_horizon": int(horizon),
        "current_volatility": float(np.sqrt(var[-1]) * ann),
        "long_run_volatility": float(np.sqrt(long_run_var) * ann),
        # How long a volatility shock takes to half-decay toward the long-run
        # level. Undefined if the process is at the stationarity boundary.
        "half_life_days": (
            float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else None
        ),
        # Above 1.0 means the market is currently more turbulent than its
        # own long-run average.
        "vol_ratio": float(np.sqrt(var[-1] / long_run_var)) if long_run_var > 0 else 1.0,
    }


def arch_lm_test(returns: pd.Series, lags: int = 5) -> dict:
    """Engle's LM test for ARCH effects.

    This is the diagnostic that justifies fitting GARCH at all. It regresses
    squared returns on their own lags: if volatility really is constant, past
    squared returns carry no information about today's and the R-squared is
    noise. Rejecting the null is the evidence that a conditional model is needed.
    """
    r = np.asarray(returns, dtype=float)
    r = r - r.mean()
    e2 = r**2
    n = e2.size
    if n <= lags + 1:
        raise ValueError("Not enough observations for the requested lags")

    y = e2[lags:]
    X = np.column_stack([np.ones(y.size)] + [e2[lags - i : n - i] for i in range(1, lags + 1)])

    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    lm_stat = float(y.size * r2)
    p = float(1.0 - stats.chi2.cdf(lm_stat, df=lags))

    return {
        "lags": int(lags),
        "lm_statistic": lm_stat,
        "p_value": p,
        "r_squared": float(r2),
        # Null is "no ARCH effects" — homoskedastic. Rejecting it means
        # volatility clusters and a constant-sigma risk model is misspecified.
        "arch_effects_present": bool(p < 0.05),
    }
