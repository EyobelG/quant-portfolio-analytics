"""Statistical inference on a return series.

Every metric in `metrics.py` is a point estimate computed from one finite,
non-normal, autocorrelated sample. This module asks the follow-up question that
a point estimate cannot answer: how much of it survives sampling error?

References
----------
Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier",
    Journal of Risk 15(2) — probabilistic and deflated Sharpe ratios.
Politis & Romano (1994), "The Stationary Bootstrap", JASA 89(428).
Kupiec (1995), "Techniques for Verifying the Accuracy of Risk Measurement
    Models", Journal of Derivatives 3(2).
Christoffersen (1998), "Evaluating Interval Forecasts",
    International Economic Review 39(4).
Cornish & Fisher (1938), "Moments and Cumulants in the Specification of
    Distributions".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Euler-Mascheroni constant, used by the deflated Sharpe ratio's expected-maximum
# term.
_EULER_GAMMA = 0.5772156649015329


def moments(returns: pd.Series) -> dict:
    """Sample moments plus a normality test.

    Skew and excess kurtosis are what break the textbook Sharpe ratio: it is a
    sufficient statistic for ranking portfolios only when returns are normal,
    and daily equity returns never are. Jarque-Bera quantifies how badly.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size

    skew = float(stats.skew(r, bias=False))
    # Fisher definition: 0 for a normal distribution.
    excess_kurtosis = float(stats.kurtosis(r, fisher=True, bias=False))

    jb_stat, jb_p = stats.jarque_bera(r)

    return {
        "observations": int(n),
        "mean": float(r.mean()),
        "std": float(r.std(ddof=1)),
        "skewness": skew,
        "excess_kurtosis": excess_kurtosis,
        "jarque_bera": float(jb_stat),
        "jarque_bera_p": float(jb_p),
        # The JB null is normality, so a small p-value rejects it.
        "normal_at_5pct": bool(jb_p > 0.05),
    }


def probabilistic_sharpe_ratio(
    returns: pd.Series, benchmark_sr: float = 0.0, periods_per_year: int = 252
) -> dict:
    """Probability that the true Sharpe ratio exceeds `benchmark_sr`.

    The estimator's standard error widens with negative skew and fat tails, so
    two strategies with identical Sharpe ratios can carry very different
    confidence. `benchmark_sr` is supplied annualized and de-annualized
    internally, because the skew/kurtosis correction is only valid in the same
    frequency as the moments it uses.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 3:
        raise ValueError("Need at least 3 observations for a PSR")

    sd = r.std(ddof=1)
    # A genuinely constant series lands near 1e-18 rather than exactly zero, and
    # the skew/kurtosis of such a sample is numerical garbage — so compare
    # against the scale of the data instead of testing for exact zero.
    scale = max(abs(float(r.mean())), 1e-12)
    if not np.isfinite(sd) or sd < 1e-10 * scale:
        raise ValueError("Cannot compute a Sharpe ratio for a constant series")

    sr = r.mean() / sd  # per-period
    skew = float(stats.skew(r, bias=False))
    # Pearson (non-excess) kurtosis: the Bailey-López de Prado formula uses
    # gamma_4 directly, which is 3 for a normal distribution.
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))

    sr_star = benchmark_sr / np.sqrt(periods_per_year)

    # Standard error of the Sharpe estimator under non-normality.
    variance = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr**2
    # Heavy negative skew with a large Sharpe can drive this non-positive, at
    # which point the normal approximation has broken down entirely.
    if variance <= 0:
        raise ValueError("PSR variance term is non-positive; sample is too extreme")

    z = (sr - sr_star) * np.sqrt(n - 1) / np.sqrt(variance)

    return {
        "sharpe_annualized": float(sr * np.sqrt(periods_per_year)),
        "benchmark_sharpe": float(benchmark_sr),
        "psr": float(stats.norm.cdf(z)),
        "z_score": float(z),
        "skewness": skew,
        "excess_kurtosis": kurt - 3.0,
        "observations": int(n),
        # Standard error of the annualized Sharpe.
        "standard_error": float(np.sqrt(variance / (n - 1)) * np.sqrt(periods_per_year)),
    }


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_sharpes: list[float],
    periods_per_year: int = 252,
) -> dict:
    """PSR against the Sharpe ratio a *lucky* search would produce by chance.

    Searching a frontier of candidate portfolios and reporting the best one is
    multiple hypothesis testing. The expected maximum Sharpe under a null of no
    skill grows with the number of trials, so the honest benchmark is not zero —
    it is that expected maximum. This is the single most common way backtests
    overstate themselves.
    """
    trials = np.asarray([t for t in trial_sharpes if np.isfinite(t)], dtype=float)
    n_trials = trials.size
    if n_trials < 2:
        raise ValueError("Need at least 2 trials to deflate against")

    trial_var = float(trials.var(ddof=1))

    # Expected maximum of n_trials draws from N(0, trial_var), via the standard
    # Gumbel approximation to the extreme-value distribution.
    q1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    expected_max = np.sqrt(trial_var) * ((1.0 - _EULER_GAMMA) * q1 + _EULER_GAMMA * q2)

    # expected_max is a per-trial (annualized) Sharpe, matching trial_sharpes.
    result = probabilistic_sharpe_ratio(
        returns, benchmark_sr=float(expected_max), periods_per_year=periods_per_year
    )

    return {
        "dsr": result["psr"],
        "expected_max_sharpe": float(expected_max),
        "n_trials": int(n_trials),
        "trial_sharpe_std": float(np.sqrt(trial_var)),
        "sharpe_annualized": result["sharpe_annualized"],
        # A strategy is only credible if it clears the bar luck alone would set.
        "beats_selection_bias": bool(result["sharpe_annualized"] > expected_max),
    }


def stationary_bootstrap_matrix(
    n: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """`n_boot` stationary-bootstrap resamples of the index set 0..n-1.

    Blocks have geometric length so the resampled series stays stationary,
    unlike a fixed-block bootstrap. Wrapping at the end keeps every observation
    equally likely to be drawn.

    Built without a sequential loop: each position's block is identified by a
    cumulative sum of the new-block flags, and its index is that block's random
    origin plus the offset into the block. Same draw as the textbook recursion,
    but it runs in numpy rather than 1.5M Python iterations.
    """
    p = 1.0 / mean_block

    new_block = rng.random((n_boot, n)) < p
    new_block[:, 0] = True  # every path starts a block

    origins = rng.integers(0, n, size=(n_boot, n))

    # Which block each position belongs to, 0-indexed.
    block_id = np.cumsum(new_block, axis=1) - 1
    # Position at which the current block started.
    pos = np.arange(n)
    block_start = np.maximum.accumulate(np.where(new_block, pos, 0), axis=1)

    return (np.take_along_axis(origins, block_id, axis=1) + (pos - block_start)) % n


def stationary_bootstrap_indices(
    n: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """A single stationary-bootstrap resample of the index set 0..n-1."""
    return stationary_bootstrap_matrix(n, 1, mean_block, rng)[0]


def bootstrap_sharpe_ci(
    returns: pd.Series,
    periods_per_year: int = 252,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> dict:
    """Confidence interval for the annualized Sharpe via stationary bootstrap.

    An IID bootstrap would ignore volatility clustering and produce an interval
    that is too narrow. Mean block length follows the n^(1/3) rule of thumb.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 30:
        raise ValueError("Need at least 30 observations to bootstrap")

    rng = np.random.default_rng(seed)
    mean_block = max(2.0, n ** (1.0 / 3.0))
    ann = np.sqrt(periods_per_year)

    samples = r[stationary_bootstrap_matrix(n, n_boot, mean_block, rng)]
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = np.where(sd > 0, samples.mean(axis=1) / sd * ann, np.nan)

    draws = draws[np.isfinite(draws)]
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.percentile(draws, [100 * alpha, 100 * (1 - alpha)])

    point = float(r.mean() / r.std(ddof=1) * ann) if r.std(ddof=1) > 0 else 0.0

    return {
        "sharpe": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "mean_block_length": float(mean_block),
        "n_resamples": int(draws.size),
        # The interval straddling zero is the honest headline: the Sharpe is not
        # distinguishable from no edge at all.
        "significant": bool(lower > 0),
    }


def cornish_fisher_var(
    returns: pd.Series, alpha: float = 0.05
) -> dict:
    """VaR/CVaR adjusted for skew and fat tails.

    Gaussian VaR understates risk for negatively-skewed, leptokurtic returns
    because it assumes away exactly the tail that hurts. The Cornish-Fisher
    expansion corrects the normal quantile using the third and fourth moments.
    """
    r = np.asarray(returns, dtype=float)
    mu, sd = r.mean(), r.std(ddof=1)
    skew = float(stats.skew(r, bias=False))
    exkurt = float(stats.kurtosis(r, fisher=True, bias=False))

    z = stats.norm.ppf(alpha)
    z_cf = (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * exkurt / 24.0
        - (2.0 * z**3 - 5.0 * z) * (skew**2) / 36.0
    )

    gaussian = float(mu + sd * z)
    modified = float(mu + sd * z_cf)
    historical = float(np.percentile(r, 100 * alpha))
    tail = r[r <= historical]

    return {
        "alpha": float(alpha),
        "gaussian_var": gaussian,
        "modified_var": modified,
        "historical_var": historical,
        "historical_cvar": float(tail.mean()) if tail.size else historical,
        "z_normal": float(z),
        "z_cornish_fisher": float(z_cf),
        # Positive when the normal model was too optimistic about the tail.
        "understatement": float(gaussian - modified),
    }


def kupiec_pof(exceptions: int, observations: int, alpha: float = 0.05) -> dict:
    """Kupiec proportion-of-failures test for VaR calibration.

    A 95% VaR should be breached on ~5% of days. Materially fewer breaches means
    the model is too conservative and the desk is leaving risk budget unused;
    materially more means it is dangerous. Both are failures.
    """
    x, n = int(exceptions), int(observations)
    if n <= 0:
        raise ValueError("Need at least one observation")
    rate = x / n

    def _xlogy(count: int, prob: float) -> float:
        # 0 * log(0) -> 0, which is the correct limit and keeps the statistic
        # finite when there are zero exceptions (or nothing but exceptions).
        return 0.0 if count == 0 else count * np.log(prob)

    ll_null = _xlogy(n - x, 1 - alpha) + _xlogy(x, alpha)
    ll_alt = _xlogy(n - x, 1 - rate) + _xlogy(x, rate)
    lr = -2.0 * (ll_null - ll_alt)

    p = float(1.0 - stats.chi2.cdf(lr, df=1))

    return {
        "exceptions": x,
        "observations": n,
        "expected_exceptions": float(n * alpha),
        "exception_rate": float(rate),
        "lr_statistic": float(lr),
        "p_value": p,
        # Null is "the VaR is correctly calibrated"; a small p-value rejects it.
        "correctly_calibrated": bool(p > 0.05),
    }


def christoffersen_independence(breaches: np.ndarray) -> dict:
    """Test whether VaR breaches cluster.

    Correct unconditional coverage is not enough. If breaches arrive in bursts
    the model is missing volatility clustering, and the risk of several
    consecutive limit violations is far higher than the headline rate implies.
    """
    b = np.asarray(breaches, dtype=int).ravel()
    if b.size < 2:
        raise ValueError("Need at least 2 observations")

    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    # Transition probabilities, guarding the empty-row cases.
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _xlogy(count: int, prob: float) -> float:
        # 0 * log(0) is 0 here, matching the likelihood's limiting behaviour.
        return 0.0 if count == 0 else count * np.log(prob)

    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = (
        _xlogy(n00, 1 - pi01)
        + _xlogy(n01, pi01)
        + _xlogy(n10, 1 - pi11)
        + _xlogy(n11, pi11)
    )
    lr = -2.0 * (ll_null - ll_alt)
    p = float(1.0 - stats.chi2.cdf(lr, df=1))

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "prob_breach_after_calm": float(pi01),
        "prob_breach_after_breach": float(pi11),
        "lr_statistic": float(lr),
        "p_value": p,
        # Null is independence; rejecting means breaches cluster.
        "independent": bool(p > 0.05),
    }


def var_backtest(returns: pd.Series, alpha: float = 0.05, window: int = 250) -> dict:
    """Rolling out-of-sample VaR backtest with Kupiec and Christoffersen tests.

    The VaR for each day is estimated from the *prior* `window` days only, so
    this measures how the model would actually have performed, not how well it
    fits in hindsight.
    """
    r = pd.Series(returns).astype(float).reset_index(drop=True)
    if r.size <= window + 10:
        raise ValueError(f"Need more than {window + 10} observations to backtest")

    # shift(1) makes the estimate strictly out-of-sample.
    var_series = r.rolling(window).quantile(alpha).shift(1)
    valid = var_series.notna()
    realized, forecast = r[valid], var_series[valid]

    breaches = (realized < forecast).to_numpy().astype(int)

    pof = kupiec_pof(int(breaches.sum()), breaches.size, alpha=alpha)
    ind = christoffersen_independence(breaches)

    # Conditional coverage is the joint test; the two statistics are independent
    # under the null, so they add and the degrees of freedom add.
    lr_cc = pof["lr_statistic"] + ind["lr_statistic"]
    p_cc = float(1.0 - stats.chi2.cdf(lr_cc, df=2))

    return {
        "alpha": float(alpha),
        "window": int(window),
        "kupiec": pof,
        "christoffersen": ind,
        "conditional_coverage_lr": float(lr_cc),
        "conditional_coverage_p": p_cc,
        "model_adequate": bool(p_cc > 0.05),
    }


def newey_west_tstats(
    y: np.ndarray, X: np.ndarray, lags: int | None = None
) -> dict:
    """OLS with heteroskedasticity- and autocorrelation-consistent errors.

    Financial residuals are neither homoskedastic nor serially independent, so
    plain OLS standard errors overstate significance — which is how a spurious
    alpha ends up with a t-stat of 3. `X` must already contain an intercept
    column if one is wanted.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    if n <= k:
        raise ValueError("Need more observations than regressors")

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    if lags is None:
        # Newey & West's automatic lag rule.
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, int(lags))

    # Meat of the sandwich: S = Gamma_0 + sum_l w_l (Gamma_l + Gamma_l').
    S = (resid[:, None] * X).T @ (resid[:, None] * X)
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)  # Bartlett kernel
        u_t = (resid[lag:, None] * X[lag:])
        u_lag = (resid[:-lag, None] * X[:-lag])
        G = u_t.T @ u_lag
        S += w * (G + G.T)

    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    p = 2.0 * (1.0 - stats.t.cdf(np.abs(t), df=n - k))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "coefficients": [float(v) for v in beta],
        "standard_errors": [float(v) for v in se],
        "t_statistics": [float(v) for v in t],
        "p_values": [float(v) for v in p],
        "r_squared": float(r2),
        "adj_r_squared": float(1.0 - (1.0 - r2) * (n - 1) / (n - k)) if n > k else 0.0,
        "lags": int(lags),
        "observations": int(n),
    }
