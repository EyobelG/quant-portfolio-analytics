"""Covariance estimation and risk decomposition.

The sample covariance matrix is the weakest input to mean-variance optimization
and the reason its output is so often unusable. With p assets and n observations
it estimates p(p+1)/2 parameters from np numbers; the smallest eigenvalues are
biased toward zero, inverting it amplifies exactly that error, and the optimizer
then concentrates the portfolio into whichever asset the noise happened to
flatter. This module implements the two standard defences — shrinkage and
random-matrix denoising — and then decomposes where the risk actually sits.

References
----------
Ledoit & Wolf (2004), "A Well-Conditioned Estimator for Large-Dimensional
    Covariance Matrices", Journal of Multivariate Analysis 88(2).
Marchenko & Pastur (1967), "Distribution of eigenvalues for some sets of
    random matrices", Mathematics of the USSR-Sbornik 1(4).
López de Prado (2020), "Machine Learning for Asset Managers", ch. 2.
Meucci (2009), "Managing Diversification", Risk 22(5) — effective number of bets.
Maillard, Roncalli & Teiletche (2010), "The Properties of Equally Weighted Risk
    Contribution Portfolios", Journal of Portfolio Management 36(4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def ledoit_wolf_shrinkage(returns: pd.DataFrame, frequency: int = 252) -> dict:
    """Shrink the sample covariance toward a scaled identity target.

    The shrinkage intensity is not tuned — it is the analytically optimal value
    that minimizes expected squared Frobenius error, derived in closed form from
    the data. That is the appeal of the estimator: no cross-validation, no
    hyperparameter, and it is guaranteed well-conditioned even when p approaches
    n and the sample matrix is singular.
    """
    X = np.asarray(returns, dtype=float)
    n, p = X.shape
    if n < 2:
        raise ValueError("Need at least 2 observations")

    Xc = X - X.mean(axis=0)
    # MLE covariance (1/n) is the estimator the Ledoit-Wolf derivation assumes.
    S = Xc.T @ Xc / n

    # Normalized Frobenius inner product <A,B> = trace(A B') / p.
    mu = np.trace(S) / p
    target = mu * np.eye(p)

    d2 = np.sum((S - target) ** 2) / p

    # b_bar2 = (1/n^2) * sum_k ||x_k x_k' - S||^2_F / p, expanded so it costs one
    # pass instead of forming n outer products.
    norms_sq = np.sum(Xc**2, axis=1)  # x_k' x_k for each observation
    S_frob_sq = np.sum(S**2)
    b_bar2 = (np.sum(norms_sq**2) - n * S_frob_sq) / (n**2 * p)

    # The optimal intensity cannot exceed 1: b2 is capped at d2.
    b2 = min(b_bar2, d2)
    a2 = d2 - b2
    intensity = float(b2 / d2) if d2 > 0 else 0.0

    shrunk = intensity * target + (1.0 - intensity) * S

    sample_ann = S * frequency
    shrunk_ann = shrunk * frequency

    return {
        "shrinkage_intensity": intensity,
        "covariance": pd.DataFrame(shrunk_ann, index=returns.columns, columns=returns.columns),
        "sample_covariance": pd.DataFrame(
            sample_ann, index=returns.columns, columns=returns.columns
        ),
        "condition_number_sample": float(np.linalg.cond(S)),
        "condition_number_shrunk": float(np.linalg.cond(shrunk)),
        "target_variance": float(mu * frequency),
        "n_observations": int(n),
        "n_assets": int(p),
    }


def marchenko_pastur_denoise(returns: pd.DataFrame, frequency: int = 252) -> dict:
    """Strip eigenvalues that are statistically indistinguishable from noise.

    Random-matrix theory gives the exact eigenvalue distribution of a
    correlation matrix built from *pure noise*. Anything inside that bulk
    carries no information, so it is replaced by the bulk's average — flattening
    the noise while preserving the trace and leaving the genuine market and
    sector factors above the edge untouched.
    """
    corr = returns.corr().to_numpy()
    p = corr.shape[0]
    n = returns.shape[0]

    eigvals, eigvecs = np.linalg.eigh(corr)
    # eigh returns ascending; work descending so index 0 is the market factor.
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    c = p / n  # aspect ratio
    lambda_plus = float((1.0 + np.sqrt(c)) ** 2)
    lambda_minus = float((1.0 - np.sqrt(c)) ** 2)

    signal_mask = eigvals > lambda_plus
    n_signal = int(signal_mask.sum())

    denoised_vals = eigvals.copy()
    if n_signal < p:
        # Constant-residual-eigenvalue method: the noise eigenvalues share their
        # mean, which preserves the trace of the correlation matrix.
        noise_avg = eigvals[~signal_mask].mean()
        denoised_vals[~signal_mask] = noise_avg

    denoised_corr = eigvecs @ np.diag(denoised_vals) @ eigvecs.T
    # Rescale so the diagonal is exactly 1 and it is a valid correlation matrix.
    d = np.sqrt(np.diag(denoised_corr))
    denoised_corr = denoised_corr / np.outer(d, d)

    vols = returns.std(ddof=1).to_numpy() * np.sqrt(frequency)
    denoised_cov = denoised_corr * np.outer(vols, vols)

    total = float(eigvals.sum())
    return {
        "eigenvalues": [float(v) for v in eigvals],
        "denoised_eigenvalues": [float(v) for v in denoised_vals],
        "variance_explained": [float(v / total) for v in eigvals],
        "cumulative_variance_explained": [float(v) for v in np.cumsum(eigvals) / total],
        "lambda_plus": lambda_plus,
        "lambda_minus": lambda_minus,
        "n_signal_factors": n_signal,
        "n_assets": int(p),
        "n_observations": int(n),
        # The top eigenvalue of an equity correlation matrix is the market
        # factor; its share is a direct read on systematic exposure.
        "market_factor_share": float(eigvals[0] / total),
        "denoised_covariance": pd.DataFrame(
            denoised_cov, index=returns.columns, columns=returns.columns
        ),
        # RMT assumes p and n are both large; with a handful of holdings the
        # bulk edge is only indicative.
        "reliable": bool(p >= 10),
    }


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> dict:
    """Decompose portfolio volatility into per-asset contributions.

    Weight is not risk. A 10% position in a volatile, highly-correlated asset can
    carry several times the risk of a 10% position in a diversifier, and this is
    the decomposition that shows it. Contributions sum exactly to portfolio
    volatility, by Euler's theorem on homogeneous functions.
    """
    w = np.asarray(weights, dtype=float).ravel()
    S = np.asarray(cov, dtype=float)

    variance = float(w @ S @ w)
    vol = float(np.sqrt(variance))
    if vol <= 0:
        raise ValueError("Portfolio volatility is zero")

    # Marginal contribution: d(sigma_p)/d(w_i).
    mctr = (S @ w) / vol
    cctr = w * mctr  # component contribution; sums to vol

    asset_vols = np.sqrt(np.diag(S))
    # Diversification ratio: 1.0 means no diversification benefit at all.
    div_ratio = float((w @ asset_vols) / vol) if vol > 0 else 1.0

    return {
        "portfolio_volatility": vol,
        "marginal_contributions": [float(v) for v in mctr],
        "component_contributions": [float(v) for v in cctr],
        "percent_contributions": [float(v) for v in (cctr / vol)],
        "diversification_ratio": div_ratio,
        # Herfindahl on risk shares: 1/HHI is the effective number of equally
        # risky positions.
        "effective_positions": float(1.0 / np.sum((cctr / vol) ** 2)),
    }


def effective_number_of_bets(weights: np.ndarray, cov: np.ndarray) -> dict:
    """Meucci's diversification measure in principal-component space.

    Counting positions overstates diversification when the positions move
    together. This projects the portfolio onto the uncorrelated principal
    components and takes the exponential entropy of their variance shares, so
    five holdings that all load on one factor score close to 1, not 5.
    """
    w = np.asarray(weights, dtype=float).ravel()
    S = np.asarray(cov, dtype=float)

    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    eigvals = np.maximum(eigvals, 0.0)

    # Exposure to each principal component.
    y = eigvecs.T @ w
    contributions = (y**2) * eigvals
    total = contributions.sum()
    if total <= 0:
        raise ValueError("Portfolio has no variance to decompose")

    shares = contributions / total
    nonzero = shares[shares > 1e-12]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))

    return {
        "effective_bets": float(np.exp(entropy)),
        "n_assets": int(w.size),
        "component_shares": [float(v) for v in shares],
        "entropy": entropy,
        # 1.0 = every unit of risk rides one factor; n = perfectly balanced.
        "concentration": float(shares.max()),
    }


def equal_risk_contribution(cov: np.ndarray, max_iter: int = 500) -> dict:
    """Solve for the long-only portfolio where every asset contributes equal risk.

    Risk parity needs no expected-return forecast at all, which is precisely why
    it survives out-of-sample where mean-variance does not: the covariance matrix
    is estimated far more reliably than the mean. Solved here as a constrained
    least-squares problem on the risk shares.
    """
    S = np.asarray(cov, dtype=float)
    n = S.shape[0]
    target = 1.0 / n

    def objective(w: np.ndarray) -> float:
        vol = np.sqrt(w @ S @ w)
        if vol <= 0:
            return 1e6
        shares = w * (S @ w) / vol**2
        return float(np.sum((shares - target) ** 2))

    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(1e-6, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": max_iter, "ftol": 1e-12},
    )

    w = np.asarray(result.x, dtype=float)
    w = w / w.sum()  # renormalize against solver drift
    rc = risk_contributions(w, S)

    return {
        "weights": [float(v) for v in w],
        "converged": bool(result.success),
        "volatility": rc["portfolio_volatility"],
        "percent_contributions": rc["percent_contributions"],
        # How far from perfect parity the solver landed, in percentage points.
        "max_deviation": float(
            np.max(np.abs(np.asarray(rc["percent_contributions"]) - target))
        ),
    }
