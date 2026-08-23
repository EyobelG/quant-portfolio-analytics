"""Unsupervised market-regime labeling.

`regimes.py` already shows correlation is conditional on the market being calm
vs. stressed, but splits the sample with a fixed benchmark-decile rule decided
in advance. This module learns the split instead: a Gaussian Mixture Model
clusters days by rolling volatility, average pairwise correlation, and trend,
the classic signature of a regime shift being volatility and correlation
moving together (everything sells off at once).

This is deliberately *not* a return predictor. It labels which kind of market
today resembles, nothing about tomorrow. Whether that label is actually useful
for picking an allocation is a separate, testable claim -- see
`walkforward.select_best_strategy` and the `regime_ml` entry in
`walkforward._STRATEGIES`, which is judged by the same out-of-sample walk
forward test as every other candidate strategy, not asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Below this many labeled (post-warmup) observations, a 3-component GMM is
# fitting noise rather than structure.
_MIN_OBSERVATIONS = 60


def compute_features(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling volatility, average pairwise correlation, and trend.

    Each row uses only the trailing `window` days ending on that date, so a
    feature at day i is unaffected by anything at day i+1 onward -- the same
    no-lookahead property every walk-forward solver in this app relies on.
    The first `window` rows are NaN (not enough trailing history) and are the
    caller's to drop.
    """
    n_obs, n_assets = returns.shape
    values = returns.to_numpy(dtype=float)
    equal_weight_rets = values.mean(axis=1)

    vol = np.full(n_obs, np.nan)
    corr = np.full(n_obs, np.nan)
    trend = np.full(n_obs, np.nan)

    for i in range(window, n_obs + 1):
        block = values[i - window : i]
        window_ew_rets = equal_weight_rets[i - window : i]
        vol[i - 1] = window_ew_rets.std(ddof=1)
        trend[i - 1] = window_ew_rets.mean()
        if n_assets >= 2:
            c = np.corrcoef(block, rowvar=False)
            iu = np.triu_indices(n_assets, k=1)
            corr[i - 1] = float(np.nanmean(c[iu]))
        else:
            corr[i - 1] = 0.0

    periods = 252
    features = pd.DataFrame(
        {
            "volatility": vol * np.sqrt(periods),
            "correlation": corr,
            "trend": trend * periods,
        },
        index=returns.index,
    )
    return features.dropna()


def fit_regimes(features: pd.DataFrame, n_regimes: int = 3, seed: int = 7) -> dict:
    """Cluster days into regimes, relabeled by ascending average volatility.

    A GMM's raw cluster indices are arbitrary and can flip between fits, which
    would make "regime 0" mean something different every time this is called.
    Relabeling by volatility gives every caller the same stable ordering:
    label 0 is always the calmest regime found, the last is always the most
    volatile, regardless of which raw cluster the model happened to number
    first.
    """
    if len(features) < _MIN_OBSERVATIONS:
        raise ValueError(
            f"Needs at least {_MIN_OBSERVATIONS} observations after the rolling-feature "
            f"warmup window; got {len(features)}. Try a longer lookback period."
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(features.to_numpy(dtype=float))

    gmm = GaussianMixture(n_components=n_regimes, random_state=seed, n_init=4)
    raw_labels = gmm.fit_predict(X)

    # Order raw cluster indices by their mean volatility feature, then map
    # each raw label to its rank -- rank 0 is calmest, rank n-1 is stormiest.
    vol_by_cluster = [
        features["volatility"].to_numpy()[raw_labels == k].mean() for k in range(n_regimes)
    ]
    rank_of_cluster = {
        cluster: rank for rank, cluster in enumerate(np.argsort(vol_by_cluster))
    }
    labels = pd.Series(
        [rank_of_cluster[c] for c in raw_labels], index=features.index, name="regime"
    )

    base_names = ["Calm", "Elevated", "Stressed"]
    if n_regimes <= len(base_names):
        level_names = base_names[:n_regimes]
    else:
        level_names = [f"Level {i}" for i in range(n_regimes)]

    posterior = gmm.predict_proba(X)
    # Reindex posterior columns to match the volatility-rank relabeling above.
    posterior = posterior[:, [c for c, _ in sorted(rank_of_cluster.items(), key=lambda kv: kv[1])]]

    return {
        "labels": labels,
        "level_names": level_names,
        "converged": bool(gmm.converged_),
        "confidence": pd.Series(posterior.max(axis=1), index=features.index),
    }
