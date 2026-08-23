"""Tests for the ML regime classifier.

The no-lookahead property is the one that matters most, mirroring
`test_walkforward.py::TestNoLookahead` -- a regime feature computed for day i
must be unaffected by anything at day i+1 onward, or the walk-forward
integration built on top of it would be quietly cheating.
"""

import numpy as np
import pandas as pd
import pytest

from app.regime_classifier import compute_features, fit_regimes


def _panel(n=300, p=4, seed=0):
    rng = np.random.default_rng(seed)
    corr = np.full((p, p), 0.3)
    np.fill_diagonal(corr, 1.0)
    z = rng.normal(0.0003, 0.01, (n, p)) @ np.linalg.cholesky(corr).T
    return pd.DataFrame(
        z, columns=[f"A{i}" for i in range(p)], index=pd.bdate_range("2020-01-01", periods=n)
    )


class TestComputeFeatures:
    def test_drops_the_warmup_window(self):
        panel = _panel(n=300)
        features = compute_features(panel, window=21)
        # The first valid rolling window ends on day 21 (1-indexed), so day 21
        # itself is the first non-NaN row -- n - window + 1 valid rows survive.
        assert len(features) == len(panel) - 21 + 1
        assert list(features.columns) == ["volatility", "correlation", "trend"]
        assert not features.isna().any().any()

    def test_no_lookahead(self):
        """A feature at day i is unchanged when the future is rewritten."""
        panel = _panel(n=300)
        base = compute_features(panel, window=21)

        tampered = panel.copy()
        tampered.iloc[200:] *= 10.0  # violently different future
        after = compute_features(tampered, window=21)

        early = base.index[base.index < panel.index[200]]
        pd.testing.assert_frame_equal(base.loc[early], after.loc[early])

    def test_high_volatility_regime_is_detected(self):
        """A synthetic vol spike should show up as elevated rolling volatility."""
        panel = _panel(n=300, seed=1)
        panel.iloc[150:180] *= 8.0
        features = compute_features(panel, window=21)
        calm_vol = features["volatility"].iloc[:100].mean()
        spike_vol = features.loc[panel.index[160:180]]["volatility"].mean()
        assert spike_vol > calm_vol * 2


class TestFitRegimes:
    def test_labels_are_ordered_by_ascending_volatility(self):
        panel = _panel(n=400, seed=2)
        panel.iloc[250:300] *= 6.0  # inject a clear high-vol block
        features = compute_features(panel, window=21)
        result = fit_regimes(features, n_regimes=3, seed=7)

        labels = result["labels"]
        vol_by_label = [features["volatility"][labels == lvl].mean() for lvl in range(3)]
        assert vol_by_label == sorted(vol_by_label)

    def test_deterministic_given_the_same_seed(self):
        panel = _panel(n=300, seed=3)
        features = compute_features(panel, window=21)
        a = fit_regimes(features, n_regimes=3, seed=7)
        b = fit_regimes(features, n_regimes=3, seed=7)
        pd.testing.assert_series_equal(a["labels"], b["labels"])

    def test_raises_on_too_few_observations(self):
        panel = _panel(n=70, seed=4)
        features = compute_features(panel, window=21)
        with pytest.raises(ValueError):
            fit_regimes(features, n_regimes=3)

    def test_level_names_match_regime_count(self):
        panel = _panel(n=300, seed=5)
        features = compute_features(panel, window=21)
        result = fit_regimes(features, n_regimes=3)
        assert result["level_names"] == ["Calm", "Elevated", "Stressed"]
        assert set(result["labels"].unique()) <= {0, 1, 2}
