"""Tests for the conditional volatility models.

The GARCH tests simulate from the model with known parameters and check that
maximum likelihood recovers them. That is the only test that actually validates
an estimator — asserting on a fitted value from real data just pins in whatever
the optimizer did the first time.
"""

import numpy as np
import pandas as pd
import pytest

from app.volatility import arch_lm_test, ewma_volatility, fit_garch


def simulate_garch(n, omega, alpha, beta, seed=0, burn=500):
    """Draw a GARCH(1,1) path with Gaussian innovations."""
    rng = np.random.default_rng(seed)
    total = n + burn
    r = np.zeros(total)
    var = np.zeros(total)
    var[0] = omega / (1 - alpha - beta)
    for t in range(1, total):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
        r[t] = rng.normal(0, np.sqrt(var[t]))
    return pd.Series(r[burn:])


class TestEWMA:
    def test_half_life_matches_closed_form(self):
        out = ewma_volatility(pd.Series(np.random.default_rng(0).normal(0, 0.01, 500)))
        # lambda = 0.94 decays to half in log(0.5)/log(0.94) ~ 11.2 days.
        assert out["half_life_days"] == pytest.approx(11.2, abs=0.2)

    def test_tracks_a_volatility_regime_shift(self):
        """Calm then turbulent: EWMA must end far above where it started."""
        rng = np.random.default_rng(1)
        calm = rng.normal(0, 0.005, 300)
        wild = rng.normal(0, 0.03, 300)
        out = ewma_volatility(pd.Series(np.concatenate([calm, wild])))
        assert out["volatility"][290] < out["current"] / 2

    def test_reacts_faster_with_smaller_lambda(self):
        rng = np.random.default_rng(2)
        r = pd.Series(np.concatenate([rng.normal(0, 0.005, 200), rng.normal(0, 0.04, 30)]))
        assert ewma_volatility(r, lam=0.80)["current"] > ewma_volatility(r, lam=0.99)["current"]

    def test_rejects_invalid_lambda(self):
        with pytest.raises(ValueError):
            ewma_volatility(pd.Series(np.zeros(100)), lam=1.5)

    def test_output_length_matches_input(self):
        r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 400))
        assert len(ewma_volatility(r)["volatility"]) == 400


class TestGarch:
    def test_recovers_known_parameters(self):
        omega, alpha, beta = 2e-6, 0.08, 0.90
        r = simulate_garch(6000, omega, alpha, beta, seed=42)
        fit = fit_garch(r)

        assert fit["converged"]
        assert fit["alpha"] == pytest.approx(alpha, abs=0.035)
        assert fit["beta"] == pytest.approx(beta, abs=0.045)
        assert fit["omega"] == pytest.approx(omega, rel=1.2)

    def test_recovers_persistence(self):
        r = simulate_garch(5000, 3e-6, 0.10, 0.85, seed=7)
        assert fit_garch(r)["persistence"] == pytest.approx(0.95, abs=0.04)

    def test_stays_inside_the_stationary_region(self):
        for seed in range(4):
            fit = fit_garch(simulate_garch(2000, 2e-6, 0.07, 0.90, seed=seed))
            assert fit["persistence"] < 1.0
            assert fit["omega"] > 0
            assert fit["alpha"] >= 0 and fit["beta"] >= 0

    def test_long_run_volatility_matches_the_unconditional_level(self):
        omega, alpha, beta = 2e-6, 0.08, 0.90
        r = simulate_garch(6000, omega, alpha, beta, seed=11)
        fit = fit_garch(r)
        true_annual = np.sqrt(omega / (1 - alpha - beta) * 252)
        assert fit["long_run_volatility"] == pytest.approx(true_annual, rel=0.25)

    def test_forecast_mean_reverts_toward_the_long_run_level(self):
        """From a calm state the term structure of vol must slope upward."""
        r = simulate_garch(4000, 2e-6, 0.08, 0.90, seed=13)
        fit = fit_garch(r, horizon=250)
        first, last = fit["forecast"][0], fit["forecast"][-1]
        long_run = fit["long_run_volatility"]
        # The far end of the forecast sits closer to the long-run level.
        assert abs(last - long_run) < abs(first - long_run) + 1e-12

    def test_forecast_length_matches_horizon(self):
        fit = fit_garch(simulate_garch(1500, 2e-6, 0.08, 0.9, seed=17), horizon=30)
        assert len(fit["forecast"]) == 30

    def test_conditional_volatility_is_annualized(self):
        r = simulate_garch(3000, 2e-6, 0.08, 0.90, seed=19)
        fit = fit_garch(r)
        realized = r.std(ddof=1) * np.sqrt(252)
        assert np.mean(fit["conditional_volatility"]) == pytest.approx(realized, rel=0.3)

    def test_half_life_is_positive_and_finite(self):
        fit = fit_garch(simulate_garch(3000, 2e-6, 0.05, 0.92, seed=23))
        assert fit["half_life_days"] is not None and 0 < fit["half_life_days"] < 500

    def test_requires_enough_observations(self):
        with pytest.raises(ValueError):
            fit_garch(pd.Series(np.random.default_rng(0).normal(0, 0.01, 50)))

    def test_information_criteria_are_finite(self):
        fit = fit_garch(simulate_garch(2000, 2e-6, 0.08, 0.9, seed=29))
        assert np.isfinite(fit["aic"]) and np.isfinite(fit["bic"])
        assert fit["bic"] > fit["aic"]  # BIC penalizes harder at n > 8


class TestArchLM:
    def test_detects_arch_effects_in_garch_data(self):
        r = simulate_garch(3000, 2e-6, 0.12, 0.85, seed=31)
        assert arch_lm_test(r)["arch_effects_present"]

    def test_finds_none_in_iid_noise(self):
        r = pd.Series(np.random.default_rng(37).normal(0, 0.01, 3000))
        assert not arch_lm_test(r)["arch_effects_present"]

    def test_statistic_is_non_negative(self):
        r = pd.Series(np.random.default_rng(41).normal(0, 0.01, 1000))
        out = arch_lm_test(r)
        assert out["lm_statistic"] >= 0
        assert 0.0 <= out["p_value"] <= 1.0

    def test_rejects_insufficient_data(self):
        with pytest.raises(ValueError):
            arch_lm_test(pd.Series([0.01, 0.02, 0.03]), lags=5)
