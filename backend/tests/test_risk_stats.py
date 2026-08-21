"""Tests for statistical inference.

Where a closed form exists, the test asserts against it rather than against a
previously-recorded output, so these catch real regressions instead of pinning
in whatever the code happened to do first.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.risk_stats import (
    bootstrap_sharpe_ci,
    christoffersen_independence,
    cornish_fisher_var,
    deflated_sharpe_ratio,
    kupiec_pof,
    moments,
    newey_west_tstats,
    probabilistic_sharpe_ratio,
    var_backtest,
)


@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0005, 0.01, 2000))


@pytest.fixture
def skewed_returns():
    rng = np.random.default_rng(7)
    # Negative skew and fat tails, like an equity index.
    return pd.Series(-stats.skewnorm.rvs(a=6, loc=-0.0005, scale=0.012, size=2000, random_state=rng))


class TestMoments:
    def test_normal_sample_is_not_rejected(self, normal_returns):
        m = moments(normal_returns)
        assert m["normal_at_5pct"]
        assert abs(m["skewness"]) < 0.2
        assert abs(m["excess_kurtosis"]) < 0.3

    def test_recovers_mean_and_std(self, normal_returns):
        m = moments(normal_returns)
        assert m["mean"] == pytest.approx(normal_returns.mean())
        assert m["std"] == pytest.approx(normal_returns.std(ddof=1))
        assert m["observations"] == 2000

    def test_fat_tails_are_rejected(self):
        rng = np.random.default_rng(3)
        t_dist = pd.Series(rng.standard_t(df=3, size=2000) * 0.01)
        m = moments(t_dist)
        assert not m["normal_at_5pct"]
        assert m["excess_kurtosis"] > 1.0


class TestProbabilisticSharpe:
    def test_matches_normal_case_closed_form(self):
        """With zero skew and kurtosis 3, PSR collapses to the textbook t-test."""
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0.001, 0.01, 5000))
        psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0, periods_per_year=252)

        n = r.size
        sr = r.mean() / r.std(ddof=1)
        # Under normality the variance term is 1 + 0.5*sr^2.
        expected_z = sr * np.sqrt(n - 1) / np.sqrt(1 + 0.5 * sr**2)
        assert psr["z_score"] == pytest.approx(expected_z, rel=0.02)

    def test_strong_track_record_is_confident(self):
        rng = np.random.default_rng(2)
        r = pd.Series(rng.normal(0.002, 0.008, 1500))  # Sharpe ~4 annualized
        assert probabilistic_sharpe_ratio(r)["psr"] > 0.99

    def test_short_noisy_record_is_not(self):
        rng = np.random.default_rng(5)
        r = pd.Series(rng.normal(0.0002, 0.02, 60))
        assert probabilistic_sharpe_ratio(r)["psr"] < 0.8

    def test_negative_skew_lowers_confidence(self):
        """Two series with the same Sharpe but different skew differ in PSR."""
        rng = np.random.default_rng(11)
        base = rng.normal(0, 0.01, 3000)
        sym = pd.Series(base + 0.0006)

        skew_raw = -stats.skewnorm.rvs(a=8, size=3000, random_state=rng)
        # Rescale so both series have an identical Sharpe ratio.
        skewed = (skew_raw - skew_raw.mean()) / skew_raw.std(ddof=1)
        skewed = pd.Series(skewed * sym.std(ddof=1) + sym.mean())

        assert sym.mean() / sym.std(ddof=1) == pytest.approx(
            skewed.mean() / skewed.std(ddof=1), rel=1e-6
        )
        assert probabilistic_sharpe_ratio(skewed)["psr"] < probabilistic_sharpe_ratio(sym)["psr"]

    def test_higher_benchmark_lowers_psr(self, normal_returns):
        low = probabilistic_sharpe_ratio(normal_returns, benchmark_sr=0.0)["psr"]
        high = probabilistic_sharpe_ratio(normal_returns, benchmark_sr=2.0)["psr"]
        assert high < low

    def test_rejects_constant_series(self):
        with pytest.raises(ValueError):
            probabilistic_sharpe_ratio(pd.Series([0.01] * 100))


class TestDeflatedSharpe:
    def test_more_trials_raise_the_bar(self, normal_returns):
        rng = np.random.default_rng(9)
        few = list(rng.normal(0.5, 0.5, 5))
        many = list(rng.normal(0.5, 0.5, 500))
        assert (
            deflated_sharpe_ratio(normal_returns, many)["expected_max_sharpe"]
            > deflated_sharpe_ratio(normal_returns, few)["expected_max_sharpe"]
        )

    def test_dsr_is_below_psr(self, normal_returns):
        """Deflating against a positive bar must reduce confidence."""
        rng = np.random.default_rng(4)
        trials = list(rng.normal(0.4, 0.6, 100))
        dsr = deflated_sharpe_ratio(normal_returns, trials)
        psr = probabilistic_sharpe_ratio(normal_returns, benchmark_sr=0.0)
        assert dsr["expected_max_sharpe"] > 0
        assert dsr["dsr"] < psr["psr"]

    def test_requires_multiple_trials(self, normal_returns):
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(normal_returns, [1.0])


class TestBootstrap:
    def test_interval_brackets_point_estimate(self, normal_returns):
        ci = bootstrap_sharpe_ci(normal_returns, n_boot=400)
        assert ci["ci_lower"] < ci["sharpe"] < ci["ci_upper"]

    def test_is_deterministic_given_a_seed(self, normal_returns):
        a = bootstrap_sharpe_ci(normal_returns, n_boot=200, seed=99)
        b = bootstrap_sharpe_ci(normal_returns, n_boot=200, seed=99)
        assert a["ci_lower"] == b["ci_lower"]

    def test_noise_is_not_significant(self):
        rng = np.random.default_rng(21)
        r = pd.Series(rng.normal(0.0, 0.01, 500))  # genuinely no edge
        assert not bootstrap_sharpe_ci(r, n_boot=500)["significant"]

    def test_wider_interval_for_shorter_sample(self):
        rng = np.random.default_rng(31)
        full = pd.Series(rng.normal(0.0008, 0.01, 2000))
        short = full.iloc[:150]
        w_full = bootstrap_sharpe_ci(full, n_boot=400)
        w_short = bootstrap_sharpe_ci(short, n_boot=400)
        assert (w_short["ci_upper"] - w_short["ci_lower"]) > (
            w_full["ci_upper"] - w_full["ci_lower"]
        )


class TestCornishFisher:
    def test_collapses_to_gaussian_for_normal_data(self, normal_returns):
        cf = cornish_fisher_var(normal_returns)
        assert cf["z_cornish_fisher"] == pytest.approx(cf["z_normal"], abs=0.12)
        assert cf["modified_var"] == pytest.approx(cf["gaussian_var"], abs=0.0015)

    def test_negative_skew_makes_var_worse(self, skewed_returns):
        cf = cornish_fisher_var(skewed_returns)
        # A left-skewed series has a deeper tail than the normal model admits.
        assert cf["modified_var"] < cf["gaussian_var"]
        assert cf["understatement"] > 0

    def test_cvar_is_worse_than_var(self, normal_returns):
        cf = cornish_fisher_var(normal_returns)
        assert cf["historical_cvar"] < cf["historical_var"]


class TestKupiec:
    def test_perfect_calibration_is_not_rejected(self):
        r = kupiec_pof(exceptions=50, observations=1000, alpha=0.05)
        assert r["lr_statistic"] == pytest.approx(0.0, abs=1e-9)
        assert r["p_value"] == pytest.approx(1.0)
        assert r["correctly_calibrated"]

    def test_far_too_many_breaches_is_rejected(self):
        assert not kupiec_pof(exceptions=150, observations=1000, alpha=0.05)["correctly_calibrated"]

    def test_far_too_few_breaches_is_also_rejected(self):
        assert not kupiec_pof(exceptions=2, observations=1000, alpha=0.05)["correctly_calibrated"]

    def test_zero_exceptions_stays_finite(self):
        r = kupiec_pof(exceptions=0, observations=500, alpha=0.05)
        assert np.isfinite(r["lr_statistic"])
        # Closed form at x=0: LR = -2 n ln(1 - alpha).
        assert r["lr_statistic"] == pytest.approx(-2 * 500 * np.log(0.95))

    def test_all_exceptions_stays_finite(self):
        r = kupiec_pof(exceptions=100, observations=100, alpha=0.05)
        assert np.isfinite(r["lr_statistic"])
        assert r["lr_statistic"] == pytest.approx(-2 * 100 * np.log(0.05))


class TestChristoffersen:
    def test_independent_breaches_are_not_rejected(self):
        rng = np.random.default_rng(17)
        breaches = (rng.random(3000) < 0.05).astype(int)
        assert christoffersen_independence(breaches)["independent"]

    def test_clustered_breaches_are_rejected(self):
        # Breaches arriving in solid blocks: strong serial dependence.
        b = np.zeros(1000, dtype=int)
        for start in range(0, 1000, 100):
            b[start : start + 12] = 1
        assert not christoffersen_independence(b)["independent"]

    def test_transition_counts_are_exhaustive(self):
        b = np.array([0, 1, 1, 0, 0, 1])
        r = christoffersen_independence(b)
        assert r["n00"] + r["n01"] + r["n10"] + r["n11"] == b.size - 1


class TestVarBacktest:
    def test_well_specified_model_passes(self):
        rng = np.random.default_rng(23)
        r = pd.Series(rng.normal(0.0004, 0.01, 1500))
        result = var_backtest(r, alpha=0.05, window=250)
        assert result["kupiec"]["exception_rate"] == pytest.approx(0.05, abs=0.025)
        assert result["model_adequate"]

    def test_requires_enough_history(self):
        with pytest.raises(ValueError):
            var_backtest(pd.Series(np.zeros(100)), window=250)

    def test_forecast_is_out_of_sample(self):
        """The VaR for day t must not depend on the return on day t."""
        rng = np.random.default_rng(29)
        r = pd.Series(rng.normal(0, 0.01, 800))
        base = var_backtest(r, window=250)

        # Perturbing only the final observation cannot change earlier breaches,
        # and can change at most the single last one.
        bumped = r.copy()
        bumped.iloc[-1] = -0.5
        after = var_backtest(bumped, window=250)
        assert after["kupiec"]["exceptions"] - base["kupiec"]["exceptions"] in (0, 1)


class TestNeweyWest:
    def test_coefficients_match_ols(self):
        rng = np.random.default_rng(13)
        n = 500
        x = rng.normal(0, 1, n)
        y = 0.3 + 1.7 * x + rng.normal(0, 0.5, n)
        X = np.column_stack([np.ones(n), x])

        result = newey_west_tstats(y, X, lags=0)
        ols = np.linalg.lstsq(X, y, rcond=None)[0]
        assert result["coefficients"] == pytest.approx(list(ols), rel=1e-10)

    def test_recovers_known_slope(self):
        rng = np.random.default_rng(19)
        n = 4000
        x = rng.normal(0, 1, n)
        y = 0.5 + 2.0 * x + rng.normal(0, 0.3, n)
        r = newey_west_tstats(y, np.column_stack([np.ones(n), x]))
        assert r["coefficients"][1] == pytest.approx(2.0, abs=0.03)
        assert r["r_squared"] > 0.9

    def test_zero_lag_equals_white_standard_errors(self):
        """With no lags the sandwich reduces to the HC0 estimator."""
        rng = np.random.default_rng(37)
        n = 300
        x = rng.normal(0, 1, n)
        y = 1.0 + 0.5 * x + rng.normal(0, 1, n) * (1 + np.abs(x))
        X = np.column_stack([np.ones(n), x])

        result = newey_west_tstats(y, X, lags=0)

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        u = y - X @ beta
        XtX_inv = np.linalg.inv(X.T @ X)
        hc0 = XtX_inv @ (X * u[:, None]).T @ (X * u[:, None]) @ XtX_inv
        assert result["standard_errors"] == pytest.approx(list(np.sqrt(np.diag(hc0))), rel=1e-9)

    def test_autocorrelation_widens_standard_errors(self):
        """The whole point: HAC errors must exceed naive ones under serial dependence."""
        rng = np.random.default_rng(41)
        n = 1000
        x = rng.normal(0, 1, n)
        # Strongly autocorrelated residuals.
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = 0.85 * u[t - 1] + rng.normal(0, 0.3)
        y = 1.0 + 0.2 * x + u
        X = np.column_stack([np.ones(n), x])

        naive = newey_west_tstats(y, X, lags=0)["standard_errors"][0]
        hac = newey_west_tstats(y, X, lags=20)["standard_errors"][0]
        assert hac > naive * 1.5

    def test_rejects_underdetermined_system(self):
        with pytest.raises(ValueError):
            newey_west_tstats(np.array([1.0, 2.0]), np.eye(2))
