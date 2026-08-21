"""Tests for covariance estimation and risk decomposition.

The decomposition identities (Euler's theorem, trace preservation, entropy
bounds) are exact, so they are asserted exactly rather than approximately.
"""

import numpy as np
import pandas as pd
import pytest

from app.risk_model import (
    effective_number_of_bets,
    equal_risk_contribution,
    ledoit_wolf_shrinkage,
    marchenko_pastur_denoise,
    risk_contributions,
)


def _correlated_returns(n=750, p=5, rho=0.5, seed=0):
    """Returns with a known equicorrelation structure."""
    rng = np.random.default_rng(seed)
    corr = np.full((p, p), rho)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)
    z = rng.normal(0, 0.01, (n, p)) @ L.T
    return pd.DataFrame(z, columns=[f"A{i}" for i in range(p)])


class TestLedoitWolf:
    def test_intensity_is_a_valid_weight(self):
        r = _correlated_returns()
        assert 0.0 <= ledoit_wolf_shrinkage(r)["shrinkage_intensity"] <= 1.0

    def test_improves_conditioning(self):
        """The estimator's headline claim: a better-conditioned matrix."""
        # Few observations relative to assets is where the sample matrix suffers.
        r = _correlated_returns(n=60, p=25, rho=0.6, seed=5)
        out = ledoit_wolf_shrinkage(r)
        assert out["condition_number_shrunk"] < out["condition_number_sample"]

    def test_shrinks_harder_when_data_is_scarce(self):
        scarce = ledoit_wolf_shrinkage(_correlated_returns(n=40, p=15, seed=1))
        plentiful = ledoit_wolf_shrinkage(_correlated_returns(n=3000, p=15, seed=1))
        assert scarce["shrinkage_intensity"] > plentiful["shrinkage_intensity"]

    def test_result_is_symmetric_positive_definite(self):
        S = ledoit_wolf_shrinkage(_correlated_returns(n=50, p=20, seed=2))["covariance"].to_numpy()
        assert np.allclose(S, S.T)
        assert np.all(np.linalg.eigvalsh(S) > 0)

    def test_preserves_average_variance(self):
        """Shrinking toward a scaled identity leaves the trace unchanged."""
        r = _correlated_returns()
        out = ledoit_wolf_shrinkage(r)
        assert np.trace(out["covariance"].to_numpy()) == pytest.approx(
            np.trace(out["sample_covariance"].to_numpy()), rel=1e-10
        )

    def test_annualization_scales_linearly(self):
        r = _correlated_returns()
        daily = ledoit_wolf_shrinkage(r, frequency=1)["covariance"].to_numpy()
        annual = ledoit_wolf_shrinkage(r, frequency=252)["covariance"].to_numpy()
        assert annual == pytest.approx(daily * 252, rel=1e-10)


class TestMarchenkoPastur:
    def test_pure_noise_has_no_signal_factors(self):
        """Independent assets should produce eigenvalues inside the MP bulk."""
        rng = np.random.default_rng(11)
        r = pd.DataFrame(rng.normal(0, 0.01, (2000, 30)), columns=[f"A{i}" for i in range(30)])
        assert marchenko_pastur_denoise(r)["n_signal_factors"] <= 2

    def test_correlated_assets_expose_a_market_factor(self):
        r = _correlated_returns(n=2000, p=30, rho=0.6, seed=3)
        out = marchenko_pastur_denoise(r)
        assert out["n_signal_factors"] >= 1
        # Equicorrelation of 0.6 across 30 assets puts ~60% of variance in PC1.
        assert out["market_factor_share"] > 0.5

    def test_denoising_preserves_the_trace(self):
        """Replacing noise eigenvalues with their mean must keep sum = p."""
        r = _correlated_returns(n=500, p=20, seed=4)
        out = marchenko_pastur_denoise(r)
        assert sum(out["denoised_eigenvalues"]) == pytest.approx(
            sum(out["eigenvalues"]), rel=1e-9
        )
        assert sum(out["eigenvalues"]) == pytest.approx(20.0, rel=1e-9)

    def test_denoised_matrix_is_a_valid_correlation_structure(self):
        r = _correlated_returns(n=400, p=12, seed=6)
        cov = marchenko_pastur_denoise(r)["denoised_covariance"].to_numpy()
        assert np.all(np.linalg.eigvalsh(cov) > -1e-10)
        assert np.allclose(cov, cov.T)

    def test_bulk_edge_matches_the_closed_form(self):
        r = _correlated_returns(n=1000, p=10, seed=8)
        out = marchenko_pastur_denoise(r)
        c = 10 / 1000
        assert out["lambda_plus"] == pytest.approx((1 + np.sqrt(c)) ** 2)

    def test_flags_itself_unreliable_for_small_portfolios(self):
        assert not marchenko_pastur_denoise(_correlated_returns(p=5))["reliable"]

    def test_variance_explained_is_a_distribution(self):
        out = marchenko_pastur_denoise(_correlated_returns(p=8))
        assert sum(out["variance_explained"]) == pytest.approx(1.0)
        assert out["cumulative_variance_explained"][-1] == pytest.approx(1.0)


class TestRiskContributions:
    def test_contributions_sum_to_volatility(self):
        """Euler's theorem — this identity is exact, not approximate."""
        rng = np.random.default_rng(0)
        A = rng.normal(0, 1, (6, 6))
        S = A @ A.T / 6
        w = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
        rc = risk_contributions(w, S)
        assert sum(rc["component_contributions"]) == pytest.approx(
            rc["portfolio_volatility"], rel=1e-12
        )
        assert sum(rc["percent_contributions"]) == pytest.approx(1.0, rel=1e-12)

    def test_matches_volatility_computed_directly(self):
        S = np.array([[0.04, 0.01], [0.01, 0.09]])
        w = np.array([0.6, 0.4])
        assert risk_contributions(w, S)["portfolio_volatility"] == pytest.approx(
            np.sqrt(w @ S @ w)
        )

    def test_uncorrelated_equal_vol_gives_no_concentration(self):
        S = np.eye(4) * 0.04
        rc = risk_contributions(np.full(4, 0.25), S)
        assert rc["percent_contributions"] == pytest.approx([0.25] * 4)
        assert rc["effective_positions"] == pytest.approx(4.0)

    def test_diversification_ratio_is_one_when_perfectly_correlated(self):
        S = np.full((3, 3), 0.04)  # correlation 1 across the board
        assert risk_contributions(np.full(3, 1 / 3), S)["diversification_ratio"] == pytest.approx(
            1.0, rel=1e-9
        )

    def test_diversification_ratio_exceeds_one_when_uncorrelated(self):
        S = np.eye(4) * 0.04
        assert risk_contributions(np.full(4, 0.25), S)["diversification_ratio"] == pytest.approx(
            2.0, rel=1e-9
        )

    def test_volatile_asset_carries_more_risk_than_weight(self):
        S = np.diag([0.01, 0.25])  # second asset is 5x as volatile
        rc = risk_contributions(np.array([0.5, 0.5]), S)
        assert rc["percent_contributions"][1] > 0.9


class TestEffectiveBets:
    def test_uncorrelated_equal_weight_uses_every_bet(self):
        S = np.eye(5) * 0.04
        assert effective_number_of_bets(np.full(5, 0.2), S)["effective_bets"] == pytest.approx(
            5.0, rel=1e-9
        )

    def test_perfectly_correlated_assets_are_one_bet(self):
        S = np.full((5, 5), 0.04)
        assert effective_number_of_bets(np.full(5, 0.2), S)["effective_bets"] == pytest.approx(
            1.0, abs=1e-6
        )

    def test_correlation_reduces_effective_bets_below_count(self):
        r = _correlated_returns(p=6, rho=0.7, seed=12)
        S = r.cov().to_numpy() * 252
        enb = effective_number_of_bets(np.full(6, 1 / 6), S)
        assert 1.0 < enb["effective_bets"] < 6.0

    def test_bounded_by_asset_count(self):
        rng = np.random.default_rng(15)
        for seed in range(5):
            r = _correlated_returns(p=7, rho=0.3, seed=seed)
            S = r.cov().to_numpy()
            w = rng.dirichlet(np.ones(7))
            assert 1.0 <= effective_number_of_bets(w, S)["effective_bets"] <= 7.0 + 1e-9


class TestEqualRiskContribution:
    def test_equalizes_risk_shares(self):
        rng = np.random.default_rng(2)
        A = rng.normal(0, 1, (5, 5))
        S = A @ A.T / 5
        out = equal_risk_contribution(S)
        assert out["converged"]
        assert out["percent_contributions"] == pytest.approx([0.2] * 5, abs=1e-4)

    def test_weights_are_a_long_only_allocation(self):
        rng = np.random.default_rng(3)
        A = rng.normal(0, 1, (6, 6))
        out = equal_risk_contribution(A @ A.T / 6)
        assert sum(out["weights"]) == pytest.approx(1.0)
        assert all(w >= 0 for w in out["weights"])

    def test_reduces_to_equal_weight_when_assets_are_identical(self):
        S = np.eye(4) * 0.04
        assert equal_risk_contribution(S)["weights"] == pytest.approx([0.25] * 4, abs=1e-4)

    def test_underweights_the_volatile_asset(self):
        """With no correlation, ERC weights are inversely proportional to vol."""
        S = np.diag([0.01, 0.04])  # vols of 0.1 and 0.2
        w = equal_risk_contribution(S)["weights"]
        assert w[0] > w[1]
        assert w[0] / w[1] == pytest.approx(2.0, rel=1e-2)
