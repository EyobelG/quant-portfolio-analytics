"""Tests for regime-conditional correlation and beta.

The correlation tests build panels whose regime structure is known by
construction — a common factor switched on only in the bottom decile — so a
detected increase is a real detection rather than a recorded output.
"""

import numpy as np
import pandas as pd
import pytest

from app.regimes import asymmetric_betas, regime_analysis


def _panel(n=1500, p=4, seed=0, stressed_rho=None, base_rho=0.2, tail=0.10):
    """Assets whose correlation optionally jumps in the benchmark's worst tail.

    When `stressed_rho` is None the correlation structure is identical in every
    regime, which is the null the detection tests are measured against.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    bench = pd.Series(rng.normal(0.0004, 0.011, n), index=idx)

    cutoff = bench.rank(method="first", pct=True) <= tail
    common = rng.normal(0, 0.011, n)
    idio = rng.normal(0, 0.011, (n, p))

    cols = {}
    for j in range(p):
        # Loading on the shared factor sets the pairwise correlation to
        # rho = load^2 / (load^2 + 1) when idiosyncratic scale matches.
        base_load = np.sqrt(base_rho / (1 - base_rho))
        load = np.full(n, base_load)
        if stressed_rho is not None:
            load[cutoff.to_numpy()] = np.sqrt(stressed_rho / (1 - stressed_rho))
        cols[f"A{j}"] = load * common + idio[:, j]

    return pd.DataFrame(cols, index=idx), bench


class TestRegimeSplit:
    def test_regimes_partition_the_sample(self):
        rets, bench = _panel(n=1000)
        out = regime_analysis(rets, bench, tail_quantile=0.10)
        counts = {k: v["observations"] for k, v in out["regimes"].items()}
        assert sum(counts.values()) == out["observations"] == 1000
        assert counts["stressed"] == 100
        assert counts["rally"] == 100
        assert counts["calm"] == 800

    def test_tail_quantile_controls_regime_size(self):
        rets, bench = _panel(n=1000)
        out = regime_analysis(rets, bench, tail_quantile=0.25)
        assert out["regimes"]["stressed"]["observations"] == 250
        assert out["regimes"]["calm"]["observations"] == 500

    def test_stressed_regime_holds_the_worst_days(self):
        rets, bench = _panel(n=1000)
        out = regime_analysis(rets, bench)
        r = out["regimes"]
        assert r["stressed"]["benchmark_mean"] < r["calm"]["benchmark_mean"]
        assert r["calm"]["benchmark_mean"] < r["rally"]["benchmark_mean"]
        # The tails must not overlap the middle.
        assert r["stressed"]["benchmark_best"] <= r["calm"]["benchmark_worst"]
        assert r["rally"]["benchmark_worst"] >= r["calm"]["benchmark_best"]

    def test_rejects_invalid_quantile(self):
        rets, bench = _panel(n=500)
        for bad in (0.0, 0.5, 0.9, -0.1):
            with pytest.raises(ValueError):
                regime_analysis(rets, bench, tail_quantile=bad)

    def test_rejects_too_little_history(self):
        rets, bench = _panel(n=60)
        with pytest.raises(ValueError):
            regime_analysis(rets, bench)

    def test_rejects_single_holding(self):
        rets, bench = _panel(n=500, p=1)
        with pytest.raises(ValueError):
            regime_analysis(rets, bench)


class TestCorrelationStructure:
    def test_detects_correlation_rising_in_the_tail(self):
        """The headline claim: a tail-only common factor must be picked up."""
        rets, bench = _panel(n=2500, stressed_rho=0.75, base_rho=0.15, seed=3)
        out = regime_analysis(rets, bench)
        calm = out["regimes"]["calm"]["average_correlation"]
        stressed = out["regimes"]["stressed"]["average_correlation"]

        assert stressed > calm + 0.3
        assert out["correlation_increase"] == pytest.approx(stressed - calm)

    def test_reports_no_increase_when_structure_is_stable(self):
        """The complement: constant correlation must not manufacture a jump."""
        rets, bench = _panel(n=3000, stressed_rho=None, base_rho=0.35, seed=5)
        out = regime_analysis(rets, bench)
        assert abs(out["correlation_increase"]) < 0.15

    def test_recovers_the_constructed_correlation_level(self):
        rets, bench = _panel(n=4000, stressed_rho=None, base_rho=0.40, seed=7)
        out = regime_analysis(rets, bench)
        assert out["regimes"]["calm"]["average_correlation"] == pytest.approx(0.40, abs=0.05)

    def test_diagonals_are_unity(self):
        rets, bench = _panel(n=1200)
        out = regime_analysis(rets, bench)
        for regime in out["regimes"].values():
            m = regime["correlation_matrix"]
            for t in out["tickers"]:
                assert m[t][t] == pytest.approx(1.0)

    def test_matrices_are_symmetric(self):
        rets, bench = _panel(n=1200)
        out = regime_analysis(rets, bench)
        m = out["regimes"]["stressed"]["correlation_matrix"]
        tickers = out["tickers"]
        for a in tickers:
            for b in tickers:
                assert m[a][b] == pytest.approx(m[b][a])

    def test_delta_matches_the_displayed_matrices_exactly(self):
        """The delta must reconcile with the two panels shown beside it."""
        rets, bench = _panel(n=2000, stressed_rho=0.7, seed=11)
        out = regime_analysis(rets, bench)
        stressed = out["regimes"]["stressed"]["correlation_matrix"]
        calm = out["regimes"]["calm"]["correlation_matrix"]
        for a in out["tickers"]:
            for b in out["tickers"]:
                assert out["delta_matrix"][a][b] == pytest.approx(
                    round(stressed[a][b] - calm[a][b], 4), abs=1e-9
                )

    def test_delta_diagonal_is_zero(self):
        rets, bench = _panel(n=1200)
        out = regime_analysis(rets, bench)
        for t in out["tickers"]:
            assert out["delta_matrix"][t][t] == pytest.approx(0.0)

    def test_average_correlation_excludes_the_diagonal(self):
        """Including the unit diagonal would bias the headline toward 1."""
        rets, bench = _panel(n=3000, stressed_rho=None, base_rho=0.20, seed=13)
        out = regime_analysis(rets, bench)
        assert out["regimes"]["calm"]["average_correlation"] < 0.5


class TestAsymmetricBeta:
    def test_symmetric_data_has_matching_up_and_down_beta(self):
        rng = np.random.default_rng(17)
        n = 4000
        bench = pd.Series(rng.normal(0, 0.011, n))
        rets = pd.DataFrame({"A": 1.2 * bench + rng.normal(0, 0.004, n)})

        b = asymmetric_betas(rets, bench)[0]
        assert b["downside_beta"] == pytest.approx(b["upside_beta"], abs=0.12)
        assert b["ratio"] == pytest.approx(1.0, abs=0.12)
        assert not b["asymmetric"]

    def test_recovers_a_known_symmetric_beta(self):
        rng = np.random.default_rng(19)
        n = 4000
        bench = pd.Series(rng.normal(0, 0.011, n))
        rets = pd.DataFrame({"A": 0.8 * bench + rng.normal(0, 0.003, n)})

        b = asymmetric_betas(rets, bench)[0]
        assert b["beta"] == pytest.approx(0.8, abs=0.03)
        assert b["downside_beta"] == pytest.approx(0.8, abs=0.10)

    def test_detects_a_constructed_downside_asymmetry(self):
        """An asset levered to losses but not gains must be flagged."""
        rng = np.random.default_rng(23)
        n = 4000
        bench = pd.Series(rng.normal(0, 0.011, n))
        loading = np.where(bench < 0, 1.6, 0.5)
        rets = pd.DataFrame({"A": loading * bench + rng.normal(0, 0.002, n)})

        b = asymmetric_betas(rets, bench)[0]
        assert b["downside_beta"] == pytest.approx(1.6, abs=0.15)
        assert b["upside_beta"] == pytest.approx(0.5, abs=0.15)
        assert b["ratio"] > 2.0
        assert b["asymmetric"]

    def test_reports_every_holding(self):
        rets, bench = _panel(n=1000, p=5)
        betas = asymmetric_betas(rets, bench)
        assert [b["ticker"] for b in betas] == [f"A{i}" for i in range(5)]

    def test_ratio_is_undefined_when_betas_straddle_zero(self):
        """An asset that rises with falls and falls with rises has no ratio.

        Dividing a positive downside beta by a negative upside one yields a
        negative number that reads like a low ratio while describing the
        opposite situation, so it must not be reported at all.
        """
        rng = np.random.default_rng(37)
        n = 3000
        bench = pd.Series(rng.normal(0, 0.011, n))
        # Positive beta on down days, negative on up days.
        loading = np.where(bench < 0, 0.5, -0.7)
        rets = pd.DataFrame({"A": loading * bench + rng.normal(0, 0.002, n)})

        b = asymmetric_betas(rets, bench)[0]
        assert b["downside_beta"] > 0
        assert b["upside_beta"] < 0
        assert np.isnan(b["ratio"])
        assert not b["asymmetric"]


class TestCapture:
    def test_capture_ratios_are_reported_when_the_portfolio_is_supplied(self):
        rets, bench = _panel(n=1500)
        port = rets.mean(axis=1)
        out = regime_analysis(rets, bench, port_ret=port)
        assert "capture" in out
        assert out["capture"]["down_days"] + out["capture"]["up_days"] <= out["observations"]

    def test_omitted_when_no_portfolio_is_supplied(self):
        rets, bench = _panel(n=1500)
        assert "capture" not in regime_analysis(rets, bench)

    def test_tracking_the_benchmark_gives_unit_capture(self):
        """A portfolio identical to the benchmark captures exactly 100% of both."""
        rng = np.random.default_rng(29)
        n = 1500
        bench = pd.Series(rng.normal(0.0004, 0.011, n))
        rets = pd.DataFrame({"A": bench, "B": bench})
        out = regime_analysis(rets, bench, port_ret=bench)
        assert out["capture"]["downside"] == pytest.approx(1.0, abs=1e-9)
        assert out["capture"]["upside"] == pytest.approx(1.0, abs=1e-9)

    def test_a_damped_portfolio_captures_less_of_the_downside(self):
        rng = np.random.default_rng(31)
        n = 2000
        bench = pd.Series(rng.normal(0.0003, 0.011, n))
        damped = 0.5 * bench
        rets = pd.DataFrame({"A": damped, "B": damped})
        out = regime_analysis(rets, bench, port_ret=damped)
        assert out["capture"]["downside"] == pytest.approx(0.5, abs=1e-9)


class TestReliability:
    def test_flags_a_thin_stressed_regime_as_unreliable(self):
        # 120 days at a 10% tail leaves 12 observations for the stressed
        # correlation matrix, which is not enough to estimate one.
        rets, bench = _panel(n=120, p=4)
        out = regime_analysis(rets, bench)
        assert out["min_regime_observations"] == 12
        assert not out["reliable"]

    def test_accepts_a_long_sample(self):
        rets, bench = _panel(n=2000, p=4)
        out = regime_analysis(rets, bench)
        assert out["reliable"]

    def test_more_assets_demand_more_observations(self):
        """The bar scales with the number of parameters being estimated."""
        wide, bench = _panel(n=400, p=20)
        assert not regime_analysis(wide, bench)["reliable"]

        narrow, bench2 = _panel(n=400, p=3, seed=1)
        assert regime_analysis(narrow, bench2)["reliable"]
