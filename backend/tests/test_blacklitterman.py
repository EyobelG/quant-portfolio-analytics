"""Tests for the Black-Litterman model.

No network access: `fetch_shares_outstanding` is the only function that touches
yfinance and it is monkeypatched everywhere below.

The assertions here are exact where Black-Litterman's algebra makes them exact —
a zero-confidence view must leave the prior untouched, a full-confidence view
must bind completely — rather than approximate comparisons against recorded
output.
"""

import numpy as np
import pandas as pd
import pytest

from app import blacklitterman as bl
from app.blacklitterman import (
    build_view_matrices,
    historical_cap_weights,
    implied_risk_aversion,
    market_cap_weights,
    run_black_litterman,
)

TICKERS = ("AAA", "BBB", "CCC", "DDD")


@pytest.fixture
def panel():
    """Correlated price and return series with a stable structure."""
    rng = np.random.default_rng(11)
    n, p = 900, len(TICKERS)
    corr = np.full((p, p), 0.4)
    np.fill_diagonal(corr, 1.0)
    rets = rng.normal(0.0004, 0.011, (n, p)) @ np.linalg.cholesky(corr).T
    idx = pd.bdate_range("2021-01-04", periods=n)
    returns = pd.DataFrame(rets, columns=list(TICKERS), index=idx)
    prices = (1 + returns).cumprod() * 100.0
    return prices, returns


@pytest.fixture
def bench(panel):
    _, returns = panel
    rng = np.random.default_rng(5)
    # A benchmark that rises over the window, so implied risk aversion is usable.
    r = pd.Series(rng.normal(0.0005, 0.009, len(returns)), index=returns.index)
    return (1 + r).cumprod() * 4000.0


@pytest.fixture(autouse=True)
def stub_shares(monkeypatch):
    """Every ticker has a share count unless a test says otherwise."""
    counts = {"AAA": 4e9, "BBB": 2e9, "CCC": 1e9, "DDD": 5e8}
    monkeypatch.setattr(bl, "fetch_shares_outstanding", lambda t: counts.get(t))
    return counts


def _run(panel, bench, views):
    prices, returns = panel
    return run_black_litterman(prices, returns, bench, views, periods_per_year=252)


class TestViewMatrices:
    def test_absolute_view_is_a_single_one(self):
        P, Q, conf = build_view_matrices(
            [{"type": "absolute", "asset": "BBB", "value": 0.12, "confidence": 0.5}], TICKERS
        )
        assert P.tolist() == [[0.0, 1.0, 0.0, 0.0]]
        assert Q.tolist() == [[0.12]]
        assert conf == [0.5]

    def test_relative_view_row_sums_to_zero(self):
        """A spread view constrains a difference and says nothing about the level."""
        P, Q, _ = build_view_matrices(
            [
                {
                    "type": "relative",
                    "asset": "AAA",
                    "versus": "CCC",
                    "value": 0.04,
                    "confidence": 0.7,
                }
            ],
            TICKERS,
        )
        assert P.sum() == pytest.approx(0.0)
        assert P[0, 0] == 1.0 and P[0, 2] == -1.0
        assert Q[0, 0] == pytest.approx(0.04)

    def test_shapes_scale_with_view_count(self):
        views = [
            {"type": "absolute", "asset": "AAA", "value": 0.1, "confidence": 0.5},
            {"type": "relative", "asset": "BBB", "versus": "DDD", "value": 0.02, "confidence": 0.3},
        ]
        P, Q, conf = build_view_matrices(views, TICKERS)
        assert P.shape == (2, 4)
        assert Q.shape == (2, 1)
        assert len(conf) == 2

    def test_no_views_gives_empty_matrices(self):
        P, Q, conf = build_view_matrices([], TICKERS)
        assert P.shape == (0, 4)
        assert conf == []


class TestPrior:
    def test_no_views_leaves_posterior_identical_to_prior(self, panel, bench):
        out = _run(panel, bench, [])
        assert out["posterior_returns"] == pytest.approx(out["prior_returns"], rel=1e-12)

    def test_market_weights_follow_capitalization(self, panel, bench):
        """Shares x price, so AAA (4e9 shares) must outweigh DDD (5e8)."""
        out = _run(panel, bench, [])
        w = dict(zip(out["tickers"], out["market_weights"]))
        assert w["AAA"] > w["BBB"] > w["CCC"] > w["DDD"]
        assert sum(out["market_weights"]) == pytest.approx(1.0)

    def test_prior_returns_are_positive_for_a_rising_market(self, panel, bench):
        # Pi = delta * S * w with delta > 0 and a positive-definite S, so every
        # implied return should be positive.
        out = _run(panel, bench, [])
        assert all(r > 0 for r in out["prior_returns"])

    def test_degrades_to_equal_weight_when_shares_are_missing(
        self, panel, bench, monkeypatch
    ):
        monkeypatch.setattr(
            bl, "fetch_shares_outstanding", lambda t: None if t == "CCC" else 1e9
        )
        out = _run(panel, bench, [])
        assert not out["market_caps_available"]
        assert out["missing_caps"] == ["CCC"]
        assert out["market_weights"] == pytest.approx([0.25] * 4)

    def test_reports_when_caps_are_real(self, panel, bench):
        out = _run(panel, bench, [])
        assert out["market_caps_available"]
        assert out["missing_caps"] == []


class TestViewsMoveThePosterior:
    def test_full_confidence_absolute_view_binds_exactly(self, panel, bench):
        """At confidence 1 Idzorek's omega is zero, so the view is an equality."""
        out = _run(
            panel,
            bench,
            [{"type": "absolute", "asset": "AAA", "value": 0.25, "confidence": 1.0}],
        )
        posterior = dict(zip(out["tickers"], out["posterior_returns"]))
        assert posterior["AAA"] == pytest.approx(0.25, abs=1e-9)
        assert out["views"][0]["adoption"] == pytest.approx(1.0, abs=1e-6)

    def test_zero_confidence_view_changes_nothing(self, panel, bench):
        """Zero confidence is near-exact, not exact.

        Idzorek's method represents "no confidence" as omega = 1e6 rather than
        infinity, so an absurd view still leaks into the posterior at around
        1e-8 relative. That is a property of the specification, not a defect
        here, and the tolerance below is sized to it — tightening it to 1e-9
        fails on the leak rather than on any real regression.
        """
        base = _run(panel, bench, [])
        out = _run(
            panel,
            bench,
            [{"type": "absolute", "asset": "AAA", "value": 0.99, "confidence": 0.0}],
        )
        assert out["posterior_returns"] == pytest.approx(base["prior_returns"], rel=1e-6)
        assert out["views"][0]["adoption"] == pytest.approx(0.0, abs=1e-6)

    def test_confidence_monotonically_increases_adoption(self, panel, bench):
        adoptions = [
            _run(
                panel,
                bench,
                [{"type": "absolute", "asset": "BBB", "value": 0.30, "confidence": c}],
            )["views"][0]["adoption"]
            for c in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        assert adoptions == sorted(adoptions)
        assert adoptions[0] == pytest.approx(0.0, abs=1e-6)
        assert adoptions[-1] == pytest.approx(1.0, abs=1e-6)

    def test_relative_view_widens_the_spread_not_the_levels(self, panel, bench):
        """The intuitive expectation here is wrong, and it looks like a bug.

        A relative view "AAA beats CCC by 8%" does NOT push AAA up and CCC down.
        Both legs routinely move the same direction — the view constrains only
        their difference, and where that difference sits relative to the prior
        is determined by the covariance structure. What must hold is that the
        spread moves toward the view, so that is what is asserted.
        """
        base = _run(panel, bench, [])
        prior = dict(zip(base["tickers"], base["prior_returns"]))
        prior_spread = prior["AAA"] - prior["CCC"]

        out = _run(
            panel,
            bench,
            [
                {
                    "type": "relative",
                    "asset": "AAA",
                    "versus": "CCC",
                    "value": prior_spread + 0.08,
                    "confidence": 0.9,
                }
            ],
        )
        post = dict(zip(out["tickers"], out["posterior_returns"]))
        assert post["AAA"] - post["CCC"] > prior_spread

    def test_full_confidence_relative_view_binds_exactly(self, panel, bench):
        out = _run(
            panel,
            bench,
            [
                {
                    "type": "relative",
                    "asset": "AAA",
                    "versus": "DDD",
                    "value": 0.06,
                    "confidence": 1.0,
                }
            ],
        )
        post = dict(zip(out["tickers"], out["posterior_returns"]))
        assert post["AAA"] - post["DDD"] == pytest.approx(0.06, abs=1e-9)

    def test_diagnostics_describe_both_view_types_uniformly(self, panel, bench):
        out = _run(
            panel,
            bench,
            [
                {"type": "absolute", "asset": "AAA", "value": 0.2, "confidence": 0.6},
                {
                    "type": "relative",
                    "asset": "BBB",
                    "versus": "CCC",
                    "value": 0.03,
                    "confidence": 0.4,
                },
            ],
        )
        assert len(out["views"]) == 2
        for d in out["views"]:
            assert {"prior_implied", "posterior_implied", "adoption", "value"} <= set(d)
        assert out["views"][1]["versus"] == "CCC"


class TestOptimization:
    def test_posterior_weights_are_long_only_and_fully_invested(self, panel, bench):
        """bl_weights() would return negatives; EfficientFrontier must not."""
        out = _run(
            panel,
            bench,
            [{"type": "absolute", "asset": "AAA", "value": 0.30, "confidence": 0.8}],
        )
        w = out["posterior"]["weights"]
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(v >= -1e-9 for v in w.values())

    def test_a_strong_view_tilts_weights_toward_that_asset(self, panel, bench):
        base = _run(panel, bench, [])
        out = _run(
            panel,
            bench,
            [{"type": "absolute", "asset": "DDD", "value": 0.40, "confidence": 0.9}],
        )
        assert out["posterior"]["weights"]["DDD"] > base["posterior"]["weights"]["DDD"]

    def test_both_frontiers_are_traced(self, panel, bench):
        out = _run(panel, bench, [])
        for side in ("prior", "posterior"):
            pts = out[side]["frontier"]
            assert len(pts) > 5
            vols = [p["volatility"] for p in pts]
            # Frontier volatility is non-decreasing in target return.
            assert vols == sorted(vols)

    def test_no_views_makes_the_two_frontiers_coincide(self, panel, bench):
        out = _run(panel, bench, [])
        assert out["prior"]["point"]["return"] == pytest.approx(
            out["posterior"]["point"]["return"], rel=1e-9
        )


class TestRiskAversion:
    def test_uses_the_market_value_when_sane(self, panel, bench):
        delta, from_market = implied_risk_aversion(bench, 252)
        assert from_market
        assert 0.5 <= delta <= 10.0

    def test_falls_back_when_the_benchmark_lost_money(self, panel):
        """A falling benchmark implies negative delta, which would invert Pi."""
        idx = pd.bdate_range("2021-01-04", periods=600)
        falling = pd.Series((1 - 0.001) ** np.arange(600) * 4000.0, index=idx)
        delta, from_market = implied_risk_aversion(falling, 252)
        assert not from_market
        assert delta == pytest.approx(2.5)

    def test_flag_is_reported_in_the_payload(self, panel, bench):
        assert _run(panel, bench, [])["risk_aversion_from_market"] is True


class TestHistoricalCapWeights:
    def test_weights_sum_to_one_each_day(self, panel):
        prices, _ = panel
        w = historical_cap_weights(prices, {t: 1e9 for t in prices.columns})
        assert w.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(prices)))

    def test_weights_move_with_relative_prices(self, panel):
        """Equal share counts make cap weights price weights, so they must drift."""
        prices, _ = panel
        w = historical_cap_weights(prices, {t: 1e9 for t in prices.columns})
        assert not np.allclose(w.iloc[0].to_numpy(), w.iloc[-1].to_numpy())

    def test_larger_share_count_earns_a_larger_weight(self, panel):
        prices, _ = panel
        shares = {"AAA": 1e10, "BBB": 1e9, "CCC": 1e9, "DDD": 1e9}
        w = historical_cap_weights(prices, shares)
        assert w["AAA"].iloc[-1] > w["BBB"].iloc[-1]

    def test_returns_none_when_a_share_count_is_missing(self, panel):
        prices, _ = panel
        shares = {"AAA": 1e9, "BBB": 1e9, "CCC": 1e9}  # DDD absent
        assert historical_cap_weights(prices, shares) is None


class TestMarketCapWeights:
    def test_uses_latest_prices(self, panel, stub_shares):
        prices, _ = panel
        out = market_cap_weights(TICKERS, prices)
        expected = np.array([stub_shares[t] * prices[t].iloc[-1] for t in TICKERS])
        assert out["weights"].to_numpy() == pytest.approx(expected / expected.sum())

    def test_preserves_ticker_order(self, panel):
        prices, _ = panel
        assert list(market_cap_weights(TICKERS, prices)["weights"].index) == list(TICKERS)
