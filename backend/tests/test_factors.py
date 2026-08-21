"""Tests for factor attribution.

Portfolios are synthesized with known betas and a known alpha so the regression
can be checked against ground truth. No network access — `fetch_factor_returns`
is the only part that touches yfinance and it is exercised separately.
"""

import numpy as np
import pandas as pd
import pytest

from app.factors import factor_regression, run_factor_analysis


@pytest.fixture
def factor_panel():
    rng = np.random.default_rng(0)
    n = 1200
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "size": rng.normal(0, 0.004, n),
            "value": rng.normal(0, 0.004, n),
            "momentum": rng.normal(0, 0.005, n),
        },
        index=idx,
    )


@pytest.fixture
def market(factor_panel):
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(0.0003, 0.01, len(factor_panel)), index=factor_panel.index)


def _build(market, factors, alpha_daily, betas, noise=0.002, seed=2):
    rng = np.random.default_rng(seed)
    r = alpha_daily + betas["market"] * market
    for name, b in betas.items():
        if name != "market":
            r = r + b * factors[name]
    return r + rng.normal(0, noise, len(market))


class TestFactorRegression:
    def test_recovers_known_betas(self, market, factor_panel):
        betas = {"market": 1.1, "size": 0.6, "value": -0.4, "momentum": 0.25}
        port = _build(market, factor_panel, 0.0002, betas)

        out = factor_regression(port, market, factor_panel)
        got = {l["factor"]: l["beta"] for l in out["loadings"]}
        for name, expected in betas.items():
            assert got[name] == pytest.approx(expected, abs=0.06)

    def test_recovers_known_alpha(self, market, factor_panel):
        alpha_daily = 0.0004
        port = _build(market, factor_panel, alpha_daily, {"market": 1.0, "size": 0.3})
        out = factor_regression(port, market, factor_panel)
        assert out["alpha_annualized"] == pytest.approx(alpha_daily * 252, abs=0.03)
        assert out["alpha_significant"]

    def test_finds_no_alpha_when_there_is_none(self, market, factor_panel):
        port = _build(market, factor_panel, 0.0, {"market": 1.0, "size": 0.5, "value": 0.3})
        out = factor_regression(port, market, factor_panel)
        assert not out["alpha_significant"]
        assert abs(out["alpha_t_statistic"]) < 2.0

    def test_pure_market_portfolio_has_unit_beta_and_no_tilts(self, market, factor_panel):
        port = market * 1.0 + np.random.default_rng(3).normal(0, 0.0005, len(market))
        out = factor_regression(port, market, factor_panel)
        got = {l["factor"]: l for l in out["loadings"]}
        assert got["market"]["beta"] == pytest.approx(1.0, abs=0.02)
        assert not got["size"]["significant"]
        assert not got["value"]["significant"]

    def test_style_tilt_is_flagged_significant(self, market, factor_panel):
        port = _build(market, factor_panel, 0.0, {"market": 1.0, "value": 1.5}, noise=0.001)
        got = {l["factor"]: l for l in factor_regression(port, market, factor_panel)["loadings"]}
        assert got["value"]["significant"]
        assert got["value"]["beta"] == pytest.approx(1.5, abs=0.06)

    def test_r_squared_is_high_for_a_constructed_portfolio(self, market, factor_panel):
        port = _build(market, factor_panel, 0.0, {"market": 1.0, "size": 0.5}, noise=0.0005)
        out = factor_regression(port, market, factor_panel)
        assert out["r_squared"] > 0.95
        assert out["unexplained_share"] < 0.05

    def test_reports_every_factor_plus_market(self, market, factor_panel):
        port = _build(market, factor_panel, 0.0, {"market": 1.0})
        out = factor_regression(port, market, factor_panel)
        assert [l["factor"] for l in out["loadings"]] == [
            "market",
            "size",
            "value",
            "momentum",
        ]

    def test_uses_newey_west_lags(self, market, factor_panel):
        port = _build(market, factor_panel, 0.0, {"market": 1.0})
        assert factor_regression(port, market, factor_panel)["newey_west_lags"] > 0

    def test_rejects_short_overlap(self, market, factor_panel):
        with pytest.raises(ValueError):
            factor_regression(market.iloc[:30], market.iloc[:30], factor_panel.iloc[:30])

    def test_only_overlapping_dates_are_used(self, market, factor_panel):
        """Misaligned calendars must inner-join, not silently misalign."""
        port = _build(market, factor_panel, 0.0, {"market": 1.0})
        out = factor_regression(port.iloc[100:], market, factor_panel)
        assert out["observations"] == len(market) - 100


class TestGracefulDegradation:
    def test_reports_unavailable_instead_of_raising(self, monkeypatch, market):
        import app.factors as mod

        monkeypatch.setattr(
            mod, "fetch_factor_returns", lambda period: (_ for _ in ()).throw(ValueError("boom"))
        )
        out = run_factor_analysis(market, market)
        assert not out["available"]
        assert "boom" in out["reason"]
        assert out["loadings"] == []
