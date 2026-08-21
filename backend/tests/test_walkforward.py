"""Tests for the walk-forward engine.

The no-lookahead tests are the important ones. Every other property of a
backtest is negotiable; using information that did not exist yet invalidates the
whole exercise, and the bug is invisible in the output because it simply makes
the results look better.
"""

import numpy as np
import pandas as pd
import pytest

from app.walkforward import STRATEGY_LABELS, walk_forward


def _panel(n=900, p=4, seed=0):
    rng = np.random.default_rng(seed)
    corr = np.full((p, p), 0.35)
    np.fill_diagonal(corr, 1.0)
    z = rng.normal(0.0004, 0.011, (n, p)) @ np.linalg.cholesky(corr).T
    return pd.DataFrame(
        z, columns=[f"A{i}" for i in range(p)], index=pd.bdate_range("2020-01-01", periods=n)
    )


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def equal_weights(panel):
    return {c: 1.0 / panel.shape[1] for c in panel.columns}


@pytest.fixture(scope="module")
def result(panel, equal_weights):
    return walk_forward(panel, equal_weights, train_window=400, rebalance_every=63)


class TestNoLookahead:
    def test_future_data_cannot_change_the_first_out_of_sample_block(
        self, panel, equal_weights
    ):
        """Rewriting the tail must leave the earliest held period untouched."""
        base = walk_forward(panel, equal_weights, train_window=400, rebalance_every=63)

        tampered = panel.copy()
        tampered.iloc[600:] *= 5.0  # violently different future
        after = walk_forward(tampered, equal_weights, train_window=400, rebalance_every=63)

        for strat in base["strategies"]:
            match = next(s for s in after["strategies"] if s["key"] == strat["key"])
            # The first rebalance block spans indices 400..463, all before 600.
            assert strat["growth"][:63] == pytest.approx(match["growth"][:63], rel=1e-12)

    def test_training_data_changes_do_propagate(self, panel, equal_weights):
        """The complement: altering the past must change the result."""
        base = walk_forward(panel, equal_weights, train_window=400, rebalance_every=63)

        tampered = panel.copy()
        tampered.iloc[:400, 0] += 0.02  # make asset 0 look spectacular in-sample
        after = walk_forward(tampered, equal_weights, train_window=400, rebalance_every=63)

        base_ms = next(s for s in base["strategies"] if s["key"] == "max_sharpe")
        after_ms = next(s for s in after["strategies"] if s["key"] == "max_sharpe")
        assert base_ms["growth"][:63] != pytest.approx(after_ms["growth"][:63])


class TestStructure:
    def test_reports_every_strategy(self, result):
        keys = {s["key"] for s in result["strategies"]}
        assert keys == set(STRATEGY_LABELS)

    def test_all_strategies_share_one_date_range(self, result):
        lengths = {len(s["growth"]) for s in result["strategies"]}
        assert len(lengths) == 1
        assert lengths.pop() == len(result["dates"])

    def test_ranked_by_sharpe(self, result):
        sharpes = [s["sharpe_ratio"] for s in result["strategies"]]
        assert sharpes == sorted(sharpes, reverse=True)
        assert result["best_strategy"] == result["strategies"][0]["label"]

    def test_out_of_sample_window_is_what_was_asked_for(self, panel, result):
        assert result["oos_days"] == panel.shape[0] - 400
        assert result["train_window"] == 400

    def test_growth_curves_start_near_one(self, result):
        for s in result["strategies"]:
            assert s["growth"][0] == pytest.approx(1.0, abs=0.1)


class TestEconomics:
    def test_equal_weight_barely_trades(self, result):
        """1/N only rebalances back to a fixed target, so turnover stays low."""
        ew = next(s for s in result["strategies"] if s["key"] == "equal_weight")
        ms = next(s for s in result["strategies"] if s["key"] == "max_sharpe")
        assert ew["annual_turnover"] < ms["annual_turnover"]

    def test_buy_and_hold_never_rebalances(self, result):
        bh = next(s for s in result["strategies"] if s["key"] == "buy_and_hold")
        assert bh["n_rebalances"] == 1

    def test_turnover_is_non_negative(self, result):
        assert all(s["annual_turnover"] >= 0 for s in result["strategies"])

    def test_costs_reduce_returns(self, panel, equal_weights):
        free = walk_forward(panel, equal_weights, train_window=400, cost_bps=0.0)
        expensive = walk_forward(panel, equal_weights, train_window=400, cost_bps=200.0)

        f = next(s for s in free["strategies"] if s["key"] == "max_sharpe")
        e = next(s for s in expensive["strategies"] if s["key"] == "max_sharpe")
        assert e["total_growth"] < f["total_growth"]

    def test_drawdowns_are_negative_or_zero(self, result):
        assert all(s["max_drawdown"] <= 0 for s in result["strategies"])

    def test_in_sample_sharpe_beats_out_of_sample(self, result):
        """The headline finding: fitting and scoring on one sample flatters."""
        assert result["in_sample_sharpe"] > result["out_of_sample_sharpe"]
        assert result["sharpe_decay"] > 0


class TestDegradation:
    def test_reports_unavailable_when_history_is_too_short(self):
        short = _panel(n=300)
        out = walk_forward(short, {c: 0.25 for c in short.columns}, train_window=250)
        assert not out["available"]
        assert "out-of-sample" in out["reason"]
        assert out["strategies"] == []

    def test_single_asset_is_rejected(self):
        one = _panel(n=900, p=1)
        out = walk_forward(one, {"A0": 1.0}, train_window=400)
        assert not out["available"]

    def test_survives_an_asset_that_never_moves(self, equal_weights):
        """A dead ticker must not take the whole run down."""
        panel = _panel(n=900)
        panel["A0"] = 0.0
        out = walk_forward(panel, {c: 0.25 for c in panel.columns}, train_window=400)
        assert out["available"]
        assert all(np.isfinite(s["sharpe_ratio"]) for s in out["strategies"])
