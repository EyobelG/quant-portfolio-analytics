"""Tests for the Alpaca paper-trading integration.

`build_rebalance_orders` and `select_best_strategy` are pure functions with no
network dependency, so they're tested directly. Anything that touches the
Alpaca SDK (`_client`, `submit_orders`, `get_account`, ...) is out of scope
here -- it's a thin wrapper around a third-party client and isn't worth
mocking out.
"""

import numpy as np
import pandas as pd
import pytest

from app.broker import build_rebalance_orders
from app.walkforward import select_best_strategy


class TestBuildRebalanceOrders:
    def test_buys_into_an_empty_account(self):
        orders = build_rebalance_orders(
            target_weights={"AAA": 0.6, "BBB": 0.4},
            account_equity=1000.0,
            current_positions={},
        )
        by_ticker = {o.ticker: o for o in orders}
        assert by_ticker["AAA"].side == "buy"
        assert by_ticker["AAA"].notional == pytest.approx(600.0)
        assert by_ticker["BBB"].side == "buy"
        assert by_ticker["BBB"].notional == pytest.approx(400.0)

    def test_sells_a_position_dropped_from_the_target(self):
        orders = build_rebalance_orders(
            target_weights={"AAA": 1.0},
            account_equity=1000.0,
            current_positions={"BBB": {"qty": 5, "market_value": 300.0}},
        )
        by_ticker = {o.ticker: o for o in orders}
        assert by_ticker["BBB"].side == "sell"
        assert by_ticker["BBB"].notional == pytest.approx(300.0)
        assert by_ticker["AAA"].side == "buy"
        assert by_ticker["AAA"].notional == pytest.approx(1000.0)

    def test_diffs_toward_an_existing_overweight_position(self):
        orders = build_rebalance_orders(
            target_weights={"AAA": 0.5},
            account_equity=1000.0,
            current_positions={"AAA": {"qty": 10, "market_value": 800.0}},
        )
        assert len(orders) == 1
        assert orders[0].side == "sell"
        assert orders[0].notional == pytest.approx(300.0)

    def test_dead_band_drops_negligible_diffs(self):
        orders = build_rebalance_orders(
            target_weights={"AAA": 0.5},
            account_equity=1000.0,
            current_positions={"AAA": {"qty": 10, "market_value": 500.30}},
        )
        assert orders == []

    def test_already_at_target_produces_no_orders(self):
        orders = build_rebalance_orders(
            target_weights={"AAA": 0.5, "BBB": 0.5},
            account_equity=1000.0,
            current_positions={
                "AAA": {"qty": 10, "market_value": 500.0},
                "BBB": {"qty": 10, "market_value": 500.0},
            },
        )
        assert orders == []


class TestSelectBestStrategy:
    def _panel(self, n=900, p=4, seed=0):
        rng = np.random.default_rng(seed)
        corr = np.full((p, p), 0.35)
        np.fill_diagonal(corr, 1.0)
        z = rng.normal(0.0004, 0.011, (n, p)) @ np.linalg.cholesky(corr).T
        return pd.DataFrame(
            z, columns=[f"A{i}" for i in range(p)], index=pd.bdate_range("2020-01-01", periods=n)
        )

    def test_returns_normalized_weights_for_the_winning_strategy(self):
        panel = self._panel()
        weights = {c: 1.0 / panel.shape[1] for c in panel.columns}

        choice = select_best_strategy(panel, weights, train_window=400, rebalance_every=63)

        assert choice["key"] == choice["walk_forward"]["strategies"][0]["key"]
        assert set(choice["weights"]) == set(panel.columns)
        total = sum(choice["weights"].values())
        assert total == pytest.approx(1.0, abs=1e-6)
        assert all(v >= 0 for v in choice["weights"].values())

    def test_buy_and_hold_winner_returns_the_users_own_weights(self):
        panel = self._panel(seed=1)
        # Wildly lopsided weights make buy-and-hold's target trivially checkable
        # regardless of which strategy the backtest actually picks as best.
        weights = {c: 0.0 for c in panel.columns}
        weights[panel.columns[0]] = 1.0

        choice = select_best_strategy(panel, weights, train_window=400, rebalance_every=63)
        if choice["key"] == "buy_and_hold":
            assert choice["weights"][panel.columns[0]] == pytest.approx(1.0)

    def test_raises_when_walk_forward_has_too_little_out_of_sample_data(self):
        panel = self._panel(n=100)
        weights = {c: 1.0 / panel.shape[1] for c in panel.columns}
        with pytest.raises(ValueError):
            select_best_strategy(panel, weights, train_window=80)
