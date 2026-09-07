"""Tests for the single-ticker snapshot the financial-advisor agent calls.

Network calls (`fetch_prices`, `fetch_returns`, `fetch_meta`, `yfinance`) are
patched out with deterministic synthetic data, matching how the rest of the
suite avoids depending on live market data.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app import ticker_metrics


def _price_series(n=300, seed=0, drift=0.0004, vol=0.01):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    prices = 100.0 * np.cumprod(1.0 + rets)
    index = pd.bdate_range("2023-01-01", periods=n)
    return pd.Series(prices, index=index), pd.Series(rets, index=index)


@pytest.fixture
def mocked_market():
    prices_a, rets_a = _price_series(seed=1)
    prices_spy, rets_spy = _price_series(seed=2)

    def fake_fetch_prices(tickers, period="3y"):
        series = {"AAPL": prices_a, "SPY": prices_spy}
        return pd.DataFrame({t: series[t] for t in tickers})

    def fake_fetch_returns(tickers, period="3y"):
        series = {"AAPL": rets_a, "SPY": rets_spy}
        return pd.DataFrame({t: series[t] for t in tickers})

    def fake_fetch_meta(ticker):
        return {"ticker": ticker, "sector": "Technology", "name": "Apple Inc.", "dividend_yield": 0.5}

    def fake_factor_analysis(port_ret, bench_ret, period, periods_per_year):
        return {"available": False, "reason": "patched out for this test"}

    with (
        patch.object(ticker_metrics, "fetch_prices", side_effect=fake_fetch_prices),
        patch.object(ticker_metrics, "fetch_returns", side_effect=fake_fetch_returns),
        patch.object(ticker_metrics, "fetch_meta", side_effect=fake_fetch_meta),
        patch.object(ticker_metrics, "run_factor_analysis", side_effect=fake_factor_analysis),
        patch.object(ticker_metrics.yf, "Ticker") as mock_ticker,
    ):
        mock_ticker.return_value.info = {"trailingPE": 28.5}
        yield


class TestBuildTickerMetrics:
    def test_returns_expected_shape(self, mocked_market):
        result = ticker_metrics.build_ticker_metrics("aapl", period="1y")

        assert result["ticker"] == "AAPL"
        assert result["name"] == "Apple Inc."
        assert result["sector"] == "Technology"
        assert result["trailing_pe"] == 28.5
        assert isinstance(result["annualized_return"], float)
        assert isinstance(result["annualized_volatility"], float)
        assert result["annualized_volatility"] > 0
        assert result["max_drawdown"] <= 0
        assert result["regime"]["available"] is True
        assert result["regime"]["label"] in {"Calm", "Elevated", "Stressed"}
        assert 0.0 <= result["regime"]["confidence"] <= 1.0
        assert result["factors"] == {"available": False, "reason": "patched out for this test"}

    def test_uppercases_ticker(self, mocked_market):
        result = ticker_metrics.build_ticker_metrics("aapl")
        assert result["ticker"] == "AAPL"

    def test_missing_pe_degrades_to_none(self, mocked_market):
        with patch.object(ticker_metrics.yf, "Ticker") as mock_ticker:
            mock_ticker.return_value.info = {}
            result = ticker_metrics.build_ticker_metrics("AAPL")
        assert result["trailing_pe"] is None
