"""Alpaca paper-trading integration.

The client is hardcoded to Alpaca's paper endpoint (`paper=True`) -- no
environment variable can point this at a live account, so a misconfigured
deploy can place unwanted orders in a sandbox at worst, never with real money.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

# Below this, a diff is almost certainly float-weight rounding noise rather
# than a real rebalance decision, and Alpaca rejects notional orders under $1
# outright.
_MIN_ORDER_NOTIONAL = 1.0


class BrokerNotConfigured(RuntimeError):
    """Raised when ALPACA_API_KEY / ALPACA_SECRET_KEY aren't set."""


def _client() -> TradingClient:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise BrokerNotConfigured(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY are not configured on this server"
        )
    return TradingClient(api_key, secret_key, paper=True)


def get_account() -> dict:
    a = _client().get_account()
    return {
        "equity": float(a.equity),
        "cash": float(a.cash),
        "buying_power": float(a.buying_power),
        "portfolio_value": float(a.portfolio_value),
    }


def get_positions() -> dict[str, dict]:
    """Current paper positions, keyed by ticker."""
    return {
        p.symbol: {"qty": float(p.qty), "market_value": float(p.market_value)}
        for p in _client().get_all_positions()
    }


@dataclass
class OrderPreview:
    ticker: str
    side: str  # "buy" | "sell"
    notional: float
    current_value: float
    target_value: float


def build_rebalance_orders(
    target_weights: dict[str, float],
    account_equity: float,
    current_positions: dict[str, dict],
) -> list[OrderPreview]:
    """Diff target dollar allocation against current positions.

    One order per ticker whose diff clears `_MIN_ORDER_NOTIONAL`, sized as a
    notional (dollar) market order so any weight is reachable without a share
    count -- fractional shares handle the rest. A ticker held but no longer in
    the target is sold to zero, since that is what "rebalance" means.
    """
    tickers = set(target_weights) | set(current_positions)
    orders: list[OrderPreview] = []
    for ticker in sorted(tickers):
        current_value = current_positions.get(ticker, {}).get("market_value", 0.0)
        target_value = target_weights.get(ticker, 0.0) * account_equity
        diff = target_value - current_value
        if abs(diff) < _MIN_ORDER_NOTIONAL:
            continue
        orders.append(
            OrderPreview(
                ticker=ticker,
                side="buy" if diff > 0 else "sell",
                notional=round(abs(diff), 2),
                current_value=round(current_value, 2),
                target_value=round(target_value, 2),
            )
        )
    return orders


@dataclass
class OrderResult:
    ticker: str
    side: str
    notional: float
    status: str
    order_id: str | None
    error: str | None = None


def submit_orders(orders: list[OrderPreview]) -> list[OrderResult]:
    """Submit each order independently -- one rejection must not block the rest."""
    client = _client()
    results: list[OrderResult] = []
    for o in orders:
        try:
            resp = client.submit_order(
                MarketOrderRequest(
                    symbol=o.ticker,
                    notional=o.notional,
                    side=OrderSide.BUY if o.side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
            results.append(
                OrderResult(
                    ticker=o.ticker,
                    side=o.side,
                    notional=o.notional,
                    status=_enum_value(resp.status),
                    order_id=str(resp.id),
                )
            )
        except Exception as exc:
            results.append(
                OrderResult(
                    ticker=o.ticker,
                    side=o.side,
                    notional=o.notional,
                    status="failed",
                    order_id=None,
                    error=str(exc)[:300],
                )
            )
    return results


def get_recent_orders(limit: int = 20) -> list[dict]:
    orders = _client().get_orders(GetOrdersRequest(limit=limit))
    return [
        {
            "id": str(o.id),
            "symbol": o.symbol,
            "side": _enum_value(o.side),
            "notional": float(o.notional) if o.notional is not None else None,
            "qty": float(o.qty) if o.qty is not None else None,
            "status": _enum_value(o.status),
            "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None,
            "filled_at": o.filled_at.isoformat() if o.filled_at else None,
        }
        for o in orders
    ]


def _enum_value(v) -> str:
    return v.value if hasattr(v, "value") else str(v)


def order_preview_dict(o: OrderPreview) -> dict:
    return asdict(o)


def order_result_dict(o: OrderResult) -> dict:
    return asdict(o)
