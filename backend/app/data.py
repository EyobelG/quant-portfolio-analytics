import pandas as pd
import yfinance as yf
from cachetools import TTLCache, cached
from cachetools.keys import hashkey

# Cache price downloads for 1 hour so repeated requests for the same
# tickers/period don't hammer the yfinance/Yahoo endpoint.
_price_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)


def _cache_key(tickers: tuple[str, ...], period: str) -> tuple:
    return hashkey(tuple(sorted(tickers)), period)


@cached(cache=_price_cache, key=lambda tickers, period: _cache_key(tickers, period))
def fetch_prices(tickers: tuple[str, ...], period: str = "3y") -> pd.DataFrame:
    """Download adjusted close prices for a set of tickers.

    Returns a DataFrame indexed by date, one column per ticker, forward-filled
    for occasional missing trading days and dropped where a ticker has no
    data at all (e.g. a bad symbol).
    """
    data = yf.download(
        list(tickers),
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if data.empty:
        raise ValueError(f"No price data found for: {', '.join(tickers)}")

    # yfinance returns a (ticker, field) MultiIndex for multi-ticker requests and,
    # depending on version, for single-ticker ones too — normalize both to a
    # plain ticker-per-column frame of closes.
    if isinstance(data.columns, pd.MultiIndex):
        available = data.columns.get_level_values(0).unique()
        prices = pd.DataFrame(
            {t: data[t]["Close"] for t in tickers if t in available and "Close" in data[t].columns}
        )
    else:
        prices = data["Close"].to_frame(name=tickers[0])

    prices = prices.ffill().dropna(how="all")
    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        raise ValueError(f"No price data found for: {', '.join(missing)}")

    return prices.dropna()


def fetch_returns(tickers: tuple[str, ...], period: str = "3y") -> pd.DataFrame:
    prices = fetch_prices(tickers, period)
    return prices.pct_change().dropna()


# Ticker metadata changes rarely, so cache it far longer than prices.
_meta_cache: TTLCache = TTLCache(maxsize=512, ttl=86400)


def _normalize_yield(raw) -> float | None:
    """Dividend yield as a percentage.

    yfinance (pinned at 1.6.0) already reports this as a percentage — AAPL comes
    back as 0.35 meaning 0.35%, not 35%. Older releases returned a fraction, so
    do not reintroduce a fraction-to-percent conversion without re-checking the
    installed version against a known yield. Values above 100% are treated as
    bad data rather than displayed.
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0 or value > 100:
        return None
    return round(value, 2)


@cached(cache=_meta_cache, key=lambda ticker: hashkey(ticker))
def fetch_meta(ticker: str) -> dict:
    """Sector, long name, and dividend yield for one ticker.

    Best-effort: this hits a slower, flakier endpoint than the price download,
    so every failure degrades to empty fields rather than raising. Funds report
    no sector, so their category is used instead.
    """
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {"ticker": ticker, "sector": None, "name": None, "dividend_yield": None}

    sector = info.get("sector") or info.get("category")
    if not sector and info.get("quoteType") in {"ETF", "MUTUALFUND", "INDEX"}:
        sector = "Fund / Index"

    return {
        "ticker": ticker,
        "sector": sector,
        "name": info.get("longName") or info.get("shortName"),
        "dividend_yield": _normalize_yield(info.get("dividendYield")),
    }
