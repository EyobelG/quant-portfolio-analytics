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
