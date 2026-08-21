from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Yahoo accepts more strings than these, but every extra option multiplies the
# cache keys and the walk-forward cost, and anything under a year is too short
# for the statistics to say anything.
Period = Literal["1y", "2y", "3y", "5y", "10y", "max"]

# Loose enough for indices (^GSPC), crypto pairs (BTC-USD), and foreign listings
# (VOD.L), strict enough to keep junk out of an upstream URL.
_TICKER_PATTERN = r"^[\^]?[A-Za-z0-9][A-Za-z0-9._\-]{0,14}$"


class Holding(BaseModel):
    ticker: str = Field(min_length=1, max_length=15, pattern=_TICKER_PATTERN)
    weight: float = Field(ge=0, le=1)


class PortfolioRequest(BaseModel):
    # Capped because each holding costs an upstream download and the covariance
    # work grows quadratically; an uncapped list is a free denial-of-service.
    holdings: list[Holding] = Field(min_length=1, max_length=25)
    benchmark: str = Field(default="^GSPC", pattern=_TICKER_PATTERN)
    period: Period = "3y"


class View(BaseModel):
    """One Black-Litterman view.

    An absolute view names a return for a single asset ("AAPL returns 12%"). A
    relative view names a spread between two ("NVDA beats MSFT by 4%") and so
    requires `versus`; the cross-field check lives on the request model, where
    both the view and the portfolio's tickers are visible.
    """

    type: Literal["absolute", "relative"]
    asset: str = Field(min_length=1, max_length=15, pattern=_TICKER_PATTERN)
    versus: str | None = Field(default=None, max_length=15, pattern=_TICKER_PATTERN)
    # Annualized expected return, or expected spread for a relative view.
    # Bounded well outside anything defensible, purely to keep the optimizer
    # away from values that make the frontier degenerate.
    value: float = Field(ge=-2.0, le=2.0)
    confidence: float = Field(ge=0.0, le=1.0)


class BlackLittermanRequest(PortfolioRequest):
    # Each view adds a row to P and Q. More than a handful stops being a set of
    # opinions and starts being an unconstrained curve fit.
    views: list[View] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _check_views(self) -> "BlackLittermanRequest":
        held = {h.ticker.strip().upper() for h in self.holdings}

        for view in self.views:
            view.asset = view.asset.strip().upper()
            if view.asset not in held:
                raise ValueError(f"View references {view.asset}, which is not in the portfolio")

            if view.type == "relative":
                if not view.versus:
                    raise ValueError("A relative view needs a `versus` asset to compare against")
                view.versus = view.versus.strip().upper()
                if view.versus not in held:
                    raise ValueError(
                        f"View references {view.versus}, which is not in the portfolio"
                    )
                if view.versus == view.asset:
                    raise ValueError("A relative view cannot compare an asset with itself")
            else:
                # Silently ignoring a stray `versus` would let a mis-typed
                # absolute view look like it had been applied as a spread.
                view.versus = None

        return self


class MetricsResponse(BaseModel):
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float


class EfficientFrontierPoint(BaseModel):
    volatility: float
    return_: float = Field(alias="return")
    sharpe: float

    class Config:
        populate_by_name = True


class OptimizeResponse(BaseModel):
    frontier: list[EfficientFrontierPoint]
    # False when no allocation beat the risk-free rate over the window, so the
    # minimum-volatility portfolio stands in for the max-Sharpe one.
    max_sharpe_available: bool = True
    max_sharpe_weights: dict[str, float]
    max_sharpe_point: EfficientFrontierPoint
    min_vol_weights: dict[str, float]
    min_vol_point: EfficientFrontierPoint
    current_point: EfficientFrontierPoint
    correlation_matrix: dict[str, dict[str, float]]


class BacktestSeries(BaseModel):
    dates: list[str]
    portfolio: list[float]
    optimized: list[float]
    benchmark: list[float]


class DrawdownSeries(BaseModel):
    dates: list[str]
    drawdown: list[float]
    max_drawdown: float
    max_drawdown_date: str
    recovery_days: int | None = None
    longest_underwater_days: int


class ReturnDistribution(BaseModel):
    bin_edges: list[float]
    counts: list[int]
    var_95: float
    cvar_95: float
    mean: float


class HoldingMeta(BaseModel):
    ticker: str
    weight: float
    sector: str | None = None
    name: str | None = None
    dividend_yield: float | None = None


class Composition(BaseModel):
    holdings: list[HoldingMeta]
    sector_weights: dict[str, float]
    # False when the upstream metadata lookup failed; the UI hides the panel.
    available: bool = True


class TradingCalendar(BaseModel):
    periods_per_year: int
    has_crypto: bool
    mixed: bool  # crypto held alongside assets that do not trade daily


class AnalyzeResponse(BaseModel):
    calendar: TradingCalendar
    metrics: MetricsResponse
    optimization: OptimizeResponse
    backtest: BacktestSeries
    drawdown: DrawdownSeries
    distribution: ReturnDistribution
    composition: Composition
