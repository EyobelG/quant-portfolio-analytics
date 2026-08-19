from pydantic import BaseModel, Field


class Holding(BaseModel):
    ticker: str
    weight: float = Field(ge=0, le=1)


class PortfolioRequest(BaseModel):
    holdings: list[Holding]
    benchmark: str = "^GSPC"
    period: str = "3y"  # yfinance period string: 1y, 2y, 3y, 5y, max


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


class AnalyzeResponse(BaseModel):
    metrics: MetricsResponse
    optimization: OptimizeResponse
    backtest: BacktestSeries
    drawdown: DrawdownSeries
    distribution: ReturnDistribution
    composition: Composition
