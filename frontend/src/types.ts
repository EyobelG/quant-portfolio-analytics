export interface Holding {
  ticker: string;
  weight: number;
}

export interface Metrics {
  annual_return: number;
  annual_volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  var_95: number;
  cvar_95: number;
  beta: number;
  alpha: number;
}

export interface FrontierPoint {
  volatility: number;
  return: number;
  sharpe: number;
}

export interface Optimization {
  frontier: FrontierPoint[];
  max_sharpe_weights: Record<string, number>;
  max_sharpe_point: FrontierPoint;
  min_vol_weights: Record<string, number>;
  min_vol_point: FrontierPoint;
  current_point: FrontierPoint;
  correlation_matrix: Record<string, Record<string, number>>;
}

export interface Backtest {
  dates: string[];
  portfolio: number[];
  optimized: number[];
  benchmark: number[];
}

export interface Drawdown {
  dates: string[];
  drawdown: number[];
  max_drawdown: number;
  max_drawdown_date: string;
  recovery_days: number | null;
  longest_underwater_days: number;
}

export interface Distribution {
  bin_edges: number[];
  counts: number[];
  var_95: number;
  cvar_95: number;
  mean: number;
}

export interface HoldingMeta {
  ticker: string;
  weight: number;
  sector: string | null;
  name: string | null;
  dividend_yield: number | null;
}

export interface Composition {
  holdings: HoldingMeta[];
  sector_weights: Record<string, number>;
  available: boolean;
}

export interface AnalyzeResponse {
  metrics: Metrics;
  optimization: Optimization;
  backtest: Backtest;
  drawdown: Drawdown;
  distribution: Distribution;
  composition: Composition;
}
