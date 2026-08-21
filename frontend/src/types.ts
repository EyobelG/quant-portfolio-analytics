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
  max_sharpe_available: boolean;
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

export interface TradingCalendar {
  periods_per_year: number;
  has_crypto: boolean;
  mixed: boolean;
}

export interface AnalyzeResponse {
  calendar: TradingCalendar;
  metrics: Metrics;
  optimization: Optimization;
  backtest: Backtest;
  drawdown: Drawdown;
  distribution: Distribution;
  composition: Composition;
}

/* ---------- Advanced analytics (POST /api/advanced) ---------- */

/** Every block either carries its payload or explains why it could not. */
type Degradable<T> = ({ available: true } & T) | { available: false; reason: string };

export interface Moments {
  observations: number;
  mean: number;
  std: number;
  skewness: number;
  excess_kurtosis: number;
  jarque_bera: number;
  jarque_bera_p: number;
  normal_at_5pct: boolean;
}

export interface Psr {
  sharpe_annualized: number;
  benchmark_sharpe: number;
  psr: number;
  z_score: number;
  skewness: number;
  excess_kurtosis: number;
  observations: number;
  standard_error: number;
}

export interface Dsr {
  dsr: number;
  expected_max_sharpe: number;
  n_trials: number;
  trial_sharpe_std: number;
  sharpe_annualized: number;
  beats_selection_bias: boolean;
}

export interface BootstrapCi {
  sharpe: number;
  ci_lower: number;
  ci_upper: number;
  confidence: number;
  mean_block_length: number;
  n_resamples: number;
  significant: boolean;
}

export interface TailRisk {
  alpha: number;
  gaussian_var: number;
  modified_var: number;
  historical_var: number;
  historical_cvar: number;
  z_normal: number;
  z_cornish_fisher: number;
  understatement: number;
}

export interface CoverageTest {
  exceptions: number;
  observations: number;
  expected_exceptions: number;
  exception_rate: number;
  lr_statistic: number;
  p_value: number;
  correctly_calibrated: boolean;
}

export interface VarBacktest {
  alpha: number;
  window: number;
  kupiec: CoverageTest;
  christoffersen: { p_value: number; independent: boolean; prob_breach_after_breach: number };
  conditional_coverage_lr: number;
  conditional_coverage_p: number;
  model_adequate: boolean;
}

export type Inference = Degradable<{
  moments: Moments;
  psr: Psr;
  bootstrap: BootstrapCi;
  tail: TailRisk;
  dsr?: Dsr;
  var_backtest?: VarBacktest;
}>;

export type RiskStructure = Degradable<{
  tickers: string[];
  weights: number[];
  shrinkage: {
    intensity: number;
    condition_number_sample: number;
    condition_number_shrunk: number;
    n_observations: number;
    n_assets: number;
  };
  eigen: {
    eigenvalues: number[];
    variance_explained: number[];
    cumulative_variance_explained: number[];
    lambda_plus: number;
    n_signal_factors: number;
    market_factor_share: number;
    reliable: boolean;
  };
  contributions: {
    portfolio_volatility: number;
    percent: number[];
    marginal: number[];
    diversification_ratio: number;
    effective_positions: number;
  };
  effective_bets: { value: number; n_assets: number; concentration: number };
  risk_parity: { weights: number[]; volatility: number; converged: boolean };
}>;

export interface Garch {
  omega: number;
  alpha: number;
  beta: number;
  persistence: number;
  converged: boolean;
  log_likelihood: number;
  aic: number;
  bic: number;
  current_volatility: number;
  long_run_volatility: number;
  half_life_days: number | null;
  vol_ratio: number;
  forecast: number[];
  forecast_horizon: number;
  conditional_volatility: number[];
}

export type VolatilityBlock = Degradable<{
  arch_test: { lags: number; lm_statistic: number; p_value: number; arch_effects_present: boolean };
  ewma: { current: number; unconditional: number; half_life_days: number; series: number[] };
  garch?: Garch;
}>;

export interface WalkForwardStrategy {
  key: string;
  label: string;
  growth: number[];
  annual_return: number;
  annual_volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  total_growth: number;
  annual_turnover: number;
  n_rebalances: number;
}

export type WalkForward = Degradable<{
  dates: string[];
  strategies: WalkForwardStrategy[];
  train_window: number;
  rebalance_every: number;
  oos_days: number;
  cost_bps: number;
  in_sample_sharpe: number;
  out_of_sample_sharpe: number;
  sharpe_decay: number;
  equal_weight_wins: boolean;
  best_strategy: string | null;
}> & { reason?: string };

export interface FactorLoading {
  factor: string;
  label: string;
  description: string;
  beta: number;
  t_statistic: number;
  p_value: number;
  significant: boolean;
}

export type Factors = Degradable<{
  alpha_annualized: number;
  alpha_t_statistic: number;
  alpha_p_value: number;
  alpha_significant: boolean;
  loadings: FactorLoading[];
  r_squared: number;
  adj_r_squared: number;
  newey_west_lags: number;
  observations: number;
  unexplained_share: number;
}> & { loadings: FactorLoading[] };

export interface AdvancedResponse {
  inference: Inference;
  risk_structure: RiskStructure;
  volatility: VolatilityBlock;
  walk_forward: WalkForward;
  factors: Factors;
}
