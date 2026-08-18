import type { Metrics } from "../types";

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
const num = (v: number) => v.toFixed(2);

interface Card {
  label: string;
  value: string;
  hint: string;
  tone?: "good" | "bad";
}

export default function MetricsGrid({ metrics }: { metrics: Metrics }) {
  const cards: Card[] = [
    {
      label: "Annualized Return",
      value: pct(metrics.annual_return),
      hint: "Compound yearly growth rate",
      tone: metrics.annual_return >= 0 ? "good" : "bad",
    },
    {
      label: "Volatility",
      value: pct(metrics.annual_volatility),
      hint: "Annualized standard deviation",
    },
    {
      label: "Sharpe Ratio",
      value: num(metrics.sharpe_ratio),
      hint: "Return per unit of total risk",
      tone: metrics.sharpe_ratio >= 1 ? "good" : undefined,
    },
    {
      label: "Sortino Ratio",
      value: num(metrics.sortino_ratio),
      hint: "Return per unit of downside risk",
    },
    {
      label: "Max Drawdown",
      value: pct(metrics.max_drawdown),
      hint: "Worst peak-to-trough loss",
      tone: "bad",
    },
    {
      label: "VaR (95%)",
      value: pct(metrics.var_95),
      hint: "Daily loss exceeded 5% of the time",
      tone: "bad",
    },
    {
      label: "CVaR (95%)",
      value: pct(metrics.cvar_95),
      hint: "Average loss in the worst 5% of days",
      tone: "bad",
    },
    {
      label: "Beta",
      value: num(metrics.beta),
      hint: "Sensitivity to the benchmark",
    },
    {
      label: "Alpha",
      value: pct(metrics.alpha),
      hint: "Excess return vs. benchmark",
      tone: metrics.alpha >= 0 ? "good" : "bad",
    },
  ];

  return (
    <div className="metrics-grid">
      {cards.map((c) => (
        <div className="metric-card" key={c.label}>
          <div className="metric-label">{c.label}</div>
          <div className={`metric-value ${c.tone ?? ""}`}>{c.value}</div>
          <div className="metric-hint">{c.hint}</div>
        </div>
      ))}
    </div>
  );
}
