import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Backtest } from "../types";

export default function BacktestChart({ bt }: { bt: Backtest }) {
  const data = bt.dates.map((d, i) => ({
    date: d,
    portfolio: bt.portfolio[i],
    optimized: bt.optimized[i],
    benchmark: bt.benchmark[i],
  }));

  // Roughly 6 x-axis labels regardless of series length.
  const tickInterval = Math.max(1, Math.floor(data.length / 6));

  return (
    <div className="panel">
      <h2>Growth of $1</h2>
      <p className="panel-sub">
        Your portfolio versus the max-Sharpe reweighting and the benchmark, over the same window.
      </p>
      <ResponsiveContainer width="100%" height={340}>
        <LineChart data={data} margin={{ top: 16, right: 24, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="#2a3142" />
          <XAxis dataKey="date" stroke="#8b95ab" interval={tickInterval} minTickGap={24} />
          <YAxis stroke="#8b95ab" tickFormatter={(v: number) => `$${v.toFixed(2)}`} domain={["auto", "auto"]} />
          <Tooltip
            formatter={(v: number) => `$${v.toFixed(3)}`}
            contentStyle={{ background: "#161b26", border: "1px solid #2a3142", borderRadius: 8 }}
          />
          <Legend verticalAlign="top" height={32} />
          <Line type="monotone" dataKey="portfolio" name="Your portfolio" stroke="#f0b429" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="optimized" name="Max-Sharpe optimized" stroke="#2fbf71" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="benchmark" name="Benchmark (S&P 500)" stroke="#6b7689" dot={false} strokeWidth={1.5} strokeDasharray="4 4" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
