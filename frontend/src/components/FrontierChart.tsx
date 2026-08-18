import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { Optimization } from "../types";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function FrontierChart({ opt }: { opt: Optimization }) {
  const frontier = opt.frontier.map((p) => ({ x: p.volatility, y: p.return }));
  const current = [{ x: opt.current_point.volatility, y: opt.current_point.return }];
  const maxSharpe = [{ x: opt.max_sharpe_point.volatility, y: opt.max_sharpe_point.return }];
  const minVol = [{ x: opt.min_vol_point.volatility, y: opt.min_vol_point.return }];

  return (
    <div className="panel">
      <h2>Efficient Frontier</h2>
      <p className="panel-sub">
        Every point on the curve is the best possible return for that level of risk. If your portfolio
        sits below the curve, the same return is available at lower volatility.
      </p>
      <ResponsiveContainer width="100%" height={360}>
        <ScatterChart margin={{ top: 16, right: 24, bottom: 32, left: 28 }}>
          <CartesianGrid stroke="#2a3142" />
          <XAxis
            type="number"
            dataKey="x"
            name="Volatility"
            tickFormatter={pct}
            stroke="#8b95ab"
            label={{ value: "Annualized volatility", position: "insideBottom", offset: -18, fill: "#8b95ab" }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Return"
            tickFormatter={pct}
            stroke="#8b95ab"
            label={{
              value: "Expected return",
              angle: -90,
              position: "insideLeft",
              offset: -16,
              fill: "#8b95ab",
            }}
          />
          <ZAxis range={[60, 61]} />
          <Tooltip
            formatter={(v: number) => pct(v)}
            contentStyle={{ background: "#161b26", border: "1px solid #2a3142", borderRadius: 8 }}
          />
          <Legend verticalAlign="top" height={32} />
          <Scatter name="Efficient frontier" data={frontier} fill="#4f7cff" line shape="circle" />
          <Scatter name="Your portfolio" data={current} fill="#f0b429" shape="diamond" />
          <Scatter name="Max Sharpe" data={maxSharpe} fill="#2fbf71" shape="star" />
          <Scatter name="Min volatility" data={minVol} fill="#b07cff" shape="triangle" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
