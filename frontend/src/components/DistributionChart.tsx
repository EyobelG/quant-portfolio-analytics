import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Distribution } from "../types";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function DistributionChart({ dist }: { dist: Distribution }) {
  // Plot each histogram bin at its midpoint so the VaR line lands in the right place.
  const data = dist.counts.map((count, i) => {
    const lo = dist.bin_edges[i];
    const hi = dist.bin_edges[i + 1];
    return { mid: (lo + hi) / 2, count, lo, hi };
  });

  return (
    <div className="panel">
      <h2>Daily Return Distribution</h2>
      <p className="panel-sub">
        Every trading day in the window, bucketed by return. Bars left of the dashed line are
        the worst 5% of days — the losses VaR is measuring.
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="#2a3142" vertical={false} />
          <XAxis
            dataKey="mid"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={pct}
            stroke="#8b95ab"
          />
          <YAxis stroke="#8b95ab" allowDecimals={false} />
          <Tooltip
            cursor={{ fill: "rgba(79,124,255,0.08)" }}
            contentStyle={{ background: "#161b26", border: "1px solid #2a3142", borderRadius: 8 }}
            formatter={(v: number) => [`${v} days`, "Count"]}
            labelFormatter={(v: number) => `Return ≈ ${pct(v)}`}
          />
          <ReferenceLine
            x={dist.var_95}
            stroke="#f0b429"
            strokeDasharray="4 4"
            label={{ value: "VaR 95%", position: "top", fill: "#f0b429", fontSize: 11 }}
          />
          <ReferenceLine x={0} stroke="#4a5468" />
          <Bar dataKey="count" name="Days">
            {data.map((d, i) => (
              <Cell key={i} fill={d.mid <= dist.var_95 ? "#dc5454" : "#4f7cff"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="chart-note">
        Average day {pct(dist.mean)} · worst 5% of days average {pct(dist.cvar_95)}
      </p>
    </div>
  );
}
