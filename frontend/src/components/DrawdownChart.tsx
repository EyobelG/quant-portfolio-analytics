import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Drawdown } from "../types";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function DrawdownChart({ dd }: { dd: Drawdown }) {
  const data = dd.dates.map((d, i) => ({ date: d, drawdown: dd.drawdown[i] }));
  const tickInterval = Math.max(1, Math.floor(data.length / 6));

  return (
    <div className="panel">
      <h2>Underwater Plot</h2>
      <p className="panel-sub">
        How far below its previous peak the portfolio sat, day by day. The depth is the loss;
        the width is how long it took to get back.
      </p>

      <div className="stat-strip">
        <div>
          <span className="stat-value bad">{pct(dd.max_drawdown)}</span>
          <span className="stat-label">Deepest drawdown</span>
        </div>
        <div>
          <span className="stat-value">{dd.max_drawdown_date}</span>
          <span className="stat-label">Trough date</span>
        </div>
        <div>
          <span className="stat-value">
            {dd.recovery_days === null ? "Not recovered" : `${dd.recovery_days} days`}
          </span>
          <span className="stat-label">Peak-to-peak recovery</span>
        </div>
        <div>
          <span className="stat-value">{dd.longest_underwater_days} days</span>
          <span className="stat-label">Longest underwater</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
          <defs>
            <linearGradient id="dd-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#dc5454" stopOpacity={0.05} />
              <stop offset="100%" stopColor="#dc5454" stopOpacity={0.55} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#2a3142" />
          <XAxis dataKey="date" stroke="#8b95ab" interval={tickInterval} minTickGap={24} />
          <YAxis stroke="#8b95ab" tickFormatter={pct} domain={["auto", 0]} />
          <Tooltip
            formatter={(v: number) => pct(v)}
            contentStyle={{ background: "#161b26", border: "1px solid #2a3142", borderRadius: 8 }}
          />
          <ReferenceLine y={0} stroke="#4a5468" />
          <Area
            type="monotone"
            dataKey="drawdown"
            name="Drawdown"
            stroke="#dc5454"
            strokeWidth={1.5}
            fill="url(#dd-fill)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
