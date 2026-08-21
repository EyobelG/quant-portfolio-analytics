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
import type { WalkForward } from "../types";

const pct = (v: number) => (Number.isFinite(v) ? `${(v * 100).toFixed(2)}%` : "—");
const num = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "—");

// One colour per strategy, held stable between the chart and the table so a row
// can be matched to its line without hunting through the legend.
const COLORS: Record<string, string> = {
  max_sharpe: "#dc5454",
  max_sharpe_shrunk: "#f0883e",
  min_volatility: "#a371f7",
  risk_parity: "#2fbf71",
  hrp: "#22d3ee",
  equal_weight: "#4f7cff",
  buy_and_hold: "#e3b341",
};

export default function WalkForwardPanel({ wf }: { wf: WalkForward }) {
  if (!wf.available) {
    return (
      <div className="panel">
        <h2>Walk-Forward Test</h2>
        <p className="panel-sub">{wf.reason}</p>
      </div>
    );
  }

  const data = wf.dates.map((date, i) => {
    const row: Record<string, string | number> = { date };
    for (const s of wf.strategies) row[s.key] = s.growth[i];
    return row;
  });

  const years = wf.oos_days / 252;

  return (
    <div className="panel">
      <h2>Walk-Forward Test</h2>
      <p className="panel-sub">
        Every allocation rule below was re-fitted at each rebalance using only data available at
        that time, then held forward through returns it had never seen. Weights drift with the
        market in between, and {wf.cost_bps} bps of transaction costs are charged on the turnover
        each rebalance generates. {wf.oos_days} out-of-sample days ({years.toFixed(1)} years) after
        a {wf.train_window}-day training window, rebalanced every {wf.rebalance_every} days.
      </p>

      <div className="finding-box">
        <div className="finding-headline">
          Max-Sharpe optimization scored{" "}
          <strong className="bad">{num(wf.in_sample_sharpe)}</strong> in-sample and{" "}
          <strong className={wf.out_of_sample_sharpe > 1 ? "good" : "bad"}>
            {num(wf.out_of_sample_sharpe)}
          </strong>{" "}
          out-of-sample.
        </div>
        <p>
          That gap — {num(wf.sharpe_decay)} of Sharpe — is the cost of judging a portfolio on the
          same data used to build it.{" "}
          {wf.equal_weight_wins ? (
            <>
              Equal weighting beat the optimizer out-of-sample, reproducing the standard result
              (DeMiguel, Garlappi &amp; Uppal, 2009): estimation error in expected returns
              outweighs the benefit of optimizing over them.
            </>
          ) : (
            <>
              The optimizer held up out-of-sample over this particular window, which is the
              less common outcome and worth treating as sample-specific rather than as evidence
              the method generalizes.
            </>
          )}
        </p>
      </div>

      <ResponsiveContainer width="100%" height={340}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
          <CartesianGrid stroke="#2a3142" strokeDasharray="3 3" />
          <XAxis dataKey="date" stroke="#8b95ab" tick={{ fontSize: 11 }} minTickGap={60} />
          <YAxis
            stroke="#8b95ab"
            tick={{ fontSize: 11 }}
            domain={["auto", "auto"]}
            tickFormatter={(v: number) => `$${v.toFixed(2)}`}
          />
          <Tooltip
            contentStyle={{ background: "#161b26", border: "1px solid #2a3142", borderRadius: 8 }}
            formatter={(v: number, name: string) => [
              `$${v.toFixed(3)}`,
              wf.strategies.find((s) => s.key === name)?.label ?? name,
            ]}
          />
          <Legend
            formatter={(name: string) => wf.strategies.find((s) => s.key === name)?.label ?? name}
            wrapperStyle={{ fontSize: 12 }}
          />
          {wf.strategies.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              stroke={COLORS[s.key] ?? "#8b95ab"}
              strokeWidth={s.key === "equal_weight" || s.key === "max_sharpe" ? 2.4 : 1.4}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      <div className="matrix-scroll">
        <table className="weights-table wf-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Sharpe</th>
              <th>Return</th>
              <th>Volatility</th>
              <th>Max DD</th>
              <th>Calmar</th>
              <th>Turnover</th>
            </tr>
          </thead>
          <tbody>
            {wf.strategies.map((s, i) => (
              <tr key={s.key} className={i === 0 ? "wf-best" : undefined}>
                <td>
                  <span className="swatch" style={{ background: COLORS[s.key] ?? "#8b95ab" }} />
                  {s.label}
                </td>
                <td className={s.sharpe_ratio >= 1 ? "good" : undefined}>{num(s.sharpe_ratio)}</td>
                <td>{pct(s.annual_return)}</td>
                <td>{pct(s.annual_volatility)}</td>
                <td className="bad">{pct(s.max_drawdown)}</td>
                <td>{num(s.calmar_ratio)}</td>
                <td className="muted-cell">{s.annual_turnover.toFixed(2)}x</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="chart-note">
        Turnover is annualized and one-way: 1.00x means the book is replaced once a year. It is
        most of why the aggressive optimizers give back their paper advantage.
      </p>
    </div>
  );
}
