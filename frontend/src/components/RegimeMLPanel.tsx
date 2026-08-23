import type { RegimeML } from "../types";

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
const num = (v: number | null) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(2) : "—");

// Calm -> good, Stressed -> bad, anything in between interpolated across a
// green -> amber -> red ramp. Same risk-color intent as the rest of the app,
// mapped onto a level index instead of a correlation value.
const LEVEL_RAMP: [number, number, number][] = [
  [47, 191, 113], // --good
  [227, 179, 65], // amber
  [220, 84, 84], // --bad
];

function levelColor(level: number, nLevels: number): string {
  if (nLevels <= 1) return "var(--accent)";
  const t = (level / (nLevels - 1)) * (LEVEL_RAMP.length - 1);
  const lo = Math.floor(t);
  const hi = Math.min(lo + 1, LEVEL_RAMP.length - 1);
  const frac = t - lo;
  const mix = LEVEL_RAMP[lo].map((c, i) => Math.round(c + (LEVEL_RAMP[hi][i] - c) * frac));
  return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
}

export default function RegimeMLPanel({ reg }: { reg: RegimeML }) {
  if (!reg.available) {
    return (
      <div className="panel">
        <h2>ML Regime Classifier</h2>
        <p className="panel-sub">{reg.reason}</p>
      </div>
    );
  }

  const currentColor = levelColor(reg.current_regime.level, reg.n_regimes);

  return (
    <div className="panel">
      <h2>ML Regime Classifier</h2>
      <p className="panel-sub">
        An unsupervised Gaussian Mixture Model clusters trading days by rolling volatility,
        average pairwise correlation, and trend into {reg.n_regimes} regimes, relabeled calmest to
        stormiest so the ordering means the same thing on every run. This is a label for what kind
        of market today resembles, not a prediction of tomorrow — whether it is actually useful for
        picking an allocation is answered by the "ML Regime Classifier" row in the walk-forward
        table above, evaluated by the same out-of-sample test as every other strategy there.
      </p>

      <div className="finding-box">
        <div className="finding-headline">
          Current regime:{" "}
          <strong style={{ color: currentColor }}>{reg.current_regime.label}</strong>
        </div>
        <p>
          {(reg.current_regime.confidence * 100).toFixed(0)}% posterior confidence.{" "}
          {reg.converged
            ? "The classifier converged cleanly on this sample."
            : "The classifier did not fully converge on this sample — treat the label as indicative rather than exact."}
        </p>
      </div>

      <div
        style={{
          display: "flex",
          height: 14,
          borderRadius: 4,
          overflow: "hidden",
          marginBottom: 8,
        }}
      >
        {reg.regime_levels.map((level, i) => (
          <div
            key={i}
            style={{
              flex: 1,
              background: levelColor(level, reg.n_regimes),
              opacity: 0.85,
            }}
            title={`${reg.dates[i]}: ${reg.level_names[level]}`}
          />
        ))}
      </div>
      <p className="chart-note">
        {reg.dates[0]} to {reg.dates[reg.dates.length - 1]} — each slice is one trading day,
        colored by regime.
      </p>

      <div className="matrix-scroll">
        <table className="weights-table">
          <thead>
            <tr>
              <th>Regime</th>
              <th>Days</th>
              <th>Best strategy in this regime</th>
              <th>Sharpe</th>
              <th>Avg volatility</th>
              <th>Avg correlation</th>
            </tr>
          </thead>
          <tbody>
            {reg.leaderboard.map((row) => (
              <tr key={row.regime}>
                <td>
                  <span
                    className="swatch"
                    style={{ background: levelColor(row.level, reg.n_regimes) }}
                  />
                  {row.regime}
                </td>
                <td className="muted-cell">{row.days}</td>
                <td>{row.best_strategy_label}</td>
                <td>{num(row.sharpe)}</td>
                <td className="muted-cell">{pct(row.avg_volatility)}</td>
                <td className="muted-cell">{row.avg_correlation.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
