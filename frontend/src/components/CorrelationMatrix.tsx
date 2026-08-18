import type { Optimization } from "../types";

// Blue (negative) → neutral → red (positive) so clusters of correlated
// holdings are visible at a glance.
function cellColor(v: number): string {
  const clamped = Math.max(-1, Math.min(1, v));
  if (clamped >= 0) {
    const a = 0.12 + clamped * 0.65;
    return `rgba(220, 84, 84, ${a})`;
  }
  const a = 0.12 + Math.abs(clamped) * 0.65;
  return `rgba(79, 124, 255, ${a})`;
}

export default function CorrelationMatrix({ opt }: { opt: Optimization }) {
  const tickers = Object.keys(opt.correlation_matrix);

  return (
    <div className="panel">
      <h2>Correlation Matrix</h2>
      <p className="panel-sub">
        Highly correlated holdings move together, which limits how much diversification they actually buy you.
      </p>
      <div className="matrix-scroll">
        <table className="corr-table">
          <thead>
            <tr>
              <th />
              {tickers.map((t) => (
                <th key={t}>{t}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tickers.map((row) => (
              <tr key={row}>
                <th>{row}</th>
                {tickers.map((col) => {
                  const v = opt.correlation_matrix[row]?.[col] ?? 0;
                  return (
                    <td key={col} style={{ background: cellColor(v) }}>
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
