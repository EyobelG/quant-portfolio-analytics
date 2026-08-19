import type { Holding, Optimization } from "../types";

export default function WeightsComparison({
  holdings,
  opt,
}: {
  holdings: Holding[];
  opt: Optimization;
}) {
  const tickers = holdings.map((h) => h.ticker);
  const currentByTicker = Object.fromEntries(holdings.map((h) => [h.ticker, h.weight]));

  return (
    <div className="panel">
      <h2>Suggested Reweighting</h2>
      <p className="panel-sub">
        What the max-Sharpe portfolio would hold instead, given the same assets.
      </p>
      {!opt.max_sharpe_available && (
        <div className="inline-note">
          No allocation of these assets earned a positive return over this window, so no
          max-Sharpe portfolio exists. The minimum-volatility portfolio is shown in its place.
        </div>
      )}
      <table className="weights-table">
        <thead>
          <tr>
            <th>Asset</th>
            <th>Current</th>
            <th>Max Sharpe</th>
            <th>Min Volatility</th>
          </tr>
        </thead>
        <tbody>
          {tickers.map((t) => {
            const cur = currentByTicker[t] ?? 0;
            const ms = opt.max_sharpe_weights[t] ?? 0;
            const delta = ms - cur;
            return (
              <tr key={t}>
                <td className="ticker-cell">{t}</td>
                <td>{(cur * 100).toFixed(1)}%</td>
                <td>
                  {(ms * 100).toFixed(1)}%
                  <span className={`delta ${delta >= 0 ? "up" : "down"}`}>
                    {delta >= 0 ? "▲" : "▼"} {Math.abs(delta * 100).toFixed(1)}
                  </span>
                </td>
                <td>{((opt.min_vol_weights[t] ?? 0) * 100).toFixed(1)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
