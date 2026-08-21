import Logo from "./Logo";
import type { Holding } from "../types";

/**
 * Title block for the exported PDF.
 *
 * Hidden on screen and revealed only by the print stylesheet, so the report
 * opens with the portfolio it describes and the date it was run — without which
 * a printed tearsheet is unattributable a week later.
 */
export default function ReportCover({
  holdings,
  period,
  benchmark,
}: {
  holdings: Holding[];
  period: string;
  benchmark: string;
}) {
  const generated = new Date();

  return (
    <section className="report-cover" aria-hidden="true">
      <div className="report-brand">
        <Logo size={30} />
        <div>
          <div className="report-title">Portfolio Analytics Report</div>
          <div className="report-sub">
            {period} lookback &middot; benchmark {benchmark}
          </div>
        </div>
      </div>

      <table className="report-holdings">
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Weight</th>
          </tr>
        </thead>
        <tbody>
          {holdings.map((h) => (
            <tr key={h.ticker}>
              <td>{h.ticker}</td>
              <td>{(h.weight * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="report-meta">
        Generated {generated.toLocaleDateString()} at {generated.toLocaleTimeString()} &middot; Data
        via Yahoo Finance &middot; Educational use only, not investment advice.
      </div>
    </section>
  );
}
