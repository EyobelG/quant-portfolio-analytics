import type { Factors } from "../types";

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
const num = (v: number) => v.toFixed(2);

export default function FactorPanel({ f }: { f: Factors }) {
  if (!f.available) {
    return (
      <div className="panel">
        <h2>Factor Attribution</h2>
        <p className="panel-sub">
          Factor analysis is unavailable for this run{f.reason ? `: ${f.reason}` : "."}
        </p>
      </div>
    );
  }

  // Scale the loading bars against the largest absolute beta so the chart reads
  // regardless of whether the portfolio is levered to a factor or barely tilted.
  const maxAbs = Math.max(...f.loadings.map((l) => Math.abs(l.beta)), 1);

  return (
    <div className="panel">
      <h2>Factor Attribution</h2>
      <p className="panel-sub">
        Jensen's alpha against a single index credits the manager for every style tilt in the book.
        This is the multi-factor version: the market plus size, value, momentum, quality and
        low-volatility spreads, all built from liquid ETFs so the leftover alpha is what an
        investor could not have replicated cheaply.
      </p>

      <div className="stat-strip">
        <div>
          <div className="stat-label">Multi-factor alpha</div>
          <div className={`stat-value ${f.alpha_annualized >= 0 ? "good" : "bad"}`}>
            {pct(f.alpha_annualized)}
          </div>
        </div>
        <div>
          <div className="stat-label">t-statistic</div>
          <div className={`stat-value ${f.alpha_significant ? "good" : ""}`}>
            {num(f.alpha_t_statistic)}
          </div>
        </div>
        <div>
          <div className="stat-label">R²</div>
          <div className="stat-value">{num(f.r_squared)}</div>
        </div>
        <div>
          <div className="stat-label">Unexplained</div>
          <div className="stat-value">{pct(f.unexplained_share)}</div>
        </div>
      </div>

      <div className="matrix-scroll">
        <table className="weights-table factor-table">
          <thead>
            <tr>
              <th>Factor</th>
              <th>Exposure</th>
              <th>Beta</th>
              <th>t-stat</th>
              <th>p-value</th>
            </tr>
          </thead>
          <tbody>
            {f.loadings.map((l) => (
              <tr key={l.factor}>
                <td>
                  <div className="factor-name">{l.label}</div>
                  <div className="factor-desc">{l.description}</div>
                </td>
                <td className="bar-cell">
                  <div className="loading-track">
                    <div className="loading-zero" />
                    <div
                      className={`loading-bar ${l.beta >= 0 ? "pos" : "neg"} ${
                        l.significant ? "" : "faded"
                      }`}
                      style={{
                        width: `${(Math.abs(l.beta) / maxAbs) * 50}%`,
                        left: l.beta >= 0 ? "50%" : undefined,
                        right: l.beta < 0 ? "50%" : undefined,
                      }}
                    />
                  </div>
                </td>
                <td>{num(l.beta)}</td>
                <td className={l.significant ? "good" : "muted-cell"}>{num(l.t_statistic)}</td>
                <td className="muted-cell">
                  {l.p_value < 0.001 ? "< 0.001" : l.p_value.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="chart-note">
        Standard errors are Newey-West with {f.newey_west_lags} lags, which corrects for the
        heteroskedasticity and serial correlation in daily returns. Without that adjustment these
        t-statistics would be inflated — it is the usual way a spurious alpha ends up looking
        significant. Estimated over {f.observations.toLocaleString()} overlapping days.
      </p>
      <p className="chart-note">
        {f.alpha_significant ? (
          <>
            Alpha of {pct(f.alpha_annualized)} survives at t = {num(f.alpha_t_statistic)}. With
            daily data and a single portfolio this clears the conventional bar, though the standard
            caution applies: a t-stat near 2 is weak evidence once you account for how many
            portfolios could have been tested.
          </>
        ) : (
          <>
            Alpha is not statistically distinguishable from zero (t = {num(f.alpha_t_statistic)}).
            The factor exposures above already account for this portfolio's returns — there is no
            residual performance left to attribute to selection.
          </>
        )}
      </p>
    </div>
  );
}
