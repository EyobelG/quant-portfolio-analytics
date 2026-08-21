import type { CorrMatrix, Regimes } from "../types";

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
// Undefined statistics arrive as null, not NaN — JSON has no NaN.
const num = (v: number | null | undefined) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(2) : "—";

// Same blue-to-red scale as the unconditional correlation matrix, so the three
// grids here read against the one already on the page.
function cellColor(v: number): string {
  const clamped = Math.max(-1, Math.min(1, v));
  if (clamped >= 0) {
    return `rgba(220, 84, 84, ${0.12 + clamped * 0.65})`;
  }
  return `rgba(79, 124, 255, ${0.12 + Math.abs(clamped) * 0.65})`;
}

// The delta spans roughly ±0.5 rather than ±1, so it is amplified to use the
// same visual range. Without this every delta cell washes out to near-neutral.
function deltaColor(v: number): string {
  return cellColor(Math.max(-1, Math.min(1, v * 2)));
}

function Matrix({
  matrix,
  tickers,
  color,
}: {
  matrix: CorrMatrix;
  tickers: string[];
  color: (v: number) => string;
}) {
  return (
    <div className="matrix-scroll">
      <table className="corr-table regime-corr">
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
                const v = matrix[row]?.[col] ?? 0;
                return (
                  <td key={col} style={{ background: color(v) }}>
                    {v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function RegimeCorrelationPanel({ reg }: { reg: Regimes }) {
  if (!reg.available) {
    return (
      <div className="panel">
        <h2>Correlation Under Stress</h2>
        <p className="panel-sub">{reg.reason}</p>
      </div>
    );
  }

  const { tickers, regimes, delta_matrix, correlation_increase, betas, capture } = reg;
  const calm = regimes.calm;
  const stressed = regimes.stressed;
  const tailPct = `${(reg.tail_quantile * 100).toFixed(0)}%`;

  // Only holdings with a defined ratio can be ranked on it.
  const worst = betas
    .filter((b) => typeof b.ratio === "number" && Number.isFinite(b.ratio))
    .sort((a, b) => (b.ratio as number) - (a.ratio as number))[0];

  return (
    <div className="panel">
      <h2>Correlation Under Stress</h2>
      <p className="panel-sub">
        A single correlation matrix averaged over the whole sample is the most misleading number on
        this page. Correlations are not stable — they rise in selloffs, so the diversification the
        average implies is largely absent on the days it is meant to protect you. The split below is
        by the <em>benchmark's</em> return, not the portfolio's: conditioning on the portfolio's own
        losses would be circular.
      </p>

      <div className="finding-box">
        <div className="finding-headline">
          Average pairwise correlation rises from <strong>{num(calm.average_correlation)}</strong> in
          calm markets to <strong className="bad">{num(stressed.average_correlation)}</strong> in the
          worst {tailPct} of days.
        </div>
        <p>
          {correlation_increase > 0.1 ? (
            <>
              That is an increase of {num(correlation_increase)}. These holdings converge exactly
              when they are supposed to offset each other, which means the portfolio is materially
              less diversified in a drawdown than the headline correlation matrix suggests.
            </>
          ) : (
            <>
              The change of {num(correlation_increase)} is small, so this portfolio's diversification
              holds up in the tail better than most — an unusual and genuinely favourable result.
            </>
          )}
        </p>
      </div>

      <div className="stat-strip">
        <div>
          <div className="stat-label">Calm ({calm.observations}d)</div>
          <div className="stat-value">{num(calm.average_correlation)}</div>
        </div>
        <div>
          <div className="stat-label">Stressed ({stressed.observations}d)</div>
          <div className="stat-value bad">{num(stressed.average_correlation)}</div>
        </div>
        <div>
          <div className="stat-label">Rally ({regimes.rally.observations}d)</div>
          <div className="stat-value">{num(regimes.rally.average_correlation)}</div>
        </div>
        {capture && (
          <>
            <div>
              <div className="stat-label">Downside capture</div>
              <div className={`stat-value ${capture.downside < 1 ? "good" : "bad"}`}>
                {num(capture.downside)}
              </div>
            </div>
            <div>
              <div className="stat-label">Upside capture</div>
              <div className={`stat-value ${capture.upside > 1 ? "good" : ""}`}>
                {num(capture.upside)}
              </div>
            </div>
          </>
        )}
      </div>

      <div className="regime-grid">
        <div>
          <h3 className="sub-heading">Calm — middle {100 - reg.tail_quantile * 200}% of days</h3>
          <Matrix matrix={calm.correlation_matrix} tickers={tickers} color={cellColor} />
          <p className="chart-note">
            Benchmark averaged {pct(calm.benchmark_mean)} a day here.
          </p>
        </div>
        <div>
          <h3 className="sub-heading">Stressed — worst {tailPct} of days</h3>
          <Matrix matrix={stressed.correlation_matrix} tickers={tickers} color={cellColor} />
          <p className="chart-note">
            Benchmark averaged {pct(stressed.benchmark_mean)} a day, worst{" "}
            {pct(stressed.benchmark_worst)}.
          </p>
        </div>
      </div>

      <h3 className="sub-heading">Change in correlation (stressed − calm)</h3>
      <Matrix matrix={delta_matrix} tickers={tickers} color={deltaColor} />
      <p className="chart-note">
        Red cells are pairs that converge under stress. Colour is amplified 2× against the matrices
        above, since the deltas span a narrower range.
      </p>

      <h3 className="sub-heading">Downside versus upside beta</h3>
      <div className="matrix-scroll">
        <table className="weights-table">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Beta</th>
              <th>Down-market beta</th>
              <th>Up-market beta</th>
              <th>Ratio</th>
            </tr>
          </thead>
          <tbody>
            {betas.map((b) => (
              <tr key={b.ticker}>
                <td className="ticker-cell">{b.ticker}</td>
                <td>{num(b.beta)}</td>
                <td className={b.asymmetric ? "bad" : undefined}>{num(b.downside_beta)}</td>
                <td>{num(b.upside_beta)}</td>
                <td className={b.asymmetric ? "bad" : "muted-cell"}>{num(b.ratio)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="chart-note">
        A single OLS beta averages the two together and hides the asymmetry. A ratio above 1 means
        the holding participates more in the market's losses than in its gains — the payoff profile
        nobody wants.{" "}
        {worst?.asymmetric && (
          <>
            {worst.ticker} is the worst offender here at {num(worst.ratio)}: it falls with a beta of{" "}
            {num(worst.downside_beta)} but only rises with {num(worst.upside_beta)}.
          </>
        )}
      </p>

      <p className="chart-note">
        <strong>How much of this is real.</strong> Conditioning on the size of the market's move
        biases correlation in the direction of the conditioning — selecting the largest moves
        inflates it, selecting the smallest deflates it, even when the true correlation is constant
        (Forbes &amp; Rigobon, 2002). So the calm figure here is biased low and the stressed figure
        high, and the gap overstates the true change. The effect is not purely mechanical though:
        Longin &amp; Solnik (2001) found genuine excess correlation in the lower tail after
        correcting for the bias. Read the direction as sound and the magnitude as an upper bound.
        {!reg.reliable && (
          <>
            {" "}
            The stressed regime holds only {reg.min_regime_observations} days here, which is thin
            for a correlation matrix over {tickers.length} assets — treat it as indicative.
          </>
        )}
      </p>
    </div>
  );
}
