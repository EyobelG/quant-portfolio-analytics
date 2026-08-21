import type { Inference } from "../types";

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
const prob = (v: number) => `${(v * 100).toFixed(1)}%`;
const num = (v: number) => v.toFixed(2);

/** A 0-1 probability drawn as a filled track, with the 95% bar marked. */
function ConfidenceBar({ value }: { value: number }) {
  return (
    <div className="conf-bar" role="img" aria-label={`${prob(value)} confidence`}>
      <div
        className={`conf-fill ${value >= 0.95 ? "good" : value >= 0.8 ? "warn" : "bad"}`}
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
      />
      <div className="conf-threshold" style={{ left: "95%" }} />
    </div>
  );
}

export default function InferencePanel({ inf }: { inf: Inference }) {
  if (!inf.available) {
    return (
      <div className="panel">
        <h2>Statistical Significance</h2>
        <p className="panel-sub">{inf.reason}</p>
      </div>
    );
  }

  const { moments, psr, bootstrap, tail, dsr, var_backtest: vb } = inf;

  return (
    <div className="panel">
      <h2>Statistical Significance</h2>
      <p className="panel-sub">
        Every figure in the metrics grid is a point estimate from one finite, non-normal,
        autocorrelated sample. These are the tests that ask how much of it survives sampling error.
      </p>

      <div className="sig-grid">
        <div className="sig-card">
          <div className="sig-label">Probabilistic Sharpe Ratio</div>
          <div className="sig-value">{prob(psr.psr)}</div>
          <ConfidenceBar value={psr.psr} />
          <p className="sig-hint">
            Probability the true Sharpe exceeds zero, once the estimator's standard error is
            widened for skew of {num(psr.skewness)} and excess kurtosis of{" "}
            {num(psr.excess_kurtosis)}. Standard error on the annualized Sharpe is ±
            {num(psr.standard_error)}.
          </p>
        </div>

        {dsr && (
          <div className="sig-card">
            <div className="sig-label">Deflated Sharpe Ratio</div>
            <div className="sig-value">{prob(dsr.dsr)}</div>
            <ConfidenceBar value={dsr.dsr} />
            <p className="sig-hint">
              The same test, but against the Sharpe a <em>lucky</em> search would produce. Sweeping{" "}
              {dsr.n_trials} frontier portfolios and reporting the best is multiple testing; under
              a null of no skill that search alone would be expected to turn up{" "}
              {num(dsr.expected_max_sharpe)}.{" "}
              {dsr.beats_selection_bias
                ? "This portfolio clears that bar."
                : "This portfolio does not clear that bar."}
            </p>
          </div>
        )}

        <div className="sig-card">
          <div className="sig-label">Bootstrap 95% Interval</div>
          <div className="sig-value">
            {num(bootstrap.ci_lower)} &ndash; {num(bootstrap.ci_upper)}
          </div>
          <div className={`sig-verdict ${bootstrap.significant ? "good" : "bad"}`}>
            {bootstrap.significant
              ? "Excludes zero — a real edge"
              : "Straddles zero — indistinguishable from no edge"}
          </div>
          <p className="sig-hint">
            {bootstrap.n_resamples.toLocaleString()} stationary-bootstrap resamples with a mean
            block length of {bootstrap.mean_block_length.toFixed(1)} days. Blocks rather than
            individual days, so volatility clustering is preserved and the interval does not come
            out artificially narrow.
          </p>
        </div>

        <div className="sig-card">
          <div className="sig-label">Return Distribution</div>
          <div className="sig-value">{moments.normal_at_5pct ? "Normal" : "Non-normal"}</div>
          <div className={`sig-verdict ${moments.normal_at_5pct ? "" : "bad"}`}>
            Jarque-Bera {num(moments.jarque_bera)} (p{" "}
            {moments.jarque_bera_p < 0.001 ? "< 0.001" : `= ${moments.jarque_bera_p.toFixed(3)}`})
          </div>
          <p className="sig-hint">
            Skew {num(moments.skewness)}, excess kurtosis {num(moments.excess_kurtosis)} over{" "}
            {moments.observations.toLocaleString()} days.{" "}
            {moments.normal_at_5pct
              ? "Normality is not rejected here, which is unusual for daily returns."
              : "Normality is rejected, so any risk figure derived from a Gaussian assumption understates the tail."}
          </p>
        </div>
      </div>

      <h3 className="sub-heading">Tail Risk</h3>
      <div className="matrix-scroll">
        <table className="weights-table">
          <thead>
            <tr>
              <th>1-day 95% VaR</th>
              <th>Gaussian</th>
              <th>Cornish-Fisher</th>
              <th>Historical</th>
              <th>Historical CVaR</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="muted-cell">Loss threshold</td>
              <td>{pct(tail.gaussian_var)}</td>
              <td className={tail.understatement > 0 ? "bad" : undefined}>
                {pct(tail.modified_var)}
              </td>
              <td>{pct(tail.historical_var)}</td>
              <td className="bad">{pct(tail.historical_cvar)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="chart-note">
        The Cornish-Fisher expansion corrects the normal quantile ({num(tail.z_normal)}) for skew
        and fat tails, moving it to {num(tail.z_cornish_fisher)}.{" "}
        {tail.understatement > 0
          ? `A Gaussian model understates this portfolio's daily tail loss by ${pct(tail.understatement)}.`
          : "For this sample the Gaussian model is not optimistic about the tail."}
      </p>

      {vb && (
        <>
          <h3 className="sub-heading">VaR Model Backtest</h3>
          <p className="panel-sub">
            Each day's VaR is estimated from the prior {vb.window} days only, so this measures how
            the model would actually have performed rather than how well it fits in hindsight.
          </p>
          <div className="matrix-scroll">
            <table className="weights-table">
              <thead>
                <tr>
                  <th>Test</th>
                  <th>Null hypothesis</th>
                  <th>Statistic</th>
                  <th>p-value</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Kupiec POF</td>
                  <td className="muted-cell">Breach rate equals 5%</td>
                  <td>
                    {vb.kupiec.exceptions} of {vb.kupiec.observations} (
                    {prob(vb.kupiec.exception_rate)})
                  </td>
                  <td>{vb.kupiec.p_value.toFixed(3)}</td>
                  <td className={vb.kupiec.correctly_calibrated ? "good" : "bad"}>
                    {vb.kupiec.correctly_calibrated ? "Calibrated" : "Miscalibrated"}
                  </td>
                </tr>
                <tr>
                  <td>Christoffersen</td>
                  <td className="muted-cell">Breaches are independent</td>
                  <td>
                    P(breach | breach) = {prob(vb.christoffersen.prob_breach_after_breach)}
                  </td>
                  <td>{vb.christoffersen.p_value.toFixed(3)}</td>
                  <td className={vb.christoffersen.independent ? "good" : "bad"}>
                    {vb.christoffersen.independent ? "Independent" : "Clustered"}
                  </td>
                </tr>
                <tr>
                  <td>Conditional coverage</td>
                  <td className="muted-cell">Both of the above</td>
                  <td>{num(vb.conditional_coverage_lr)}</td>
                  <td>{vb.conditional_coverage_p.toFixed(3)}</td>
                  <td className={vb.model_adequate ? "good" : "bad"}>
                    {vb.model_adequate ? "Adequate" : "Rejected"}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
