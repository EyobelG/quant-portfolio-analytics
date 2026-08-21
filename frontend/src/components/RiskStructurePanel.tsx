import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { RiskStructure } from "../types";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const num = (v: number) => v.toFixed(2);

export default function RiskStructurePanel({ rs }: { rs: RiskStructure }) {
  if (!rs.available) {
    return (
      <div className="panel">
        <h2>Risk Decomposition</h2>
        <p className="panel-sub">{rs.reason}</p>
      </div>
    );
  }

  const { tickers, weights, shrinkage, eigen, contributions, effective_bets, risk_parity } = rs;

  // Weight versus risk share, per holding — the comparison that shows a 10%
  // position in a volatile name is not a 10% allocation of risk.
  const riskData = tickers.map((t, i) => ({
    ticker: t,
    weight: weights[i],
    risk: contributions.percent[i],
    gap: contributions.percent[i] - weights[i],
    parity: risk_parity.weights[i],
  }));

  const screeData = eigen.eigenvalues.map((v, i) => ({
    pc: `PC${i + 1}`,
    eigenvalue: v,
    share: eigen.variance_explained[i],
    signal: v > eigen.lambda_plus,
  }));

  return (
    <div className="panel">
      <h2>Risk Decomposition</h2>
      <p className="panel-sub">
        Weight is not risk. These are the diagnostics that separate how the capital is allocated
        from where the volatility actually comes from.
      </p>

      <div className="stat-strip">
        <div>
          <div className="stat-label">Effective bets</div>
          <div className="stat-value">
            {num(effective_bets.value)}{" "}
            <span className="muted-cell">of {effective_bets.n_assets}</span>
          </div>
        </div>
        <div>
          <div className="stat-label">Diversification ratio</div>
          <div className="stat-value">{num(contributions.diversification_ratio)}</div>
        </div>
        <div>
          <div className="stat-label">Ledoit-Wolf shrinkage</div>
          <div className="stat-value">{pct(shrinkage.intensity)}</div>
        </div>
        <div>
          <div className="stat-label">Market factor share</div>
          <div className="stat-value">{pct(eigen.market_factor_share)}</div>
        </div>
      </div>

      <p className="chart-note">
        Meucci's effective number of bets projects the portfolio onto its uncorrelated principal
        components and takes the exponential entropy of their variance shares.{" "}
        {effective_bets.value < effective_bets.n_assets * 0.5 ? (
          <>
            At {num(effective_bets.value)} against {effective_bets.n_assets} holdings, this
            portfolio is far less diversified than its position count suggests — the names move
            together, so most of the risk rides a single factor.
          </>
        ) : (
          <>
            At {num(effective_bets.value)} against {effective_bets.n_assets} holdings, the
            positions are carrying genuinely distinct risk.
          </>
        )}
      </p>

      <h3 className="sub-heading">Weight versus risk contribution</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={riskData} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
          <CartesianGrid stroke="#2a3142" strokeDasharray="3 3" />
          <XAxis dataKey="ticker" stroke="#8b95ab" tick={{ fontSize: 12 }} />
          <YAxis
            stroke="#8b95ab"
            tick={{ fontSize: 11 }}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={{ background: "#161b26", border: "1px solid #2a3142", borderRadius: 8 }}
            formatter={(v: number, name: string) => [
              pct(v),
              name === "weight" ? "Capital weight" : "Risk contribution",
            ]}
          />
          <Bar dataKey="weight" fill="#4f7cff" name="weight" radius={[3, 3, 0, 0]} />
          <Bar dataKey="risk" fill="#dc5454" name="risk" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>

      <div className="matrix-scroll">
        <table className="weights-table">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Capital weight</th>
              <th>Risk contribution</th>
              <th>Difference</th>
              <th>Risk-parity weight</th>
            </tr>
          </thead>
          <tbody>
            {riskData.map((r) => (
              <tr key={r.ticker}>
                <td className="ticker-cell">{r.ticker}</td>
                <td>{pct(r.weight)}</td>
                <td>{pct(r.risk)}</td>
                <td className={r.gap > 0.02 ? "bad" : r.gap < -0.02 ? "good" : "muted-cell"}>
                  {r.gap >= 0 ? "+" : ""}
                  {pct(r.gap)}
                </td>
                <td className="muted-cell">{pct(r.parity)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="chart-note">
        Contributions are the Euler decomposition of portfolio volatility, so they sum exactly to
        it. The final column is the long-only allocation where every holding contributes identical
        risk — a portfolio that needs no expected-return forecast at all, which is why it tends to
        survive out-of-sample where mean-variance does not.
      </p>

      <h3 className="sub-heading">Correlation eigenvalues</h3>
      <ResponsiveContainer width="100%" height={230}>
        <BarChart data={screeData} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
          <CartesianGrid stroke="#2a3142" strokeDasharray="3 3" />
          <XAxis dataKey="pc" stroke="#8b95ab" tick={{ fontSize: 12 }} />
          <YAxis stroke="#8b95ab" tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: "#161b26", border: "1px solid #2a3142", borderRadius: 8 }}
            formatter={(v: number, _n, item) => {
              // Recharts types `payload` as optional, so read the variance share
              // defensively rather than asserting a shape onto it.
              const share = (item?.payload as { share?: number } | undefined)?.share;
              return [
                share == null ? v.toFixed(3) : `${v.toFixed(3)} (${pct(share)} of variance)`,
                "Eigenvalue",
              ];
            }}
          />
          <ReferenceLine
            y={eigen.lambda_plus}
            stroke="#e3b341"
            strokeDasharray="4 4"
            label={{
              value: `Marchenko-Pastur bound ${eigen.lambda_plus.toFixed(2)}`,
              fill: "#e3b341",
              fontSize: 11,
              position: "insideTopRight",
            }}
          />
          <Bar dataKey="eigenvalue" radius={[3, 3, 0, 0]}>
            {screeData.map((d) => (
              <Cell key={d.pc} fill={d.signal ? "#22d3ee" : "#3a4358"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="chart-note">
        Random-matrix theory gives the eigenvalue distribution of a correlation matrix built from
        pure noise. Components above the bound ({eigen.n_signal_factors} here, in cyan) carry
        genuine structure; those inside the bulk are statistically indistinguishable from noise and
        would be flattened before the matrix is inverted.{" "}
        {!eigen.reliable && (
          <>
            With only {shrinkage.n_assets} assets the bound is indicative rather than reliable — the
            asymptotics assume many assets and many observations.
          </>
        )}
      </p>
      <p className="chart-note">
        Shrinkage moved the covariance {pct(shrinkage.intensity)} of the way toward a scaled
        identity target, improving its condition number from{" "}
        {num(shrinkage.condition_number_sample)} to {num(shrinkage.condition_number_shrunk)}. The
        intensity is not tuned — it is the closed-form value that minimizes expected squared error,
        derived from {shrinkage.n_observations} observations across {shrinkage.n_assets} assets.
      </p>
    </div>
  );
}
