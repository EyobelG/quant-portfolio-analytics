import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  ComposedChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { BlackLittermanResponse, Holding, PortfolioView } from "../types";

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`;

/** A fresh view defaults to the first holding at moderate confidence. */
const blankView = (asset: string): PortfolioView => ({
  type: "absolute",
  asset,
  versus: null,
  value: 0.1,
  confidence: 0.5,
});

export default function BlackLittermanPanel({
  holdings,
  period,
  apiBase,
}: {
  holdings: Holding[];
  period: string;
  apiBase: string;
}) {
  const tickers = holdings.map((h) => h.ticker).filter(Boolean);
  const [views, setViews] = useState<PortfolioView[]>([]);
  const [result, setResult] = useState<BlackLittermanResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const update = (i: number, patch: Partial<PortfolioView>) =>
    setViews((vs) => vs.map((v, j) => (j === i ? { ...v, ...patch } : v)));

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/api/black-litterman`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          holdings: holdings.filter((h) => h.ticker.trim() !== ""),
          period,
          // The API rejects a stray `versus` on an absolute view rather than
          // ignoring it, so strip it here instead of sending a contradiction.
          views: views.map((v) =>
            v.type === "relative" ? v : { ...v, versus: null }
          ),
        }),
      });
      const body = await res.json();
      if (!res.ok) {
        const detail = body.detail;
        throw new Error(
          typeof detail === "string" ? detail : detail?.[0]?.msg ?? `Request failed (${res.status})`
        );
      }
      setResult(body);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const returnsData =
    result?.tickers.map((t, i) => ({
      ticker: t,
      equilibrium: result.prior_returns[i],
      posterior: result.posterior_returns[i],
    })) ?? [];

  const weightsData =
    result?.tickers.map((t) => ({
      ticker: t,
      market: result.prior.weights[t] ?? 0,
      posterior: result.posterior.weights[t] ?? 0,
    })) ?? [];

  // Both frontiers on one axis pair, so the shift is a shape change rather than
  // two charts the reader has to hold in their head at once.
  const frontierData = [
    ...(result?.prior.frontier ?? []).map((p) => ({
      volatility: p.volatility,
      priorReturn: p.return,
    })),
    ...(result?.posterior.frontier ?? []).map((p) => ({
      volatility: p.volatility,
      posteriorReturn: p.return,
    })),
  ].sort((a, b) => a.volatility - b.volatility);

  return (
    <div className="panel">
      <h2>Black-Litterman</h2>
      <p className="panel-sub">
        The walk-forward test above shows mean-variance optimization losing to equal weighting,
        because the historical mean is a hopeless estimator of expected return. This is the
        principled repair, and it never estimates a mean at all: it reverse-optimizes the market's
        own capitalization weights into the returns the market must already be assuming, then moves
        off them only as far as an explicit view at an explicit confidence justifies.
      </p>

      <div className="bl-views">
        <div className="bl-views-head">
          <h3 className="sub-heading">Your views</h3>
          <button
            className="ghost-btn"
            onClick={() => setViews((vs) => [...vs, blankView(tickers[0] ?? "")])}
            disabled={tickers.length < 2 || views.length >= 10}
          >
            + Add view
          </button>
        </div>

        {views.length === 0 && (
          <p className="chart-note bl-empty">
            With no views the posterior is <em>exactly</em> the prior and the optimal portfolio is
            the market portfolio. That is the honest default — run it as-is to see what the market
            is implying, then add a view to see how far your opinion moves it.
          </p>
        )}

        {views.map((v, i) => (
          <div className="bl-view-row" key={i}>
            <select
              value={v.type}
              onChange={(e) =>
                update(i, {
                  type: e.target.value as PortfolioView["type"],
                  versus:
                    e.target.value === "relative"
                      ? v.versus ?? tickers.find((t) => t !== v.asset) ?? null
                      : null,
                })
              }
            >
              <option value="absolute">Absolute</option>
              <option value="relative">Relative</option>
            </select>

            <select value={v.asset} onChange={(e) => update(i, { asset: e.target.value })}>
              {tickers.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>

            {v.type === "relative" ? (
              <>
                <span className="bl-op">beats</span>
                <select
                  value={v.versus ?? ""}
                  onChange={(e) => update(i, { versus: e.target.value })}
                >
                  {tickers
                    .filter((t) => t !== v.asset)
                    .map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                </select>
                <span className="bl-op">by</span>
              </>
            ) : (
              <span className="bl-op">returns</span>
            )}

            <input
              type="number"
              step="0.01"
              value={v.value}
              onChange={(e) => update(i, { value: Number(e.target.value) })}
            />
            <span className="bl-op">a year</span>

            <label className="bl-conf">
              Confidence {pct1(v.confidence)}
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={v.confidence}
                onChange={(e) => update(i, { confidence: Number(e.target.value) })}
              />
            </label>

            <button
              className="icon-btn"
              aria-label="Remove view"
              onClick={() => setViews((vs) => vs.filter((_, j) => j !== i))}
            >
              ✕
            </button>
          </div>
        ))}

        <button className="primary-btn bl-run" onClick={run} disabled={loading || tickers.length < 2}>
          {loading ? "Solving…" : result ? "Re-run with these views" : "Compute equilibrium"}
        </button>
        {tickers.length < 2 && (
          <p className="chart-note">Black-Litterman needs at least two holdings.</p>
        )}
      </div>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <>
          {!result.market_caps_available && (
            <div className="calendar-note">
              <strong>Market caps unavailable.</strong> Share counts could not be retrieved for{" "}
              {result.missing_caps.join(", ") || "these holdings"}, so the prior falls back to equal
              weight. Mixing real capitalizations with assumed ones would produce a prior that looks
              authoritative and is quietly wrong, so the whole thing degrades together.
            </div>
          )}

          <div className="stat-strip">
            <div>
              <div className="stat-label">Risk aversion (δ)</div>
              <div className="stat-value">{result.risk_aversion.toFixed(2)}</div>
            </div>
            <div>
              <div className="stat-label">τ</div>
              <div className="stat-value">{result.tau}</div>
            </div>
            <div>
              <div className="stat-label">Prior Sharpe</div>
              <div className="stat-value">{result.prior.point.sharpe.toFixed(2)}</div>
            </div>
            <div>
              <div className="stat-label">Posterior Sharpe</div>
              <div className="stat-value">{result.posterior.point.sharpe.toFixed(2)}</div>
            </div>
          </div>

          {!result.risk_aversion_from_market && (
            <p className="chart-note">
              The benchmark lost money over this window, which implies a negative risk aversion and
              would invert every equilibrium return. δ has fallen back to the long-run equity value
              of 2.5.
            </p>
          )}

          <h3 className="sub-heading">Equilibrium versus posterior returns</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={returnsData} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
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
                  name === "equilibrium" ? "Equilibrium (prior)" : "Posterior",
                ]}
              />
              <Legend
                formatter={(n: string) => (n === "equilibrium" ? "Equilibrium (prior)" : "Posterior")}
                wrapperStyle={{ fontSize: 12 }}
              />
              <Bar dataKey="equilibrium" fill="#4f7cff" radius={[3, 3, 0, 0]} />
              <Bar dataKey="posterior" fill="#22d3ee" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>

          {result.views.length > 0 && (
            <div className="matrix-scroll">
              <table className="weights-table">
                <thead>
                  <tr>
                    <th>View</th>
                    <th>Your number</th>
                    <th>Equilibrium said</th>
                    <th>Posterior says</th>
                    <th>Adoption</th>
                  </tr>
                </thead>
                <tbody>
                  {result.views.map((v, i) => (
                    <tr key={i}>
                      <td className="ticker-cell">
                        {v.asset}
                        {v.versus ? ` − ${v.versus}` : ""}
                      </td>
                      <td>{pct(v.value)}</td>
                      <td className="muted-cell">{pct(v.prior_implied)}</td>
                      <td>{pct(v.posterior_implied)}</td>
                      <td>
                        <div className="conf-bar">
                          <div
                            className="conf-fill good"
                            style={{
                              width: `${Math.max(0, Math.min(1, v.adoption)) * 100}%`,
                            }}
                          />
                        </div>
                        <span className="bl-adoption">{pct1(v.adoption)}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {result.views.length > 0 && (
            <p className="chart-note">
              Adoption is the share of the distance from the equilibrium to your view that the
              posterior actually travelled. For an absolute view it tracks the confidence slider
              almost exactly. For a relative view it usually does not: the view's uncertainty scales
              with the variance of the <em>spread</em>, which can dwarf the prior's uncertainty
              about it, so a half-confidence spread view often barely moves the posterior. Both ends
              still behave — zero changes nothing, one binds exactly — but the middle is non-linear.
            </p>
          )}

          <h3 className="sub-heading">Market weights versus the posterior optimum</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={weightsData} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
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
                  name === "market" ? "Market portfolio" : "Posterior optimum",
                ]}
              />
              <Legend
                formatter={(n: string) => (n === "market" ? "Market portfolio" : "Posterior optimum")}
                wrapperStyle={{ fontSize: 12 }}
              />
              <Bar dataKey="market" fill="#8b95ab" radius={[3, 3, 0, 0]} />
              <Bar dataKey="posterior" fill="#2fbf71" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>

          <h3 className="sub-heading">Frontier shift</h3>
          <ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={frontierData} margin={{ top: 8, right: 16, bottom: 12, left: 4 }}>
              <CartesianGrid stroke="#2a3142" strokeDasharray="3 3" />
              <XAxis
                dataKey="volatility"
                type="number"
                stroke="#8b95ab"
                tick={{ fontSize: 11 }}
                domain={["dataMin", "dataMax"]}
                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                label={{ value: "Volatility", fill: "#8b95ab", fontSize: 11, dy: 12 }}
              />
              <YAxis
                stroke="#8b95ab"
                tick={{ fontSize: 11 }}
                domain={["auto", "auto"]}
                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{ background: "#161b26", border: "1px solid #2a3142", borderRadius: 8 }}
                formatter={(v: number, name: string) => [
                  pct(v),
                  name === "priorReturn" ? "Equilibrium frontier" : "Posterior frontier",
                ]}
                labelFormatter={(v: number) => `Volatility ${pct(v)}`}
              />
              <Legend
                formatter={(n: string) =>
                  n === "priorReturn" ? "Equilibrium frontier" : "Posterior frontier"
                }
                wrapperStyle={{ fontSize: 12 }}
              />
              <Line
                type="monotone"
                dataKey="priorReturn"
                stroke="#4f7cff"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="posteriorReturn"
                stroke="#22d3ee"
                strokeWidth={2}
                strokeDasharray="5 3"
                dot={false}
                connectNulls
              />
              <Scatter dataKey="priorReturn" fill="transparent" />
            </ComposedChart>
          </ResponsiveContainer>

          <p className="chart-note">
            δ = {result.risk_aversion.toFixed(2)} is the market-implied price of risk, backed out of
            the benchmark as (return − risk-free) ÷ variance. Because this app uses a 0% risk-free
            rate everywhere for consistency, δ comes out higher than the textbook figure and the
            equilibrium returns scale up with it — read them as relative magnitudes rather than
            forecasts. τ = {result.tau} scales the prior's own uncertainty; the posterior is famously
            insensitive to it once confidences are specified directly, as they are here.
          </p>
        </>
      )}
    </div>
  );
}
