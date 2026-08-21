import {
  Area,
  ComposedChart,
  CartesianGrid,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { VolatilityBlock } from "../types";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const num = (v: number, d = 3) => v.toFixed(d);

export default function VolatilityPanel({
  vol,
  dates,
}: {
  vol: VolatilityBlock;
  dates: string[];
}) {
  if (!vol.available) {
    return (
      <div className="panel">
        <h2>Conditional Volatility</h2>
        <p className="panel-sub">{vol.reason}</p>
      </div>
    );
  }

  const { arch_test: arch, ewma, garch } = vol;

  // The fitted path is aligned to the tail of the return series, then the
  // forecast is appended so the handoff is visible as one continuous line.
  const fitted = garch?.conditional_volatility ?? ewma.series;
  const offset = Math.max(0, dates.length - fitted.length);
  const history = fitted.map((v, i) => ({
    label: dates[offset + i] ?? "",
    garch: v,
    ewma: ewma.series[ewma.series.length - fitted.length + i],
  }));

  const forecast = (garch?.forecast ?? []).map((v, i) => ({
    label: `+${i + 1}d`,
    forecast: v,
  }));

  const data = [
    ...history,
    // Stitch the two segments so the line does not break at the boundary.
    ...forecast.map((f, i) =>
      i === 0 ? { ...f, garch: history[history.length - 1]?.garch } : f
    ),
  ];

  return (
    <div className="panel">
      <h2>Conditional Volatility</h2>
      <p className="panel-sub">
        Annualized volatility is a single number describing a quantity that is not constant.
        Volatility clusters, so a risk model built on one unconditional sigma is too loose in calm
        regimes and far too tight in a crisis.
      </p>

      <div className="stat-strip">
        <div>
          <div className="stat-label">Current (GARCH)</div>
          <div className="stat-value">
            {garch ? pct(garch.current_volatility) : pct(ewma.current)}
          </div>
        </div>
        <div>
          <div className="stat-label">Long-run level</div>
          <div className="stat-value">
            {garch ? pct(garch.long_run_volatility) : pct(ewma.unconditional)}
          </div>
        </div>
        {garch && (
          <div>
            <div className="stat-label">Persistence (α + β)</div>
            <div className="stat-value">{num(garch.persistence)}</div>
          </div>
        )}
        {garch?.half_life_days != null && (
          <div>
            <div className="stat-label">Shock half-life</div>
            <div className="stat-value">{garch.half_life_days.toFixed(0)}d</div>
          </div>
        )}
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
          <CartesianGrid stroke="#2a3142" strokeDasharray="3 3" />
          <XAxis dataKey="label" stroke="#8b95ab" tick={{ fontSize: 11 }} minTickGap={70} />
          <YAxis
            stroke="#8b95ab"
            tick={{ fontSize: 11 }}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={{ background: "#161b26", border: "1px solid #2a3142", borderRadius: 8 }}
            formatter={(v: number, name: string) => [
              pct(v),
              name === "garch" ? "GARCH(1,1)" : name === "ewma" ? "EWMA (λ=0.94)" : "Forecast",
            ]}
          />
          {garch && (
            <ReferenceLine
              y={garch.long_run_volatility}
              stroke="#8b95ab"
              strokeDasharray="4 4"
              label={{ value: "Long-run", fill: "#8b95ab", fontSize: 11, position: "insideLeft" }}
            />
          )}
          <Area
            type="monotone"
            dataKey="ewma"
            stroke="#2a3142"
            fill="#1b2130"
            strokeWidth={1}
            dot={false}
          />
          <Line type="monotone" dataKey="garch" stroke="#22d3ee" strokeWidth={1.8} dot={false} />
          <Line
            type="monotone"
            dataKey="forecast"
            stroke="#e3b341"
            strokeWidth={2}
            strokeDasharray="5 3"
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {garch && (
        <p className="chart-note">
          GARCH(1,1) fitted by maximum likelihood: ω = {garch.omega.toExponential(2)}, α ={" "}
          {num(garch.alpha)}, β = {num(garch.beta)}. α is how sharply volatility reacts to news, β
          how long it remembers. Their sum, {num(garch.persistence)}, is the persistence — a shock
          decays halfway back to the long-run level in about{" "}
          {garch.half_life_days?.toFixed(0) ?? "—"} trading days. The dashed gold line is the{" "}
          {garch.forecast_horizon}-day forecast, mean-reverting toward that level. Log-likelihood{" "}
          {garch.log_likelihood.toFixed(1)}, AIC {garch.aic.toFixed(1)}, BIC {garch.bic.toFixed(1)}.
        </p>
      )}

      <p className="chart-note">
        <strong>Engle ARCH-LM test</strong> ({arch.lags} lags): statistic{" "}
        {arch.lm_statistic.toFixed(2)}, p{" "}
        {arch.p_value < 0.001 ? "< 0.001" : `= ${arch.p_value.toFixed(3)}`}.{" "}
        {arch.arch_effects_present ? (
          <>
            The null of constant variance is rejected — volatility genuinely clusters here, which
            is what justifies fitting a conditional model rather than reporting one sigma.
          </>
        ) : (
          <>
            The null of constant variance is not rejected over this window, so the conditional
            model is not adding much beyond the unconditional estimate.
          </>
        )}{" "}
        {garch && garch.vol_ratio > 1.15 && (
          <>
            The market is currently {num(garch.vol_ratio, 2)}× its own long-run volatility.
          </>
        )}
      </p>
    </div>
  );
}
