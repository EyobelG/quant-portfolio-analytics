import { Cell, Pie, PieChart, Tooltip } from "recharts";
import type { Composition } from "../types";
import AssetIcon from "./AssetIcon";

const SECTOR_COLORS = [
  "#4f7cff",
  "#22d3ee",
  "#2fbf71",
  "#f0b429",
  "#b07cff",
  "#ff7ab6",
  "#7dd3fc",
  "#facc15",
  "#34d399",
  "#fb923c",
];

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function CompositionPanel({ comp }: { comp: Composition }) {
  if (!comp.available || comp.holdings.length === 0) return null;

  const sectors = Object.entries(comp.sector_weights).map(([name, weight]) => ({ name, weight }));
  const topShare = sectors[0]?.weight ?? 0;
  // Yields come from an endpoint that is unavailable in some environments; an
  // always-empty column reads as broken, so only show it when there is data.
  const hasYields = comp.holdings.some((h) => h.dividend_yield !== null);

  return (
    <div className="panel">
      <h2>Composition &amp; Concentration</h2>
      <p className="panel-sub">
        Sector exposure and per-holding detail. Diversification across names means little if
        they all sit in the same sector.
      </p>

      <div className="composition-grid">
        {/* Fixed dimensions rather than ResponsiveContainer: inside a grid track
            it can measure zero width and render nothing in a production build,
            where React's dev-only double render is not there to force a
            re-measure. The donut is a fixed size anyway; the grid stacks on
            narrow screens. */}
        <div className="donut-wrap">
          <PieChart width={230} height={230}>
            <Pie
              data={sectors}
              dataKey="weight"
              nameKey="name"
              cx={115}
              cy={115}
              innerRadius={58}
              outerRadius={92}
              paddingAngle={2}
              stroke="none"
              isAnimationActive={false}
            >
              {sectors.map((_, i) => (
                <Cell key={i} fill={SECTOR_COLORS[i % SECTOR_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              formatter={(v: number) => pct(v)}
              contentStyle={{
                background: "#161b26",
                border: "1px solid #2a3142",
                borderRadius: 8,
              }}
            />
          </PieChart>
          <div className="donut-center">
            <span className="donut-value">{pct(topShare)}</span>
            <span className="donut-label">largest sector</span>
          </div>
        </div>

        <div>
          <ul className="sector-legend">
            {sectors.map((s, i) => (
              <li key={s.name}>
                <span
                  className="swatch"
                  style={{ background: SECTOR_COLORS[i % SECTOR_COLORS.length] }}
                />
                <span className="sector-name">{s.name}</span>
                <span className="sector-weight">{pct(s.weight)}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <table className="weights-table holdings-table">
        <thead>
          <tr>
            <th>Asset</th>
            <th>Sector</th>
            <th>Weight</th>
            {hasYields && <th>Div. yield</th>}
          </tr>
        </thead>
        <tbody>
          {comp.holdings.map((h) => (
            <tr key={h.ticker}>
              <td className="ticker-cell" title={h.name ?? undefined}>
                <span className="asset-cell">
                  <AssetIcon ticker={h.ticker} sector={h.sector} />
                  {h.ticker}
                </span>
              </td>
              <td className="muted-cell">{h.sector ?? "—"}</td>
              <td>{pct(h.weight)}</td>
              {hasYields && (
                <td>{h.dividend_yield === null ? "—" : `${h.dividend_yield.toFixed(2)}%`}</td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
