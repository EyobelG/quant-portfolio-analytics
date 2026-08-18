import type { Holding } from "../types";

// A spread of liquid, well-known symbols so people can try the app without
// having to think up tickers. Broad-market ETFs sit first — they make the
// clearest starting portfolio.
const SUGGESTIONS: { ticker: string; name: string }[] = [
  { ticker: "SPY", name: "S&P 500 ETF" },
  { ticker: "QQQ", name: "Nasdaq 100 ETF" },
  { ticker: "VTI", name: "Total US Market ETF" },
  { ticker: "AGG", name: "US Aggregate Bond ETF" },
  { ticker: "GLD", name: "Gold ETF" },
  { ticker: "NVDA", name: "NVIDIA" },
  { ticker: "AAPL", name: "Apple" },
  { ticker: "MSFT", name: "Microsoft" },
  { ticker: "AMZN", name: "Amazon" },
  { ticker: "GOOGL", name: "Alphabet" },
  { ticker: "TSLA", name: "Tesla" },
  { ticker: "BRK-B", name: "Berkshire Hathaway" },
  { ticker: "JPM", name: "JPMorgan Chase" },
  { ticker: "JNJ", name: "Johnson & Johnson" },
  { ticker: "XOM", name: "Exxon Mobil" },
  { ticker: "KO", name: "Coca-Cola" },
];

interface Props {
  holdings: Holding[];
  period: string;
  onChange: (holdings: Holding[]) => void;
  onPeriodChange: (period: string) => void;
  onAnalyze: () => void;
  loading: boolean;
}

export default function PortfolioBuilder({
  holdings,
  period,
  onChange,
  onPeriodChange,
  onAnalyze,
  loading,
}: Props) {
  const totalWeight = holdings.reduce((sum, h) => sum + h.weight, 0);
  const balanced = Math.abs(totalWeight - 1) < 0.001;

  const update = (i: number, patch: Partial<Holding>) => {
    onChange(holdings.map((h, idx) => (idx === i ? { ...h, ...patch } : h)));
  };

  const addRow = () => onChange([...holdings, { ticker: "", weight: 0 }]);
  const removeRow = (i: number) => onChange(holdings.filter((_, idx) => idx !== i));

  const equalize = () => {
    const w = 1 / holdings.length;
    onChange(holdings.map((h) => ({ ...h, weight: Number(w.toFixed(4)) })));
  };

  // Drop a suggested ticker into the first blank row if there is one, otherwise
  // append it. Either way the weights need rebalancing, which "Equal weight" does.
  const addSuggestion = (ticker: string) => {
    if (holdings.some((h) => h.ticker === ticker)) return;
    const blank = holdings.findIndex((h) => h.ticker.trim() === "");
    if (blank !== -1) {
      update(blank, { ticker });
    } else {
      onChange([...holdings, { ticker, weight: 0 }]);
    }
  };

  const owned = new Set(holdings.map((h) => h.ticker));

  return (
    <div className="panel">
      <h2>Portfolio</h2>

      <div className="holdings">
        {holdings.map((h, i) => (
          <div className="holding-row" key={i}>
            <input
              className="ticker-input"
              placeholder="AAPL"
              value={h.ticker}
              onChange={(e) => update(i, { ticker: e.target.value.toUpperCase() })}
            />
            <input
              className="weight-input"
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={h.weight}
              onChange={(e) => update(i, { weight: parseFloat(e.target.value) || 0 })}
            />
            <span className="weight-pct">{(h.weight * 100).toFixed(1)}%</span>
            <button className="icon-btn" onClick={() => removeRow(i)} disabled={holdings.length <= 2}>
              ×
            </button>
          </div>
        ))}
      </div>

      <div className="builder-actions">
        <button className="ghost-btn" onClick={addRow}>
          + Add holding
        </button>
        <button className="ghost-btn" onClick={equalize}>
          Equal weight
        </button>
      </div>

      <div className={`weight-total ${balanced ? "ok" : "warn"}`}>
        Total weight: {(totalWeight * 100).toFixed(1)}%
        {!balanced && " — must equal 100%"}
      </div>

      <div className="suggestions">
        <span className="suggestions-label">Quick add</span>
        <div className="chip-row">
          {SUGGESTIONS.map((s) => (
            <button
              key={s.ticker}
              className="chip"
              title={s.name}
              disabled={owned.has(s.ticker)}
              onClick={() => addSuggestion(s.ticker)}
            >
              {s.ticker}
            </button>
          ))}
        </div>
        <p className="suggestions-hint">
          Any Yahoo Finance symbol works — try <code>^GSPC</code>, <code>BTC-USD</code>, or{" "}
          <code>VOO</code>. Add a few, then hit Equal weight.
        </p>
      </div>

      <label className="field">
        <span>Lookback period</span>
        <select value={period} onChange={(e) => onPeriodChange(e.target.value)}>
          <option value="1y">1 year</option>
          <option value="2y">2 years</option>
          <option value="3y">3 years</option>
          <option value="5y">5 years</option>
          <option value="10y">10 years</option>
        </select>
      </label>

      <button className="primary-btn" onClick={onAnalyze} disabled={!balanced || loading}>
        {loading ? "Analyzing…" : "Analyze portfolio"}
      </button>
    </div>
  );
}
