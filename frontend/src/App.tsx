import { useState } from "react";
import BacktestChart from "./components/BacktestChart";
import CorrelationMatrix from "./components/CorrelationMatrix";
import FrontierChart from "./components/FrontierChart";
import MetricsGrid from "./components/MetricsGrid";
import PortfolioBuilder from "./components/PortfolioBuilder";
import WeightsComparison from "./components/WeightsComparison";
import type { AnalyzeResponse, Holding } from "./types";

// Empty in dev (Vite proxies /api to localhost:8000). In production this is the
// deployed API's address, which Render injects as a bare hostname — so add the
// scheme when it's missing.
const RAW_API_BASE = import.meta.env.VITE_API_BASE ?? "";
const API_BASE =
  RAW_API_BASE && !/^https?:\/\//.test(RAW_API_BASE) ? `https://${RAW_API_BASE}` : RAW_API_BASE;

const DEFAULT_HOLDINGS: Holding[] = [
  { ticker: "AAPL", weight: 0.25 },
  { ticker: "MSFT", weight: 0.25 },
  { ticker: "JNJ", weight: 0.2 },
  { ticker: "JPM", weight: 0.15 },
  { ticker: "XOM", weight: 0.15 },
];

export default function App() {
  const [holdings, setHoldings] = useState<Holding[]>(DEFAULT_HOLDINGS);
  const [period, setPeriod] = useState("3y");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          holdings: holdings.filter((h) => h.ticker.trim() !== ""),
          period,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Request failed (${res.status})`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Quant Portfolio Analytics</h1>
          <p>
            Risk metrics, mean-variance optimization, and backtesting on live market data.
          </p>
        </div>
      </header>

      <main className="layout">
        <aside className="sidebar">
          <PortfolioBuilder
            holdings={holdings}
            period={period}
            onChange={setHoldings}
            onPeriodChange={setPeriod}
            onAnalyze={analyze}
            loading={loading}
          />
        </aside>

        <section className="results">
          {error && <div className="error-box">{error}</div>}

          {!result && !error && !loading && (
            <div className="empty-state">
              <h2>Build a portfolio to get started</h2>
              <p>
                Enter tickers and weights on the left, then run the analysis. The default is a
                five-stock sample portfolio across tech, healthcare, financials, and energy.
              </p>
            </div>
          )}

          {loading && <div className="empty-state">Fetching prices and running optimization…</div>}

          {result && (
            <>
              <MetricsGrid metrics={result.metrics} />
              <BacktestChart bt={result.backtest} />
              <FrontierChart opt={result.optimization} />
              <div className="two-col">
                <WeightsComparison holdings={holdings} opt={result.optimization} />
                <CorrelationMatrix opt={result.optimization} />
              </div>
            </>
          )}
        </section>
      </main>

      <footer className="app-footer">
        Data via Yahoo Finance. Educational use only — not investment advice.
      </footer>
    </div>
  );
}
