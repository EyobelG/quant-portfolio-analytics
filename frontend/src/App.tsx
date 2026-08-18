import { useState } from "react";
import BacktestChart from "./components/BacktestChart";
import CorrelationMatrix from "./components/CorrelationMatrix";
import FrontierChart from "./components/FrontierChart";
import Header from "./components/Header";
import Logo from "./components/Logo";
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
  // The holdings the current result was computed from — editing the builder
  // afterwards must not retroactively change what the results claim to show.
  const [analyzed, setAnalyzed] = useState<Holding[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = async () => {
    setLoading(true);
    setError(null);
    const submitted = holdings.filter((h) => h.ticker.trim() !== "");
    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ holdings: submitted, period }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Request failed (${res.status})`);
      }
      setResult(await res.json());
      setAnalyzed(submitted);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Header hasResults={!!result} />
      <div className="app" id="top">
        <section className="hero">
          <h1>
            Quantitative <span className="gradient-text">Portfolio Analytics</span>
          </h1>
          <p>
            Risk metrics, mean-variance optimization, and backtesting on live market data.
          </p>
        </section>

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
              <div id="metrics">
                <MetricsGrid metrics={result.metrics} />
              </div>
              <div id="backtest">
                <BacktestChart bt={result.backtest} />
              </div>
              <div id="frontier">
                <FrontierChart opt={result.optimization} />
              </div>
              <div className="two-col" id="allocation">
                <WeightsComparison holdings={analyzed} opt={result.optimization} />
                <CorrelationMatrix opt={result.optimization} />
              </div>
            </>
          )}
          </section>
        </main>

        <footer className="app-footer">
          <div className="footer-credit">
            <Logo size={26} />
            <div>
              <div className="footer-name">
                Built by <strong>Eyobel Gebre</strong>
              </div>
              <div className="footer-links">
                <a href="https://github.com/EyobelG/quant-portfolio-analytics" target="_blank" rel="noreferrer noopener">
                  GitHub
                </a>
                <span className="dot">·</span>
                <a href="https://www.linkedin.com/in/eyobelgebre/" target="_blank" rel="noreferrer noopener">
                  LinkedIn
                </a>
              </div>
            </div>
          </div>
          <p className="footer-disclaimer">
            Data via Yahoo Finance. Educational use only — not investment advice.
          </p>
        </footer>
      </div>
    </>
  );
}
