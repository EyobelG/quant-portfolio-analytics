import { useCallback, useEffect, useRef, useState } from "react";
import BacktestChart from "./components/BacktestChart";
import BlackLittermanPanel from "./components/BlackLittermanPanel";
import CompositionPanel from "./components/CompositionPanel";
import CorrelationMatrix from "./components/CorrelationMatrix";
import DistributionChart from "./components/DistributionChart";
import DrawdownChart from "./components/DrawdownChart";
import FactorPanel from "./components/FactorPanel";
import FrontierChart from "./components/FrontierChart";
import Header from "./components/Header";
import InferencePanel from "./components/InferencePanel";
import Logo from "./components/Logo";
import MetricsGrid from "./components/MetricsGrid";
import PortfolioBuilder from "./components/PortfolioBuilder";
import ReportCover from "./components/ReportCover";
import RiskStructurePanel from "./components/RiskStructurePanel";
import VolatilityPanel from "./components/VolatilityPanel";
import WalkForwardPanel from "./components/WalkForwardPanel";
import WeightsComparison from "./components/WeightsComparison";
import { encodePortfolio, parsePortfolio } from "./share";
import type { AdvancedResponse, AnalyzeResponse, Holding } from "./types";

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

const shared = parsePortfolio(window.location.search);

export default function App() {
  const [holdings, setHoldings] = useState<Holding[]>(shared?.holdings ?? DEFAULT_HOLDINGS);
  const [period, setPeriod] = useState(shared?.period ?? "3y");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [advanced, setAdvanced] = useState<AdvancedResponse | null>(null);
  const [advancedLoading, setAdvancedLoading] = useState(false);
  // The holdings the current result was computed from — editing the builder
  // afterwards must not retroactively change what the results claim to show.
  const [analyzed, setAnalyzed] = useState<Holding[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [slow, setSlow] = useState(false);
  const [copied, setCopied] = useState(false);

  const analyze = useCallback(async () => {
    setLoading(true);
    setError(null);
    setAdvanced(null);
    const submitted = holdings.filter((h) => h.ticker.trim() !== "");
    // Keep the address bar in sync so the current portfolio is always linkable.
    window.history.replaceState(null, "", encodePortfolio(submitted, period));
    const body = JSON.stringify({ holdings: submitted, period });
    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload.detail ?? `Request failed (${res.status})`);
      }
      setResult(await res.json());
      setAnalyzed(submitted);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      setResult(null);
      setLoading(false);
      return;
    }
    setLoading(false);

    // The advanced block fits GARCH by MLE, bootstraps 2000 resamples and
    // re-runs six optimizers at every rebalance, so it is fetched separately
    // and streamed in behind the headline metrics rather than delaying them.
    setAdvancedLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/advanced`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (res.ok) setAdvanced(await res.json());
    } catch {
      // Non-fatal: the core analysis already rendered.
    } finally {
      setAdvancedLoading(false);
    }
  }, [holdings, period]);

  // The API sleeps on Render's free tier and takes ~50s to wake. Without this,
  // a cold start is indistinguishable from the app being broken.
  useEffect(() => {
    if (!loading) {
      setSlow(false);
      return;
    }
    const t = setTimeout(() => setSlow(true), 5000);
    return () => clearTimeout(t);
  }, [loading]);

  // A shared link should show its results without the visitor pressing anything.
  const autoRan = useRef(false);
  useEffect(() => {
    if (shared && !autoRan.current) {
      autoRan.current = true;
      analyze();
    }
  }, [analyze]);

  const copyLink = async () => {
    const url = `${window.location.origin}${window.location.pathname}${encodePortfolio(
      analyzed.length ? analyzed : holdings,
      period
    )}`;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setError("Could not copy — your browser blocked clipboard access.");
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

          {loading && (
            <div className="empty-state">
              {slow ? (
                <>
                  <h2>Waking up the server…</h2>
                  <p>
                    The API sleeps after 15 minutes idle on the free tier and takes about 50
                    seconds to start. Subsequent runs are fast.
                  </p>
                </>
              ) : (
                "Fetching prices and running optimization…"
              )}
            </div>
          )}

          {result && (
            <>
              <div className="results-bar">
                <span className="results-summary">
                  {analyzed.length} holdings · {period} lookback
                </span>
                <span className="results-actions">
                  <button className="ghost-btn" onClick={copyLink}>
                    {copied ? "✓ Link copied" : "Copy shareable link"}
                  </button>
                  <button
                    className="ghost-btn"
                    onClick={() => window.print()}
                    disabled={advancedLoading}
                    title={
                      advancedLoading
                        ? "Waiting for the advanced analytics to finish"
                        : "Opens your browser's print dialog — choose “Save as PDF”"
                    }
                  >
                    {advancedLoading ? "Preparing report…" : "Export PDF"}
                  </button>
                </span>
              </div>
              <ReportCover holdings={analyzed} period={period} benchmark="^GSPC" />
              {result.calendar.has_crypto && (
                <div className="calendar-note">
                  <strong>₿ Crypto detected.</strong>{" "}
                  {result.calendar.mixed
                    ? "Crypto trades daily while equities do not, so returns are aligned to the 252-day equity calendar — forward-filling stocks across weekends would invent flat days and understate their volatility."
                    : "All holdings trade every day, so figures are annualized over 365 days rather than the 252-day equity calendar."}
                </div>
              )}
              <div id="metrics">
                <MetricsGrid metrics={result.metrics} />
              </div>
              <div id="backtest">
                <BacktestChart bt={result.backtest} />
              </div>
              <div id="risk">
                <DrawdownChart dd={result.drawdown} />
                <DistributionChart dist={result.distribution} />
              </div>
              <div id="frontier">
                <FrontierChart opt={result.optimization} />
              </div>
              <div id="composition">
                <CompositionPanel comp={result.composition} />
              </div>
              <div className="two-col" id="allocation">
                <WeightsComparison holdings={analyzed} opt={result.optimization} />
                <CorrelationMatrix opt={result.optimization} />
              </div>

              {advancedLoading && !advanced && (
                <div className="panel advanced-pending">
                  <h2>Running the advanced analytics…</h2>
                  <p className="panel-sub">
                    Fitting GARCH(1,1) by maximum likelihood, bootstrapping 2,000 resamples, and
                    walking six allocation rules forward through out-of-sample data. A few seconds.
                  </p>
                </div>
              )}

              {advanced && (
                <>
                  <div id="walkforward">
                    <WalkForwardPanel wf={advanced.walk_forward} />
                  </div>
                  <div id="inference">
                    <InferencePanel inf={advanced.inference} />
                  </div>
                  <div id="riskstructure">
                    <RiskStructurePanel rs={advanced.risk_structure} />
                  </div>
                  <div id="volatility">
                    <VolatilityPanel vol={advanced.volatility} dates={result.backtest.dates} />
                  </div>
                  <div id="factors">
                    <FactorPanel f={advanced.factors} />
                  </div>
                </>
              )}

              {/* Outside the `advanced` gate: it fetches independently, so it
                  must not wait on the slow path to become usable. */}
              <div id="blacklitterman">
                <BlackLittermanPanel
                  holdings={analyzed}
                  period={period}
                  apiBase={API_BASE}
                />
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
