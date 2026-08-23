import { useEffect, useState } from "react";
import type {
  BrokerAccount,
  BrokerOrder,
  Holding,
  OrderResult,
  RebalancePreview,
} from "../types";

const money = (v: number) =>
  v.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function BrokerPanel({
  holdings,
  period,
  apiBase,
}: {
  holdings: Holding[];
  period: string;
  apiBase: string;
}) {
  const [account, setAccount] = useState<BrokerAccount | null>(null);
  const [configured, setConfigured] = useState(true);
  const [preview, setPreview] = useState<RebalancePreview | null>(null);
  const [results, setResults] = useState<OrderResult[] | null>(null);
  const [orders, setOrders] = useState<BrokerOrder[]>([]);
  const [previewing, setPreviewing] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadAccount = async () => {
    try {
      const res = await fetch(`${apiBase}/api/broker/account`);
      if (res.status === 503) {
        setConfigured(false);
        return;
      }
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      setAccount(await res.json());
      setConfigured(true);
    } catch {
      // A transient Alpaca hiccup shouldn't block the rest of the page — the
      // "Preview rebalance" button surfaces the same error if it recurs.
    }
  };

  const loadOrders = async () => {
    try {
      const res = await fetch(`${apiBase}/api/broker/orders`);
      if (!res.ok) return;
      const body = await res.json();
      setOrders(body.orders ?? []);
    } catch {
      // Non-fatal — the history table just stays empty.
    }
  };

  useEffect(() => {
    loadAccount();
    loadOrders();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase]);

  const runPreview = async () => {
    setPreviewing(true);
    setError(null);
    setResults(null);
    setPreview(null);
    try {
      const res = await fetch(`${apiBase}/api/broker/rebalance/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          holdings: holdings.filter((h) => h.ticker.trim() !== ""),
          period,
        }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail ?? `Request failed (${res.status})`);
      setPreview(body);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setPreviewing(false);
    }
  };

  const submit = async () => {
    if (!preview) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/api/broker/rebalance/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          orders: preview.orders.map((o) => ({
            ticker: o.ticker,
            side: o.side,
            notional: o.notional,
          })),
        }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail ?? `Request failed (${res.status})`);
      setResults(body.results);
      setPreview(null);
      loadAccount();
      loadOrders();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setSubmitting(false);
    }
  };

  if (!configured) {
    return (
      <div className="panel">
        <h2>Paper Trading (Alpaca)</h2>
        <p className="panel-sub">
          Not configured on this server. Set <code>ALPACA_API_KEY</code> and{" "}
          <code>ALPACA_SECRET_KEY</code> — a paper key pair from the Alpaca dashboard, not a live
          one — as environment variables on the API service to enable this panel.
        </p>
      </div>
    );
  }

  return (
    <div className="panel">
      <h2>Paper Trading (Alpaca)</h2>
      <p className="panel-sub">
        Sends real orders to your Alpaca <strong>paper</strong> account — simulated fills, no real
        money. "Preview rebalance" picks whichever allocation currently backtests best
        out-of-sample in the walk-forward test above and diffs it against your paper positions;
        nothing is sent to Alpaca until you confirm.
      </p>

      {account && (
        <div className="finding-box">
          <div className="finding-headline">Paper account</div>
          <p>
            Equity <strong>{money(account.equity)}</strong> · Cash{" "}
            <strong>{money(account.cash)}</strong> · Buying power{" "}
            <strong>{money(account.buying_power)}</strong>
          </p>
        </div>
      )}

      {error && <div className="error-box">{error}</div>}

      <div className="results-actions" style={{ marginBottom: 16 }}>
        <button className="primary-btn" onClick={runPreview} disabled={previewing}>
          {previewing ? "Computing rebalance…" : "Preview rebalance"}
        </button>
        {preview && (
          <button className="ghost-btn" onClick={submit} disabled={submitting}>
            {submitting ? "Sending…" : "Send to Alpaca (paper)"}
          </button>
        )}
      </div>

      {preview && (
        <>
          <p className="panel-sub">
            Best out-of-sample strategy: <strong>{preview.strategy_label}</strong> (Sharpe{" "}
            {preview.strategy_sharpe.toFixed(2)})
          </p>
          {preview.orders.length === 0 ? (
            <p className="panel-sub">
              Already within a dollar of target on every position — nothing to trade.
            </p>
          ) : (
            <div className="matrix-scroll">
              <table className="weights-table">
                <thead>
                  <tr>
                    <th>Ticker</th>
                    <th>Side</th>
                    <th>Notional</th>
                    <th>Current</th>
                    <th>Target</th>
                    <th>Target weight</th>
                  </tr>
                </thead>
                <tbody>
                  {preview.orders.map((o) => (
                    <tr key={o.ticker}>
                      <td>{o.ticker}</td>
                      <td className={o.side === "buy" ? "good" : "bad"}>{o.side.toUpperCase()}</td>
                      <td>{money(o.notional)}</td>
                      <td className="muted-cell">{money(o.current_value)}</td>
                      <td>{money(o.target_value)}</td>
                      <td className="muted-cell">{pct(preview.target_weights[o.ticker] ?? 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {results && (
        <>
          <p className="panel-sub">Order results</p>
          <div className="matrix-scroll">
            <table className="weights-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Side</th>
                  <th>Notional</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={`${r.ticker}-${i}`}>
                    <td>{r.ticker}</td>
                    <td className={r.side === "buy" ? "good" : "bad"}>{r.side.toUpperCase()}</td>
                    <td>{money(r.notional)}</td>
                    <td className={r.status === "failed" ? "bad" : undefined}>
                      {r.error ?? r.status}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {orders.length > 0 && (
        <>
          <p className="panel-sub">Recent orders</p>
          <div className="matrix-scroll">
            <table className="weights-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Side</th>
                  <th>Notional</th>
                  <th>Status</th>
                  <th>Submitted</th>
                </tr>
              </thead>
              <tbody>
                {orders.map((o) => (
                  <tr key={o.id}>
                    <td>{o.symbol}</td>
                    <td className={o.side === "buy" ? "good" : "bad"}>{o.side.toUpperCase()}</td>
                    <td>{o.notional != null ? money(o.notional) : "—"}</td>
                    <td className="muted-cell">{o.status}</td>
                    <td className="muted-cell">
                      {o.submitted_at ? new Date(o.submitted_at).toLocaleString() : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
