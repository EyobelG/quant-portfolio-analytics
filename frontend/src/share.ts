import type { Holding } from "./types";

const VALID_PERIODS = new Set(["1y", "2y", "3y", "5y", "10y"]);

/**
 * Portfolios are shareable as `?p=AAPL:0.4,MSFT:0.3,JNJ:0.3&period=3y`, so a
 * specific allocation can be linked to directly instead of rebuilt by hand.
 */
export function encodePortfolio(holdings: Holding[], period: string): string {
  const p = holdings
    .filter((h) => h.ticker.trim() !== "")
    .map((h) => `${h.ticker}:${Number(h.weight.toFixed(4))}`)
    .join(",");
  return `?p=${encodeURIComponent(p)}&period=${period}`;
}

export function parsePortfolio(
  search: string
): { holdings: Holding[]; period: string } | null {
  const params = new URLSearchParams(search);
  const raw = params.get("p");
  if (!raw) return null;

  const holdings: Holding[] = [];
  for (const part of raw.split(",")) {
    const [ticker, weight] = part.split(":");
    const parsed = Number(weight);
    if (!ticker || !Number.isFinite(parsed) || parsed < 0 || parsed > 1) continue;
    holdings.push({ ticker: ticker.trim().toUpperCase(), weight: parsed });
  }
  if (holdings.length < 2) return null;

  const period = params.get("period") ?? "3y";
  return { holdings, period: VALID_PERIODS.has(period) ? period : "3y" };
}
