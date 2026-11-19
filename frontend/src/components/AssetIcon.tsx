import { useState } from "react";

// Crypto gets recognizable currency colors; major companies get their real
// logo (fetched from a public logo CDN, keyed by domain); everything else
// falls back to a monogram tinted by sector.

const CRYPTO_BRAND: Record<string, { bg: string; symbol: string }> = {
  "BTC-USD": { bg: "#f7931a", symbol: "₿" },
  "ETH-USD": { bg: "#627eea", symbol: "Ξ" },
  "SOL-USD": { bg: "#14f195", symbol: "◎" },
  "DOGE-USD": { bg: "#c2a633", symbol: "Ð" },
  "XRP-USD": { bg: "#23292f", symbol: "✕" },
  "ADA-USD": { bg: "#0033ad", symbol: "₳" },
  "LTC-USD": { bg: "#a6a9aa", symbol: "Ł" },
};

const LOGO_DOMAIN: Record<string, string> = {
  AAPL: "apple.com",
  MSFT: "microsoft.com",
  NVDA: "nvidia.com",
  AMZN: "amazon.com",
  GOOGL: "abc.xyz",
  GOOG: "abc.xyz",
  META: "meta.com",
  TSLA: "tesla.com",
  NFLX: "netflix.com",
  "BRK-B": "berkshirehathaway.com",
  JPM: "jpmorganchase.com",
  JNJ: "jnj.com",
  XOM: "exxonmobil.com",
  KO: "coca-colacompany.com",
  V: "visa.com",
  MA: "mastercard.com",
  WMT: "walmart.com",
  DIS: "disney.com",
  ORCL: "oracle.com",
  CRM: "salesforce.com",
  ADBE: "adobe.com",
  AMD: "amd.com",
  INTC: "intel.com",
  CSCO: "cisco.com",
  IBM: "ibm.com",
  ANT: "anthropic.com",
};

const SECTOR_TINT: Record<string, string> = {
  Technology: "#4f7cff",
  "Communication Services": "#7dd3fc",
  "Consumer Cyclical": "#fb923c",
  "Consumer Defensive": "#f0b429",
  Healthcare: "#2fbf71",
  "Financial Services": "#b07cff",
  Energy: "#dc5454",
  Industrials: "#8b95ab",
  Utilities: "#22d3ee",
  "Basic Materials": "#a3a3a3",
  "Real Estate": "#ff7ab6",
  "Fund / Index": "#4f7cff",
  Bonds: "#34d399",
  Commodities: "#facc15",
  Crypto: "#f7931a",
};

export default function AssetIcon({
  ticker,
  sector,
  size = 26,
}: {
  ticker: string;
  sector?: string | null;
  size?: number;
}) {
  const [logoFailed, setLogoFailed] = useState(false);
  const upper = ticker.toUpperCase();
  const brand = CRYPTO_BRAND[upper];
  const domain = LOGO_DOMAIN[upper];
  const bg = brand?.bg ?? SECTOR_TINT[sector ?? ""] ?? "#2a3142";
  // Crypto shows its currency symbol; everything else the first two letters.
  const label = brand?.symbol ?? ticker.replace(/[^A-Z]/gi, "").slice(0, 2).toUpperCase();

  if (domain && !logoFailed) {
    return (
      <img
        className="asset-icon asset-icon-logo"
        src={`https://logo.clearbit.com/${domain}?size=${size * 2}`}
        alt=""
        width={size}
        height={size}
        style={{ width: size, height: size, borderRadius: "50%", background: "#fff" }}
        onError={() => setLogoFailed(true)}
      />
    );
  }

  return (
    <span
      className="asset-icon"
      style={{
        width: size,
        height: size,
        background: bg,
        fontSize: brand ? size * 0.55 : size * 0.4,
      }}
      aria-hidden="true"
    >
      {label}
    </span>
  );
}
