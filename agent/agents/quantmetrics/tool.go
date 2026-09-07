// Package quantmetrics wraps the QuantFolio FastAPI backend's single-ticker
// snapshot endpoint as an ADK function tool, so the data analyst persona
// pulls real quantitative numbers instead of only web search.
package quantmetrics

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// Args is the input schema for the quant_metrics tool.
type Args struct {
	Ticker string `json:"ticker"`           // ticker symbol, e.g. AAPL
	Period string `json:"period,omitempty"` // lookback window, e.g. 1y, 3y (default 3y)
}

// Result is the raw JSON snapshot returned by /api/ticker/{ticker}/metrics.
type Result map[string]any

// New builds the quant_metrics tool against the given backend base URL
// (e.g. https://quant-portfolio-api.onrender.com).
func New(backendBaseURL string) (tool.Tool, error) {
	client := &http.Client{Timeout: 20 * time.Second}

	fetch := func(_ agent.Context, args Args) (Result, error) {
		if args.Ticker == "" {
			return nil, fmt.Errorf("ticker is required")
		}
		period := args.Period
		if period == "" {
			period = "3y"
		}
		endpoint := fmt.Sprintf("%s/api/ticker/%s/metrics?period=%s",
			backendBaseURL, url.PathEscape(args.Ticker), url.QueryEscape(period))
		resp, err := client.Get(endpoint)
		if err != nil {
			return nil, fmt.Errorf("quant_metrics backend request failed: %w", err)
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("quant_metrics backend read failed: %w", err)
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("quant_metrics backend returned %d: %s", resp.StatusCode, string(body))
		}

		var result Result
		if err := json.Unmarshal(body, &result); err != nil {
			return nil, fmt.Errorf("quant_metrics backend returned invalid JSON: %w", err)
		}
		return result, nil
	}

	return functiontool.New(functiontool.Config{
		Name: "quant_metrics",
		Description: "Fetches a quantitative snapshot for one ticker from QuantFolio's own " +
			"analytics backend: last price, annualized return and volatility, max drawdown, " +
			"trailing P/E, dividend yield, market-regime classification, and factor loadings.",
	}, fetch)
}
