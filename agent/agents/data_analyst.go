package agents

import (
	"context"
	_ "embed"
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"

	"github.com/EyobelG/quant-portfolio-analytics/agent/agents/quantmetrics"
)

//go:embed prompts/data_analyst.md
var dataAnalystInstruction string

// NewDataAnalyst builds the data analyst persona. It calls QuantFolio's own
// analytics backend (quant_metrics) for hard numbers.
//
// Gemini rejects a request that combines a built-in tool (e.g. GoogleSearch)
// with function calling in the same agent, so this cannot also carry a web
// search tool alongside quant_metrics.
func NewDataAnalyst(ctx context.Context, backendBaseURL string) (agent.Agent, error) {
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", nil)
	if err != nil {
		return nil, fmt.Errorf("data analyst model: %w", err)
	}

	quantTool, err := quantmetrics.New(backendBaseURL)
	if err != nil {
		return nil, fmt.Errorf("quant_metrics tool: %w", err)
	}

	dataAnalyst, err := llmagent.New(llmagent.Config{
		Name:        "data_analyst_agent",
		Model:       model,
		Description: "Pull quantitative metrics for a ticker from QuantFolio's backend and analyze them.",
		Instruction: dataAnalystInstruction,
		Tools:       []tool.Tool{quantTool},
		OutputKey:   "market_data_analysis_output",
	})
	if err != nil {
		return nil, fmt.Errorf("data analyst agent: %w", err)
	}
	return dataAnalyst, nil
}
