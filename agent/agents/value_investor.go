package agents

import (
	"context"
	_ "embed"
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/gemini"
)

//go:embed prompts/value_investor.md
var valueInvestorInstruction string

// NewValueInvestor builds the fundamental-value analyst persona, conditioned
// on the data analyst's output already present in session state.
func NewValueInvestor(ctx context.Context) (agent.Agent, error) {
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash-lite", nil)
	if err != nil {
		return nil, fmt.Errorf("value investor model: %w", err)
	}
	valueInvestor, err := llmagent.New(llmagent.Config{
		Name:        "value_investor_agent",
		Model:       model,
		Description: "Evaluate a ticker's fundamental value and margin of safety from quantitative data already gathered.",
		Instruction: valueInvestorInstruction,
		OutputKey:   "value_investor_output",
	})
	if err != nil {
		return nil, fmt.Errorf("value investor agent: %w", err)
	}
	return valueInvestor, nil
}
