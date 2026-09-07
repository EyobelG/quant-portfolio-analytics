package agents

import (
	"context"
	_ "embed"
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/gemini"
)

//go:embed prompts/growth_strategist.md
var growthStrategistInstruction string

// NewGrowthStrategist builds the growth-focused analyst persona, conditioned
// on the data analyst's output already present in session state.
func NewGrowthStrategist(ctx context.Context) (agent.Agent, error) {
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash-lite", nil)
	if err != nil {
		return nil, fmt.Errorf("growth strategist model: %w", err)
	}
	growthStrategist, err := llmagent.New(llmagent.Config{
		Name:        "growth_strategist_agent",
		Model:       model,
		Description: "Evaluate a ticker's growth thesis from quantitative data already gathered.",
		Instruction: growthStrategistInstruction,
		OutputKey:   "growth_strategist_output",
	})
	if err != nil {
		return nil, fmt.Errorf("growth strategist agent: %w", err)
	}
	return growthStrategist, nil
}
