package agents

import (
	"context"
	_ "embed"
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/gemini"
)

//go:embed prompts/macro_risk_assessor.md
var macroRiskAssessorInstruction string

// NewMacroRiskAssessor builds the macro/systemic-risk analyst persona,
// conditioned on the data analyst's output already present in session state.
func NewMacroRiskAssessor(ctx context.Context) (agent.Agent, error) {
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash-lite", nil)
	if err != nil {
		return nil, fmt.Errorf("macro risk assessor model: %w", err)
	}
	macroRiskAssessor, err := llmagent.New(llmagent.Config{
		Name:        "macro_risk_assessor_agent",
		Model:       model,
		Description: "Evaluate a ticker's macro and systemic risk exposure from quantitative data already gathered.",
		Instruction: macroRiskAssessorInstruction,
		OutputKey:   "macro_risk_assessor_output",
	})
	if err != nil {
		return nil, fmt.Errorf("macro risk assessor agent: %w", err)
	}
	return macroRiskAssessor, nil
}
