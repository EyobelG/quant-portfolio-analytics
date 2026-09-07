// Command financial-advisor runs QuantFolio's AI Portfolio Strategist: a
// multi-agent financial advisor (coordinator + data analyst + three analyst
// personas) exposed over ADK's HTTP/SSE API, adapted from
// github.com/google/adk-samples/tree/main/go/agents/financial-advisor.
package main

import (
	"context"
	_ "embed"
	"log"
	"os"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"

	"github.com/EyobelG/quant-portfolio-analytics/agent/agents"
)

//go:embed agents/prompts/financial_coordinator.md
var instruction string

func main() {
	ctx := context.Background()

	backendBaseURL := os.Getenv("QUANTFOLIO_BACKEND_URL")
	if backendBaseURL == "" {
		backendBaseURL = "http://localhost:8000"
	}

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", nil)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	dataAnalyst, err := agents.NewDataAnalyst(ctx, backendBaseURL)
	if err != nil {
		log.Fatalf("Failed to create data analyst: %v", err)
	}
	growthStrategist, err := agents.NewGrowthStrategist(ctx)
	if err != nil {
		log.Fatalf("Failed to create growth strategist: %v", err)
	}
	macroRiskAssessor, err := agents.NewMacroRiskAssessor(ctx)
	if err != nil {
		log.Fatalf("Failed to create macro risk assessor: %v", err)
	}
	valueInvestor, err := agents.NewValueInvestor(ctx)
	if err != nil {
		log.Fatalf("Failed to create value investor: %v", err)
	}

	financialCoordinator, err := llmagent.New(llmagent.Config{
		Name:  "financial_coordinator",
		Model: model,
		Description: "guide users through a structured process to receive financial " +
			"advice by orchestrating a data analyst and three specialist analyst " +
			"personas (growth, macro risk, value) grounded in QuantFolio's own " +
			"quantitative analytics.",
		Instruction: instruction,
		OutputKey:   "financial_coordinator_output",
		Tools: []tool.Tool{
			agenttool.New(dataAnalyst, nil),
			agenttool.New(growthStrategist, nil),
			agenttool.New(macroRiskAssessor, nil),
			agenttool.New(valueInvestor, nil),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(financialCoordinator),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
