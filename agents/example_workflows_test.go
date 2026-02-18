// Copyright 2025 The NLP Odyssey Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package agents_test

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type EvaluationFeedback struct {
	Feedback string `json:"feedback"`
	Score    string `json:"score"`
}

type OutlineCheckerOutput struct {
	GoodQuality bool `json:"good_quality"`
	IsScifi     bool `json:"is_scifi"`
}

type MathHomeworkOutput struct {
	Reasoning      string `json:"reasoning"`
	IsMathHomework bool   `json:"is_math_homework"`
}

type MessageOutput struct {
	Reasoning string  `json:"reasoning"`
	Response  string  `json:"response"`
	UserName  string  `json:"user_name,omitempty"`
}

type AppContext struct {
	LanguagePreference string `json:"language_preference"`
}

func mustWorkflowJSON(t *testing.T, value any) string {
	t.Helper()
	raw, err := json.Marshal(value)
	require.NoError(t, err)
	return string(raw)
}

func languagePreferenceFromContext(ctx context.Context) (string, bool) {
	value, ok := agents.RunContextValueFromContext(ctx)
	if !ok {
		return "", false
	}
	switch v := value.(type) {
	case AppContext:
		return v.LanguagePreference, true
	case *AppContext:
		if v == nil {
			return "", false
		}
		return v.LanguagePreference, true
	default:
		return "", false
	}
}

func TestLLMAsJudgeLoopHandlesStructFeedback(t *testing.T) {
	outlineModel := agentstesting.NewFakeModel(false, nil)
	outlineModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Outline v1")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Outline v2")}},
	})

	judgeModel := agentstesting.NewFakeModel(false, nil)
	judgeModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFinalOutputMessage(mustWorkflowJSON(t, EvaluationFeedback{
				Feedback: "Add more suspense",
				Score:    "needs_improvement",
			})),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFinalOutputMessage(mustWorkflowJSON(t, EvaluationFeedback{
				Feedback: "Looks good",
				Score:    "pass",
			})),
		}},
	})

	outlineAgent := agents.New("outline").WithModelInstance(outlineModel)
	judgeAgent := agents.New("judge").
		WithModelInstance(judgeModel).
		WithOutputType(agents.OutputType[EvaluationFeedback]())

	conversation := []agents.TResponseInputItem{
		agentstesting.GetTextInputItem("Tell me a space story"),
	}
	var latestOutline string

	for _, step := range []struct {
		Outline string
		Score   string
	}{
		{Outline: "Outline v1", Score: "needs_improvement"},
		{Outline: "Outline v2", Score: "pass"},
	} {
		outlineResult, err := agents.Runner{}.RunInputs(t.Context(), outlineAgent, conversation)
		require.NoError(t, err)
		latestOutline = agents.ItemHelpers().TextMessageOutputs(outlineResult.NewItems)
		assert.Equal(t, step.Outline, latestOutline)

		conversation = outlineResult.ToInputList()

		judgeResult, err := agents.Runner{}.RunInputs(t.Context(), judgeAgent, conversation)
		require.NoError(t, err)
		feedback, ok := judgeResult.FinalOutput.(EvaluationFeedback)
		require.True(t, ok)
		assert.Equal(t, step.Score, feedback.Score)

		if feedback.Score == "pass" {
			break
		}
		conversation = append(conversation, agentstesting.GetTextInputItem(
			fmt.Sprintf("Feedback: %s", feedback.Feedback),
		))
	}

	assert.Equal(t, "Outline v2", latestOutline)
	assert.Len(t, conversation, 4)
	assert.Equal(t, agents.InputItems(conversation), judgeModel.LastTurnArgs.Input)
}

func TestParallelTranslationFlowReusesRunnerOutputs(t *testing.T) {
	translationModel := agentstesting.NewFakeModel(false, nil)
	translationModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Uno")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Dos")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Tres")}},
	})
	spanishAgent := agents.New("spanish_agent").WithModelInstance(translationModel)

	pickerModel := agentstesting.NewFakeModel(false, nil)
	pickerModel.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Pick: Dos")},
	})
	pickerAgent := agents.New("picker").WithModelInstance(pickerModel)

	translations := make([]string, 0, 3)
	for range 3 {
		result, err := agents.Runner{}.Run(t.Context(), spanishAgent, "Hello")
		require.NoError(t, err)
		translations = append(translations, agents.ItemHelpers().TextMessageOutputs(result.NewItems))
	}

	combined := strings.Join(translations, "\n\n")
	pickerResult, err := agents.Runner{}.Run(
		t.Context(),
		pickerAgent,
		fmt.Sprintf("Input: Hello\n\nTranslations:\n%s", combined),
	)
	require.NoError(t, err)

	assert.Equal(t, []string{"Uno", "Dos", "Tres"}, translations)
	assert.Equal(t, "Pick: Dos", pickerResult.FinalOutput)
	assert.Equal(
		t,
		agents.InputItems{agentstesting.GetTextInputItem(
			fmt.Sprintf("Input: Hello\n\nTranslations:\n%s", combined),
		)},
		pickerModel.LastTurnArgs.Input,
	)
}

func TestDeterministicStoryFlowStopsWhenCheckerBlocks(t *testing.T) {
	outlineModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Outline v1")},
	})
	checkerModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetFinalOutputMessage(
			mustWorkflowJSON(t, OutlineCheckerOutput{GoodQuality: false, IsScifi: true}),
		)},
	})
	storyModel := agentstesting.NewFakeModel(false, nil)
	storyModel.SetNextOutput(agentstesting.FakeModelTurnOutput{Error: fmt.Errorf("story should not run")})

	outlineAgent := agents.New("outline").WithModelInstance(outlineModel)
	checkerAgent := agents.New("checker").
		WithModelInstance(checkerModel).
		WithOutputType(agents.OutputType[OutlineCheckerOutput]())
	storyAgent := agents.New("story").WithModelInstance(storyModel)

	inputs := []agents.TResponseInputItem{agentstesting.GetTextInputItem("Sci-fi please")}
	outlineResult, err := agents.Runner{}.RunInputs(t.Context(), outlineAgent, inputs)
	require.NoError(t, err)

	checkerResult, err := agents.Runner{}.RunInputs(t.Context(), checkerAgent, outlineResult.ToInputList())
	require.NoError(t, err)
	decision, ok := checkerResult.FinalOutput.(OutlineCheckerOutput)
	require.True(t, ok)

	if decision.GoodQuality && decision.IsScifi {
		outlineText, ok := outlineResult.FinalOutput.(string)
		require.True(t, ok)
		_, _ = agents.Runner{}.Run(t.Context(), storyAgent, outlineText)
	}
	assert.Nil(t, storyModel.FirstTurnArgs)
}

func TestDeterministicStoryFlowRunsStoryOnPass(t *testing.T) {
	outlineModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Outline ready")},
	})
	checkerModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetFinalOutputMessage(
			mustWorkflowJSON(t, OutlineCheckerOutput{GoodQuality: true, IsScifi: true}),
		)},
	})
	storyModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Final story")},
	})

	outlineAgent := agents.New("outline").WithModelInstance(outlineModel)
	checkerAgent := agents.New("checker").
		WithModelInstance(checkerModel).
		WithOutputType(agents.OutputType[OutlineCheckerOutput]())
	storyAgent := agents.New("story").WithModelInstance(storyModel)

	inputs := []agents.TResponseInputItem{agentstesting.GetTextInputItem("Sci-fi please")}
	outlineResult, err := agents.Runner{}.RunInputs(t.Context(), outlineAgent, inputs)
	require.NoError(t, err)

	checkerResult, err := agents.Runner{}.RunInputs(t.Context(), checkerAgent, outlineResult.ToInputList())
	require.NoError(t, err)
	decision, ok := checkerResult.FinalOutput.(OutlineCheckerOutput)
	require.True(t, ok)
	assert.True(t, decision.GoodQuality)
	assert.True(t, decision.IsScifi)

	outlineText, ok := outlineResult.FinalOutput.(string)
	require.True(t, ok)
	storyResult, err := agents.Runner{}.Run(t.Context(), storyAgent, outlineText)
	require.NoError(t, err)
	assert.Equal(t, "Final story", storyResult.FinalOutput)
	assert.Equal(
		t,
		agents.InputItems{agentstesting.GetTextInputItem("Outline ready")},
		storyModel.LastTurnArgs.Input,
	)
}

func TestInputGuardrailAgentTripsAndReturnsInfo(t *testing.T) {
	guardrailModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetFinalOutputMessage(
			mustWorkflowJSON(t, MathHomeworkOutput{Reasoning: "math detected", IsMathHomework: true}),
		)},
	})
	guardrailAgent := agents.New("guardrail").
		WithModelInstance(guardrailModel).
		WithOutputType(agents.OutputType[MathHomeworkOutput]())

	mathGuardrail := agents.InputGuardrail{
		Name: "math_guardrail",
		GuardrailFunction: func(ctx context.Context, _ *agents.Agent, input agents.Input) (agents.GuardrailFunctionOutput, error) {
			var result *agents.RunResult
			var err error
			switch v := input.(type) {
			case agents.InputString:
				result, err = agents.Runner{}.Run(ctx, guardrailAgent, v.String())
			case agents.InputItems:
				result, err = agents.Runner{}.RunInputs(ctx, guardrailAgent, v)
			default:
				return agents.GuardrailFunctionOutput{}, fmt.Errorf("unexpected input type %T", input)
			}
			if err != nil {
				return agents.GuardrailFunctionOutput{}, err
			}
			output, ok := result.FinalOutput.(MathHomeworkOutput)
			if !ok {
				return agents.GuardrailFunctionOutput{}, fmt.Errorf("unexpected guardrail output type %T", result.FinalOutput)
			}
			return agents.GuardrailFunctionOutput{
				OutputInfo:        output,
				TripwireTriggered: output.IsMathHomework,
			}, nil
		},
	}

	mainModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Should not run")},
	})
	mainAgent := agents.New("main").WithModelInstance(mainModel).WithInputGuardrails([]agents.InputGuardrail{mathGuardrail})

	_, err := agents.Runner{}.Run(t.Context(), mainAgent, "Solve 2x+5=11")
	require.Error(t, err)

	var tripwireErr agents.InputGuardrailTripwireTriggeredError
	require.ErrorAs(t, err, &tripwireErr)
	info, ok := tripwireErr.GuardrailResult.Output.OutputInfo.(MathHomeworkOutput)
	require.True(t, ok)
	assert.True(t, info.IsMathHomework)
	assert.Equal(t, "math detected", info.Reasoning)
}

func TestOutputGuardrailBlocksSensitiveData(t *testing.T) {
	sensitiveCheck := agents.OutputGuardrail{
		Name: "sensitive_check",
		GuardrailFunction: func(_ context.Context, _ *agents.Agent, output any) (agents.GuardrailFunctionOutput, error) {
			msg, ok := output.(MessageOutput)
			if !ok {
				return agents.GuardrailFunctionOutput{}, fmt.Errorf("unexpected output type %T", output)
			}
			containsPhone := strings.Contains(msg.Response, "650") || strings.Contains(msg.Reasoning, "650")
			return agents.GuardrailFunctionOutput{
				OutputInfo:        map[string]any{"contains_phone": containsPhone},
				TripwireTriggered: containsPhone,
			}, nil
		},
	}

	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetFinalOutputMessage(
			mustWorkflowJSON(t, MessageOutput{
				Reasoning: "User shared phone 650-123-4567",
				Response:  "Thanks!",
				UserName:  "guest",
			}),
		)},
	})
	agent := agents.New("assistant").
		WithModelInstance(model).
		WithOutputType(agents.OutputType[MessageOutput]()).
		WithOutputGuardrails([]agents.OutputGuardrail{sensitiveCheck})

	_, err := agents.Runner{}.Run(t.Context(), agent, "My phone number is 650-123-4567.")
	require.Error(t, err)

	var tripwireErr agents.OutputGuardrailTripwireTriggeredError
	require.ErrorAs(t, err, &tripwireErr)
	info, ok := tripwireErr.GuardrailResult.Output.OutputInfo.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, true, info["contains_phone"])
}

func TestForcingToolUseBehaviorsAlignWithExample(t *testing.T) {
	type WeatherArgs struct {
		City string `json:"city"`
	}
	getWeather := agents.NewFunctionTool("get_weather", "", func(_ context.Context, args WeatherArgs) (string, error) {
		return fmt.Sprintf("%s: Sunny", args.City), nil
	})

	defaultModel := agentstesting.NewFakeModel(false, nil)
	defaultModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Tool call coming"),
			agentstesting.GetFunctionToolCall("get_weather", `{"city":"Tokyo"}`),
		}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Done after tool")}},
	})
	defaultAgent := agents.New("default").
		WithModelInstance(defaultModel).
		WithTools(getWeather).
		WithToolUseBehavior(agents.RunLLMAgain())

	defaultResult, err := agents.Runner{}.Run(t.Context(), defaultAgent, "Weather?")
	require.NoError(t, err)
	assert.Equal(t, "Done after tool", defaultResult.FinalOutput)
	assert.Len(t, defaultResult.RawResponses, 2)

	firstModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Tool call coming"),
			agentstesting.GetFunctionToolCall("get_weather", `{"city":"Paris"}`),
		},
	})
	firstAgent := agents.New("first").
		WithModelInstance(firstModel).
		WithTools(getWeather).
		WithToolUseBehavior(agents.StopOnFirstTool()).
		WithModelSettings(modelsettings.ModelSettings{ToolChoice: modelsettings.ToolChoiceRequired})

	firstResult, err := agents.Runner{}.Run(t.Context(), firstAgent, "Weather?")
	require.NoError(t, err)
	assert.Equal(t, "Paris: Sunny", firstResult.FinalOutput)
	assert.Len(t, firstResult.RawResponses, 1)

	customBehavior := agents.ToolsToFinalOutputFunction(func(_ context.Context, results []agents.FunctionToolResult) (agents.ToolsToFinalOutputResult, error) {
		return agents.ToolsToFinalOutputResult{
			IsFinalOutput: true,
			FinalOutput:   param.NewOpt[any](fmt.Sprintf("Custom:%v", results[0].Output)),
		}, nil
	})
	customModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Tool call coming"),
			agentstesting.GetFunctionToolCall("get_weather", `{"city":"Berlin"}`),
		},
	})
	customAgent := agents.New("custom").
		WithModelInstance(customModel).
		WithTools(getWeather).
		WithToolUseBehavior(customBehavior).
		WithModelSettings(modelsettings.ModelSettings{ToolChoice: modelsettings.ToolChoiceRequired})

	customResult, err := agents.Runner{}.Run(t.Context(), customAgent, "Weather?")
	require.NoError(t, err)
	assert.Equal(t, "Custom:Berlin: Sunny", customResult.FinalOutput)
}

func TestRoutingMultiTurnContinuesWithHandoffAgent(t *testing.T) {
	delegateModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Bonjour")},
	})
	delegateAgent := agents.New("delegate").WithModelInstance(delegateModel)

	triageModel := agentstesting.NewFakeModel(false, nil)
	triageModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetHandoffToolCall(delegateAgent, "", "")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("handoff completed")}},
	})
	triageAgent := agents.New("triage").
		WithModelInstance(triageModel).
		WithAgentHandoffs(delegateAgent)

	firstResult, err := agents.Runner{}.Run(t.Context(), triageAgent, "Help me in French")
	require.NoError(t, err)
	assert.Equal(t, "Bonjour", firstResult.FinalOutput)
	assert.Same(t, delegateAgent, firstResult.LastAgent)

	delegateModel.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Encore?")},
	})
	followUp := firstResult.ToInputList()
	followUp = append(followUp, agentstesting.GetTextInputItem("Encore!"))

	secondResult, err := agents.Runner{}.RunInputs(t.Context(), delegateAgent, followUp)
	require.NoError(t, err)
	assert.Equal(t, "Encore?", secondResult.FinalOutput)
	assert.Equal(t, agents.InputItems(followUp), delegateModel.LastTurnArgs.Input)
}

func TestAgentsAsToolsConditionalEnablingMatchesPreference(t *testing.T) {
	frenchSpanishEnabled := agents.FunctionToolEnablerFunc(func(ctx context.Context, _ *agents.Agent) (bool, error) {
		preference, ok := languagePreferenceFromContext(ctx)
		if !ok {
			return false, nil
		}
		return preference == "french_spanish" || preference == "european", nil
	})
	europeanEnabled := agents.FunctionToolEnablerFunc(func(ctx context.Context, _ *agents.Agent) (bool, error) {
		preference, ok := languagePreferenceFromContext(ctx)
		if !ok {
			return false, nil
		}
		return preference == "european", nil
	})

	scenarios := []struct {
		preference   string
		expectedTool []string
	}{
		{preference: "spanish_only", expectedTool: []string{"respond_spanish"}},
		{preference: "french_spanish", expectedTool: []string{"respond_spanish", "respond_french"}},
		{preference: "european", expectedTool: []string{"respond_spanish", "respond_french", "respond_italian"}},
	}

	for _, scenario := range scenarios {
		spanishModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("ES hola")},
		})
		spanishAgent := agents.New("spanish").WithModelInstance(spanishModel)

		frenchModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("FR bonjour")},
		})
		frenchAgent := agents.New("french").WithModelInstance(frenchModel)

		italianModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("IT ciao")},
		})
		italianAgent := agents.New("italian").WithModelInstance(italianModel)

		toolNames := slices.Clone(scenario.expectedTool)
		slices.Sort(toolNames)
		toolCalls := make([]agents.TResponseOutputItem, 0, len(toolNames))
		for _, toolName := range toolNames {
			toolCalls = append(toolCalls, agentstesting.GetFunctionToolCall(toolName, `{"input":"Hi"}`))
		}

		orchestratorModel := agentstesting.NewFakeModel(false, nil)
		orchestratorModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
			{Value: toolCalls},
			{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Done")}},
		})

		orchestrator := agents.New("orchestrator").
			WithModelInstance(orchestratorModel).
			WithTools(
				spanishAgent.AsTool(agents.AgentAsToolParams{
					ToolName:        "respond_spanish",
					ToolDescription: "Spanish",
					IsEnabled:       agents.FunctionToolEnabled(),
				}),
				frenchAgent.AsTool(agents.AgentAsToolParams{
					ToolName:        "respond_french",
					ToolDescription: "French",
					IsEnabled:       frenchSpanishEnabled,
				}),
				italianAgent.AsTool(agents.AgentAsToolParams{
					ToolName:        "respond_italian",
					ToolDescription: "Italian",
					IsEnabled:       europeanEnabled,
				}),
			).
			WithModelSettings(modelsettings.ModelSettings{ToolChoice: modelsettings.ToolChoiceRequired})

		ctx := agents.ContextWithRunContextValue(t.Context(), AppContext{LanguagePreference: scenario.preference})
		result, err := agents.Runner{}.Run(ctx, orchestrator, "Hello")
		require.NoError(t, err)
		assert.Equal(t, "Done", result.FinalOutput)

		assert.Equal(t, slices.Contains(scenario.expectedTool, "respond_spanish"), spanishModel.FirstTurnArgs != nil)
		assert.Equal(t, slices.Contains(scenario.expectedTool, "respond_french"), frenchModel.FirstTurnArgs != nil)
		assert.Equal(t, slices.Contains(scenario.expectedTool, "respond_italian"), italianModel.FirstTurnArgs != nil)
	}
}

func TestAgentsAsToolsOrchestratorRunsMultipleTranslations(t *testing.T) {
	spanishModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("ES hola")},
	})
	spanishAgent := agents.New("spanish").WithModelInstance(spanishModel)

	frenchModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("FR bonjour")},
	})
	frenchAgent := agents.New("french").WithModelInstance(frenchModel)

	orchestratorModel := agentstesting.NewFakeModel(false, nil)
	orchestratorModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("translate_to_spanish", `{"input":"Hi"}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("translate_to_french", `{"input":"Hi"}`),
		}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Summary complete")}},
	})

	orchestrator := agents.New("orchestrator").
		WithModelInstance(orchestratorModel).
		WithTools(
			spanishAgent.AsTool(agents.AgentAsToolParams{ToolName: "translate_to_spanish", ToolDescription: "Spanish"}),
			frenchAgent.AsTool(agents.AgentAsToolParams{ToolName: "translate_to_french", ToolDescription: "French"}),
		)

	result, err := agents.Runner{}.Run(t.Context(), orchestrator, "Hi")
	require.NoError(t, err)

	assert.Equal(t, "Summary complete", result.FinalOutput)
	assert.Equal(
		t,
		agents.InputItems{agentstesting.GetTextInputItem("Hi")},
		spanishModel.LastTurnArgs.Input,
	)
	assert.Equal(
		t,
		agents.InputItems{agentstesting.GetTextInputItem("Hi")},
		frenchModel.LastTurnArgs.Input,
	)
	assert.Len(t, result.RawResponses, 3)
}
