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

package agentstesting

import (
	"context"
	"fmt"
	"reflect"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/denggeng/openai-agents-go-plus/tracing"
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

type FakeModel struct {
	TracingEnabled bool
	TurnOutputs    []FakeModelTurnOutput
	LastTurnArgs   FakeModelLastTurnArgs
	FirstTurnArgs  *FakeModelLastTurnArgs
	HardcodedUsage *usage.Usage
	ResponseID     string
}

type FakeModelTurnOutput struct {
	Value []agents.TResponseOutputItem
	Error error
}

type FakeModelLastTurnArgs struct {
	SystemInstructions param.Opt[string]
	Input              agents.Input
	ModelSettings      modelsettings.ModelSettings
	Tools              []agents.Tool
	OutputType         agents.OutputTypeInterface
	// optional
	PreviousResponseID string
	ConversationID     string
}

func NewFakeModel(tracingEnabled bool, initialOutput *FakeModelTurnOutput) *FakeModel {
	var turnOutputs []FakeModelTurnOutput
	if initialOutput != nil && !reflect.ValueOf(*initialOutput).IsZero() {
		turnOutputs = []FakeModelTurnOutput{*initialOutput}
	}

	return &FakeModel{
		TracingEnabled: tracingEnabled,
		TurnOutputs:    turnOutputs,
	}
}

func (m *FakeModel) SetHardcodedUsage(u usage.Usage) {
	m.HardcodedUsage = &u
}

func (m *FakeModel) SetNextOutput(output FakeModelTurnOutput) {
	m.TurnOutputs = append(m.TurnOutputs, output)
}

func (m *FakeModel) AddMultipleTurnOutputs(outputs []FakeModelTurnOutput) {
	m.TurnOutputs = append(m.TurnOutputs, outputs...)
}

func (m *FakeModel) GetNextOutput() FakeModelTurnOutput {
	if len(m.TurnOutputs) == 0 {
		return FakeModelTurnOutput{}
	}
	v := m.TurnOutputs[0]
	m.TurnOutputs = m.TurnOutputs[1:]
	return v
}

func (m *FakeModel) GetResponse(ctx context.Context, params agents.ModelResponseParams) (*agents.ModelResponse, error) {
	m.LastTurnArgs = FakeModelLastTurnArgs{
		SystemInstructions: params.SystemInstructions,
		Input:              params.Input,
		ModelSettings:      params.ModelSettings,
		Tools:              params.Tools,
		OutputType:         params.OutputType,
		PreviousResponseID: params.PreviousResponseID,
		ConversationID:     params.ConversationID,
	}
	if m.FirstTurnArgs == nil {
		first := m.LastTurnArgs
		m.FirstTurnArgs = &first
	}

	var modelResponse *agents.ModelResponse
	err := tracing.GenerationSpan(
		ctx, tracing.GenerationSpanParams{Disabled: !m.TracingEnabled},
		func(ctx context.Context, span tracing.Span) error {
			output := m.GetNextOutput()

			if err := output.Error; err != nil {
				span.SetError(tracing.SpanError{
					Message: "Error",
					Data: map[string]any{
						"name":    fmt.Sprintf("%T", err),
						"message": err.Error(),
					},
				})
				return err
			}

			u := m.HardcodedUsage
			if u == nil {
				u = usage.NewUsage()
			}

			responseID := m.ResponseID
			modelResponse = &agents.ModelResponse{
				Output:     output.Value,
				Usage:      u,
				ResponseID: responseID,
			}
			return nil
		},
	)
	if err != nil {
		return nil, err
	}
	return modelResponse, nil
}

func (m *FakeModel) StreamResponse(
	ctx context.Context,
	params agents.ModelResponseParams,
	yield agents.ModelStreamResponseCallback,
) error {
	m.LastTurnArgs = FakeModelLastTurnArgs{
		SystemInstructions: params.SystemInstructions,
		Input:              params.Input,
		ModelSettings:      params.ModelSettings,
		Tools:              params.Tools,
		OutputType:         params.OutputType,
		PreviousResponseID: params.PreviousResponseID,
		ConversationID:     params.ConversationID,
	}
	if m.FirstTurnArgs == nil {
		first := m.LastTurnArgs
		m.FirstTurnArgs = &first
	}

	return tracing.GenerationSpan(
		ctx, tracing.GenerationSpanParams{Disabled: !m.TracingEnabled},
		func(ctx context.Context, span tracing.Span) error {
			output := m.GetNextOutput()

			if err := output.Error; err != nil {
				span.SetError(tracing.SpanError{
					Message: "Error",
					Data: map[string]any{
						"name":    fmt.Sprintf("%T", err),
						"message": err.Error(),
					},
				})
				return err
			}

			responseID := m.ResponseID
			return yield(ctx, agents.TResponseStreamEvent{ // responses.ResponseCompletedEvent
				Response:       GetResponseObj(output.Value, responseID, m.HardcodedUsage),
				Type:           "response.completed",
				SequenceNumber: 0,
			})
		})
}

func GetResponseObj(
	output []agents.TResponseOutputItem,
	responseID string,
	u *usage.Usage,
) responses.Response {
	if responseID == "" {
		responseID = "123"
	}

	var responseUsage responses.ResponseUsage
	if u != nil {
		responseUsage = responses.ResponseUsage{
			InputTokens: int64(u.InputTokens),
			InputTokensDetails: responses.ResponseUsageInputTokensDetails{
				CachedTokens: 0,
			},
			OutputTokens: int64(u.OutputTokens),
			OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
				ReasoningTokens: 0,
			},
			TotalTokens: int64(u.TotalTokens),
		}
	}

	return responses.Response{
		ID:        responseID,
		CreatedAt: 123,
		Model:     "test_model",
		Object:    "response",
		Output:    output,
		ToolChoice: responses.ResponseToolChoiceUnion{
			OfToolChoiceMode: responses.ToolChoiceOptionsNone,
		},
		Tools:             nil,
		TopP:              0,
		ParallelToolCalls: false,
		Usage:             responseUsage,
	}
}
