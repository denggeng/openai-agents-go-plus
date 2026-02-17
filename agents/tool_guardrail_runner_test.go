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
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunResultIncludesToolGuardrailResults(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)

	tool := agentstesting.GetFunctionTool("foo", "tool_output")
	tool.ToolInputGuardrails = []agents.ToolInputGuardrail{
		{
			Name: "tool_input_guard",
			GuardrailFunction: func(context.Context, agents.ToolInputGuardrailData) (agents.ToolGuardrailFunctionOutput, error) {
				return agents.ToolGuardrailAllow("input_checked"), nil
			},
		},
	}
	tool.ToolOutputGuardrails = []agents.ToolOutputGuardrail{
		{
			Name: "tool_output_guard",
			GuardrailFunction: func(context.Context, agents.ToolOutputGuardrailData) (agents.ToolGuardrailFunctionOutput, error) {
				return agents.ToolGuardrailRejectContent("filtered_by_guardrail", "output_checked"), nil
			},
		},
	}

	agent := agents.New("test").
		WithModelInstance(model).
		WithTools(tool)

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("foo", `{"a":"b"}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	result, err := agents.Runner{}.Run(t.Context(), agent, "hello")
	require.NoError(t, err)

	require.Len(t, result.ToolInputGuardrailResults, 1)
	require.Len(t, result.ToolOutputGuardrailResults, 1)
	assert.Equal(t, agents.ToolGuardrailBehaviorTypeAllow, result.ToolInputGuardrailResults[0].Output.BehaviorType())
	assert.Equal(t, agents.ToolGuardrailBehaviorTypeRejectContent, result.ToolOutputGuardrailResults[0].Output.BehaviorType())
	assert.Equal(t, "filtered_by_guardrail", result.ToolOutputGuardrailResults[0].Output.BehaviorMessage())
}

func TestRunResultStreamingIncludesToolGuardrailResults(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)

	tool := agentstesting.GetFunctionTool("foo", "tool_output")
	tool.ToolInputGuardrails = []agents.ToolInputGuardrail{
		{
			Name: "tool_input_guard",
			GuardrailFunction: func(context.Context, agents.ToolInputGuardrailData) (agents.ToolGuardrailFunctionOutput, error) {
				return agents.ToolGuardrailAllow("input_checked"), nil
			},
		},
	}
	tool.ToolOutputGuardrails = []agents.ToolOutputGuardrail{
		{
			Name: "tool_output_guard",
			GuardrailFunction: func(context.Context, agents.ToolOutputGuardrailData) (agents.ToolGuardrailFunctionOutput, error) {
				return agents.ToolGuardrailRejectContent("filtered_by_guardrail", "output_checked"), nil
			},
		},
	}

	agent := agents.New("test").
		WithModelInstance(model).
		WithTools(tool)

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("foo", `{"a":"b"}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "hello")
	require.NoError(t, err)

	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	require.Len(t, result.ToolInputGuardrailResults(), 1)
	require.Len(t, result.ToolOutputGuardrailResults(), 1)
	assert.Equal(t, agents.ToolGuardrailBehaviorTypeAllow, result.ToolInputGuardrailResults()[0].Output.BehaviorType())
	assert.Equal(t, agents.ToolGuardrailBehaviorTypeRejectContent, result.ToolOutputGuardrailResults()[0].Output.BehaviorType())
	assert.Equal(t, "filtered_by_guardrail", result.ToolOutputGuardrailResults()[0].Output.BehaviorMessage())
}
