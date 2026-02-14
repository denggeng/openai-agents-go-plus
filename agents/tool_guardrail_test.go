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

package agents

import (
	"context"
	"fmt"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolGuardrailFunctionOutputHelpers(t *testing.T) {
	t.Run("default behavior is allow", func(t *testing.T) {
		out := ToolGuardrailFunctionOutput{OutputInfo: "safe"}
		assert.Equal(t, ToolGuardrailBehaviorTypeAllow, out.BehaviorType())
		assert.Equal(t, "", out.BehaviorMessage())
	})

	t.Run("allow helper", func(t *testing.T) {
		out := ToolGuardrailAllow("ok")
		assert.Equal(t, ToolGuardrailBehaviorTypeAllow, out.BehaviorType())
		assert.Equal(t, "ok", out.OutputInfo)
	})

	t.Run("reject helper", func(t *testing.T) {
		out := ToolGuardrailRejectContent("blocked", "reason")
		assert.Equal(t, ToolGuardrailBehaviorTypeRejectContent, out.BehaviorType())
		assert.Equal(t, "blocked", out.BehaviorMessage())
		assert.Equal(t, "reason", out.OutputInfo)
	})

	t.Run("raise_exception helper", func(t *testing.T) {
		out := ToolGuardrailRaiseException("danger")
		assert.Equal(t, ToolGuardrailBehaviorTypeRaiseException, out.BehaviorType())
		assert.Equal(t, "danger", out.OutputInfo)
	})
}

func TestToolInputGuardrailRun(t *testing.T) {
	guardrail := ToolInputGuardrail{
		Name: "input_guard",
		GuardrailFunction: func(_ context.Context, data ToolInputGuardrailData) (ToolGuardrailFunctionOutput, error) {
			assert.Equal(t, "tool_1", data.Context.ToolName)
			assert.Equal(t, "call_1", data.Context.ToolCallID)
			assert.Equal(t, `{"value":"x"}`, data.Context.ToolArguments)
			return ToolGuardrailAllow("checked"), nil
		},
	}

	result, err := guardrail.Run(t.Context(), ToolInputGuardrailData{
		Context: ToolContextData{
			ToolName:      "tool_1",
			ToolCallID:    "call_1",
			ToolArguments: `{"value":"x"}`,
		},
		Agent: &Agent{Name: "agent"},
	})
	require.NoError(t, err)
	assert.Equal(t, ToolGuardrailBehaviorTypeAllow, result.Output.BehaviorType())
	assert.Equal(t, "checked", result.Output.OutputInfo)
	assert.Equal(t, "input_guard", result.Guardrail.GetName())
}

func TestToolInputGuardrailRunNilFunction(t *testing.T) {
	guardrail := ToolInputGuardrail{Name: "broken_input_guard"}
	_, err := guardrail.Run(t.Context(), ToolInputGuardrailData{
		Context: ToolContextData{
			ToolName:      "tool_1",
			ToolCallID:    "call_1",
			ToolArguments: "{}",
		},
		Agent: &Agent{Name: "agent"},
	})
	require.Error(t, err)
	var userErr UserError
	require.ErrorAs(t, err, &userErr)
}

func TestToolOutputGuardrailRunNilFunction(t *testing.T) {
	guardrail := ToolOutputGuardrail{Name: "broken_output_guard"}
	_, err := guardrail.Run(t.Context(), ToolOutputGuardrailData{
		ToolInputGuardrailData: ToolInputGuardrailData{
			Context: ToolContextData{
				ToolName:      "tool_1",
				ToolCallID:    "call_1",
				ToolArguments: "{}",
			},
			Agent: &Agent{Name: "agent"},
		},
		Output: "raw",
	})
	require.Error(t, err)
	var userErr UserError
	require.ErrorAs(t, err, &userErr)
}

func TestToolGuardrailTripwireErrors(t *testing.T) {
	t.Run("input tripwire", func(t *testing.T) {
		guardrail := ToolInputGuardrail{Name: "danger_input"}
		output := ToolGuardrailRaiseException("test")
		err := NewToolInputGuardrailTripwireTriggeredError(guardrail, output)

		assert.Equal(t, "danger_input", err.Guardrail.Name)
		assert.Equal(t, output, err.Output)
		assert.Contains(t, err.Error(), "tool input guardrail")
	})

	t.Run("output tripwire", func(t *testing.T) {
		guardrail := ToolOutputGuardrail{Name: "danger_output"}
		output := ToolGuardrailRaiseException("test")
		err := NewToolOutputGuardrailTripwireTriggeredError(guardrail, output)

		assert.Equal(t, "danger_output", err.Guardrail.Name)
		assert.Equal(t, output, err.Output)
		assert.Contains(t, err.Error(), "tool output guardrail")
	})
}

func TestToolInputGuardrailRunsOnInvalidJSON(t *testing.T) {
	type EchoArgs struct {
		Value string `json:"value"`
	}

	guardrailCalls := make([]string, 0, 1)
	guardrail := ToolInputGuardrail{
		Name: "input_guard",
		GuardrailFunction: func(_ context.Context, data ToolInputGuardrailData) (ToolGuardrailFunctionOutput, error) {
			guardrailCalls = append(guardrailCalls, data.Context.ToolArguments)
			return ToolGuardrailAllow("checked"), nil
		},
	}

	tool := NewFunctionTool("guarded", "", func(_ context.Context, args EchoArgs) (string, error) {
		return args.Value, nil
	})
	tool.ToolInputGuardrails = []ToolInputGuardrail{guardrail}

	agent := &Agent{
		Name:  "test",
		Tools: []Tool{tool},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getFunctionToolCall("guarded", "bad_json", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}

	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent,
		response: response,
	})

	require.Equal(t, []string{"bad_json"}, guardrailCalls)
	require.Len(t, result.ToolInputGuardrailResults, 1)
	assert.Equal(t, "checked", result.ToolInputGuardrailResults[0].Output.OutputInfo)

	require.Len(t, result.GeneratedItems(), 2)
	outputItem, ok := result.GeneratedItems()[1].(ToolCallOutputItem)
	require.True(t, ok)
	assert.Contains(t, fmt.Sprint(outputItem.Output), "An error occurred while running the tool")
}

func TestToolInputGuardrailRejectContentSkipsToolInvocation(t *testing.T) {
	invoked := false
	guardrail := ToolInputGuardrail{
		Name: "input_guard",
		GuardrailFunction: func(_ context.Context, _ ToolInputGuardrailData) (ToolGuardrailFunctionOutput, error) {
			return ToolGuardrailRejectContent("blocked by guardrail", map[string]any{"code": "blocked"}), nil
		},
	}

	tool := FunctionTool{
		Name:             "guarded",
		ParamsJSONSchema: map[string]any{"type": "object"},
		OnInvokeTool: func(context.Context, string) (any, error) {
			invoked = true
			return "tool_output", nil
		},
		ToolInputGuardrails: []ToolInputGuardrail{guardrail},
	}

	agent := &Agent{
		Name:  "test",
		Tools: []Tool{tool},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getFunctionToolCall("guarded", `{"value":"x"}`, ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}

	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent,
		response: response,
	})

	assert.False(t, invoked)
	require.Len(t, result.ToolInputGuardrailResults, 1)
	assert.Equal(t, ToolGuardrailBehaviorTypeRejectContent, result.ToolInputGuardrailResults[0].Output.BehaviorType())
	assertItemIsFunctionToolCallOutput(t, result.GeneratedItems()[1], agent, "blocked by guardrail", "")
}

func TestToolOutputGuardrailRejectContentReplacesToolOutput(t *testing.T) {
	tool := FunctionTool{
		Name:             "guarded",
		ParamsJSONSchema: map[string]any{"type": "object"},
		OnInvokeTool: func(context.Context, string) (any, error) {
			return "tool_output", nil
		},
		ToolOutputGuardrails: []ToolOutputGuardrail{
			{
				Name: "output_guard",
				GuardrailFunction: func(_ context.Context, data ToolOutputGuardrailData) (ToolGuardrailFunctionOutput, error) {
					assert.Equal(t, "tool_output", data.Output)
					return ToolGuardrailRejectContent("filtered output", map[string]any{"reason": "sensitive"}), nil
				},
			},
		},
	}

	agent := &Agent{
		Name:  "test",
		Tools: []Tool{tool},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getFunctionToolCall("guarded", `{"value":"x"}`, ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}

	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent,
		response: response,
	})

	require.Len(t, result.ToolOutputGuardrailResults, 1)
	assert.Equal(t, ToolGuardrailBehaviorTypeRejectContent, result.ToolOutputGuardrailResults[0].Output.BehaviorType())
	assertItemIsFunctionToolCallOutput(t, result.GeneratedItems()[1], agent, "filtered output", "")
}

func TestToolGuardrailResultsFollowToolCallOrder(t *testing.T) {
	inputRelease := make(chan struct{})
	outputRelease := make(chan struct{})

	slowTool := FunctionTool{
		Name:             "slow",
		ParamsJSONSchema: map[string]any{"type": "object"},
		OnInvokeTool: func(context.Context, string) (any, error) {
			return "slow_output", nil
		},
		ToolInputGuardrails: []ToolInputGuardrail{
			{
				Name: "slow_input",
				GuardrailFunction: func(context.Context, ToolInputGuardrailData) (ToolGuardrailFunctionOutput, error) {
					<-inputRelease
					return ToolGuardrailAllow("slow_input_checked"), nil
				},
			},
		},
		ToolOutputGuardrails: []ToolOutputGuardrail{
			{
				Name: "slow_output",
				GuardrailFunction: func(context.Context, ToolOutputGuardrailData) (ToolGuardrailFunctionOutput, error) {
					<-outputRelease
					return ToolGuardrailAllow("slow_output_checked"), nil
				},
			},
		},
	}

	fastTool := FunctionTool{
		Name:             "fast",
		ParamsJSONSchema: map[string]any{"type": "object"},
		OnInvokeTool: func(context.Context, string) (any, error) {
			return "fast_output", nil
		},
		ToolInputGuardrails: []ToolInputGuardrail{
			{
				Name: "fast_input",
				GuardrailFunction: func(context.Context, ToolInputGuardrailData) (ToolGuardrailFunctionOutput, error) {
					close(inputRelease)
					return ToolGuardrailAllow("fast_input_checked"), nil
				},
			},
		},
		ToolOutputGuardrails: []ToolOutputGuardrail{
			{
				Name: "fast_output",
				GuardrailFunction: func(context.Context, ToolOutputGuardrailData) (ToolGuardrailFunctionOutput, error) {
					close(outputRelease)
					return ToolGuardrailAllow("fast_output_checked"), nil
				},
			},
		},
	}

	agent := &Agent{
		Name:  "test",
		Tools: []Tool{slowTool, fastTool},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getFunctionToolCall("slow", `{"value":"x"}`, ""),
			getFunctionToolCall("fast", `{"value":"y"}`, ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}

	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent,
		response: response,
	})

	require.Len(t, result.ToolInputGuardrailResults, 2)
	assert.Equal(t, "slow_input", result.ToolInputGuardrailResults[0].Guardrail.Name)
	assert.Equal(t, "fast_input", result.ToolInputGuardrailResults[1].Guardrail.Name)

	require.Len(t, result.ToolOutputGuardrailResults, 2)
	assert.Equal(t, "slow_output", result.ToolOutputGuardrailResults[0].Guardrail.Name)
	assert.Equal(t, "fast_output", result.ToolOutputGuardrailResults[1].Guardrail.Name)
}

func TestToolInputGuardrailRaiseExceptionReturnsError(t *testing.T) {
	invoked := false
	tool := FunctionTool{
		Name:             "guarded",
		ParamsJSONSchema: map[string]any{"type": "object"},
		OnInvokeTool: func(context.Context, string) (any, error) {
			invoked = true
			return "tool_output", nil
		},
		ToolInputGuardrails: []ToolInputGuardrail{
			{
				Name: "input_tripwire",
				GuardrailFunction: func(context.Context, ToolInputGuardrailData) (ToolGuardrailFunctionOutput, error) {
					return ToolGuardrailRaiseException("input blocked"), nil
				},
			},
		},
	}

	agent := &Agent{
		Name:  "test",
		Tools: []Tool{tool},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getFunctionToolCall("guarded", `{"value":"x"}`, ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}

	result, err := getExecuteResultAllowingError(t, getExecuteResultParams{
		agent:    agent,
		response: response,
	})
	require.Error(t, err)
	assert.Nil(t, result)
	assert.False(t, invoked)

	var tripwireErr ToolInputGuardrailTripwireTriggeredError
	require.ErrorAs(t, err, &tripwireErr)
	assert.Equal(t, "input_tripwire", tripwireErr.Guardrail.Name)
}

func TestToolOutputGuardrailRaiseExceptionReturnsError(t *testing.T) {
	tool := FunctionTool{
		Name:             "guarded",
		ParamsJSONSchema: map[string]any{"type": "object"},
		OnInvokeTool: func(context.Context, string) (any, error) {
			return "tool_output", nil
		},
		ToolOutputGuardrails: []ToolOutputGuardrail{
			{
				Name: "output_tripwire",
				GuardrailFunction: func(context.Context, ToolOutputGuardrailData) (ToolGuardrailFunctionOutput, error) {
					return ToolGuardrailRaiseException("output blocked"), nil
				},
			},
		},
	}

	agent := &Agent{
		Name:  "test",
		Tools: []Tool{tool},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getFunctionToolCall("guarded", `{"value":"x"}`, ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}

	result, err := getExecuteResultAllowingError(t, getExecuteResultParams{
		agent:    agent,
		response: response,
	})
	require.Error(t, err)
	assert.Nil(t, result)

	var tripwireErr ToolOutputGuardrailTripwireTriggeredError
	require.ErrorAs(t, err, &tripwireErr)
	assert.Equal(t, "output_tripwire", tripwireErr.Guardrail.Name)
}

func getExecuteResultAllowingError(t *testing.T, params getExecuteResultParams) (*SingleStepResult, error) {
	t.Helper()

	handoffs, err := Runner{}.getHandoffs(t.Context(), params.agent)
	require.NoError(t, err)

	allTools, err := params.agent.GetAllTools(t.Context())
	require.NoError(t, err)

	processedResponse, err := RunImpl().ProcessModelResponse(
		t.Context(),
		params.agent,
		allTools,
		params.response,
		handoffs,
	)
	require.NoError(t, err)

	hooks := params.hooks
	if hooks == nil {
		hooks = NoOpRunHooks{}
	}

	input := params.originalInput
	if input == nil {
		input = InputString("hello")
	}

	return RunImpl().ExecuteToolsAndSideEffects(
		t.Context(),
		params.agent,
		input,
		params.generatedItems,
		params.response,
		*processedResponse,
		params.agent.OutputType,
		hooks,
		params.runConfig,
	)
}
