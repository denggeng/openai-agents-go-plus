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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func parityToolContext(toolArguments string) ToolContextData {
	return ToolContextData{
		ToolName:      "test_tool",
		ToolCallID:    "call_123",
		ToolArguments: toolArguments,
	}
}

func parityInputGuardrail(ctx context.Context, _ ToolInputGuardrailData) (ToolGuardrailFunctionOutput, error) {
	return ToolGuardrailAllow("ok"), nil
}

func parityOutputGuardrail(ctx context.Context, _ ToolOutputGuardrailData) (ToolGuardrailFunctionOutput, error) {
	return ToolGuardrailAllow("ok"), nil
}

func TestToolInputGuardrailRunVariants(t *testing.T) {
	data := ToolInputGuardrailData{
		Context: parityToolContext(`{"param":"value"}`),
		Agent:   &Agent{Name: "test"},
	}

	allowGuardrail := ToolInputGuardrail{
		GuardrailFunction: func(context.Context, ToolInputGuardrailData) (ToolGuardrailFunctionOutput, error) {
			return ToolGuardrailAllow(nil), nil
		},
	}
	allowResult, err := allowGuardrail.Run(t.Context(), data)
	require.NoError(t, err)
	assert.Equal(t, ToolGuardrailBehaviorTypeAllow, allowResult.Output.BehaviorType())
	assert.Nil(t, allowResult.Output.OutputInfo)

	raiseGuardrail := ToolInputGuardrail{
		GuardrailFunction: func(context.Context, ToolInputGuardrailData) (ToolGuardrailFunctionOutput, error) {
			return ToolGuardrailRaiseException("test_info"), nil
		},
	}
	raiseResult, err := raiseGuardrail.Run(t.Context(), data)
	require.NoError(t, err)
	assert.Equal(t, ToolGuardrailBehaviorTypeRaiseException, raiseResult.Output.BehaviorType())
	assert.Equal(t, "test_info", raiseResult.Output.OutputInfo)
}

func TestToolOutputGuardrailRunVariants(t *testing.T) {
	data := ToolOutputGuardrailData{
		ToolInputGuardrailData: ToolInputGuardrailData{
			Context: parityToolContext(`{"param":"value"}`),
			Agent:   &Agent{Name: "test"},
		},
		Output: "test output",
	}

	allowGuardrail := ToolOutputGuardrail{
		GuardrailFunction: func(context.Context, ToolOutputGuardrailData) (ToolGuardrailFunctionOutput, error) {
			return ToolGuardrailAllow(nil), nil
		},
	}
	allowResult, err := allowGuardrail.Run(t.Context(), data)
	require.NoError(t, err)
	assert.Equal(t, ToolGuardrailBehaviorTypeAllow, allowResult.Output.BehaviorType())
	assert.Nil(t, allowResult.Output.OutputInfo)

	raiseGuardrail := ToolOutputGuardrail{
		GuardrailFunction: func(context.Context, ToolOutputGuardrailData) (ToolGuardrailFunctionOutput, error) {
			return ToolGuardrailRaiseException("test_info"), nil
		},
	}
	raiseResult, err := raiseGuardrail.Run(t.Context(), data)
	require.NoError(t, err)
	assert.Equal(t, ToolGuardrailBehaviorTypeRaiseException, raiseResult.Output.BehaviorType())
	assert.Equal(t, "test_info", raiseResult.Output.OutputInfo)
}

func TestToolGuardrailNameResolution(t *testing.T) {
	inputGuardrail := ToolInputGuardrail{GuardrailFunction: parityInputGuardrail}
	assert.Equal(t, "parityInputGuardrail", inputGuardrail.GetName())

	outputGuardrail := ToolOutputGuardrail{GuardrailFunction: parityOutputGuardrail}
	assert.Equal(t, "parityOutputGuardrail", outputGuardrail.GetName())

	inputGuardrail.Name = "Custom input name"
	outputGuardrail.Name = "Custom output name"
	assert.Equal(t, "Custom input name", inputGuardrail.GetName())
	assert.Equal(t, "Custom output name", outputGuardrail.GetName())
}

func TestPasswordBlockingInputGuardrail(t *testing.T) {
	guardrail := ToolInputGuardrail{
		GuardrailFunction: func(_ context.Context, data ToolInputGuardrailData) (ToolGuardrailFunctionOutput, error) {
			if strings.Contains(strings.ToLower(data.Context.ToolArguments), "password") {
				return ToolGuardrailRejectContent(
					"Tool call blocked: contains password",
					map[string]any{"blocked_word": "password"},
				), nil
			}
			return ToolGuardrailAllow("safe_input"), nil
		},
	}

	data := ToolInputGuardrailData{
		Context: parityToolContext(`{"message": "Hello password world"}`),
		Agent:   &Agent{Name: "test"},
	}
	result, err := guardrail.Run(t.Context(), data)
	require.NoError(t, err)
	assert.Equal(t, ToolGuardrailBehaviorTypeRejectContent, result.Output.BehaviorType())
	assert.Equal(t, "Tool call blocked: contains password", result.Output.BehaviorMessage())
	assert.Equal(t, "password", result.Output.OutputInfo.(map[string]any)["blocked_word"])

	data = ToolInputGuardrailData{
		Context: parityToolContext(`{"message": "Hello safe world"}`),
		Agent:   &Agent{Name: "test"},
	}
	result, err = guardrail.Run(t.Context(), data)
	require.NoError(t, err)
	assert.Equal(t, ToolGuardrailBehaviorTypeAllow, result.Output.BehaviorType())
	assert.Equal(t, "safe_input", result.Output.OutputInfo)
}

func TestSSNBlockingOutputGuardrail(t *testing.T) {
	guardrail := ToolOutputGuardrail{
		GuardrailFunction: func(_ context.Context, data ToolOutputGuardrailData) (ToolGuardrailFunctionOutput, error) {
			outputStr := strings.ToLower(data.Output.(string))
			if strings.Contains(outputStr, "ssn") || strings.Contains(outputStr, "123-45-6789") {
				return ToolGuardrailRaiseException(map[string]any{"blocked_pattern": "SSN"}), nil
			}
			return ToolGuardrailAllow("safe_output"), nil
		},
	}

	data := ToolOutputGuardrailData{
		ToolInputGuardrailData: ToolInputGuardrailData{
			Context: parityToolContext(`{"param":"value"}`),
			Agent:   &Agent{Name: "test"},
		},
		Output: "User SSN is 123-45-6789",
	}
	result, err := guardrail.Run(t.Context(), data)
	require.NoError(t, err)
	assert.Equal(t, ToolGuardrailBehaviorTypeRaiseException, result.Output.BehaviorType())
	assert.Equal(t, "SSN", result.Output.OutputInfo.(map[string]any)["blocked_pattern"])

	data.Output = "User name is John Doe"
	result, err = guardrail.Run(t.Context(), data)
	require.NoError(t, err)
	assert.Equal(t, ToolGuardrailBehaviorTypeAllow, result.Output.BehaviorType())
	assert.Equal(t, "safe_output", result.Output.OutputInfo)
}

func TestMixedBehaviorOutputGuardrail(t *testing.T) {
	guardrail := ToolOutputGuardrail{
		GuardrailFunction: func(_ context.Context, data ToolOutputGuardrailData) (ToolGuardrailFunctionOutput, error) {
			outputStr := strings.ToLower(data.Output.(string))
			if strings.Contains(outputStr, "dangerous") {
				return ToolGuardrailRaiseException(map[string]any{"reason": "dangerous_content"}), nil
			}
			if strings.Contains(outputStr, "sensitive") {
				return ToolGuardrailRejectContent("Content was filtered", map[string]any{"reason": "sensitive_content"}), nil
			}
			return ToolGuardrailAllow(map[string]any{"status": "clean"}), nil
		},
	}

	data := ToolOutputGuardrailData{
		ToolInputGuardrailData: ToolInputGuardrailData{
			Context: parityToolContext(`{"param":"value"}`),
			Agent:   &Agent{Name: "test"},
		},
		Output: "This is dangerous content",
	}
	result, err := guardrail.Run(t.Context(), data)
	require.NoError(t, err)
	assert.Equal(t, ToolGuardrailBehaviorTypeRaiseException, result.Output.BehaviorType())
	assert.Equal(t, "dangerous_content", result.Output.OutputInfo.(map[string]any)["reason"])

	data.Output = "This is sensitive data"
	result, err = guardrail.Run(t.Context(), data)
	require.NoError(t, err)
	assert.Equal(t, ToolGuardrailBehaviorTypeRejectContent, result.Output.BehaviorType())
	assert.Equal(t, "Content was filtered", result.Output.BehaviorMessage())
	assert.Equal(t, "sensitive_content", result.Output.OutputInfo.(map[string]any)["reason"])

	data.Output = "This is clean content"
	result, err = guardrail.Run(t.Context(), data)
	require.NoError(t, err)
	assert.Equal(t, ToolGuardrailBehaviorTypeAllow, result.Output.BehaviorType())
	assert.Equal(t, "clean", result.Output.OutputInfo.(map[string]any)["status"])
}
