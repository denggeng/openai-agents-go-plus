package agents_test

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func float64Ptr(v float64) *float64 { return &v }

func TestFunctionToolInvokeTimeoutReturnsDefaultMessage(t *testing.T) {
	tool := agents.FunctionTool{
		Name:             "slow_tool",
		Description:      "slow",
		ParamsJSONSchema: map[string]any{"type": "object", "properties": map[string]any{}},
		OnInvokeTool: func(context.Context, string) (any, error) {
			time.Sleep(200 * time.Millisecond)
			return "slow", nil
		},
		TimeoutSeconds: float64Ptr(0.01),
	}

	result, err := tool.Invoke(t.Context(), "{}")
	require.NoError(t, err)

	output, ok := result.(string)
	require.True(t, ok)
	assert.Contains(t, strings.ToLower(output), "timed out")
	assert.Contains(t, output, "0.01")
}

func TestFunctionToolInvokeTimeoutUsesCustomErrorFunction(t *testing.T) {
	timeoutErrFn := agents.ToolErrorFunction(func(_ context.Context, err error) (any, error) {
		var timeoutErr agents.ToolTimeoutError
		require.ErrorAs(t, err, &timeoutErr)
		return "custom_timeout:" + timeoutErr.ToolName + ":0.01", nil
	})

	tool := agents.FunctionTool{
		Name:             "slow_tool",
		Description:      "slow",
		ParamsJSONSchema: map[string]any{"type": "object", "properties": map[string]any{}},
		OnInvokeTool: func(context.Context, string) (any, error) {
			time.Sleep(200 * time.Millisecond)
			return "slow", nil
		},
		TimeoutSeconds:       float64Ptr(0.01),
		TimeoutErrorFunction: &timeoutErrFn,
	}

	result, err := tool.Invoke(t.Context(), "{}")
	require.NoError(t, err)
	assert.Equal(t, "custom_timeout:slow_tool:0.01", result)
}

func TestFunctionToolInvokeTimeoutCanRaiseException(t *testing.T) {
	tool := agents.FunctionTool{
		Name:             "slow_tool",
		Description:      "slow",
		ParamsJSONSchema: map[string]any{"type": "object", "properties": map[string]any{}},
		OnInvokeTool: func(context.Context, string) (any, error) {
			time.Sleep(200 * time.Millisecond)
			return "slow", nil
		},
		TimeoutSeconds:  float64Ptr(0.01),
		TimeoutBehavior: agents.ToolTimeoutBehaviorRaiseException,
	}

	_, err := tool.Invoke(t.Context(), "{}")
	require.Error(t, err)
	var timeoutErr agents.ToolTimeoutError
	require.ErrorAs(t, err, &timeoutErr)
	assert.Equal(t, "slow_tool", timeoutErr.ToolName)
	assert.InDelta(t, 0.01, timeoutErr.TimeoutSeconds, 1e-9)
}

func TestFunctionToolInvokeDoesNotRewriteToolRaisedTimeoutError(t *testing.T) {
	internalErr := errors.New("tool_internal_timeout")
	tool := agents.FunctionTool{
		Name:             "timeout_tool",
		Description:      "timeout",
		ParamsJSONSchema: map[string]any{"type": "object", "properties": map[string]any{}},
		OnInvokeTool: func(context.Context, string) (any, error) {
			return nil, internalErr
		},
		TimeoutSeconds: float64Ptr(1.0),
	}

	_, err := tool.Invoke(t.Context(), "{}")
	require.ErrorIs(t, err, internalErr)
}

func TestFunctionToolTimeoutRejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name            string
		timeoutSeconds  *float64
		timeoutBehavior agents.ToolTimeoutBehavior
		wantMessage     string
	}{
		{
			name:           "zero timeout",
			timeoutSeconds: float64Ptr(0),
			wantMessage:    "greater than 0",
		},
		{
			name:           "negative timeout",
			timeoutSeconds: float64Ptr(-1),
			wantMessage:    "greater than 0",
		},
		{
			name:           "nan timeout",
			timeoutSeconds: float64Ptr(math.NaN()),
			wantMessage:    "finite number",
		},
		{
			name:           "inf timeout",
			timeoutSeconds: float64Ptr(math.Inf(1)),
			wantMessage:    "finite number",
		},
		{
			name:            "unsupported behavior",
			timeoutBehavior: agents.ToolTimeoutBehavior("unsupported"),
			wantMessage:     "must be one of",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			called := false
			tool := agents.FunctionTool{
				Name:             "bad_tool",
				Description:      "bad",
				ParamsJSONSchema: map[string]any{"type": "object", "properties": map[string]any{}},
				OnInvokeTool: func(context.Context, string) (any, error) {
					called = true
					return "ok", nil
				},
				TimeoutSeconds:  tt.timeoutSeconds,
				TimeoutBehavior: tt.timeoutBehavior,
			}

			_, err := tool.Invoke(t.Context(), "{}")
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantMessage)
			assert.False(t, called)
		})
	}
}
