// Copyright 2026 The NLP Odyssey Authors
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
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const shellRejectionMessage = DefaultApprovalRejectionMessage

func makeShellCall(callID string) map[string]any {
	return map[string]any{
		"type":    "shell_call",
		"id":      "shell_call",
		"call_id": callID,
		"status":  "completed",
		"action": map[string]any{
			"type":       "exec",
			"commands":   []string{"echo hi"},
			"timeout_ms": 1000,
		},
	}
}

func requireShellApproval(
	context.Context,
	*RunContextWrapper[any],
	ShellActionRequest,
	string,
) (bool, error) {
	return true, nil
}

func makeShellOnApprovalCallback(approve bool, reason string) ShellOnApprovalFunc {
	return func(_ *RunContextWrapper[any], _ ToolApprovalItem) (any, error) {
		payload := map[string]any{"approve": approve}
		if reason != "" {
			payload["reason"] = reason
		}
		return payload, nil
	}
}

func rejectShellToolCall(
	contextWrapper *RunContextWrapper[any],
	rawItem any,
	toolName string,
) ToolApprovalItem {
	approvalItem := ToolApprovalItem{
		ToolName: toolName,
		RawItem:  rawItem,
	}
	contextWrapper.RejectTool(approvalItem, false)
	return approvalItem
}

func TestShellToolDefaultsToLocalEnvironment(t *testing.T) {
	shellTool := ShellTool{
		Executor: func(context.Context, ShellCommandRequest) (any, error) {
			return "ok", nil
		},
	}

	require.NoError(t, shellTool.Normalize())
	assert.Equal(t, map[string]any{"type": "local"}, shellTool.Environment)
	assert.NotNil(t, shellTool.Executor)
}

func TestShellToolSupportsHostedEnvironmentWithoutExecutor(t *testing.T) {
	shellTool := ShellTool{
		Environment: map[string]any{
			"type":         "container_reference",
			"container_id": "cntr_123",
		},
	}

	require.NoError(t, shellTool.Normalize())
	assert.Equal(t, map[string]any{
		"type":         "container_reference",
		"container_id": "cntr_123",
	}, shellTool.Environment)
	assert.Nil(t, shellTool.Executor)
}

func TestShellToolNormalizesContainerAutoEnvironment(t *testing.T) {
	shellTool := ShellTool{
		Environment: map[string]any{
			"type":         "container_auto",
			"file_ids":     []string{"file_123"},
			"memory_limit": "4g",
			"network_policy": map[string]any{
				"type":            "allowlist",
				"allowed_domains": []string{"example.com"},
				"domain_secrets": []map[string]any{
					{
						"domain": "example.com",
						"name":   "API_TOKEN",
						"value":  "secret",
					},
				},
			},
			"skills": []map[string]any{
				{"type": "skill_reference", "skill_id": "skill_123", "version": "latest"},
				{
					"type":        "inline",
					"name":        "csv-workbench",
					"description": "Analyze CSV files.",
					"source": map[string]any{
						"type":       "base64",
						"media_type": "application/zip",
						"data":       "ZmFrZS16aXA=",
					},
				},
			},
		},
	}

	require.NoError(t, shellTool.Normalize())
	assert.Equal(t, map[string]any{
		"type":         "container_auto",
		"file_ids":     []string{"file_123"},
		"memory_limit": "4g",
		"network_policy": map[string]any{
			"type":            "allowlist",
			"allowed_domains": []string{"example.com"},
			"domain_secrets": []map[string]any{
				{
					"domain": "example.com",
					"name":   "API_TOKEN",
					"value":  "secret",
				},
			},
		},
		"skills": []map[string]any{
			{"type": "skill_reference", "skill_id": "skill_123", "version": "latest"},
			{
				"type":        "inline",
				"name":        "csv-workbench",
				"description": "Analyze CSV files.",
				"source": map[string]any{
					"type":       "base64",
					"media_type": "application/zip",
					"data":       "ZmFrZS16aXA=",
				},
			},
		},
	}, shellTool.Environment)
}

func TestShellToolRejectsLocalModeWithoutExecutor(t *testing.T) {
	shellTool := ShellTool{}
	err := shellTool.Normalize()
	assert.ErrorAs(t, err, &UserError{})
	assert.Contains(t, err.Error(), "requires an executor")

	shellTool = ShellTool{Environment: map[string]any{"type": "local"}}
	err = shellTool.Normalize()
	assert.ErrorAs(t, err, &UserError{})
	assert.Contains(t, err.Error(), "requires an executor")
}

func TestShellToolAllowsUnvalidatedHostedEnvironmentShapes(t *testing.T) {
	shellTool := ShellTool{Environment: map[string]any{"type": "container_reference"}}
	require.NoError(t, shellTool.Normalize())
	assert.Equal(t, map[string]any{"type": "container_reference"}, shellTool.Environment)

	shellTool = ShellTool{
		Environment: map[string]any{
			"type": "container_auto",
			"network_policy": map[string]any{
				"type":            "future_mode",
				"allowed_domains": []string{"example.com"},
				"some_new_field":  true,
			},
			"skills": []map[string]any{
				{"type": "skill_reference"},
			},
		},
	}
	require.NoError(t, shellTool.Normalize())
	assert.IsType(t, map[string]any{}, shellTool.Environment)
	assert.Equal(t, "container_auto", shellTool.Environment["type"])
}

func TestShellToolRejectsLocalExecutorAndApprovalForHostedEnvironment(t *testing.T) {
	shellTool := ShellTool{
		Executor: func(context.Context, ShellCommandRequest) (any, error) {
			return "ok", nil
		},
		Environment: map[string]any{
			"type":         "container_reference",
			"container_id": "cntr_123",
		},
	}
	err := shellTool.Normalize()
	assert.ErrorAs(t, err, &UserError{})
	assert.Contains(t, err.Error(), "does not accept an executor")

	shellTool = ShellTool{
		Environment: map[string]any{
			"type":         "container_reference",
			"container_id": "cntr_123",
		},
		NeedsApproval: ShellNeedsApprovalEnabled(),
	}
	err = shellTool.Normalize()
	assert.ErrorAs(t, err, &UserError{})
	assert.Contains(t, err.Error(), "does not support needs_approval or on_approval")

	shellTool = ShellTool{
		Environment: map[string]any{
			"type":         "container_reference",
			"container_id": "cntr_123",
		},
		OnApproval: func(_ *RunContextWrapper[any], _ ToolApprovalItem) (any, error) {
			return map[string]any{"approve": true}, nil
		},
	}
	err = shellTool.Normalize()
	assert.ErrorAs(t, err, &UserError{})
	assert.Contains(t, err.Error(), "does not support needs_approval or on_approval")
}

func TestExecuteShellCallsSurfacesMissingLocalExecutor(t *testing.T) {
	shellTool := ShellTool{
		Environment: map[string]any{
			"type":         "container_reference",
			"container_id": "cntr_123",
		},
	}
	require.NoError(t, shellTool.Normalize())

	toolRun := ToolRunShellCall{
		ToolCall:  makeShellCall("call_shell"),
		ShellTool: shellTool,
	}
	agent := New("shell-agent").WithTools(shellTool)
	contextWrapper := NewRunContextWrapper[any](nil)

	results, interruptions, err := RunImpl().ExecuteShellCalls(
		t.Context(),
		agent,
		[]ToolRunShellCall{toolRun},
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)
	assert.Empty(t, interruptions)
	require.Len(t, results, 1)

	outputItem, ok := results[0].(ToolCallOutputItem)
	require.True(t, ok)
	assert.Equal(t, "Shell tool has no local executor configured.", outputItem.Output)
	rawItem := outputItem.RawItem.(ShellCallOutputRawItem)
	assert.Equal(t, "shell_call_output", rawItem["type"])
	assert.Equal(t, "call_shell", rawItem["call_id"])
	assert.Equal(t, "failed", rawItem["status"])
}

func TestShellToolStructuredOutputIsRendered(t *testing.T) {
	shellTool := ShellTool{
		Executor: func(context.Context, ShellCommandRequest) (any, error) {
			return ShellResult{
				Output: []ShellCommandOutput{
					{
						Command: strPtr("echo hi"),
						Stdout:  "hi\n",
						Outcome: ShellCallOutcome{Type: ShellCallOutcomeExit, ExitCode: intPtr(0)},
					},
					{
						Command: strPtr("ls"),
						Stdout:  "README.md\nsrc\n",
						Stderr:  "warning",
						Outcome: ShellCallOutcome{Type: ShellCallOutcomeExit, ExitCode: intPtr(1)},
					},
				},
				ProviderData: map[string]any{"runner": "demo"},
				MaxOutputLen: intPtr(4096),
			}, nil
		},
	}
	require.NoError(t, shellTool.Normalize())

	toolCall := makeShellCall("call_shell")
	action := toolCall["action"].(map[string]any)
	action["commands"] = []string{"echo hi", "ls"}
	action["max_output_length"] = 4096

	toolRun := ToolRunShellCall{ToolCall: toolCall, ShellTool: shellTool}
	agent := New("shell-agent").WithTools(shellTool)
	contextWrapper := NewRunContextWrapper[any](nil)

	result, err := ShellAction().Execute(
		t.Context(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Contains(t, outputItem.Output, "$ echo hi")
	assert.Contains(t, outputItem.Output, "stderr:\nwarning")

	rawItem := outputItem.RawItem.(ShellCallOutputRawItem)
	assert.Equal(t, "shell_call_output", rawItem["type"])
	assert.Equal(t, "completed", rawItem["status"])
	assert.Equal(t, map[string]any{"runner": "demo"}, rawItem["provider_data"])
	assert.EqualValues(t, 4096, rawItem["max_output_length"])
	shellOutput := rawItem["shell_output"].([]any)
	assert.EqualValues(t, 1, shellOutput[1].(map[string]any)["exit_code"])
	assert.IsType(t, []any{}, rawItem["output"])
	firstOutput := rawItem["output"].([]any)[0].(map[string]any)
	assert.Contains(t, firstOutput["stdout"], "hi")
	assert.Equal(t, "exit", firstOutput["outcome"].(map[string]any)["type"])
	assert.EqualValues(t, 0, firstOutput["outcome"].(map[string]any)["exit_code"])
	_, hasCommand := firstOutput["command"]
	assert.False(t, hasCommand)

	inputPayload := outputItem.ToInputItem()
	payloadJSON, err := json.Marshal(inputPayload)
	require.NoError(t, err)
	var payload map[string]any
	require.NoError(t, json.Unmarshal(payloadJSON, &payload))
	assert.Equal(t, "shell_call_output", payload["type"])
	_, ok = payload["status"]
	assert.False(t, ok)
	_, ok = payload["shell_output"]
	assert.False(t, ok)
	_, ok = payload["provider_data"]
	assert.False(t, ok)
}

func TestShellToolExecutorFailureReturnsError(t *testing.T) {
	shellTool := ShellTool{
		Executor: func(context.Context, ShellCommandRequest) (any, error) {
			return nil, fmt.Errorf("%s", strings.Repeat("boom", 10))
		},
	}
	require.NoError(t, shellTool.Normalize())

	toolCall := map[string]any{
		"type":    "shell_call",
		"id":      "shell_call_fail",
		"call_id": "call_shell_fail",
		"status":  "completed",
		"action": map[string]any{
			"commands":          []string{"echo boom"},
			"timeout_ms":        1000,
			"max_output_length": 6,
		},
	}
	toolRun := ToolRunShellCall{ToolCall: toolCall, ShellTool: shellTool}
	agent := New("shell-agent").WithTools(shellTool)
	contextWrapper := NewRunContextWrapper[any](nil)

	result, err := ShellAction().Execute(
		t.Context(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Equal(t, "boombo", outputItem.Output)
	rawItem := outputItem.RawItem.(ShellCallOutputRawItem)
	assert.Equal(t, "shell_call_output", rawItem["type"])
	assert.Equal(t, "failed", rawItem["status"])
	assert.EqualValues(t, 6, rawItem["max_output_length"])
	assert.IsType(t, []any{}, rawItem["output"])
	assert.Equal(t, "boombo", rawItem["output"].([]any)[0].(map[string]any)["stdout"])
	firstOutput := rawItem["output"].([]any)[0].(map[string]any)
	assert.Equal(t, "exit", firstOutput["outcome"].(map[string]any)["type"])
	assert.EqualValues(t, 1, firstOutput["outcome"].(map[string]any)["exit_code"])
	inputPayload := outputItem.ToInputItem()
	payloadJSON, err := json.Marshal(inputPayload)
	require.NoError(t, err)
	var payload map[string]any
	require.NoError(t, json.Unmarshal(payloadJSON, &payload))
	assert.Equal(t, "shell_call_output", payload["type"])
	_, ok = payload["status"]
	assert.False(t, ok)
	_, ok = payload["shell_output"]
	assert.False(t, ok)
	_, ok = payload["provider_data"]
	assert.False(t, ok)
}

func TestShellToolOutputRespectsMaxOutputLength(t *testing.T) {
	shellTool := ShellTool{
		Executor: func(context.Context, ShellCommandRequest) (any, error) {
			return ShellResult{
				Output: []ShellCommandOutput{
					{
						Stdout:  "0123456789",
						Stderr:  "abcdef",
						Outcome: ShellCallOutcome{Type: ShellCallOutcomeExit, ExitCode: intPtr(0)},
					},
				},
			}, nil
		},
	}
	require.NoError(t, shellTool.Normalize())

	toolCall := map[string]any{
		"type":    "shell_call",
		"id":      "shell_call",
		"call_id": "call_shell",
		"status":  "completed",
		"action": map[string]any{
			"commands":          []string{"echo hi"},
			"timeout_ms":        1000,
			"max_output_length": 6,
		},
	}

	toolRun := ToolRunShellCall{ToolCall: toolCall, ShellTool: shellTool}
	agent := New("shell-agent").WithTools(shellTool)
	contextWrapper := NewRunContextWrapper[any](nil)

	result, err := ShellAction().Execute(
		t.Context(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Equal(t, "012345", outputItem.Output)
	rawItem := outputItem.RawItem.(ShellCallOutputRawItem)
	assert.EqualValues(t, 6, rawItem["max_output_length"])
	assert.Equal(t, "012345", rawItem["output"].([]any)[0].(map[string]any)["stdout"])
	assert.Equal(t, "", rawItem["output"].([]any)[0].(map[string]any)["stderr"])
}

func TestShellToolUsesSmallerMaxOutputLength(t *testing.T) {
	shellTool := ShellTool{
		Executor: func(context.Context, ShellCommandRequest) (any, error) {
			return ShellResult{
				Output: []ShellCommandOutput{
					{
						Stdout:  "0123456789",
						Stderr:  "abcdef",
						Outcome: ShellCallOutcome{Type: ShellCallOutcomeExit, ExitCode: intPtr(0)},
					},
				},
				MaxOutputLen: intPtr(8),
			}, nil
		},
	}
	require.NoError(t, shellTool.Normalize())

	toolCall := map[string]any{
		"type":    "shell_call",
		"id":      "shell_call",
		"call_id": "call_shell",
		"status":  "completed",
		"action": map[string]any{
			"commands":          []string{"echo hi"},
			"timeout_ms":        1000,
			"max_output_length": 6,
		},
	}

	toolRun := ToolRunShellCall{ToolCall: toolCall, ShellTool: shellTool}
	agent := New("shell-agent").WithTools(shellTool)
	contextWrapper := NewRunContextWrapper[any](nil)

	result, err := ShellAction().Execute(
		t.Context(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Equal(t, "012345", outputItem.Output)
	rawItem := outputItem.RawItem.(ShellCallOutputRawItem)
	assert.EqualValues(t, 6, rawItem["max_output_length"])
	assert.Equal(t, "012345", rawItem["output"].([]any)[0].(map[string]any)["stdout"])
	assert.Equal(t, "", rawItem["output"].([]any)[0].(map[string]any)["stderr"])
}

func TestShellToolExecutorCanOverrideMaxOutputLengthToZero(t *testing.T) {
	shellTool := ShellTool{
		Executor: func(context.Context, ShellCommandRequest) (any, error) {
			return ShellResult{
				Output: []ShellCommandOutput{
					{
						Stdout:  "0123456789",
						Stderr:  "abcdef",
						Outcome: ShellCallOutcome{Type: ShellCallOutcomeExit, ExitCode: intPtr(0)},
					},
				},
				MaxOutputLen: intPtr(0),
			}, nil
		},
	}
	require.NoError(t, shellTool.Normalize())

	toolCall := map[string]any{
		"type":    "shell_call",
		"id":      "shell_call",
		"call_id": "call_shell",
		"status":  "completed",
		"action": map[string]any{
			"commands":          []string{"echo hi"},
			"timeout_ms":        1000,
			"max_output_length": 6,
		},
	}

	toolRun := ToolRunShellCall{ToolCall: toolCall, ShellTool: shellTool}
	agent := New("shell-agent").WithTools(shellTool)
	contextWrapper := NewRunContextWrapper[any](nil)

	result, err := ShellAction().Execute(
		t.Context(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Equal(t, "", outputItem.Output)
	rawItem := outputItem.RawItem.(ShellCallOutputRawItem)
	assert.EqualValues(t, 0, rawItem["max_output_length"])
	assert.Equal(t, "", rawItem["output"].([]any)[0].(map[string]any)["stdout"])
	assert.Equal(t, "", rawItem["output"].([]any)[0].(map[string]any)["stderr"])
}

func TestShellToolActionCanRequestZeroMaxOutputLength(t *testing.T) {
	shellTool := ShellTool{
		Executor: func(context.Context, ShellCommandRequest) (any, error) {
			return ShellResult{
				Output: []ShellCommandOutput{
					{
						Stdout:  "0123456789",
						Stderr:  "abcdef",
						Outcome: ShellCallOutcome{Type: ShellCallOutcomeExit, ExitCode: intPtr(0)},
					},
				},
			}, nil
		},
	}
	require.NoError(t, shellTool.Normalize())

	toolCall := map[string]any{
		"type":    "shell_call",
		"id":      "shell_call",
		"call_id": "call_shell",
		"status":  "completed",
		"action": map[string]any{
			"commands":          []string{"echo hi"},
			"timeout_ms":        1000,
			"max_output_length": 0,
		},
	}

	toolRun := ToolRunShellCall{ToolCall: toolCall, ShellTool: shellTool}
	agent := New("shell-agent").WithTools(shellTool)
	contextWrapper := NewRunContextWrapper[any](nil)

	result, err := ShellAction().Execute(
		t.Context(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Equal(t, "", outputItem.Output)
	rawItem := outputItem.RawItem.(ShellCallOutputRawItem)
	assert.EqualValues(t, 0, rawItem["max_output_length"])
	assert.Equal(t, "", rawItem["output"].([]any)[0].(map[string]any)["stdout"])
	assert.Equal(t, "", rawItem["output"].([]any)[0].(map[string]any)["stderr"])
}

func TestShellToolActionNegativeMaxOutputLengthClampsToZero(t *testing.T) {
	shellTool := ShellTool{
		Executor: func(context.Context, ShellCommandRequest) (any, error) {
			return ShellResult{
				Output: []ShellCommandOutput{
					{
						Stdout:  "0123456789",
						Stderr:  "abcdef",
						Outcome: ShellCallOutcome{Type: ShellCallOutcomeExit, ExitCode: intPtr(0)},
					},
				},
			}, nil
		},
	}
	require.NoError(t, shellTool.Normalize())

	toolCall := map[string]any{
		"type":    "shell_call",
		"id":      "shell_call",
		"call_id": "call_shell",
		"status":  "completed",
		"action": map[string]any{
			"commands":          []string{"echo hi"},
			"timeout_ms":        1000,
			"max_output_length": -5,
		},
	}

	toolRun := ToolRunShellCall{ToolCall: toolCall, ShellTool: shellTool}
	agent := New("shell-agent").WithTools(shellTool)
	contextWrapper := NewRunContextWrapper[any](nil)

	result, err := ShellAction().Execute(
		t.Context(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Equal(t, "", outputItem.Output)
	rawItem := outputItem.RawItem.(ShellCallOutputRawItem)
	assert.EqualValues(t, 0, rawItem["max_output_length"])
	assert.Equal(t, "", rawItem["output"].([]any)[0].(map[string]any)["stdout"])
	assert.Equal(t, "", rawItem["output"].([]any)[0].(map[string]any)["stderr"])
}

func TestShellToolNeedsApprovalReturnsApprovalItem(t *testing.T) {
	shellTool := ShellTool{
		Executor:      func(context.Context, ShellCommandRequest) (any, error) { return "output", nil },
		NeedsApproval: ShellNeedsApprovalFunc(requireShellApproval),
	}
	require.NoError(t, shellTool.Normalize())

	toolRun := ToolRunShellCall{ToolCall: makeShellCall("call_shell"), ShellTool: shellTool}
	agent := New("shell-agent").WithTools(shellTool)
	contextWrapper := NewRunContextWrapper[any](nil)

	result, err := ShellAction().Execute(
		t.Context(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	approvalItem, ok := result.(ToolApprovalItem)
	require.True(t, ok)
	assert.Equal(t, "shell", approvalItem.ToolName)
}

func TestShellToolNeedsApprovalRejectedReturnsRejection(t *testing.T) {
	shellTool := ShellTool{
		Executor:      func(context.Context, ShellCommandRequest) (any, error) { return "output", nil },
		NeedsApproval: ShellNeedsApprovalFunc(requireShellApproval),
	}
	require.NoError(t, shellTool.Normalize())

	toolCall := makeShellCall("call_shell")
	toolRun := ToolRunShellCall{ToolCall: toolCall, ShellTool: shellTool}
	agent := New("shell-agent").WithTools(shellTool)
	contextWrapper := NewRunContextWrapper[any](nil)

	rejectShellToolCall(contextWrapper, toolCall, "shell")

	result, err := ShellAction().Execute(
		t.Context(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Contains(t, outputItem.Output, shellRejectionMessage)
	rawItem := outputItem.RawItem.(ShellCallOutputRawItem)
	outputList := rawItem["output"].([]any)
	assert.Len(t, outputList, 1)
	assert.Equal(t, shellRejectionMessage, outputList[0].(map[string]any)["stderr"])
}

func TestShellToolRejectionUsesRunLevelFormatter(t *testing.T) {
	shellTool := ShellTool{
		Executor:      func(context.Context, ShellCommandRequest) (any, error) { return "output", nil },
		NeedsApproval: ShellNeedsApprovalFunc(requireShellApproval),
	}
	require.NoError(t, shellTool.Normalize())

	toolCall := makeShellCall("call_shell")
	toolRun := ToolRunShellCall{ToolCall: toolCall, ShellTool: shellTool}
	agent := New("shell-agent").WithTools(shellTool)
	contextWrapper := NewRunContextWrapper[any](nil)

	rejectShellToolCall(contextWrapper, toolCall, "shell")

	result, err := ShellAction().Execute(
		t.Context(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{
			ToolErrorFormatter: func(args ToolErrorFormatterArgs) any {
				return fmt.Sprintf("%s denied (%s)", args.ToolName, args.CallID)
			},
		},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Equal(t, "shell denied (call_shell)", outputItem.Output)
	rawItem := outputItem.RawItem.(ShellCallOutputRawItem)
	assert.Equal(t, "shell denied (call_shell)", rawItem["output"].([]any)[0].(map[string]any)["stderr"])
}

func TestShellToolOnApprovalCallbackAutoApproves(t *testing.T) {
	shellTool := ShellTool{
		Executor:      func(context.Context, ShellCommandRequest) (any, error) { return "output", nil },
		NeedsApproval: ShellNeedsApprovalFunc(requireShellApproval),
		OnApproval:    makeShellOnApprovalCallback(true, ""),
	}
	require.NoError(t, shellTool.Normalize())

	toolRun := ToolRunShellCall{ToolCall: makeShellCall("call_shell"), ShellTool: shellTool}
	agent := New("shell-agent").WithTools(shellTool)
	contextWrapper := NewRunContextWrapper[any](nil)

	result, err := ShellAction().Execute(
		t.Context(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Equal(t, "output", outputItem.Output)
}

func TestShellToolOnApprovalCallbackAutoRejects(t *testing.T) {
	shellTool := ShellTool{
		Executor:      func(context.Context, ShellCommandRequest) (any, error) { return "output", nil },
		NeedsApproval: ShellNeedsApprovalFunc(requireShellApproval),
		OnApproval:    makeShellOnApprovalCallback(false, "Not allowed"),
	}
	require.NoError(t, shellTool.Normalize())

	toolRun := ToolRunShellCall{ToolCall: makeShellCall("call_shell"), ShellTool: shellTool}
	agent := New("shell-agent").WithTools(shellTool)
	contextWrapper := NewRunContextWrapper[any](nil)

	result, err := ShellAction().Execute(
		t.Context(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Contains(t, outputItem.Output, shellRejectionMessage)
}

func intPtr(v int) *int {
	return &v
}

func strPtr(v string) *string {
	return &v
}
