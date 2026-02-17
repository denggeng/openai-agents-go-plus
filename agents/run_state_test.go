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
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunStateRoundTripToolGuardrailResults(t *testing.T) {
	inputGuardrail := agents.InputGuardrail{
		Name: "input_guardrail",
		GuardrailFunction: func(context.Context, *agents.Agent, agents.Input) (agents.GuardrailFunctionOutput, error) {
			return agents.GuardrailFunctionOutput{}, nil
		},
	}
	outputGuardrail := agents.OutputGuardrail{
		Name: "output_guardrail",
		GuardrailFunction: func(context.Context, *agents.Agent, any) (agents.GuardrailFunctionOutput, error) {
			return agents.GuardrailFunctionOutput{}, nil
		},
	}
	toolInputGuardrail := agents.ToolInputGuardrail{
		Name: "tool_input_guardrail",
		GuardrailFunction: func(context.Context, agents.ToolInputGuardrailData) (agents.ToolGuardrailFunctionOutput, error) {
			return agents.ToolGuardrailAllow(nil), nil
		},
	}
	toolOutputGuardrail := agents.ToolOutputGuardrail{
		Name: "tool_output_guardrail",
		GuardrailFunction: func(context.Context, agents.ToolOutputGuardrailData) (agents.ToolGuardrailFunctionOutput, error) {
			return agents.ToolGuardrailAllow(nil), nil
		},
	}

	runResult := agents.RunResult{
		Input: agents.InputItems{
			agentstesting.GetTextInputItem("hello"),
		},
		NewItems: nil,
		RawResponses: []agents.ModelResponse{
			{
				Output:     nil,
				Usage:      usage.NewUsage(),
				ResponseID: "resp_123",
			},
		},
		FinalOutput: "done",
		InputGuardrailResults: []agents.InputGuardrailResult{
			{
				Guardrail: inputGuardrail,
				Output: agents.GuardrailFunctionOutput{
					OutputInfo:        map[string]any{"input": "info"},
					TripwireTriggered: false,
				},
			},
		},
		OutputGuardrailResults: []agents.OutputGuardrailResult{
			{
				Guardrail: outputGuardrail,
				Output: agents.GuardrailFunctionOutput{
					OutputInfo:        map[string]any{"output": "info"},
					TripwireTriggered: true,
				},
			},
		},
		ToolInputGuardrailResults: []agents.ToolInputGuardrailResult{
			{
				Guardrail: toolInputGuardrail,
				Output:    agents.ToolGuardrailAllow(map[string]any{"tool_input": "info"}),
			},
		},
		ToolOutputGuardrailResults: []agents.ToolOutputGuardrailResult{
			{
				Guardrail: toolOutputGuardrail,
				Output:    agents.ToolGuardrailRejectContent("filtered", map[string]any{"tool_output": "info"}),
			},
		},
		Interruptions: []agents.ToolApprovalItem{
			{
				ToolName: "add",
				RawItem: map[string]any{
					"id":   "approval_1",
					"type": "mcp_approval_request",
					"name": "add",
				},
			},
		},
		LastAgent: &agents.Agent{Name: "test_agent"},
	}

	state := agents.NewRunStateFromResult(runResult, 2, 10)
	require.Equal(t, agents.CurrentRunStateSchemaVersion, state.SchemaVersion)
	require.Equal(t, "test_agent", state.CurrentAgentName)
	require.Equal(t, "resp_123", state.PreviousResponseID)

	encoded, err := state.ToJSON()
	require.NoError(t, err)

	decoded, err := agents.RunStateFromJSON(encoded)
	require.NoError(t, err)

	require.Len(t, decoded.InputGuardrailResults, 1)
	require.Len(t, decoded.OutputGuardrailResults, 1)
	require.Len(t, decoded.ToolInputGuardrailResults, 1)
	require.Len(t, decoded.ToolOutputGuardrailResults, 1)
	require.Len(t, decoded.Interruptions, 1)
	assert.Equal(t, "tool_input_guardrail", decoded.ToolInputGuardrailResults[0].Name)
	assert.Equal(t, agents.ToolGuardrailBehaviorTypeAllow, decoded.ToolInputGuardrailResults[0].Output.Behavior.Type)
	assert.Equal(t, "tool_output_guardrail", decoded.ToolOutputGuardrailResults[0].Name)
	assert.Equal(t, agents.ToolGuardrailBehaviorTypeRejectContent, decoded.ToolOutputGuardrailResults[0].Output.Behavior.Type)
	assert.Equal(t, "filtered", decoded.ToolOutputGuardrailResults[0].Output.Behavior.Message)
	assert.Equal(t, "add", decoded.Interruptions[0].ToolName)
}

func TestRunStateResumeHelpers(t *testing.T) {
	state := agents.RunState{
		SchemaVersion:      agents.CurrentRunStateSchemaVersion,
		MaxTurns:           42,
		PreviousResponseID: "resp_abc",
		OriginalInput: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem("original"),
		},
		GeneratedItems: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem("generated"),
		},
	}

	merged := state.ResumeInputItems()
	require.Len(t, merged, 2)

	cfg := state.ResumeRunConfig(agents.RunConfig{})
	assert.Equal(t, uint64(42), cfg.MaxTurns)
	assert.Equal(t, "resp_abc", cfg.PreviousResponseID)
}

func TestRunStateFromJSONRejectsUnsupportedSchemaVersion(t *testing.T) {
	_, err := agents.RunStateFromJSONString(`{"$schemaVersion":"9.9"}`)
	require.Error(t, err)
	assert.ErrorContains(t, err, "unsupported run state schema version")
}

func TestRunStateRoundTripToolApprovals(t *testing.T) {
	runContext := agents.NewRunContextWrapper[any](nil)
	runContext.ApproveTool(agents.ToolApprovalItem{
		ToolName: "tool_1",
		RawItem:  map[string]any{"call_id": "call-1"},
	}, false)
	runContext.RejectTool(agents.ToolApprovalItem{
		ToolName: "tool_2",
		RawItem:  map[string]any{"call_id": "call-2"},
	}, true)

	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
	}
	state.SetToolApprovalsFromContext(runContext)
	require.NotNil(t, state.ToolApprovals)

	encoded, err := state.ToJSON()
	require.NoError(t, err)

	decoded, err := agents.RunStateFromJSON(encoded)
	require.NoError(t, err)
	require.NotNil(t, decoded.ToolApprovals)

	restoredContext := agents.NewRunContextWrapper[any](nil)
	decoded.ApplyToolApprovalsToContext(restoredContext)

	approved, known := restoredContext.IsToolApproved("tool_1", "call-1")
	require.True(t, known)
	assert.True(t, approved)

	approved, known = restoredContext.IsToolApproved("tool_2", "any")
	require.True(t, known)
	assert.False(t, approved)
}

func TestRunStateApproveAndRejectTool(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
	}
	approvalItem := agents.ToolApprovalItem{
		ToolName: "add",
		RawItem: responses.ResponseOutputItemMcpApprovalRequest{
			ID:          "approval_1",
			Name:        "add",
			ServerLabel: "mcp_server",
			Type:        constant.ValueOf[constant.McpApprovalRequest](),
		},
	}

	err := state.ApproveTool(approvalItem)
	require.NoError(t, err)
	require.Len(t, state.GeneratedItems, 1)
	require.NotNil(t, state.ToolApprovals)
	assert.Equal(t, "mcp_approval_response", *state.GeneratedItems[0].GetType())
	assert.Equal(t, "approval_1", *state.GeneratedItems[0].GetApprovalRequestID())
	assert.True(t, *state.GeneratedItems[0].GetApprove())

	err = state.RejectTool(approvalItem, "denied")
	require.NoError(t, err)
	require.Len(t, state.GeneratedItems, 2)
	assert.Equal(t, "mcp_approval_response", *state.GeneratedItems[1].GetType())
	assert.Equal(t, "approval_1", *state.GeneratedItems[1].GetApprovalRequestID())
	assert.False(t, *state.GeneratedItems[1].GetApprove())
	assert.Equal(t, "denied", *state.GeneratedItems[1].GetReason())

	restoredContext := agents.NewRunContextWrapper[any](nil)
	state.ApplyToolApprovalsToContext(restoredContext)
	approved, known := restoredContext.IsToolApproved("add", "approval_1")
	require.True(t, known)
	assert.False(t, approved)
}

func TestRunStateApproveToolUpdatesApprovalsForNonMCP(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
	}
	approvalItem := agents.ToolApprovalItem{
		ToolName: "apply_patch",
		RawItem: map[string]any{
			"type":    "apply_patch_call",
			"call_id": "call-apply-1",
		},
	}

	err := state.ApproveTool(approvalItem)
	require.NoError(t, err)
	assert.Empty(t, state.GeneratedItems)

	ctx := agents.NewRunContextWrapper[any](nil)
	state.ApplyToolApprovalsToContext(ctx)
	approved, known := ctx.IsToolApproved("apply_patch", "call-apply-1")
	require.True(t, known)
	assert.True(t, approved)
}

func TestRunStateRejectToolUpdatesApprovalsForNonMCP(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
	}
	approvalItem := agents.ToolApprovalItem{
		ToolName: "apply_patch",
		RawItem: map[string]any{
			"type":    "apply_patch_call",
			"call_id": "call-apply-2",
		},
	}

	err := state.RejectTool(approvalItem, "denied")
	require.NoError(t, err)
	assert.Empty(t, state.GeneratedItems)

	ctx := agents.NewRunContextWrapper[any](nil)
	state.ApplyToolApprovalsToContext(ctx)
	approved, known := ctx.IsToolApproved("apply_patch", "call-apply-2")
	require.True(t, known)
	assert.False(t, approved)
}

func TestRunStateApplyStoredToolApprovals(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
		Interruptions: []agents.ToolApprovalItem{
			{
				ToolName: "add",
				RawItem: responses.ResponseOutputItemMcpApprovalRequest{
					ID:          "approval_1",
					Name:        "add",
					ServerLabel: "mcp_server",
					Type:        constant.ValueOf[constant.McpApprovalRequest](),
				},
			},
			{
				ToolName: "subtract",
				RawItem: responses.ResponseOutputItemMcpApprovalRequest{
					ID:          "approval_2",
					Name:        "subtract",
					ServerLabel: "mcp_server",
					Type:        constant.ValueOf[constant.McpApprovalRequest](),
				},
			},
		},
	}
	state.SetToolApprovalsFromContext(agents.NewRunContextWrapper[any](nil))

	runContext := agents.NewRunContextWrapper[any](nil)
	runContext.ApproveTool(state.Interruptions[0], false)
	state.SetToolApprovalsFromContext(runContext)
	require.NotEmpty(t, state.ToolApprovals)

	err := state.ApplyStoredToolApprovals()
	require.NoError(t, err)
	require.Len(t, state.GeneratedItems, 1)
	assert.Equal(t, "mcp_approval_response", *state.GeneratedItems[0].GetType())
	assert.Equal(t, "approval_1", *state.GeneratedItems[0].GetApprovalRequestID())
	assert.True(t, *state.GeneratedItems[0].GetApprove())
}

func TestRunStateApplyStoredToolApprovalsSkipsExistingResponses(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
		GeneratedItems: []agents.TResponseInputItem{
			{
				OfMcpApprovalResponse: &responses.ResponseInputItemMcpApprovalResponseParam{
					ApprovalRequestID: "approval_1",
					Approve:           true,
					Type:              constant.ValueOf[constant.McpApprovalResponse](),
				},
			},
		},
		Interruptions: []agents.ToolApprovalItem{
			{
				ToolName: "add",
				RawItem: responses.ResponseOutputItemMcpApprovalRequest{
					ID:          "approval_1",
					Name:        "add",
					ServerLabel: "mcp_server",
					Type:        constant.ValueOf[constant.McpApprovalRequest](),
				},
			},
		},
	}

	runContext := agents.NewRunContextWrapper[any](nil)
	runContext.ApproveTool(state.Interruptions[0], false)
	state.SetToolApprovalsFromContext(runContext)

	err := state.ApplyStoredToolApprovals()
	require.NoError(t, err)
	require.Len(t, state.GeneratedItems, 1)
}

func TestRunStateApplyStoredToolApprovalsSkipsNonMCP(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
		Interruptions: []agents.ToolApprovalItem{
			{
				ToolName: "apply_patch",
				RawItem: map[string]any{
					"type":    "apply_patch_call",
					"call_id": "call-apply-pending",
				},
			},
		},
	}
	ctx := agents.NewRunContextWrapper[any](nil)
	ctx.ApproveTool(state.Interruptions[0], false)
	state.SetToolApprovalsFromContext(ctx)

	err := state.ApplyStoredToolApprovals()
	require.NoError(t, err)
	assert.Empty(t, state.GeneratedItems)
}

func TestRunStateResumeGuardrailResultHelpers(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
		InputGuardrailResults: []agents.GuardrailResultState{
			{
				Name: "input_guard",
				Output: agents.GuardrailFunctionOutputState{
					OutputInfo:        map[string]any{"k": "v"},
					TripwireTriggered: true,
				},
			},
		},
		OutputGuardrailResults: []agents.GuardrailResultState{
			{
				Name: "output_guard",
				Output: agents.GuardrailFunctionOutputState{
					OutputInfo:        "ok",
					TripwireTriggered: false,
				},
			},
		},
		ToolInputGuardrailResults: []agents.ToolGuardrailResultState{
			{
				Name: "tool_input_guard",
				Output: agents.ToolGuardrailFunctionOutputState{
					OutputInfo: "ti",
					Behavior: agents.ToolGuardrailBehaviorState{
						Type:    agents.ToolGuardrailBehaviorTypeRejectContent,
						Message: "blocked",
					},
				},
			},
		},
		ToolOutputGuardrailResults: []agents.ToolGuardrailResultState{
			{
				Name: "tool_output_guard",
				Output: agents.ToolGuardrailFunctionOutputState{
					OutputInfo: "to",
					Behavior: agents.ToolGuardrailBehaviorState{
						Type: agents.ToolGuardrailBehaviorTypeAllow,
					},
				},
			},
		},
	}

	inputResults := state.ResumeInputGuardrailResults()
	require.Len(t, inputResults, 1)
	assert.Equal(t, "input_guard", inputResults[0].Guardrail.Name)
	assert.True(t, inputResults[0].Output.TripwireTriggered)

	outputResults := state.ResumeOutputGuardrailResults()
	require.Len(t, outputResults, 1)
	assert.Equal(t, "output_guard", outputResults[0].Guardrail.Name)
	assert.Equal(t, "ok", outputResults[0].Output.OutputInfo)

	toolInputResults := state.ResumeToolInputGuardrailResults()
	require.Len(t, toolInputResults, 1)
	assert.Equal(t, "tool_input_guard", toolInputResults[0].Guardrail.GetName())
	assert.Equal(t, agents.ToolGuardrailBehaviorTypeRejectContent, toolInputResults[0].Output.BehaviorType())
	assert.Equal(t, "blocked", toolInputResults[0].Output.BehaviorMessage())

	toolOutputResults := state.ResumeToolOutputGuardrailResults()
	require.Len(t, toolOutputResults, 1)
	assert.Equal(t, "tool_output_guard", toolOutputResults[0].Guardrail.GetName())
	assert.Equal(t, agents.ToolGuardrailBehaviorTypeAllow, toolOutputResults[0].Output.BehaviorType())
}
