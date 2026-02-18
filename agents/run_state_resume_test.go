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
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunFromStateUsesResumeInputAndConfig(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		},
	})

	inputGuardrailCalls := 0
	agent := agents.New("resume_agent").WithModelInstance(model).WithInputGuardrails([]agents.InputGuardrail{
		{
			Name: "input_guard",
			GuardrailFunction: func(context.Context, *agents.Agent, agents.Input) (agents.GuardrailFunctionOutput, error) {
				inputGuardrailCalls++
				return agents.GuardrailFunctionOutput{}, nil
			},
		},
	})

	state := agents.RunState{
		SchemaVersion:      agents.CurrentRunStateSchemaVersion,
		CurrentTurn:        2,
		MaxTurns:           3,
		CurrentAgentName:   "resume_agent",
		PreviousResponseID: "resp_prev",
		OriginalInput: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem("original"),
		},
		GeneratedItems: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem("generated"),
		},
	}

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "done", result.FinalOutput)

	assert.Equal(t, "resp_prev", model.LastTurnArgs.PreviousResponseID)
	assert.Equal(t, state.ResumeInputItems(), agents.ItemHelpers().InputToNewInputList(model.LastTurnArgs.Input))
	assert.Equal(t, 0, inputGuardrailCalls)
}

func TestRunFromStateErrorsWhenCurrentAgentMismatches(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("actual_agent").WithModelInstance(model)

	state := agents.RunState{
		SchemaVersion:    agents.CurrentRunStateSchemaVersion,
		CurrentAgentName: "different_agent",
	}

	_, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.Error(t, err)
	assert.ErrorAs(t, err, &agents.UserError{})
	assert.ErrorContains(t, err, "does not match provided starting agent")
}

func TestRunFromStateHonorsMaxTurnsFromSnapshot(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("resume_agent").WithModelInstance(model)

	state := agents.RunState{
		SchemaVersion:    agents.CurrentRunStateSchemaVersion,
		CurrentTurn:      3,
		MaxTurns:         3,
		CurrentAgentName: "resume_agent",
	}

	_, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.Error(t, err)
	assert.ErrorAs(t, err, &agents.MaxTurnsExceededError{})
}

func TestRunFromStateStreamedUsesResumeInputAndConfig(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		},
	})

	inputGuardrailCalls := 0
	agent := agents.New("resume_agent").WithModelInstance(model).WithInputGuardrails([]agents.InputGuardrail{
		{
			Name: "input_guard",
			GuardrailFunction: func(context.Context, *agents.Agent, agents.Input) (agents.GuardrailFunctionOutput, error) {
				inputGuardrailCalls++
				return agents.GuardrailFunctionOutput{}, nil
			},
		},
	})

	state := agents.RunState{
		SchemaVersion:      agents.CurrentRunStateSchemaVersion,
		CurrentTurn:        1,
		MaxTurns:           2,
		CurrentAgentName:   "resume_agent",
		PreviousResponseID: "resp_prev",
		OriginalInput: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem("original"),
		},
		GeneratedItems: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem("generated"),
		},
	}

	result, err := agents.Runner{}.RunFromStateStreamed(t.Context(), agent, state)
	require.NoError(t, err)

	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, "done", result.FinalOutput())
	assert.Equal(t, "resp_prev", model.LastTurnArgs.PreviousResponseID)
	assert.Equal(t, state.ResumeInputItems(), agents.ItemHelpers().InputToNewInputList(model.LastTurnArgs.Input))
	assert.Equal(t, uint64(2), result.CurrentTurn())
	assert.Equal(t, 0, inputGuardrailCalls)
}

func TestRunFromStateAutoAppliesStoredToolApprovals(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		},
	})

	agent := agents.New("resume_agent").WithModelInstance(model)
	interruption := agents.ToolApprovalItem{
		ToolName: "add",
		RawItem: responses.ResponseOutputItemMcpApprovalRequest{
			ID:          "approval_1",
			Name:        "add",
			ServerLabel: "mcp_server",
			Type:        constant.ValueOf[constant.McpApprovalRequest](),
		},
	}
	runContext := agents.NewRunContextWrapper[any](nil)
	runContext.ApproveTool(interruption, false)

	state := agents.RunState{
		SchemaVersion:    agents.CurrentRunStateSchemaVersion,
		CurrentTurn:      1,
		MaxTurns:         3,
		CurrentAgentName: "resume_agent",
		OriginalInput: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem("original"),
		},
		Interruptions: []agents.ToolApprovalItem{interruption},
	}
	state.SetToolApprovalsFromContext(runContext)

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "done", result.FinalOutput)

	resumeInput := agents.ItemHelpers().InputToNewInputList(model.LastTurnArgs.Input)
	assert.Equal(t, 1, countMCPApprovalResponses(resumeInput, "approval_1", true))
}

func TestRunFromStateAutoAppliesStoredToolApprovalsSkipsExistingResponses(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		},
	})

	agent := agents.New("resume_agent").WithModelInstance(model)
	interruption := agents.ToolApprovalItem{
		ToolName: "add",
		RawItem: responses.ResponseOutputItemMcpApprovalRequest{
			ID:          "approval_1",
			Name:        "add",
			ServerLabel: "mcp_server",
			Type:        constant.ValueOf[constant.McpApprovalRequest](),
		},
	}
	runContext := agents.NewRunContextWrapper[any](nil)
	runContext.ApproveTool(interruption, false)

	state := agents.RunState{
		SchemaVersion:    agents.CurrentRunStateSchemaVersion,
		CurrentTurn:      1,
		MaxTurns:         3,
		CurrentAgentName: "resume_agent",
		OriginalInput: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem("original"),
		},
		GeneratedItems: []agents.TResponseInputItem{
			{
				OfMcpApprovalResponse: &responses.ResponseInputItemMcpApprovalResponseParam{
					ApprovalRequestID: "approval_1",
					Approve:           true,
					Type:              constant.ValueOf[constant.McpApprovalResponse](),
				},
			},
		},
		Interruptions: []agents.ToolApprovalItem{interruption},
	}
	state.SetToolApprovalsFromContext(runContext)

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "done", result.FinalOutput)

	resumeInput := agents.ItemHelpers().InputToNewInputList(model.LastTurnArgs.Input)
	assert.Equal(t, 1, countMCPApprovalResponses(resumeInput, "approval_1", true))
}

func TestRunFromStateResumesApprovedApplyPatch(t *testing.T) {
	editor := &recordingApplyPatchEditor{}
	applyPatchTool := agents.ApplyPatchTool{
		Editor:        editor,
		NeedsApproval: agents.ApplyPatchNeedsApprovalEnabled(),
	}

	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{
			Value: []agents.TResponseOutputItem{
				mustApplyPatchCustomCall(t, "call_apply_1"),
			},
		},
		{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetTextMessage("done"),
			},
		},
	})

	agent := agents.New("apply-agent").WithModelInstance(model).WithTools(applyPatchTool)
	first, err := agents.Runner{}.Run(t.Context(), agent, "update")
	require.NoError(t, err)
	require.NotNil(t, first)
	require.Len(t, first.Interruptions, 1)

	state := agents.NewRunStateFromResult(*first, 1, 3)
	require.NoError(t, state.ApproveTool(first.Interruptions[0]))

	resumed, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	require.NotNil(t, resumed)
	assert.Equal(t, "done", resumed.FinalOutput)
	assert.Empty(t, resumed.Interruptions)
	assert.Len(t, editor.operations, 1)
	assert.NotEmpty(t, collectApplyPatchOutputs(resumed.NewItems))
}

func TestRunFromStatePendingApplyPatchRemainsPending(t *testing.T) {
	editor := &recordingApplyPatchEditor{}
	applyPatchTool := agents.ApplyPatchTool{
		Editor:        editor,
		NeedsApproval: agents.ApplyPatchNeedsApprovalEnabled(),
	}

	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{
			Value: []agents.TResponseOutputItem{
				mustApplyPatchCustomCall(t, "call_apply_pending"),
			},
		},
	})

	agent := agents.New("apply-agent").WithModelInstance(model).WithTools(applyPatchTool)
	first, err := agents.Runner{}.Run(t.Context(), agent, "update")
	require.NoError(t, err)
	require.NotNil(t, first)
	require.Len(t, first.Interruptions, 1)

	state := agents.NewRunStateFromResult(*first, 1, 3)
	resumed, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	require.NotNil(t, resumed)
	assert.NotEmpty(t, resumed.Interruptions)
	assert.Empty(t, editor.operations)
	assert.Empty(t, collectApplyPatchOutputs(resumed.NewItems))
}

type recordingApplyPatchEditor struct {
	operations []agents.ApplyPatchOperation
}

func (r *recordingApplyPatchEditor) CreateFile(operation agents.ApplyPatchOperation) (any, error) {
	r.operations = append(r.operations, operation)
	return nil, nil
}

func (r *recordingApplyPatchEditor) UpdateFile(operation agents.ApplyPatchOperation) (any, error) {
	r.operations = append(r.operations, operation)
	return nil, nil
}

func (r *recordingApplyPatchEditor) DeleteFile(operation agents.ApplyPatchOperation) (any, error) {
	r.operations = append(r.operations, operation)
	return nil, nil
}

func mustApplyPatchCustomCall(t *testing.T, callID string) responses.ResponseOutputItemUnion {
	t.Helper()
	payload := map[string]any{
		"type":    "custom_tool_call",
		"name":    "apply_patch",
		"call_id": callID,
		"input":   `{"type":"update_file","path":"test.md","diff":"-a\n+b\n"}`,
	}
	raw, err := json.Marshal(payload)
	require.NoError(t, err)
	var item responses.ResponseOutputItemUnion
	require.NoError(t, json.Unmarshal(raw, &item))
	return item
}

func collectApplyPatchOutputs(items []agents.RunItem) []agents.ToolCallOutputItem {
	if len(items) == 0 {
		return nil
	}
	out := make([]agents.ToolCallOutputItem, 0)
	for _, item := range items {
		outputItem, ok := item.(agents.ToolCallOutputItem)
		if !ok {
			continue
		}
		raw, err := json.Marshal(outputItem.RawItem)
		if err != nil {
			continue
		}
		var payload map[string]any
		if err := json.Unmarshal(raw, &payload); err != nil {
			continue
		}
		if payload["type"] == "apply_patch_call_output" {
			out = append(out, outputItem)
		}
	}
	return out
}

func TestRunFromStatePreservesAndAppendsToolGuardrailResults(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("foo", `{"a":"b"}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	tool := agentstesting.GetFunctionTool("foo", "tool_output")
	tool.ToolInputGuardrails = []agents.ToolInputGuardrail{
		{
			Name: "new_tool_input_guardrail",
			GuardrailFunction: func(context.Context, agents.ToolInputGuardrailData) (agents.ToolGuardrailFunctionOutput, error) {
				return agents.ToolGuardrailAllow("new_input"), nil
			},
		},
	}
	tool.ToolOutputGuardrails = []agents.ToolOutputGuardrail{
		{
			Name: "new_tool_output_guardrail",
			GuardrailFunction: func(context.Context, agents.ToolOutputGuardrailData) (agents.ToolGuardrailFunctionOutput, error) {
				return agents.ToolGuardrailAllow("new_output"), nil
			},
		},
	}

	agent := agents.New("resume_agent").WithModelInstance(model).WithTools(tool)
	state := agents.RunState{
		SchemaVersion:    agents.CurrentRunStateSchemaVersion,
		CurrentTurn:      1,
		MaxTurns:         4,
		CurrentAgentName: "resume_agent",
		OriginalInput: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem("original"),
		},
		InputGuardrailResults: []agents.GuardrailResultState{
			{
				Name: "state_input_guardrail",
				Output: agents.GuardrailFunctionOutputState{
					OutputInfo:        map[string]any{"source": "state"},
					TripwireTriggered: false,
				},
			},
		},
		ToolInputGuardrailResults: []agents.ToolGuardrailResultState{
			{
				Name: "state_tool_input_guardrail",
				Output: agents.ToolGuardrailFunctionOutputState{
					OutputInfo: map[string]any{"source": "state"},
					Behavior: agents.ToolGuardrailBehaviorState{
						Type: agents.ToolGuardrailBehaviorTypeAllow,
					},
				},
			},
		},
		ToolOutputGuardrailResults: []agents.ToolGuardrailResultState{
			{
				Name: "state_tool_output_guardrail",
				Output: agents.ToolGuardrailFunctionOutputState{
					OutputInfo: map[string]any{"source": "state"},
					Behavior: agents.ToolGuardrailBehaviorState{
						Type: agents.ToolGuardrailBehaviorTypeAllow,
					},
				},
			},
		},
	}

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "done", result.FinalOutput)

	require.Len(t, result.InputGuardrailResults, 1)
	assert.Equal(t, "state_input_guardrail", result.InputGuardrailResults[0].Guardrail.Name)

	require.Len(t, result.ToolInputGuardrailResults, 2)
	assert.Equal(t, "state_tool_input_guardrail", result.ToolInputGuardrailResults[0].Guardrail.GetName())
	assert.Equal(t, "new_tool_input_guardrail", result.ToolInputGuardrailResults[1].Guardrail.GetName())

	require.Len(t, result.ToolOutputGuardrailResults, 2)
	assert.Equal(t, "state_tool_output_guardrail", result.ToolOutputGuardrailResults[0].Guardrail.GetName())
	assert.Equal(t, "new_tool_output_guardrail", result.ToolOutputGuardrailResults[1].Guardrail.GetName())
}

func TestRunFromStateStreamedPreservesAndAppendsToolGuardrailResults(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("foo", `{"a":"b"}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	tool := agentstesting.GetFunctionTool("foo", "tool_output")
	tool.ToolInputGuardrails = []agents.ToolInputGuardrail{
		{
			Name: "new_tool_input_guardrail",
			GuardrailFunction: func(context.Context, agents.ToolInputGuardrailData) (agents.ToolGuardrailFunctionOutput, error) {
				return agents.ToolGuardrailAllow("new_input"), nil
			},
		},
	}
	tool.ToolOutputGuardrails = []agents.ToolOutputGuardrail{
		{
			Name: "new_tool_output_guardrail",
			GuardrailFunction: func(context.Context, agents.ToolOutputGuardrailData) (agents.ToolGuardrailFunctionOutput, error) {
				return agents.ToolGuardrailAllow("new_output"), nil
			},
		},
	}

	agent := agents.New("resume_agent").WithModelInstance(model).WithTools(tool)
	state := agents.RunState{
		SchemaVersion:    agents.CurrentRunStateSchemaVersion,
		CurrentTurn:      1,
		MaxTurns:         4,
		CurrentAgentName: "resume_agent",
		OriginalInput: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem("original"),
		},
		ToolInputGuardrailResults: []agents.ToolGuardrailResultState{
			{
				Name: "state_tool_input_guardrail",
				Output: agents.ToolGuardrailFunctionOutputState{
					OutputInfo: map[string]any{"source": "state"},
					Behavior: agents.ToolGuardrailBehaviorState{
						Type: agents.ToolGuardrailBehaviorTypeAllow,
					},
				},
			},
		},
		ToolOutputGuardrailResults: []agents.ToolGuardrailResultState{
			{
				Name: "state_tool_output_guardrail",
				Output: agents.ToolGuardrailFunctionOutputState{
					OutputInfo: map[string]any{"source": "state"},
					Behavior: agents.ToolGuardrailBehaviorState{
						Type: agents.ToolGuardrailBehaviorTypeAllow,
					},
				},
			},
		},
	}

	result, err := agents.Runner{}.RunFromStateStreamed(t.Context(), agent, state)
	require.NoError(t, err)

	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)
	assert.Equal(t, "done", result.FinalOutput())

	require.Len(t, result.ToolInputGuardrailResults(), 2)
	assert.Equal(t, "state_tool_input_guardrail", result.ToolInputGuardrailResults()[0].Guardrail.GetName())
	assert.Equal(t, "new_tool_input_guardrail", result.ToolInputGuardrailResults()[1].Guardrail.GetName())

	require.Len(t, result.ToolOutputGuardrailResults(), 2)
	assert.Equal(t, "state_tool_output_guardrail", result.ToolOutputGuardrailResults()[0].Guardrail.GetName())
	assert.Equal(t, "new_tool_output_guardrail", result.ToolOutputGuardrailResults()[1].Guardrail.GetName())
}

func TestRunFromStateUsesConversationID(t *testing.T) {
	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("first"),
		},
	})
	agent := agents.New("resume_agent").WithModelInstance(model)

	result1, err := (agents.Runner{Config: agents.RunConfig{ConversationID: "conv123"}}).
		Run(t.Context(), agent, "First input")
	require.NoError(t, err)
	assert.Equal(t, "conv123", model.LastTurnArgs.ConversationID)

	state := agents.NewRunStateFromResult(*result1, 1, 3)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("second"),
		},
	})

	result2, err := (agents.Runner{Config: agents.RunConfig{ConversationID: "conv123"}}).
		RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	assert.Equal(t, "second", result2.FinalOutput)
	assert.Equal(t, "conv123", model.LastTurnArgs.ConversationID)
}

func TestRunFromStateUsesAutoPreviousResponseID(t *testing.T) {
	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("first"),
		},
	})
	model.ResponseID = "resp-123"
	agent := agents.New("resume_agent").WithModelInstance(model)

	result1, err := (agents.Runner{Config: agents.RunConfig{AutoPreviousResponseID: true}}).
		Run(t.Context(), agent, "First input")
	require.NoError(t, err)
	assert.Equal(t, "resp-123", result1.PreviousResponseID)

	state := agents.NewRunStateFromResult(*result1, 1, 3)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("second"),
		},
	})

	result2, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	assert.Equal(t, "second", result2.FinalOutput)
	assert.Equal(t, "resp-123", model.LastTurnArgs.PreviousResponseID)
}

func countMCPApprovalResponses(items []agents.TResponseInputItem, approvalRequestID string, approve bool) int {
	count := 0
	for _, item := range items {
		itemType := item.GetType()
		if itemType == nil || *itemType != "mcp_approval_response" {
			continue
		}
		itemApprovalRequestID := item.GetApprovalRequestID()
		if itemApprovalRequestID == nil || *itemApprovalRequestID != approvalRequestID {
			continue
		}
		itemApprove := item.GetApprove()
		if itemApprove == nil || *itemApprove != approve {
			continue
		}
		count++
	}
	return count
}
