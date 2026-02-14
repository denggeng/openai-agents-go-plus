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

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
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
