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

package agents_test

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/denggeng/openai-agents-go-plus/computer"
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const hitlRejectionMessage = agents.DefaultApprovalRejectionMessage

type trackingComputer struct {
	calls []string
}

type approveArgs struct {
	Reason string `json:"reason"`
}

func (t *trackingComputer) Environment(context.Context) (computer.Environment, error) {
	return computer.EnvironmentMac, nil
}
func (t *trackingComputer) Dimensions(context.Context) (computer.Dimensions, error) {
	return computer.Dimensions{Width: 1, Height: 1}, nil
}
func (t *trackingComputer) Screenshot(context.Context) (string, error) {
	t.calls = append(t.calls, "screenshot")
	return "img", nil
}
func (t *trackingComputer) Click(context.Context, int64, int64, computer.Button) error {
	t.calls = append(t.calls, "click")
	return nil
}
func (t *trackingComputer) DoubleClick(context.Context, int64, int64) error {
	t.calls = append(t.calls, "double_click")
	return nil
}
func (t *trackingComputer) Scroll(context.Context, int64, int64, int64, int64) error {
	t.calls = append(t.calls, "scroll")
	return nil
}
func (t *trackingComputer) Type(context.Context, string) error {
	t.calls = append(t.calls, "type")
	return nil
}
func (t *trackingComputer) Wait(context.Context) error {
	t.calls = append(t.calls, "wait")
	return nil
}
func (t *trackingComputer) Move(context.Context, int64, int64) error {
	t.calls = append(t.calls, "move")
	return nil
}
func (t *trackingComputer) Keypress(context.Context, []string) error {
	t.calls = append(t.calls, "keypress")
	return nil
}
func (t *trackingComputer) Drag(context.Context, []computer.Position) error {
	t.calls = append(t.calls, "drag")
	return nil
}

func requireFunctionApproval(
	context.Context,
	*agents.RunContextWrapper[any],
	agents.FunctionTool,
	map[string]any,
	string,
) (bool, error) {
	return true, nil
}

func requireShellApproval(
	context.Context,
	*agents.RunContextWrapper[any],
	agents.ShellActionRequest,
	string,
) (bool, error) {
	return true, nil
}

func requireApplyPatchApproval(
	context.Context,
	*agents.RunContextWrapper[any],
	agents.ApplyPatchOperation,
	string,
) (bool, error) {
	return true, nil
}

func makeModelAndAgent(tools []agents.Tool, name string) (*agentstesting.FakeModel, *agents.Agent) {
	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New(name).WithModelInstance(model).WithTools(tools...)
	return model, agent
}

func makeAgent(model *agentstesting.FakeModel, tools []agents.Tool, name string) *agents.Agent {
	return agents.New(name).WithModelInstance(model).WithTools(tools...)
}

func makeContextWrapper() *agents.RunContextWrapper[any] {
	return agents.NewRunContextWrapper[any](nil)
}

func makeShellCall(t *testing.T, callID string, idValue string, commands []string, status string) responses.ResponseOutputItemUnion {
	t.Helper()
	if status == "" {
		status = "in_progress"
	}
	if len(commands) == 0 {
		commands = []string{"echo test"}
	}
	payload := map[string]any{
		"type":    "shell_call",
		"id":      idValue,
		"call_id": callID,
		"status":  status,
		"action": map[string]any{
			"type":       "exec",
			"commands":   commands,
			"timeout_ms": 1000,
		},
	}
	if idValue == "" {
		payload["id"] = callID
	}
	return mustOutputItem(t, payload)
}

func makeApplyPatchCall(t *testing.T, callID string, diff string) responses.ResponseOutputItemUnion {
	t.Helper()
	if diff == "" {
		diff = "-a\n+b\n"
	}
	operationJSON, err := json.Marshal(map[string]any{
		"type": "update_file",
		"path": "test.md",
		"diff": diff,
	})
	require.NoError(t, err)
	payload := map[string]any{
		"type":    "custom_tool_call",
		"name":    "apply_patch",
		"call_id": callID,
		"input":   string(operationJSON),
	}
	return mustOutputItem(t, payload)
}

func makeApplyPatchDict(t *testing.T, callID string, diff string) responses.ResponseOutputItemUnion {
	t.Helper()
	if diff == "" {
		diff = "-a\n+b\n"
	}
	payload := map[string]any{
		"type":    "apply_patch_call",
		"call_id": callID,
		"operation": map[string]any{
			"type": "update_file",
			"path": "test.md",
			"diff": diff,
		},
	}
	return mustOutputItem(t, payload)
}

func makeFunctionToolCall(t *testing.T, name string, callID string, arguments string) responses.ResponseOutputItemUnion {
	t.Helper()
	if callID == "" {
		callID = "call-1"
	}
	if arguments == "" {
		arguments = "{}"
	}
	payload := map[string]any{
		"type":      "function_call",
		"name":      name,
		"call_id":   callID,
		"arguments": arguments,
	}
	return mustOutputItem(t, payload)
}

func queueFunctionCallAndText(
	model *agentstesting.FakeModel,
	functionCall responses.ResponseOutputItemUnion,
	firstTurnExtra []responses.ResponseOutputItemUnion,
	followup []responses.ResponseOutputItemUnion,
) {
	first := append([]responses.ResponseOutputItemUnion{functionCall}, firstTurnExtra...)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: first},
		{Value: followup},
	})
}

func collectToolOutputs(items []agents.RunItem, outputType string) []agents.ToolCallOutputItem {
	out := make([]agents.ToolCallOutputItem, 0)
	for _, item := range items {
		output, ok := item.(agents.ToolCallOutputItem)
		if !ok {
			continue
		}
		var raw map[string]any
		switch typed := output.RawItem.(type) {
		case agents.ShellCallOutputRawItem:
			raw = map[string]any(typed)
		default:
			raw = mustJSONMap(nil, output.RawItem)
		}
		if len(raw) == 0 {
			continue
		}
		if rawType, ok := raw["type"].(string); ok && rawType == outputType {
			out = append(out, output)
		}
	}
	return out
}

func mustJSONMap(t *testing.T, value any) map[string]any {
	if value == nil {
		return nil
	}
	raw, err := json.Marshal(value)
	if t != nil {
		require.NoError(t, err)
	} else if err != nil {
		return nil
	}
	var out map[string]any
	if t != nil {
		require.NoError(t, json.Unmarshal(raw, &out))
	} else if err := json.Unmarshal(raw, &out); err != nil {
		return nil
	}
	return out
}

func mustOutputItem(t *testing.T, payload any) responses.ResponseOutputItemUnion {
	t.Helper()
	raw, err := json.Marshal(payload)
	require.NoError(t, err)
	var item responses.ResponseOutputItemUnion
	require.NoError(t, json.Unmarshal(raw, &item))
	return item
}

func approvalCallID(item agents.ToolApprovalItem) string {
	raw := mustJSONMap(nil, item.RawItem)
	if raw == nil {
		return ""
	}
	if callID, ok := raw["call_id"].(string); ok && callID != "" {
		return callID
	}
	if id, ok := raw["id"].(string); ok && id != "" {
		return id
	}
	return ""
}

func stateFromResult(result *agents.RunResult, maxTurns uint64) agents.RunState {
	currentTurn := uint64(len(result.RawResponses))
	if currentTurn == 0 {
		currentTurn = 1
	}
	return agents.NewRunStateFromResult(*result, currentTurn, maxTurns)
}

func stateFromResultWithApprovals(prev agents.RunState, result *agents.RunResult, maxTurns uint64) agents.RunState {
	state := stateFromResult(result, maxTurns)
	state.ToolApprovals = prev.ToolApprovals
	return state
}

func approveFirstInterruption(t *testing.T, result *agents.RunResult, always bool) agents.RunState {
	t.Helper()
	require.NotEmpty(t, result.Interruptions)
	state := stateFromResult(result, agents.DefaultMaxTurns)
	err := state.ApproveTool(result.Interruptions[0])
	require.NoError(t, err)
	return state
}

func runAndResumeAfterApproval(
	t *testing.T,
	agent *agents.Agent,
	model *agentstesting.FakeModel,
	rawCall responses.ResponseOutputItemUnion,
	finalOutput responses.ResponseOutputItemUnion,
	userInput string,
) *agents.RunResult {
	t.Helper()
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []responses.ResponseOutputItemUnion{rawCall}},
		{Value: []responses.ResponseOutputItemUnion{finalOutput}},
	})
	first, err := agents.Runner{}.Run(t.Context(), agent, userInput)
	require.NoError(t, err)
	require.NotEmpty(t, first.Interruptions)
	state := stateFromResult(first, agents.DefaultMaxTurns)
	require.NoError(t, state.ApproveTool(first.Interruptions[0]))
	resumed, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	return resumed
}

func makeStateWithInterruptions(
	agent *agents.Agent,
	interruptions []agents.ToolApprovalItem,
	originalInput string,
	maxTurns uint64,
) agents.RunState {
	if maxTurns == 0 {
		maxTurns = agents.DefaultMaxTurns
	}
	state := agents.RunState{
		SchemaVersion:    agents.CurrentRunStateSchemaVersion,
		CurrentTurn:      1,
		MaxTurns:         maxTurns,
		CurrentAgentName: agent.Name,
		OriginalInput: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem(originalInput),
		},
		Interruptions: interruptions,
	}
	return state
}

func assertRoundtripToolName(
	t *testing.T,
	agent *agents.Agent,
	model *agentstesting.FakeModel,
	rawCall responses.ResponseOutputItemUnion,
	expectedToolName string,
	userInput string,
) {
	t.Helper()
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{Value: []responses.ResponseOutputItemUnion{rawCall}})
	result, err := agents.Runner{}.Run(t.Context(), agent, userInput)
	require.NoError(t, err)
	require.NotEmpty(t, result.Interruptions)
	state := stateFromResult(result, agents.DefaultMaxTurns)
	rawState, err := state.ToJSON()
	require.NoError(t, err)
	decoded, err := agents.RunStateFromJSON(rawState)
	require.NoError(t, err)
	require.NotEmpty(t, decoded.Interruptions)
	assert.Equal(t, expectedToolName, decoded.Interruptions[0].ToolName)
}

func assertToolOutputRoundtrip(
	t *testing.T,
	agent *agents.Agent,
	rawOutput any,
	expectedType string,
	output any,
) {
	t.Helper()
	rawMap, ok := rawOutput.(map[string]any)
	if !ok {
		rawMap = mustJSONMap(t, rawOutput)
	}
	rawItem := agents.ShellCallOutputRawItem(rawMap)
	state := agents.RunState{
		SchemaVersion:    agents.CurrentRunStateSchemaVersion,
		CurrentTurn:      1,
		MaxTurns:         1,
		CurrentAgentName: agent.Name,
		GeneratedRunItems: []agents.RunItem{
			agents.ToolCallOutputItem{
				Agent:   agent,
				RawItem: rawItem,
				Output:  output,
				Type:    "tool_call_output_item",
			},
		},
	}

	rawState, err := state.ToJSON()
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(rawState, &payload))
	generated, _ := payload["generated_items"].([]any)
	require.Len(t, generated, 1)
	itemPayload, _ := generated[0].(map[string]any)
	rawItemPayload, _ := itemPayload["raw_item"].(map[string]any)
	require.NotNil(t, rawItemPayload)
	assert.Equal(t, expectedType, rawItemPayload["type"])

	decoded, err := agents.RunStateFromJSON(rawState)
	require.NoError(t, err)
	require.Len(t, decoded.GeneratedRunItems, 1)
	outputItem, ok := decoded.GeneratedRunItems[0].(agents.ToolCallOutputItem)
	require.True(t, ok)
	switch typed := outputItem.RawItem.(type) {
	case agents.ShellCallOutputRawItem:
		assert.Equal(t, expectedType, map[string]any(typed)["type"])
	default:
		rawJSON := mustJSONMap(t, outputItem.RawItem)
		assert.Equal(t, expectedType, rawJSON["type"])
	}
}

func makeMcpApprovalItem(
	agent *agents.Agent,
	callID string,
	includeProviderData bool,
	toolName *string,
	providerData map[string]any,
	includeName bool,
	useCallID bool,
) agents.ToolApprovalItem {
	rawItem := map[string]any{
		"type": "hosted_tool_call",
	}
	if includeName {
		name := "test_mcp_tool"
		if toolName != nil && *toolName != "" {
			name = *toolName
		}
		rawItem["name"] = name
	}
	if includeProviderData {
		if useCallID {
			rawItem["call_id"] = callID
		} else {
			rawItem["id"] = callID
		}
		if providerData == nil {
			providerData = map[string]any{
				"type":         "mcp_approval_request",
				"id":           "req-1",
				"server_label": "test_server",
			}
		}
		rawItem["provider_data"] = providerData
	} else {
		rawItem["id"] = callID
	}

	approvalToolName := ""
	if toolName != nil {
		approvalToolName = *toolName
	}
	return agents.ToolApprovalItem{ToolName: approvalToolName, RawItem: rawItem}
}

func TestResumedHITLExecutesApprovedTools(t *testing.T) {
	t.Run("shell", func(t *testing.T) {
		shellTool := agents.ShellTool{
			Executor:      func(context.Context, agents.ShellCommandRequest) (any, error) { return "shell_output", nil },
			NeedsApproval: agents.ShellNeedsApprovalFunc(requireShellApproval),
		}
		model, agent := makeModelAndAgent([]agents.Tool{shellTool}, "shell-agent")
		result := runAndResumeAfterApproval(
			t,
			agent,
			model,
			makeShellCall(t, "call_shell_1", "shell_1", []string{"echo test"}, "in_progress"),
			agentstesting.GetTextMessage("done"),
			"run shell command",
		)
		outputs := collectToolOutputs(result.NewItems, "shell_call_output")
		require.NotEmpty(t, outputs)
	})

	t.Run("apply_patch", func(t *testing.T) {
		editor := &recordingEditor{}
		tool := agents.ApplyPatchTool{
			Editor:        editor,
			NeedsApproval: agents.ApplyPatchNeedsApprovalFunc(requireApplyPatchApproval),
		}
		model, agent := makeModelAndAgent([]agents.Tool{tool}, "apply-agent")
		result := runAndResumeAfterApproval(
			t,
			agent,
			model,
			makeApplyPatchCall(t, "call_apply_1", ""),
			agentstesting.GetTextMessage("done"),
			"update file",
		)
		outputs := collectToolOutputs(result.NewItems, "apply_patch_call_output")
		require.NotEmpty(t, outputs)
		assert.NotEmpty(t, editor.operations)
	})
}

func TestResumingSkipsApprovalsForNonHITLTools(t *testing.T) {
	t.Run("shell_auto", func(t *testing.T) {
		var shellRuns []string
		autoShell := agents.ShellTool{
			Executor: func(context.Context, agents.ShellCommandRequest) (any, error) {
				shellRuns = append(shellRuns, "run")
				return "shell_output", nil
			},
		}

		approvalTool := agents.NewFunctionTool("needs_hitl", "", func(context.Context, struct{}) (string, error) {
			return "approved", nil
		})
		approvalTool.NeedsApproval = agents.FunctionToolNeedsApprovalFunc(requireFunctionApproval)

		model, agent := makeModelAndAgent([]agents.Tool{autoShell, approvalTool}, "auto-shell-agent")
		functionCall := makeFunctionToolCall(t, approvalTool.Name, "call-func-auto", "{}")
		queueFunctionCallAndText(
			model,
			functionCall,
			[]responses.ResponseOutputItemUnion{
				makeShellCall(t, "call_shell_auto", "shell_auto", []string{"echo auto"}, "in_progress"),
			},
			[]responses.ResponseOutputItemUnion{agentstesting.GetTextMessage("done")},
		)

		first, err := agents.Runner{}.Run(t.Context(), agent, "resume approvals")
		require.NoError(t, err)
		require.NotEmpty(t, first.Interruptions)
		state := approveFirstInterruption(t, first, true)
		resumed, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
		require.NoError(t, err)

		assert.Empty(t, resumed.Interruptions)
		allOutputs := append([]agents.RunItem{}, first.NewItems...)
		allOutputs = append(allOutputs, resumed.NewItems...)
		outputs := collectToolOutputs(allOutputs, "shell_call_output")
		assert.Len(t, outputs, 1)
		assert.Len(t, shellRuns, 1)
	})

	t.Run("apply_patch_auto", func(t *testing.T) {
		editor := &recordingEditor{}
		autoPatch := agents.ApplyPatchTool{Editor: editor}

		approvalTool := agents.NewFunctionTool("needs_hitl", "", func(context.Context, struct{}) (string, error) {
			return "approved", nil
		})
		approvalTool.NeedsApproval = agents.FunctionToolNeedsApprovalFunc(requireFunctionApproval)

		model, agent := makeModelAndAgent([]agents.Tool{autoPatch, approvalTool}, "auto-apply-agent")
		functionCall := makeFunctionToolCall(t, approvalTool.Name, "call-func-auto", "{}")
		queueFunctionCallAndText(
			model,
			functionCall,
			[]responses.ResponseOutputItemUnion{
				makeApplyPatchCall(t, "call_apply_auto", ""),
			},
			[]responses.ResponseOutputItemUnion{agentstesting.GetTextMessage("done")},
		)

		first, err := agents.Runner{}.Run(t.Context(), agent, "resume approvals")
		require.NoError(t, err)
		require.NotEmpty(t, first.Interruptions)
		state := approveFirstInterruption(t, first, true)
		resumed, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
		require.NoError(t, err)

		assert.Empty(t, resumed.Interruptions)
		allOutputs := append([]agents.RunItem{}, first.NewItems...)
		allOutputs = append(allOutputs, resumed.NewItems...)
		outputs := collectToolOutputs(allOutputs, "apply_patch_call_output")
		assert.Len(t, outputs, 1)
		assert.Len(t, editor.operations, 1)
	})
}

func TestNestedAgentToolResumesAfterRejection(t *testing.T) {
	innerTool := agents.NewFunctionTool("inner_hitl_tool", "", func(context.Context, struct{}) (string, error) {
		return "ok", nil
	})
	innerTool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()

	innerModel := agentstesting.NewFakeModel(false, nil)
	innerAgent := agents.New("Inner").WithModelInstance(innerModel).WithTools(innerTool)
	innerModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, innerTool.Name, "inner-1", "{}")}},
		{Value: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, innerTool.Name, "inner-2", "{}")}},
		{Value: []responses.ResponseOutputItemUnion{agentstesting.GetTextMessage("done")}},
	})

	agentTool := innerAgent.AsTool(agents.AgentAsToolParams{
		ToolName:        "inner_agent_tool",
		ToolDescription: "Inner agent tool with HITL",
		NeedsApproval:   agents.FunctionToolNeedsApprovalEnabled(),
	})

	outerModel := agentstesting.NewFakeModel(false, nil)
	outerAgent := agents.New("Outer").WithModelInstance(outerModel).WithTools(agentTool)
	outerModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, "inner_agent_tool", "outer-1", `{"input":"hi"}`)}},
	})

	first, err := agents.Runner{}.Run(t.Context(), outerAgent, "start")
	require.NoError(t, err)
	require.NotEmpty(t, first.Interruptions)
	state := stateFromResult(first, agents.DefaultMaxTurns)
	require.NoError(t, state.ApproveTool(first.Interruptions[0]))

	second, err := agents.Runner{}.RunFromState(t.Context(), outerAgent, state)
	require.NoError(t, err)
	require.NotEmpty(t, second.Interruptions)
	assert.Equal(t, innerTool.Name, second.Interruptions[0].ToolName)

	state2 := stateFromResultWithApprovals(state, second, agents.DefaultMaxTurns)
	require.NoError(t, state2.RejectTool(second.Interruptions[0], ""))
	third, err := agents.Runner{}.RunFromState(t.Context(), outerAgent, state2)
	require.NoError(t, err)
	require.NotEmpty(t, third.Interruptions)
	assert.Equal(t, innerTool.Name, third.Interruptions[0].ToolName)
}

func TestNestedAgentToolInterruptionsDontCollideOnDuplicateCallIDs(t *testing.T) {
	innerTool := agents.NewFunctionTool("inner_hitl_tool", "", func(context.Context, struct{}) (string, error) {
		return "ok", nil
	})
	innerTool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()

	innerModel := agentstesting.NewFakeModel(false, nil)
	innerAgent := agents.New("Inner").WithModelInstance(innerModel).WithTools(innerTool)
	innerModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, innerTool.Name, "inner-1", "{}")}},
		{Value: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, innerTool.Name, "inner-2", "{}")}},
	})

	agentTool := innerAgent.AsTool(agents.AgentAsToolParams{
		ToolName:        "inner_agent_tool",
		ToolDescription: "Inner agent tool",
	})

	outerModel := agentstesting.NewFakeModel(false, nil)
	outerAgent := agents.New("Outer").WithModelInstance(outerModel).WithTools(agentTool)
	outerModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []responses.ResponseOutputItemUnion{
			makeFunctionToolCall(t, agentTool.ToolName(), "outer-dup", `{"input":"a"}`),
			makeFunctionToolCall(t, agentTool.ToolName(), "outer-dup", `{"input":"b"}`),
		}},
	})

	result, err := agents.Runner{}.Run(t.Context(), outerAgent, "start")
	require.NoError(t, err)
	require.NotEmpty(t, result.Interruptions)
	var nested []agents.ToolApprovalItem
	for _, item := range result.Interruptions {
		if item.ToolName == innerTool.Name {
			nested = append(nested, item)
		}
	}
	assert.Len(t, nested, 2)
}

func TestNestedAgentToolDoesNotInheritParentApprovals(t *testing.T) {
	outerShared := agents.NewFunctionTool("shared_tool", "", func(context.Context, struct{}) (string, error) {
		return "outer", nil
	})
	outerShared.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()

	innerShared := agents.NewFunctionTool("shared_tool", "", func(context.Context, struct{}) (string, error) {
		return "inner", nil
	})
	innerShared.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()

	innerModel := agentstesting.NewFakeModel(false, nil)
	innerAgent := agents.New("Inner").WithModelInstance(innerModel).WithTools(innerShared)
	innerModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, innerShared.Name, "dup", "{}")}},
	})

	agentTool := innerAgent.AsTool(agents.AgentAsToolParams{
		ToolName:        "inner_agent_tool",
		ToolDescription: "Inner agent tool",
	})

	outerModel := agentstesting.NewFakeModel(false, nil)
	outerAgent := agents.New("Outer").WithModelInstance(outerModel).WithTools(outerShared, agentTool)
	outerModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, outerShared.Name, "dup", "{}")}},
		{Value: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, agentTool.ToolName(), "outer-agent", `{"input":"hi"}`)}},
	})

	first, err := agents.Runner{}.Run(t.Context(), outerAgent, "start")
	require.NoError(t, err)
	require.NotEmpty(t, first.Interruptions)
	state := stateFromResult(first, agents.DefaultMaxTurns)
	require.NoError(t, state.ApproveTool(first.Interruptions[0]))

	second, err := agents.Runner{}.RunFromState(t.Context(), outerAgent, state)
	require.NoError(t, err)
	require.NotEmpty(t, second.Interruptions)
	assert.Equal(t, innerShared.Name, second.Interruptions[0].ToolName)
}

func TestPendingApprovalsStayPendingOnResume(t *testing.T) {
	t.Run("shell_pending", func(t *testing.T) {
		shellTool := agents.ShellTool{
			Executor:      func(context.Context, agents.ShellCommandRequest) (any, error) { return "shell_output", nil },
			NeedsApproval: agents.ShellNeedsApprovalEnabled(),
		}
		model, agent := makeModelAndAgent([]agents.Tool{shellTool}, "shell-pending")
		model.SetNextOutput(agentstesting.FakeModelTurnOutput{
			Value: []responses.ResponseOutputItemUnion{makeShellCall(t, "call_shell_pending", "shell_pending", []string{"echo pending"}, "in_progress")},
		})
		first, err := agents.Runner{}.Run(t.Context(), agent, "resume pending approval")
		require.NoError(t, err)
		require.NotEmpty(t, first.Interruptions)
		state := stateFromResult(first, agents.DefaultMaxTurns)
		resumed, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
		require.NoError(t, err)
		assert.NotEmpty(t, resumed.Interruptions)
		assert.Empty(t, collectToolOutputs(resumed.NewItems, "shell_call_output"))
	})

	t.Run("apply_patch_pending", func(t *testing.T) {
		editor := &recordingEditor{}
		applyPatchTool := agents.ApplyPatchTool{
			Editor:        editor,
			NeedsApproval: agents.ApplyPatchNeedsApprovalEnabled(),
		}
		model, agent := makeModelAndAgent([]agents.Tool{applyPatchTool}, "apply-pending")
		model.SetNextOutput(agentstesting.FakeModelTurnOutput{
			Value: []responses.ResponseOutputItemUnion{makeApplyPatchCall(t, "call_apply_pending", "")},
		})
		first, err := agents.Runner{}.Run(t.Context(), agent, "resume pending approval")
		require.NoError(t, err)
		require.NotEmpty(t, first.Interruptions)
		state := stateFromResult(first, agents.DefaultMaxTurns)
		resumed, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
		require.NoError(t, err)
		assert.NotEmpty(t, resumed.Interruptions)
		assert.Empty(t, editor.operations)
		assert.Empty(t, collectToolOutputs(resumed.NewItems, "apply_patch_call_output"))
	})
}

func TestResumeDoesNotDuplicatePendingShellApprovals(t *testing.T) {
	shellTool := agents.ShellTool{
		Executor:      func(context.Context, agents.ShellCommandRequest) (any, error) { return "shell_output", nil },
		NeedsApproval: agents.ShellNeedsApprovalEnabled(),
	}
	model, agent := makeModelAndAgent([]agents.Tool{shellTool}, "shell-pending-dup")
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []responses.ResponseOutputItemUnion{
			makeShellCall(t, "call_shell_pending_dup", "shell_pending_dup", []string{"echo pending"}, "in_progress"),
		},
	})
	first, err := agents.Runner{}.Run(t.Context(), agent, "run shell")
	require.NoError(t, err)
	require.NotEmpty(t, first.Interruptions)
	state := stateFromResult(first, agents.DefaultMaxTurns)
	resumed, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	callID := "call_shell_pending_dup"
	var pending []agents.ToolApprovalItem
	for _, item := range resumed.Interruptions {
		if id := approvalCallID(item); id == callID {
			pending = append(pending, item)
		}
	}
	assert.Len(t, pending, 1)
}

func TestRouteLocalShellCallsToLocalShellTool(t *testing.T) {
	var remoteExecuted []any
	var localExecuted []any

	shellTool := agents.ShellTool{
		Executor: func(_ context.Context, request agents.ShellCommandRequest) (any, error) {
			remoteExecuted = append(remoteExecuted, request)
			return "remote_output", nil
		},
	}
	localShellTool := agents.LocalShellTool{
		Executor: func(_ context.Context, request agents.LocalShellCommandRequest) (string, error) {
			localExecuted = append(localExecuted, request)
			return "local_output", nil
		},
	}

	model, agent := makeModelAndAgent([]agents.Tool{shellTool, localShellTool}, "local-shell-agent")
	localShellCall := mustOutputItem(t, map[string]any{
		"type":    "local_shell_call",
		"id":      "local_1",
		"call_id": "call_local_1",
		"status":  "in_progress",
		"action": map[string]any{
			"type":    "exec",
			"command": []string{"echo", "test"},
			"env":     map[string]any{},
		},
	})
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{Value: []responses.ResponseOutputItemUnion{localShellCall}})

	_, err := agents.Runner{}.Run(t.Context(), agent, "run local shell")
	require.NoError(t, err)
	assert.NotEmpty(t, localExecuted)
	assert.Empty(t, remoteExecuted)
}

func TestPreserveMaxTurnsWhenResumingFromRunResultState(t *testing.T) {
	tool := agents.NewFunctionTool("test_tool", "", func(context.Context, struct{}) (string, error) {
		return "tool_result", nil
	})
	tool.NeedsApproval = agents.FunctionToolNeedsApprovalFunc(requireFunctionApproval)

	model, agent := makeModelAndAgent([]agents.Tool{tool}, "max-turns-agent")
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, "test_tool", "call-1", "{}")},
	})

	runner := agents.Runner{Config: agents.RunConfig{MaxTurns: 20}}
	result1, err := runner.Run(t.Context(), agent, "call test_tool")
	require.NoError(t, err)
	require.NotEmpty(t, result1.Interruptions)

	state := stateFromResult(result1, 20)
	require.NoError(t, state.ApproveTool(result1.Interruptions[0]))

	for i := 0; i < 10; i++ {
		model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
			{
				Value: []responses.ResponseOutputItemUnion{
					agentstesting.GetTextMessage(fmt.Sprintf("turn %d", i+2)),
					makeFunctionToolCall(t, "test_tool", fmt.Sprintf("call-%d", i+2), "{}"),
				},
			},
		})
	}

	result2, err := runner.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	assert.NotNil(t, result2)
}

func TestCurrentTurnPreservedInRunState(t *testing.T) {
	tool := agents.NewFunctionTool("test_tool", "", func(context.Context, struct{}) (string, error) {
		return "tool_result", nil
	})
	tool.NeedsApproval = agents.FunctionToolNeedsApprovalFunc(requireFunctionApproval)

	model, agent := makeModelAndAgent([]agents.Tool{tool}, "turn-agent")
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, "test_tool", "call-1", "{}")},
	})
	result1, err := agents.Runner{}.Run(t.Context(), agent, "call test_tool")
	require.NoError(t, err)
	require.NotEmpty(t, result1.Interruptions)

	state := stateFromResult(result1, agents.DefaultMaxTurns)
	assert.Equal(t, uint64(1), state.CurrentTurn)
}

func TestDeserializeInterruptionsPreserveToolCalls(t *testing.T) {
	t.Run("shell", func(t *testing.T) {
		shellTool := agents.ShellTool{
			Executor:      func(context.Context, agents.ShellCommandRequest) (any, error) { return "output", nil },
			NeedsApproval: agents.ShellNeedsApprovalFunc(requireShellApproval),
		}
		model, agent := makeModelAndAgent([]agents.Tool{shellTool}, "shell-roundtrip")
		assertRoundtripToolName(t, agent, model, makeShellCall(t, "call_shell_1", "shell_1", []string{"echo"}, "in_progress"), "shell", "run shell")
	})

	t.Run("apply_patch", func(t *testing.T) {
		tool := agents.ApplyPatchTool{
			Editor:        &recordingEditor{},
			NeedsApproval: agents.ApplyPatchNeedsApprovalFunc(requireApplyPatchApproval),
		}
		model, agent := makeModelAndAgent([]agents.Tool{tool}, "apply-roundtrip")
		assertRoundtripToolName(t, agent, model, makeApplyPatchDict(t, "call_apply_1", ""), "apply_patch", "update file")
	})
}

func TestDeserializeInterruptionsPreserveMCPTools(t *testing.T) {
	_, agent := makeModelAndAgent(nil, "mcp-roundtrip")
	toolName := "test_mcp_tool"
	approval := makeMcpApprovalItem(agent, "mcp-approval-1", true, &toolName, nil, true, true)
	state := makeStateWithInterruptions(agent, []agents.ToolApprovalItem{approval}, "test", agents.DefaultMaxTurns)

	rawState, err := state.ToJSON()
	require.NoError(t, err)
	decoded, err := agents.RunStateFromJSON(rawState)
	require.NoError(t, err)
	require.NotEmpty(t, decoded.Interruptions)
	assert.Equal(t, toolName, decoded.Interruptions[0].ToolName)
}

func TestMCPApprovalFallbacksToCallID(t *testing.T) {
	ctx := makeContextWrapper()
	item := agents.ToolApprovalItem{
		RawItem: map[string]any{
			"type": "hosted_tool_call",
			"provider_data": map[string]any{
				"type": "mcp_approval_request",
				"id":   "mcp-123",
			},
		},
	}
	ctx.ApproveTool(item, false)
	approved, known := ctx.GetApprovalStatus("missing", "mcp-123", &item)
	assert.True(t, known)
	assert.True(t, approved)
}

func TestShellCallWithoutCallIDRaises(t *testing.T) {
	agent := agents.New("shell-missing")
	shellTool := agents.ShellTool{Executor: func(context.Context, agents.ShellCommandRequest) (any, error) { return "", nil }}
	toolRun := agents.ToolRunShellCall{
		ToolCall:  map[string]any{"type": "shell_call", "action": map[string]any{"commands": []string{"echo", "hi"}}},
		ShellTool: shellTool,
	}
	_, _, err := agents.RunImpl().ExecuteShellCalls(t.Context(), agent, []agents.ToolRunShellCall{toolRun}, agents.NoOpRunHooks{}, makeContextWrapper(), agents.RunConfig{})
	var modelErr agents.ModelBehaviorError
	assert.ErrorAs(t, err, &modelErr)
}

func TestPreserveToolOutputTypesDuringSerialization(t *testing.T) {
	_, agent := makeModelAndAgent(nil, "serialization")

	computerOutput := map[string]any{
		"type":    "computer_call_output",
		"call_id": "call_computer_1",
		"output": map[string]any{
			"type":      "computer_screenshot",
			"image_url": "base64_screenshot_data",
		},
	}
	assertToolOutputRoundtrip(t, agent, computerOutput, "computer_call_output", "screenshot_data")

	shellOutput := map[string]any{
		"type":    "local_shell_call_output",
		"id":      "shell_1",
		"call_id": "call_shell_1",
		"output":  "command output",
	}
	assertToolOutputRoundtrip(t, agent, shellOutput, "local_shell_call_output", "command output")
}

type invalidNeedsApproval struct{}

func (invalidNeedsApproval) NeedsApproval(
	context.Context,
	*agents.RunContextWrapper[any],
	agents.FunctionTool,
	map[string]any,
	string,
) (bool, error) {
	return false, agents.UserErrorf("needs_approval")
}

func TestFunctionNeedsApprovalInvalidTypeRaises(t *testing.T) {
	badTool := agents.NewFunctionTool("bad_tool", "", func(context.Context, struct{}) (string, error) {
		return "ok", nil
	})
	badTool.NeedsApproval = invalidNeedsApproval{}

	model, agent := makeModelAndAgent([]agents.Tool{badTool}, "bad-needs-approval")
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, "bad_tool", "call-1", "{}")},
	})

	_, err := agents.Runner{}.Run(t.Context(), agent, "run invalid")
	var userErr agents.UserError
	assert.ErrorAs(t, err, &userErr)
}

func TestResumeInvalidNeedsApprovalRaises(t *testing.T) {
	badTool := agents.NewFunctionTool("bad_tool", "", func(context.Context, struct{}) (string, error) {
		return "ok", nil
	})
	badTool.NeedsApproval = invalidNeedsApproval{}

	agent := agents.New("bad-approval-resume").WithTools(badTool)
	contextWrapper := makeContextWrapper()
	toolCall := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      badTool.Name,
		CallID:    "call-1",
		Arguments: "{}",
	}
	processed := agents.ProcessedResponse{
		NewItems: []agents.RunItem{},
		Functions: []agents.ToolRunFunction{
			{FunctionTool: badTool, ToolCall: agents.ResponseFunctionToolCall(toolCall)},
		},
	}
	_, err := agents.RunImpl().ExecuteToolsAndSideEffects(
		t.Context(),
		agent,
		agents.InputString("resume invalid"),
		nil,
		agents.ModelResponse{Output: []responses.ResponseOutputItemUnion{}, Usage: usage.NewUsage(), ResponseID: "resp"},
		processed,
		nil,
		agents.NoOpRunHooks{},
		agents.RunConfig{},
		contextWrapper,
	)
	var userErr agents.UserError
	assert.ErrorAs(t, err, &userErr)
}

func TestAgentAsToolWithNestedApprovalsPropagates(t *testing.T) {
	nestedModel := agentstesting.NewFakeModel(false, nil)
	spanishAgent := agents.New("spanish_agent").WithModelInstance(nestedModel)

	var toolCalls []string
	nestedTool := agents.NewFunctionTool("get_current_timestamp", "", func(context.Context, struct{}) (string, error) {
		toolCalls = append(toolCalls, "called")
		return "timestamp", nil
	})
	nestedTool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()
	spanishAgent.Tools = []agents.Tool{nestedTool}

	nestedModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, "get_current_timestamp", "call-1", "{}")}},
		{Value: []responses.ResponseOutputItemUnion{agentstesting.GetTextMessage("hola")}},
	})

	orchestratorModel := agentstesting.NewFakeModel(false, nil)
	orchestrator := agents.New("orchestrator").WithModelInstance(orchestratorModel).WithTools(
		spanishAgent.AsTool(agents.AgentAsToolParams{
			ToolName:        "respond_spanish",
			ToolDescription: "Respond in Spanish",
			NeedsApproval:   agents.FunctionToolNeedsApprovalEnabled(),
		}),
	)

	orchestratorModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, "respond_spanish", "spanish-call", `{"input": "hola"}`)}},
		{Value: []responses.ResponseOutputItemUnion{agentstesting.GetTextMessage("done")}},
	})

	first, err := agents.Runner{}.Run(t.Context(), orchestrator, "hola")
	require.NoError(t, err)
	require.NotEmpty(t, first.Interruptions)

	state := stateFromResult(first, agents.DefaultMaxTurns)
	require.NoError(t, state.ApproveTool(first.Interruptions[0]))
	resumed, err := agents.Runner{}.RunFromState(t.Context(), orchestrator, state)
	require.NoError(t, err)
	require.NotEmpty(t, resumed.Interruptions)
	assert.Equal(t, nestedTool.Name, resumed.Interruptions[0].ToolName)
	assert.Empty(t, toolCalls)

	finalState := stateFromResultWithApprovals(state, resumed, agents.DefaultMaxTurns)
	require.NoError(t, finalState.ApproveTool(resumed.Interruptions[0]))
	final, err := agents.Runner{}.RunFromState(t.Context(), orchestrator, finalState)
	require.NoError(t, err)
	assert.Equal(t, "done", final.FinalOutput)
	assert.Equal(t, []string{"called"}, toolCalls)
}

func TestResumeRebuildsFunctionRunsFromPendingApprovals(t *testing.T) {
	tool := agents.NewFunctionTool("approve_me", "", func(_ context.Context, args approveArgs) (string, error) {
		if args.Reason != "" {
			return "approved:" + args.Reason, nil
		}
		return "approved", nil
	})
	tool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()

	_, agent := makeModelAndAgent([]agents.Tool{tool}, "rebuild-approvals")
	approvalRaw := map[string]any{
		"type":      "function_call",
		"name":      tool.Name,
		"call_id":   "call-rebuild-1",
		"arguments": `{"reason":"ok"}`,
		"status":    "completed",
	}
	approvalItem := agents.ToolApprovalItem{ToolName: tool.Name, RawItem: approvalRaw}
	contextWrapper := makeContextWrapper()
	contextWrapper.ApproveTool(approvalItem, false)

	state := makeStateWithInterruptions(agent, []agents.ToolApprovalItem{approvalItem}, "resume approvals", agents.DefaultMaxTurns)
	state.ModelResponses = []agents.ModelResponse{
		{Output: []responses.ResponseOutputItemUnion{mustOutputItem(t, approvalRaw)}, Usage: usage.NewUsage(), ResponseID: "resp"},
	}
	state.SetToolApprovalsFromContext(contextWrapper)

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	executed := collectToolOutputs(result.NewItems, "function_call_output")
	assert.NotEmpty(t, executed)
}

func TestResumeRebuildsFunctionRunsFromObjectApprovals(t *testing.T) {
	tool := agents.NewFunctionTool("approve_me", "", func(_ context.Context, args approveArgs) (string, error) {
		if args.Reason != "" {
			return "approved:" + args.Reason, nil
		}
		return "approved", nil
	})
	tool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()

	_, agent := makeModelAndAgent([]agents.Tool{tool}, "rebuild-obj")
	toolCall := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      tool.Name,
		CallID:    "call-rebuild-obj",
		Arguments: `{"reason":"ok"}`,
	}
	approvalItem := agents.ToolApprovalItem{ToolName: tool.Name, RawItem: toolCall}
	contextWrapper := makeContextWrapper()
	contextWrapper.ApproveTool(approvalItem, false)

	state := makeStateWithInterruptions(agent, []agents.ToolApprovalItem{approvalItem}, "resume approvals", agents.DefaultMaxTurns)
	state.ModelResponses = []agents.ModelResponse{
		{Output: []responses.ResponseOutputItemUnion{mustOutputItem(t, toolCall)}, Usage: usage.NewUsage(), ResponseID: "resp"},
	}
	state.SetToolApprovalsFromContext(contextWrapper)

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	executed := collectToolOutputs(result.NewItems, "function_call_output")
	assert.NotEmpty(t, executed)
}

func TestRebuildFunctionRunsHandlesObjectPendingAndRejections(t *testing.T) {
	rejectTool := agents.NewFunctionTool("reject_me", "", func(context.Context, struct{}) (string, error) {
		return "nope", nil
	})
	rejectTool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()
	pendingTool := agents.NewFunctionTool("pending_me", "", func(context.Context, struct{}) (string, error) {
		return "wait", nil
	})
	pendingTool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()

	_, agent := makeModelAndAgent([]agents.Tool{rejectTool, pendingTool}, "rebuild-obj-pending")
	contextWrapper := makeContextWrapper()

	rejectedCall := responses.ResponseFunctionToolCall{
		Type:   constant.ValueOf[constant.FunctionCall](),
		Name:   rejectTool.Name,
		CallID: "obj-reject",
	}
	pendingCall := responses.ResponseFunctionToolCall{
		Type:   constant.ValueOf[constant.FunctionCall](),
		Name:   pendingTool.Name,
		CallID: "obj-pending",
	}
	rejectedItem := agents.ToolApprovalItem{ToolName: rejectTool.Name, RawItem: rejectedCall}
	pendingItem := agents.ToolApprovalItem{ToolName: pendingTool.Name, RawItem: pendingCall}
	contextWrapper.RejectTool(rejectedItem, false)

	state := makeStateWithInterruptions(agent, []agents.ToolApprovalItem{rejectedItem, pendingItem}, "resume approvals", agents.DefaultMaxTurns)
	state.ModelResponses = []agents.ModelResponse{
		{Output: []responses.ResponseOutputItemUnion{mustOutputItem(t, rejectedCall), mustOutputItem(t, pendingCall)}, Usage: usage.NewUsage(), ResponseID: "resp"},
	}
	state.SetToolApprovalsFromContext(contextWrapper)

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	require.NotEmpty(t, result.Interruptions)
	var rejectionOutputs []agents.ToolCallOutputItem
	for _, item := range result.NewItems {
		if output, ok := item.(agents.ToolCallOutputItem); ok && output.Output == hitlRejectionMessage {
			rejectionOutputs = append(rejectionOutputs, output)
		}
	}
	assert.NotEmpty(t, rejectionOutputs)
}

func TestResumeKeepsUnmatchedPendingApprovalsWithFunctionRuns(t *testing.T) {
	outerTool := agents.NewFunctionTool("outer_tool", "", func(context.Context, struct{}) (string, error) {
		return "outer", nil
	})
	innerTool := agents.NewFunctionTool("inner_tool", "", func(context.Context, struct{}) (string, error) {
		return "inner", nil
	})
	innerTool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()

	_, agent := makeModelAndAgent([]agents.Tool{outerTool, innerTool}, "pending-approvals")
	contextWrapper := makeContextWrapper()

	pendingCall := responses.ResponseFunctionToolCall{
		Type:   constant.ValueOf[constant.FunctionCall](),
		Name:   innerTool.Name,
		CallID: "call-inner",
	}
	pendingItem := agents.ToolApprovalItem{ToolName: innerTool.Name, RawItem: pendingCall}

	state := makeStateWithInterruptions(agent, []agents.ToolApprovalItem{pendingItem}, "resume approvals", agents.DefaultMaxTurns)
	state.ModelResponses = []agents.ModelResponse{
		{Output: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, outerTool.Name, "call-outer", "{}")}, Usage: usage.NewUsage(), ResponseID: "resp"},
	}
	state.SetToolApprovalsFromContext(contextWrapper)

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	require.NotEmpty(t, result.Interruptions)
	assert.Equal(t, innerTool.Name, result.Interruptions[0].ToolName)
}

func TestResumeExecutesNonHITLFunctionCallsWithoutOutput(t *testing.T) {
	tool := agents.NewFunctionTool("already_ran", "", func(context.Context, struct{}) (string, error) {
		return "done", nil
	})
	_, agent := makeModelAndAgent([]agents.Tool{tool}, "resume-non-hitl")
	functionCall := makeFunctionToolCall(t, tool.Name, "call-skip", "{}")

	state := makeStateWithInterruptions(agent, nil, "resume run", agents.DefaultMaxTurns)
	state.ModelResponses = []agents.ModelResponse{
		{Output: []responses.ResponseOutputItemUnion{functionCall}, Usage: usage.NewUsage(), ResponseID: "resp"},
	}

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	assert.NotEmpty(t, collectToolOutputs(result.NewItems, "function_call_output"))
}

func TestResumeSkipsNonHITLFunctionCallsWithExistingOutput(t *testing.T) {
	tool := agents.NewFunctionTool("already_ran", "", func(context.Context, struct{}) (string, error) {
		return "done", nil
	})
	_, agent := makeModelAndAgent([]agents.Tool{tool}, "resume-non-hitl-existing")

	state := makeStateWithInterruptions(agent, nil, "resume run", agents.DefaultMaxTurns)
	state.ModelResponses = []agents.ModelResponse{
		{Output: []responses.ResponseOutputItemUnion{makeFunctionToolCall(t, tool.Name, "call-skip", "{}")}, Usage: usage.NewUsage(), ResponseID: "resp"},
	}
	functionCall := agents.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      tool.Name,
		CallID:    "call-skip",
		Arguments: "{}",
	}
	functionOutput := agents.ResponseInputItemFunctionCallOutputParam(
		agents.ItemHelpers().ToolCallOutputItem(functionCall, "prior run"),
	)
	state.GeneratedItems = []agents.TResponseInputItem{
		{OfFunctionCallOutput: (*responses.ResponseInputItemFunctionCallOutputParam)(&functionOutput)},
	}

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	assert.Empty(t, result.NewItems)
}

func TestResumeSkipsShellCallsWithExistingOutput(t *testing.T) {
	shellTool := agents.ShellTool{
		Executor:      func(context.Context, agents.ShellCommandRequest) (any, error) { return "should_not_run", nil },
		NeedsApproval: agents.ShellNeedsApprovalEnabled(),
	}
	_, agent := makeModelAndAgent([]agents.Tool{shellTool}, "shell-existing")

	shellCall := makeShellCall(t, "call_shell_resume", "shell_resume", []string{"echo done"}, "completed")
	state := makeStateWithInterruptions(agent, nil, "resume shell", agents.DefaultMaxTurns)
	state.ModelResponses = []agents.ModelResponse{
		{Output: []responses.ResponseOutputItemUnion{shellCall}, Usage: usage.NewUsage(), ResponseID: "resp"},
	}
	shellOutput := responses.ResponseInputItemShellCallOutputParam{
		CallID: "call_shell_resume",
		Type:   constant.ValueOf[constant.ShellCallOutput](),
	}
	state.GeneratedItems = []agents.TResponseInputItem{
		{OfShellCallOutput: &shellOutput},
	}

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	assert.Empty(t, result.NewItems)
}

func TestResumeKeepsApprovedShellOutputsWithPendingInterruptions(t *testing.T) {
	pendingTool := agents.NewFunctionTool("pending_tool", "", func(context.Context, struct{}) (string, error) {
		return "ok", nil
	})
	pendingTool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()
	shellTool := agents.ShellTool{
		Executor:      func(context.Context, agents.ShellCommandRequest) (any, error) { return "shell-ok", nil },
		NeedsApproval: agents.ShellNeedsApprovalEnabled(),
	}

	_, agent := makeModelAndAgent([]agents.Tool{pendingTool, shellTool}, "shell-pending")
	contextWrapper := makeContextWrapper()

	functionCall := makeFunctionToolCall(t, pendingTool.Name, "call-pending", "{}")
	shellCall := makeShellCall(t, "call_shell_ok", "shell_ok", []string{"echo ok"}, "completed")

	shellApproval := agents.ToolApprovalItem{ToolName: shellTool.ToolName(), RawItem: shellCall}
	contextWrapper.ApproveTool(shellApproval, false)

	pendingApproval := agents.ToolApprovalItem{ToolName: pendingTool.Name, RawItem: functionCall}
	state := makeStateWithInterruptions(agent, []agents.ToolApprovalItem{pendingApproval}, "resume shell", agents.DefaultMaxTurns)
	state.ModelResponses = []agents.ModelResponse{
		{Output: []responses.ResponseOutputItemUnion{functionCall, shellCall}, Usage: usage.NewUsage(), ResponseID: "resp"},
	}
	state.SetToolApprovalsFromContext(contextWrapper)

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	require.NotEmpty(t, result.Interruptions)
	outputs := collectToolOutputs(result.NewItems, "shell_call_output")
	assert.NotEmpty(t, outputs)
}

func TestResumeExecutesPendingComputerActions(t *testing.T) {
	comp := &trackingComputer{}
	computerTool := agents.ComputerTool{Computer: comp}
	_, agent := makeModelAndAgent([]agents.Tool{computerTool}, "computer-resume")

	computerCall := responses.ResponseComputerToolCall{
		Type:   "computer_call",
		ID:     "comp_pending",
		CallID: "comp_pending",
		Status: "in_progress",
		Action: responses.ResponseComputerToolCallActionUnion{Type: "screenshot"},
	}
	state := makeStateWithInterruptions(agent, nil, "resume computer", agents.DefaultMaxTurns)
	state.ModelResponses = []agents.ModelResponse{
		{Output: []responses.ResponseOutputItemUnion{mustOutputItem(t, computerCall)}, Usage: usage.NewUsage(), ResponseID: "resp"},
	}

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	outputs := collectToolOutputs(result.NewItems, "computer_call_output")
	assert.NotEmpty(t, outputs)
	assert.NotEmpty(t, comp.calls)
}

func TestResumeSkipsComputerActionsWithExistingOutput(t *testing.T) {
	comp := &trackingComputer{}
	computerTool := agents.ComputerTool{Computer: comp}
	_, agent := makeModelAndAgent([]agents.Tool{computerTool}, "computer-skip")

	computerCall := responses.ResponseComputerToolCall{
		Type:   "computer_call",
		ID:     "comp_skip",
		CallID: "comp_skip",
		Status: "completed",
		Action: responses.ResponseComputerToolCallActionUnion{Type: "screenshot"},
	}
	state := makeStateWithInterruptions(agent, nil, "resume computer", agents.DefaultMaxTurns)
	state.ModelResponses = []agents.ModelResponse{
		{Output: []responses.ResponseOutputItemUnion{mustOutputItem(t, computerCall)}, Usage: usage.NewUsage(), ResponseID: "resp"},
	}
	computerOutput := responses.ResponseInputItemComputerCallOutputParam{
		CallID: "comp_skip",
		Output: responses.ResponseComputerToolCallOutputScreenshotParam{
			ImageURL: param.NewOpt("data:image/png;base64,ok"),
			Type:     constant.ValueOf[constant.ComputerScreenshot](),
		},
		Type: constant.ValueOf[constant.ComputerCallOutput](),
	}
	state.GeneratedItems = []agents.TResponseInputItem{
		{OfComputerCallOutput: &computerOutput},
	}

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	assert.Empty(t, result.NewItems)
	assert.Empty(t, comp.calls)
}

func TestRebuildFunctionRunsHandlesPendingAndRejections(t *testing.T) {
	rejectTool := agents.NewFunctionTool("reject_me", "", func(context.Context, struct{}) (string, error) {
		return "nope", nil
	})
	rejectTool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()
	pendingTool := agents.NewFunctionTool("pending_me", "", func(context.Context, struct{}) (string, error) {
		return "wait", nil
	})
	pendingTool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()

	_, agent := makeModelAndAgent([]agents.Tool{rejectTool, pendingTool}, "rebuild-pending")
	contextWrapper := makeContextWrapper()

	rejectedRaw := map[string]any{
		"type":      "function_call",
		"name":      rejectTool.Name,
		"call_id":   "call-reject",
		"arguments": "{}",
	}
	pendingRaw := map[string]any{
		"type":      "function_call",
		"name":      pendingTool.Name,
		"call_id":   "call-pending",
		"arguments": "{}",
	}
	rejectedItem := agents.ToolApprovalItem{ToolName: rejectTool.Name, RawItem: rejectedRaw}
	pendingItem := agents.ToolApprovalItem{ToolName: pendingTool.Name, RawItem: pendingRaw}
	contextWrapper.RejectTool(rejectedItem, false)

	state := makeStateWithInterruptions(agent, []agents.ToolApprovalItem{rejectedItem, pendingItem}, "resume approvals", agents.DefaultMaxTurns)
	state.ModelResponses = []agents.ModelResponse{
		{Output: []responses.ResponseOutputItemUnion{mustOutputItem(t, rejectedRaw), mustOutputItem(t, pendingRaw)}, Usage: usage.NewUsage(), ResponseID: "resp"},
	}
	state.SetToolApprovalsFromContext(contextWrapper)

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	require.NotEmpty(t, result.Interruptions)
	var rejectionOutputs []agents.ToolCallOutputItem
	for _, item := range result.NewItems {
		if output, ok := item.(agents.ToolCallOutputItem); ok && output.Output == hitlRejectionMessage {
			rejectionOutputs = append(rejectionOutputs, output)
		}
	}
	assert.NotEmpty(t, rejectionOutputs)
}

func TestRebuildPreservesUnmatchedPendingApprovals(t *testing.T) {
	rawItems := []struct {
		raw      any
		toolName string
	}{
		{
			raw:      makeShellCall(t, "call_shell_pending_rebuild", "shell_pending_rebuild", []string{"echo pending"}, "in_progress"),
			toolName: "shell",
		},
		{
			raw:      makeApplyPatchDict(t, "call_apply_pending_rebuild", ""),
			toolName: "apply_patch",
		},
		{
			raw: map[string]any{
				"type":      "function_call",
				"name":      "missing_tool",
				"call_id":   "call_missing_tool",
				"arguments": "{}",
			},
			toolName: "missing_tool",
		},
	}

	for _, entry := range rawItems {
		t.Run(entry.toolName, func(t *testing.T) {
			_, agent := makeModelAndAgent(nil, "unmatched")
			approvalItem := agents.ToolApprovalItem{ToolName: entry.toolName, RawItem: entry.raw}
			state := makeStateWithInterruptions(agent, []agents.ToolApprovalItem{approvalItem}, "resume approvals", agents.DefaultMaxTurns)

			result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
			require.NoError(t, err)
			require.NotEmpty(t, result.Interruptions)
			assert.Equal(t, approvalItem.ToolName, result.Interruptions[0].ToolName)
		})
	}
}

func TestRejectedShellCallsEmitRejectionOutput(t *testing.T) {
	shellTool := agents.ShellTool{
		Executor:      func(context.Context, agents.ShellCommandRequest) (any, error) { return "should_not_run", nil },
		NeedsApproval: agents.ShellNeedsApprovalEnabled(),
	}
	_, agent := makeModelAndAgent([]agents.Tool{shellTool}, "shell-reject")
	contextWrapper := makeContextWrapper()

	shellCall := makeShellCall(t, "call_reject_shell", "shell_reject", []string{"echo test"}, "in_progress")
	approvalItem := agents.ToolApprovalItem{ToolName: shellTool.ToolName(), RawItem: shellCall}
	contextWrapper.RejectTool(approvalItem, false)

	state := makeStateWithInterruptions(agent, []agents.ToolApprovalItem{approvalItem}, "resume shell", agents.DefaultMaxTurns)
	state.ModelResponses = []agents.ModelResponse{
		{Output: []responses.ResponseOutputItemUnion{shellCall}, Usage: usage.NewUsage(), ResponseID: "resp"},
	}
	state.SetToolApprovalsFromContext(contextWrapper)

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	var rejectionOutputs []agents.ToolCallOutputItem
	for _, item := range result.NewItems {
		outputItem, ok := item.(agents.ToolCallOutputItem)
		if !ok {
			continue
		}
		var raw map[string]any
		switch typed := outputItem.RawItem.(type) {
		case agents.ShellCallOutputRawItem:
			raw = map[string]any(typed)
		default:
			raw = mustJSONMap(t, outputItem.RawItem)
		}
		if raw == nil || raw["type"] != "shell_call_output" {
			continue
		}
		outputEntries, _ := raw["output"].([]any)
		if len(outputEntries) == 0 {
			continue
		}
		first, _ := outputEntries[0].(map[string]any)
		if first != nil && first["stderr"] == hitlRejectionMessage {
			rejectionOutputs = append(rejectionOutputs, outputItem)
		}
	}
	assert.NotEmpty(t, rejectionOutputs)
}

func TestRejectedShellCallsWithExistingOutputAreNotDuplicated(t *testing.T) {
	shellTool := agents.ShellTool{
		Executor:      func(context.Context, agents.ShellCommandRequest) (any, error) { return "should_not_run", nil },
		NeedsApproval: agents.ShellNeedsApprovalEnabled(),
	}
	_, agent := makeModelAndAgent([]agents.Tool{shellTool}, "shell-reject-dup")
	contextWrapper := makeContextWrapper()

	shellCall := makeShellCall(t, "call_reject_shell_dup", "shell_reject_dup", []string{"echo test"}, "in_progress")
	approvalItem := agents.ToolApprovalItem{ToolName: shellTool.ToolName(), RawItem: shellCall}
	contextWrapper.RejectTool(approvalItem, false)

	state := makeStateWithInterruptions(agent, nil, "resume shell rejection existing", agents.DefaultMaxTurns)
	state.ModelResponses = []agents.ModelResponse{
		{Output: []responses.ResponseOutputItemUnion{shellCall}, Usage: usage.NewUsage(), ResponseID: "resp"},
	}
	rejectionOutput := responses.ResponseInputItemShellCallOutputParam{
		CallID: "call_reject_shell_dup",
		Type:   constant.ValueOf[constant.ShellCallOutput](),
	}
	state.GeneratedItems = []agents.TResponseInputItem{
		{OfShellCallOutput: &rejectionOutput},
	}
	state.SetToolApprovalsFromContext(contextWrapper)

	result, err := agents.Runner{}.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	assert.Empty(t, result.NewItems)
}

func TestMCPCallbackApprovalsProcessed(t *testing.T) {
	agent := agents.New("mcp-callback")
	callback := func(_ context.Context, _ responses.ResponseOutputItemMcpApprovalRequest) (agents.MCPToolApprovalFunctionResult, error) {
		return agents.MCPToolApprovalFunctionResult{
			Approve: true,
			Reason:  "ok",
		}, nil
	}

	request := agents.ToolRunMCPApprovalRequest{
		RequestItem: responses.ResponseOutputItemMcpApprovalRequest{
			ID:          "mcp-callback-1",
			Type:        constant.ValueOf[constant.McpApprovalRequest](),
			ServerLabel: "server",
			Name:        "hosted_mcp",
			Arguments:   "{}",
		},
		MCPTool: agents.HostedMCPTool{
			OnApprovalRequest: callback,
			ToolConfig: responses.ToolMcpParam{
				ServerLabel: "server",
				Type:        constant.ValueOf[constant.Mcp](),
			},
		},
	}

	items, err := agents.RunImpl().ExecuteMCPApprovalRequests(t.Context(), agent, []agents.ToolRunMCPApprovalRequest{request})
	require.NoError(t, err)
	require.Len(t, items, 1)
	_, ok := items[0].(agents.MCPApprovalResponseItem)
	assert.True(t, ok)
}
