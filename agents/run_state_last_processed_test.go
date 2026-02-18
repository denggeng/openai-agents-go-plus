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
	"encoding/json"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunStateToJSONIncludesToolCallFromLastProcessedResponse(t *testing.T) {
	toolCall := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "test_tool",
		CallID:    "call_1",
		Arguments: "{}",
	}
	item := agents.ToolCallItem{
		Agent:   &agents.Agent{Name: "agent"},
		RawItem: agents.ResponseFunctionToolCall(toolCall),
		Type:    "tool_call_item",
	}
	processed := agents.ProcessedResponse{
		NewItems: []agents.RunItem{item},
	}
	state := agents.RunState{
		SchemaVersion:         agents.CurrentRunStateSchemaVersion,
		LastProcessedResponse: &processed,
	}

	raw, err := state.ToJSON()
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(raw, &payload))

	generated, ok := payload["generated_items"].([]any)
	require.True(t, ok)
	require.Len(t, generated, 1)

	entry, ok := generated[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "tool_call_item", entry["type"])
	rawItem, ok := entry["raw_item"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "test_tool", rawItem["name"])
}

func TestRunStateToJSONDeduplicatesGeneratedAndLastProcessed(t *testing.T) {
	toolCall := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "test_tool",
		CallID:    "call_1",
		Arguments: "{}",
	}
	item := agents.ToolCallItem{
		RawItem: agents.ResponseFunctionToolCall(toolCall),
		Type:    "tool_call_item",
	}
	processed := agents.ProcessedResponse{
		NewItems: []agents.RunItem{item},
	}
	state := agents.RunState{
		SchemaVersion:         agents.CurrentRunStateSchemaVersion,
		GeneratedRunItems:     []agents.RunItem{item},
		LastProcessedResponse: &processed,
	}

	raw, err := state.ToJSON()
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(raw, &payload))

	generated, ok := payload["generated_items"].([]any)
	require.True(t, ok)
	require.Len(t, generated, 1)

	restored, err := agents.RunStateFromJSON(raw)
	require.NoError(t, err)
	require.Len(t, restored.GeneratedRunItems, 1)
}

func TestRunStateFromJSONRestoresLastProcessedResponse(t *testing.T) {
	toolCall := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "test_tool",
		CallID:    "call_1",
		Arguments: "{}",
	}
	item := agents.ToolCallItem{
		RawItem: agents.ResponseFunctionToolCall(toolCall),
		Type:    "tool_call_item",
	}
	processed := agents.ProcessedResponse{
		NewItems: []agents.RunItem{item},
	}
	state := agents.RunState{
		SchemaVersion:         agents.CurrentRunStateSchemaVersion,
		LastProcessedResponse: &processed,
	}

	raw, err := state.ToJSON()
	require.NoError(t, err)

	restored, err := agents.RunStateFromJSON(raw)
	require.NoError(t, err)
	require.NotNil(t, restored.LastProcessedResponse)
	require.Len(t, restored.LastProcessedResponse.NewItems, 1)

	itemOut, ok := restored.LastProcessedResponse.NewItems[0].(agents.ToolCallItem)
	require.True(t, ok)
	rawOut, ok := itemOut.RawItem.(agents.ResponseFunctionToolCall)
	require.True(t, ok)
	assert.Equal(t, "call_1", rawOut.CallID)
}

func TestRunStateLastProcessedResponseSerializesLocalShellActions(t *testing.T) {
	call := responses.ResponseOutputItemLocalShellCall{
		ID:     "ls1",
		CallID: "call_local",
		Status: "completed",
		Type:   constant.ValueOf[constant.LocalShellCall](),
		Action: responses.ResponseOutputItemLocalShellCallAction{
			Command: []string{"echo", "hi"},
			Env:     map[string]string{},
			Type:    constant.ValueOf[constant.Exec](),
		},
	}
	processed := agents.ProcessedResponse{
		LocalShellCalls: []agents.ToolRunLocalShellCall{{
			ToolCall:       call,
			LocalShellTool: agents.LocalShellTool{},
		}},
	}
	state := agents.RunState{
		SchemaVersion:         agents.CurrentRunStateSchemaVersion,
		LastProcessedResponse: &processed,
	}

	raw, err := state.ToJSON()
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(raw, &payload))
	lastProcessed, ok := payload["last_processed_response"].(map[string]any)
	require.True(t, ok)
	actions, ok := lastProcessed["local_shell_actions"].([]any)
	require.True(t, ok)
	require.Len(t, actions, 1)
	action, ok := actions[0].(map[string]any)
	require.True(t, ok)
	localShell, ok := action["local_shell"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "local_shell", localShell["name"])

	restored, err := agents.RunStateFromJSON(raw)
	require.NoError(t, err)
	require.NotNil(t, restored.LastProcessedResponse)
	require.Len(t, restored.LastProcessedResponse.LocalShellCalls, 1)
	assert.Equal(t, "call_local", restored.LastProcessedResponse.LocalShellCalls[0].ToolCall.CallID)
}

func TestRunStateSessionItemsFallbackToMerge(t *testing.T) {
	toolCall1 := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "tool_one",
		CallID:    "call_1",
		Arguments: "{}",
	}
	toolCall2 := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "tool_two",
		CallID:    "call_2",
		Arguments: "{}",
	}
	item1 := agents.ToolCallItem{RawItem: agents.ResponseFunctionToolCall(toolCall1), Type: "tool_call_item"}
	item2 := agents.ToolCallItem{RawItem: agents.ResponseFunctionToolCall(toolCall2), Type: "tool_call_item"}

	processed := agents.ProcessedResponse{NewItems: []agents.RunItem{item2}}
	state := agents.RunState{
		SchemaVersion:         agents.CurrentRunStateSchemaVersion,
		GeneratedRunItems:     []agents.RunItem{item1},
		LastProcessedResponse: &processed,
	}

	raw, err := state.ToJSON()
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(raw, &payload))
	delete(payload, "session_items")
	raw, err = json.Marshal(payload)
	require.NoError(t, err)

	restored, err := agents.RunStateFromJSON(raw)
	require.NoError(t, err)
	require.Len(t, restored.SessionItems, 2)
}
