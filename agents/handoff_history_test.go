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
	"strings"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func createMockAgent() *agents.Agent {
	return &agents.Agent{Name: "test_agent"}
}

func createToolCallItem(agent *agents.Agent) agents.ToolCallItem {
	raw := responses.ResponseFunctionToolCall{
		ID:        "call_tool_123",
		CallID:    "call_tool_123",
		Name:      "get_weather",
		Arguments: `{"city": "London"}`,
		Type:      constant.ValueOf[constant.FunctionCall](),
	}
	return agents.ToolCallItem{
		Agent:   agent,
		RawItem: agents.ResponseFunctionToolCall(raw),
		Type:    "tool_call_item",
	}
}

func createToolOutputItem(agent *agents.Agent) agents.ToolCallOutputItem {
	raw := responses.ResponseInputItemFunctionCallOutputParam{
		CallID: "call_tool_123",
		Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
			OfString: param.NewOpt("Sunny, 22C"),
		},
		Type: constant.ValueOf[constant.FunctionCallOutput](),
	}
	return agents.ToolCallOutputItem{
		Agent:   agent,
		RawItem: agents.ResponseInputItemFunctionCallOutputParam(raw),
		Output:  "Sunny, 22C",
		Type:    "tool_call_output_item",
	}
}

func createHandoffCallItem(agent *agents.Agent) agents.HandoffCallItem {
	raw := responses.ResponseFunctionToolCall{
		ID:        "call_handoff_456",
		CallID:    "call_handoff_456",
		Name:      "transfer_to_agent_b",
		Arguments: "{}",
		Type:      constant.ValueOf[constant.FunctionCall](),
	}
	return agents.HandoffCallItem{
		Agent:   agent,
		RawItem: raw,
		Type:    "handoff_call_item",
	}
}

func createHandoffOutputItem(agent *agents.Agent) agents.HandoffOutputItem {
	raw := agents.TResponseInputItem{
		OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
			CallID: "call_handoff_456",
			Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
				OfString: param.NewOpt(`{"assistant": "agent_b"}`),
			},
			Type: constant.ValueOf[constant.FunctionCallOutput](),
		},
	}
	return agents.HandoffOutputItem{
		Agent:       agent,
		RawItem:     raw,
		SourceAgent: agent,
		TargetAgent: agent,
		Type:        "handoff_output_item",
	}
}

func createMessageItem(agent *agents.Agent) agents.MessageOutputItem {
	raw := responses.ResponseOutputMessage{
		ID: "msg_123",
		Content: []responses.ResponseOutputMessageContentUnion{{
			Text:        "Hello!",
			Type:        "output_text",
			Annotations: nil,
		}},
		Role:   constant.ValueOf[constant.Assistant](),
		Status: responses.ResponseOutputMessageStatusCompleted,
		Type:   constant.ValueOf[constant.Message](),
	}
	return agents.MessageOutputItem{
		Agent:   agent,
		RawItem: raw,
		Type:    "message_output_item",
	}
}

func createToolApprovalItem() agents.ToolApprovalItem {
	return agents.ToolApprovalItem{
		RawItem: map[string]any{
			"type":      "function_call",
			"call_id":   "call_tool_approve",
			"name":      "needs_approval",
			"arguments": "{}",
		},
	}
}

func extractHistoryItems(t *testing.T, input agents.Input) []agents.TResponseInputItem {
	t.Helper()
	history, ok := input.(agents.InputItems)
	require.True(t, ok, "expected InputItems history")
	return history
}

func extractSummaryContent(t *testing.T, item agents.TResponseInputItem) string {
	t.Helper()
	if item.OfMessage == nil || !item.OfMessage.Content.OfString.Valid() {
		require.FailNow(t, "expected message summary content")
	}
	return item.OfMessage.Content.OfString.Value
}

func TestPreHandoffToolItemsAreFiltered(t *testing.T) {
	agent := createMockAgent()
	handoffData := agents.HandoffInputData{
		InputHistory: agents.InputItems{agentstesting.GetTextInputItem("Hello")},
		PreHandoffItems: []agents.RunItem{
			createToolCallItem(agent),
			createToolOutputItem(agent),
		},
		NewItems: nil,
	}

	nested := agents.NestHandoffHistory(handoffData, nil)

	assert.Len(t, nested.PreHandoffItems, 0)

	history := extractHistoryItems(t, nested.InputHistory)
	require.Len(t, history, 1)
	summary := extractSummaryContent(t, history[0])
	assert.Contains(t, summary, "<CONVERSATION HISTORY>")
}

func TestToolApprovalItemsAreSkipped(t *testing.T) {
	handoffData := agents.HandoffInputData{
		InputHistory: agents.InputItems{agentstesting.GetTextInputItem("Hello")},
		PreHandoffItems: []agents.RunItem{
			createToolApprovalItem(),
		},
		NewItems: nil,
	}

	nested := agents.NestHandoffHistory(handoffData, nil)

	assert.Len(t, nested.PreHandoffItems, 0)
	require.NotNil(t, nested.InputItems)
	assert.Len(t, nested.InputItems, 0)
}

func TestNewItemsHandoffOutputIsFilteredForInput(t *testing.T) {
	agent := createMockAgent()
	handoffData := agents.HandoffInputData{
		InputHistory:    agents.InputItems{agentstesting.GetTextInputItem("Hello")},
		PreHandoffItems: nil,
		NewItems: []agents.RunItem{
			createHandoffCallItem(agent),
			createHandoffOutputItem(agent),
		},
	}

	nested := agents.NestHandoffHistory(handoffData, nil)

	assert.Len(t, nested.NewItems, 2)
	require.NotNil(t, nested.InputItems)

	hasHandoffOutput := false
	for _, item := range nested.InputItems {
		switch item.(type) {
		case agents.HandoffOutputItem, *agents.HandoffOutputItem:
			hasHandoffOutput = true
		}
	}
	assert.False(t, hasHandoffOutput)
}

func TestMessageItemsArePreservedInNewItems(t *testing.T) {
	agent := createMockAgent()
	handoffData := agents.HandoffInputData{
		InputHistory:    agents.InputItems{agentstesting.GetTextInputItem("Hello")},
		PreHandoffItems: nil,
		NewItems: []agents.RunItem{
			createMessageItem(agent),
		},
	}

	nested := agents.NestHandoffHistory(handoffData, nil)

	assert.Len(t, nested.NewItems, 1)
	require.NotNil(t, nested.InputItems)
	require.Len(t, nested.InputItems, 1)
	_, ok := nested.InputItems[0].(agents.MessageOutputItem)
	assert.True(t, ok)
}

func TestSummaryContainsFilteredItemsAsText(t *testing.T) {
	agent := createMockAgent()
	handoffData := agents.HandoffInputData{
		InputHistory: agents.InputItems{agentstesting.GetTextInputItem("Hello")},
		PreHandoffItems: []agents.RunItem{
			createToolCallItem(agent),
			createToolOutputItem(agent),
		},
		NewItems: nil,
	}

	nested := agents.NestHandoffHistory(handoffData, nil)

	history := extractHistoryItems(t, nested.InputHistory)
	require.Len(t, history, 1)
	summary := extractSummaryContent(t, history[0])

	assert.True(t, strings.Contains(summary, "function_call") || strings.Contains(summary, "get_weather"))
}

func TestInputItemsFieldExistsAfterNesting(t *testing.T) {
	agent := createMockAgent()
	handoffData := agents.HandoffInputData{
		InputHistory:    agents.InputItems{agentstesting.GetTextInputItem("Hello")},
		PreHandoffItems: nil,
		NewItems: []agents.RunItem{
			createHandoffCallItem(agent),
		},
	}

	nested := agents.NestHandoffHistory(handoffData, nil)

	require.NotNil(t, nested.InputItems)
}

func TestFullHandoffScenarioNoDuplication(t *testing.T) {
	agent := createMockAgent()
	handoffData := agents.HandoffInputData{
		InputHistory: agents.InputItems{agentstesting.GetTextInputItem("What's the weather?")},
		PreHandoffItems: []agents.RunItem{
			createToolCallItem(agent),
			createToolOutputItem(agent),
		},
		NewItems: []agents.RunItem{
			createMessageItem(agent),
			createHandoffCallItem(agent),
			createHandoffOutputItem(agent),
		},
	}

	nested := agents.NestHandoffHistory(handoffData, nil)

	history := extractHistoryItems(t, nested.InputHistory)
	totalModelItems := len(history) + len(nested.PreHandoffItems) + len(nested.InputItems)
	assert.LessOrEqual(t, totalModelItems, 3)

	allInputItems := append([]agents.RunItem{}, nested.PreHandoffItems...)
	allInputItems = append(allInputItems, nested.InputItems...)
	for _, item := range allInputItems {
		switch item.(type) {
		case agents.ToolCallOutputItem, *agents.ToolCallOutputItem, agents.HandoffOutputItem, *agents.HandoffOutputItem:
			assert.Fail(t, "function_call_output items should not be in model input")
		}
	}
}
