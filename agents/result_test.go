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
	"testing"

	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunResultToInputListNormalizedUsesModelInputItems(t *testing.T) {
	toolCallRaw := ResponseFunctionToolCall{
		ID:        "fc_1",
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "delegate",
		CallID:    "call_1",
		Arguments: `{}`,
	}
	toolCallItem := ToolCallItem{
		RawItem: toolCallRaw,
		Type:    "tool_call_item",
	}
	toolOutputItem := ToolCallOutputItem{
		RawItem: ResponseInputItemFunctionCallOutputParam(
			ItemHelpers().ToolCallOutputItem(toolCallRaw, "delegated"),
		),
		Output: "delegated",
		Type:   "tool_call_output_item",
	}
	messageItem := createMessageOutputItem(nil, "final answer")

	result := RunResult{
		Input:           InputString("hello"),
		NewItems:        []RunItem{toolCallItem, toolOutputItem, messageItem},
		ModelInputItems: []RunItem{messageItem},
	}

	preserveAll := result.ToInputList()
	normalized := result.ToInputList(ToInputListModeNormalized)

	require.Len(t, preserveAll, 4)
	require.Len(t, normalized, 2)

	require.NotNil(t, preserveAll[1].GetType())
	assert.Equal(t, "function_call", *preserveAll[1].GetType())
	require.NotNil(t, preserveAll[2].GetType())
	assert.Equal(t, "function_call_output", *preserveAll[2].GetType())
	require.NotNil(t, normalized[1].GetType())
	assert.Equal(t, "message", *normalized[1].GetType())
}

func TestRunResultToInputListNormalizedFallsBackToPreserveAll(t *testing.T) {
	messageItem := createMessageOutputItem(nil, "final answer")
	result := RunResult{
		Input:    InputString("hello"),
		NewItems: []RunItem{messageItem},
	}

	assert.Equal(t, result.ToInputList(), result.ToInputList(ToInputListModeNormalized))
}

func TestRunResultStreamingToInputListNormalizedUsesModelInputItems(t *testing.T) {
	streamed := newRunResultStreaming(t.Context())
	streamed.setInput(InputString("hello"))

	toolCallRaw := ResponseFunctionToolCall{
		ID:        "fc_2",
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "delegate",
		CallID:    "call_2",
		Arguments: `{}`,
	}
	toolCallItem := ToolCallItem{
		RawItem: toolCallRaw,
		Type:    "tool_call_item",
	}
	toolOutputItem := ToolCallOutputItem{
		RawItem: ResponseInputItemFunctionCallOutputParam(
			ItemHelpers().ToolCallOutputItem(toolCallRaw, "delegated"),
		),
		Output: "delegated",
		Type:   "tool_call_output_item",
	}
	messageItem := createMessageOutputItem(nil, "streamed answer")

	streamed.setNewItems([]RunItem{toolCallItem, toolOutputItem, messageItem})
	streamed.setModelInputItems([]RunItem{messageItem})

	preserveAll := streamed.ToInputList()
	normalized := streamed.ToInputList(ToInputListModeNormalized)

	require.Len(t, preserveAll, 4)
	require.Len(t, normalized, 2)
	require.NotNil(t, normalized[1].GetType())
	assert.Equal(t, "message", *normalized[1].GetType())
}
