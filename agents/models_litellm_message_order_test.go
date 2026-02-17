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
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFixToolMessageOrderingBasicReorder(t *testing.T) {
	messages := []openai.ChatCompletionMessageParamUnion{
		userMessage("Hello"),
		toolMessage("call_123", "Result for call_123"),
		assistantWithToolCalls("", toolCall("call_123", "test")),
		userMessage("Thanks"),
	}

	result := fixToolMessageOrdering(messages)

	require.Len(t, result, 4)
	assert.Equal(t, "user", roleOfMessage(result[0]))
	assert.Equal(t, "assistant", roleOfMessage(result[1]))
	assert.Equal(t, "call_123", assistantToolCallID(result[1], 0))
	assert.Equal(t, "tool", roleOfMessage(result[2]))
	assert.Equal(t, "call_123", toolResultCallID(result[2]))
	assert.Equal(t, "user", roleOfMessage(result[3]))
}

func TestFixToolMessageOrderingConsecutiveToolCalls(t *testing.T) {
	messages := []openai.ChatCompletionMessageParamUnion{
		userMessage("Hello"),
		assistantWithToolCalls("", toolCall("call_1", "test1")),
		assistantWithToolCalls("", toolCall("call_2", "test2")),
		toolMessage("call_1", "Result 1"),
		toolMessage("call_2", "Result 2"),
	}

	result := fixToolMessageOrdering(messages)

	require.Len(t, result, 5)
	assert.Equal(t, "user", roleOfMessage(result[0]))
	assert.Equal(t, "assistant", roleOfMessage(result[1]))
	assert.Equal(t, "call_1", assistantToolCallID(result[1], 0))
	assert.Equal(t, "tool", roleOfMessage(result[2]))
	assert.Equal(t, "call_1", toolResultCallID(result[2]))
	assert.Equal(t, "assistant", roleOfMessage(result[3]))
	assert.Equal(t, "call_2", assistantToolCallID(result[3], 0))
	assert.Equal(t, "tool", roleOfMessage(result[4]))
	assert.Equal(t, "call_2", toolResultCallID(result[4]))
}

func TestFixToolMessageOrderingUnmatchedToolResultPreserved(t *testing.T) {
	messages := []openai.ChatCompletionMessageParamUnion{
		userMessage("Hello"),
		assistantWithToolCalls("", toolCall("call_1", "test")),
		toolMessage("call_1", "Matched result"),
		toolMessage("call_orphan", "Orphaned result"),
		userMessage("End"),
	}

	result := fixToolMessageOrdering(messages)

	require.Len(t, result, 5)
	assert.Equal(t, "user", roleOfMessage(result[0]))
	assert.Equal(t, "assistant", roleOfMessage(result[1]))
	assert.Equal(t, "call_1", assistantToolCallID(result[1], 0))
	assert.Equal(t, "tool", roleOfMessage(result[2]))
	assert.Equal(t, "call_1", toolResultCallID(result[2]))
	assert.Equal(t, "tool", roleOfMessage(result[3]))
	assert.Equal(t, "call_orphan", toolResultCallID(result[3]))
	assert.Equal(t, "user", roleOfMessage(result[4]))
}

func TestFixToolMessageOrderingToolCallWithoutResultPreserved(t *testing.T) {
	messages := []openai.ChatCompletionMessageParamUnion{
		userMessage("Hello"),
		assistantWithToolCalls("", toolCall("call_1", "test")),
		userMessage("End"),
	}

	result := fixToolMessageOrdering(messages)

	require.Len(t, result, 3)
	assert.Equal(t, "user", roleOfMessage(result[0]))
	assert.Equal(t, "assistant", roleOfMessage(result[1]))
	assert.Equal(t, "call_1", assistantToolCallID(result[1], 0))
	assert.Equal(t, "user", roleOfMessage(result[2]))
}

func TestFixToolMessageOrderingAlreadyOrderedUnchanged(t *testing.T) {
	messages := []openai.ChatCompletionMessageParamUnion{
		userMessage("Hello"),
		assistantWithToolCalls("", toolCall("call_1", "test")),
		toolMessage("call_1", "Result"),
		assistantMessage("Done"),
	}

	result := fixToolMessageOrdering(messages)

	require.Len(t, result, 4)
	assert.Equal(t, "user", roleOfMessage(result[0]))
	assert.Equal(t, "assistant", roleOfMessage(result[1]))
	assert.Equal(t, "call_1", assistantToolCallID(result[1], 0))
	assert.Equal(t, "tool", roleOfMessage(result[2]))
	assert.Equal(t, "call_1", toolResultCallID(result[2]))
	assert.Equal(t, "assistant", roleOfMessage(result[3]))
}

func TestFixToolMessageOrderingMultipleToolCallsSingleMessage(t *testing.T) {
	messages := []openai.ChatCompletionMessageParamUnion{
		userMessage("Hello"),
		assistantWithToolCalls("", toolCall("call_1", "test1"), toolCall("call_2", "test2")),
		toolMessage("call_1", "Result 1"),
		toolMessage("call_2", "Result 2"),
	}

	result := fixToolMessageOrdering(messages)

	require.Len(t, result, 5)
	assert.Equal(t, "user", roleOfMessage(result[0]))
	assert.Equal(t, "assistant", roleOfMessage(result[1]))
	assert.Equal(t, "call_1", assistantToolCallID(result[1], 0))
	assert.Equal(t, "tool", roleOfMessage(result[2]))
	assert.Equal(t, "call_1", toolResultCallID(result[2]))
	assert.Equal(t, "assistant", roleOfMessage(result[3]))
	assert.Equal(t, "call_2", assistantToolCallID(result[3], 0))
	assert.Equal(t, "tool", roleOfMessage(result[4]))
	assert.Equal(t, "call_2", toolResultCallID(result[4]))
}

func TestFixToolMessageOrderingEmptyMessages(t *testing.T) {
	messages := []openai.ChatCompletionMessageParamUnion{}
	assert.Empty(t, fixToolMessageOrdering(messages))
}

func TestFixToolMessageOrderingNoToolMessagesUnchanged(t *testing.T) {
	messages := []openai.ChatCompletionMessageParamUnion{
		userMessage("Hello"),
		assistantMessage("Hi there"),
		userMessage("How are you?"),
	}

	result := fixToolMessageOrdering(messages)

	assert.Equal(t, messages, result)
}

func TestFixToolMessageOrderingComplexScenario(t *testing.T) {
	messages := []openai.ChatCompletionMessageParamUnion{
		userMessage("Start"),
		toolMessage("call_out_of_order", "Out of order result"),
		assistantMessage("Regular response"),
		assistantWithToolCalls("", toolCall("call_out_of_order", "test")),
		assistantWithToolCalls("", toolCall("call_normal", "test2")),
		toolMessage("call_normal", "Normal result"),
		toolMessage("call_orphan", "Orphaned result"),
		userMessage("End"),
	}

	result := fixToolMessageOrdering(messages)

	require.Len(t, result, 8)
	assert.Equal(t, "user", roleOfMessage(result[0]))
	assert.Equal(t, "assistant", roleOfMessage(result[1]))
	assert.Equal(t, "assistant", roleOfMessage(result[2]))
	assert.Equal(t, "call_out_of_order", assistantToolCallID(result[2], 0))
	assert.Equal(t, "tool", roleOfMessage(result[3]))
	assert.Equal(t, "call_out_of_order", toolResultCallID(result[3]))
	assert.Equal(t, "assistant", roleOfMessage(result[4]))
	assert.Equal(t, "call_normal", assistantToolCallID(result[4], 0))
	assert.Equal(t, "tool", roleOfMessage(result[5]))
	assert.Equal(t, "call_normal", toolResultCallID(result[5]))
	assert.Equal(t, "tool", roleOfMessage(result[6]))
	assert.Equal(t, "call_orphan", toolResultCallID(result[6]))
	assert.Equal(t, "user", roleOfMessage(result[7]))
}

func userMessage(content string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{
		OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfString: param.NewOpt(content),
			},
			Role: constant.ValueOf[constant.User](),
		},
	}
}

func assistantMessage(content string) openai.ChatCompletionMessageParamUnion {
	return assistantWithToolCalls(content)
}

func assistantWithToolCalls(content string, calls ...openai.ChatCompletionMessageToolCallUnionParam) openai.ChatCompletionMessageParamUnion {
	assistant := &openai.ChatCompletionAssistantMessageParam{
		Role: constant.ValueOf[constant.Assistant](),
	}
	if content != "" {
		assistant.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
			OfString: param.NewOpt(content),
		}
	}
	if len(calls) > 0 {
		assistant.ToolCalls = calls
	}
	return openai.ChatCompletionMessageParamUnion{OfAssistant: assistant}
}

func toolMessage(callID string, content string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{
		OfTool: &openai.ChatCompletionToolMessageParam{
			Content: openai.ChatCompletionToolMessageParamContentUnion{
				OfString: param.NewOpt(content),
			},
			ToolCallID: callID,
			Role:       constant.ValueOf[constant.Tool](),
		},
	}
}

func toolCall(id string, name string) openai.ChatCompletionMessageToolCallUnionParam {
	return openai.ChatCompletionMessageToolCallUnionParam{
		OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
			ID: id,
			Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
				Name:      name,
				Arguments: "{}",
			},
			Type: constant.ValueOf[constant.Function](),
		},
	}
}

func roleOfMessage(msg openai.ChatCompletionMessageParamUnion) string {
	switch {
	case msg.OfUser != nil:
		return "user"
	case msg.OfAssistant != nil:
		return "assistant"
	case msg.OfTool != nil:
		return "tool"
	case msg.OfSystem != nil:
		return "system"
	case msg.OfDeveloper != nil:
		return "developer"
	case msg.OfFunction != nil:
		return "function"
	default:
		return ""
	}
}

func assistantToolCallID(msg openai.ChatCompletionMessageParamUnion, index int) string {
	if msg.OfAssistant == nil {
		return ""
	}
	if index < 0 || index >= len(msg.OfAssistant.ToolCalls) {
		return ""
	}
	return toolCallID(msg.OfAssistant.ToolCalls[index])
}

func toolResultCallID(msg openai.ChatCompletionMessageParamUnion) string {
	if msg.OfTool == nil {
		return ""
	}
	return msg.OfTool.ToolCallID
}
