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
	"encoding/json"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDeepSeekReasoningContentPreservedInToolCalls(t *testing.T) {
	message := chatCompletionMessageWithReasoningContent(t,
		"Let me think about getting the weather for Tokyo...",
		"call_123",
		"get_weather",
		`{"city":"Tokyo"}`,
	)

	outputItems, err := agents.ChatCmplConverter().MessageToOutputItems(message, map[string]any{
		"model": "deepseek/deepseek-reasoner",
	})
	require.NoError(t, err)

	inputItems := []agents.TResponseInputItem{
		userMessageItem("What's the weather in Tokyo?"),
	}
	inputItems = append(inputItems, agents.ModelResponse{Output: outputItems}.ToInputItems()...)
	inputItems = append(inputItems, functionCallOutputItem("call_123", "The weather in Tokyo is sunny."))

	messages, err := agents.ChatCmplConverter().ItemsToMessagesWithModel(agents.InputItems(inputItems), "deepseek/deepseek-reasoner")
	require.NoError(t, err)

	assistantMsg := findAssistantWithToolCalls(t, messages)
	require.NotNil(t, assistantMsg)
	_, ok := assistantMsg["reasoning_content"]
	assert.True(t, ok)
}

func TestDeepSeekReasoningContentInMultiTurnConversation(t *testing.T) {
	message := chatCompletionMessageWithReasoningContent(t,
		"I need to get the weather for Tokyo first.",
		"call_weather_123",
		"get_weather",
		`{"city":"Tokyo"}`,
	)

	outputItems, err := agents.ChatCmplConverter().MessageToOutputItems(message, map[string]any{
		"model": "deepseek/deepseek-reasoner",
	})
	require.NoError(t, err)

	inputItems := []agents.TResponseInputItem{
		userMessageItem("What's the weather in Tokyo?"),
	}
	inputItems = append(inputItems, agents.ModelResponse{Output: outputItems}.ToInputItems()...)
	inputItems = append(inputItems, functionCallOutputItem("call_weather_123", "The weather in Tokyo is sunny and 22C."))

	messages, err := agents.ChatCmplConverter().ItemsToMessagesWithModel(agents.InputItems(inputItems), "deepseek/deepseek-reasoner")
	require.NoError(t, err)

	assistantMsg := findAssistantWithToolCalls(t, messages)
	require.NotNil(t, assistantMsg)
	_, ok := assistantMsg["reasoning_content"]
	assert.True(t, ok)
}

func TestDeepSeekReasoningContentWithOpenAIChatCompletionsPath(t *testing.T) {
	reasoning := responses.ResponseReasoningItemParam{
		ID: "__fake_id__",
		Summary: []responses.ResponseReasoningItemSummaryParam{
			{
				Text: "I need to check the weather in Paris.",
				Type: constant.ValueOf[constant.SummaryText](),
			},
		},
		Type: constant.ValueOf[constant.Reasoning](),
	}
	reasoning.SetExtraFields(map[string]any{
		"provider_data": map[string]any{
			"model":       "deepseek-reasoner",
			"response_id": "chatcmpl-test",
		},
	})

	funcCall := responses.ResponseFunctionToolCallParam{
		ID:        param.NewOpt("__fake_id__"),
		CallID:    "call_weather_456",
		Name:      "get_weather",
		Arguments: `{"city": "Paris"}`,
		Type:      constant.ValueOf[constant.FunctionCall](),
	}
	funcCall.SetExtraFields(map[string]any{
		"provider_data": map[string]any{
			"model": "deepseek-reasoner",
		},
	})

	inputItems := []agents.TResponseInputItem{
		userMessageItem("What's the weather in Paris?"),
		{OfReasoning: &reasoning},
		{OfFunctionCall: &funcCall},
		functionCallOutputItem("call_weather_456", "The weather in Paris is cloudy and 15C."),
	}

	messages, err := agents.ChatCmplConverter().ItemsToMessagesWithModel(agents.InputItems(inputItems), "deepseek-reasoner")
	require.NoError(t, err)

	assistantMsg := findAssistantWithToolCalls(t, messages)
	require.NotNil(t, assistantMsg)
	assert.Equal(t, "I need to check the weather in Paris.", assistantMsg["reasoning_content"])
}

func TestReasoningContentFromOtherProviderNotAttachedToDeepseek(t *testing.T) {
	reasoning := responses.ResponseReasoningItemParam{
		ID: "__fake_id__",
		Summary: []responses.ResponseReasoningItemSummaryParam{
			{
				Text: "Claude's reasoning about the weather.",
				Type: constant.ValueOf[constant.SummaryText](),
			},
		},
		Type: constant.ValueOf[constant.Reasoning](),
	}
	reasoning.SetExtraFields(map[string]any{
		"provider_data": map[string]any{
			"model":       "claude-sonnet-4-20250514",
			"response_id": "chatcmpl-test",
		},
	})

	funcCall := responses.ResponseFunctionToolCallParam{
		ID:        param.NewOpt("__fake_id__"),
		CallID:    "call_weather_789",
		Name:      "get_weather",
		Arguments: `{"city": "Paris"}`,
		Type:      constant.ValueOf[constant.FunctionCall](),
	}
	funcCall.SetExtraFields(map[string]any{
		"provider_data": map[string]any{
			"model": "claude-sonnet-4-20250514",
		},
	})

	inputItems := []agents.TResponseInputItem{
		userMessageItem("What's the weather in Paris?"),
		{OfReasoning: &reasoning},
		{OfFunctionCall: &funcCall},
		functionCallOutputItem("call_weather_789", "The weather in Paris is cloudy."),
	}

	messages, err := agents.ChatCmplConverter().ItemsToMessagesWithModel(agents.InputItems(inputItems), "deepseek-reasoner")
	require.NoError(t, err)

	assistantMsg := findAssistantWithToolCalls(t, messages)
	require.NotNil(t, assistantMsg)
	_, ok := assistantMsg["reasoning_content"]
	assert.False(t, ok)
}

func TestReasoningContentWithoutProviderDataAttachedForBackwardCompat(t *testing.T) {
	reasoning := responses.ResponseReasoningItemParam{
		ID: "__fake_id__",
		Summary: []responses.ResponseReasoningItemSummaryParam{
			{
				Text: "Reasoning without provider info.",
				Type: constant.ValueOf[constant.SummaryText](),
			},
		},
		Type: constant.ValueOf[constant.Reasoning](),
	}

	funcCall := responses.ResponseFunctionToolCallParam{
		ID:        param.NewOpt("__fake_id__"),
		CallID:    "call_weather_101",
		Name:      "get_weather",
		Arguments: `{"city": "Tokyo"}`,
		Type:      constant.ValueOf[constant.FunctionCall](),
	}

	inputItems := []agents.TResponseInputItem{
		userMessageItem("What's the weather in Tokyo?"),
		{OfReasoning: &reasoning},
		{OfFunctionCall: &funcCall},
		functionCallOutputItem("call_weather_101", "The weather in Tokyo is sunny."),
	}

	messages, err := agents.ChatCmplConverter().ItemsToMessagesWithModel(agents.InputItems(inputItems), "deepseek-reasoner")
	require.NoError(t, err)

	assistantMsg := findAssistantWithToolCalls(t, messages)
	require.NotNil(t, assistantMsg)
	assert.Equal(t, "Reasoning without provider info.", assistantMsg["reasoning_content"])
}

func chatCompletionMessageWithReasoningContent(t *testing.T, reasoningContent, callID, toolName, toolArguments string) openai.ChatCompletionMessage {
	t.Helper()

	rawMessage := map[string]any{
		"role":    "assistant",
		"content": nil,
		"tool_calls": []map[string]any{
			{
				"id":   callID,
				"type": "function",
				"function": map[string]any{
					"name":      toolName,
					"arguments": toolArguments,
				},
			},
		},
		"reasoning_content": reasoningContent,
	}
	rawBytes, err := json.Marshal(rawMessage)
	require.NoError(t, err)

	var message openai.ChatCompletionMessage
	require.NoError(t, json.Unmarshal(rawBytes, &message))
	return message
}

func userMessageItem(content string) agents.TResponseInputItem {
	return agents.TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{
				OfString: param.NewOpt(content),
			},
			Role: responses.EasyInputMessageRoleUser,
			Type: responses.EasyInputMessageTypeMessage,
		},
	}
}

func functionCallOutputItem(callID, output string) agents.TResponseInputItem {
	return agents.TResponseInputItem{
		OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
			CallID: callID,
			Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
				OfString: param.NewOpt(output),
			},
			Type: constant.ValueOf[constant.FunctionCallOutput](),
		},
	}
}

func findAssistantWithToolCalls(t *testing.T, messages []openai.ChatCompletionMessageParamUnion) map[string]any {
	t.Helper()

	for _, msg := range messages {
		raw, err := json.Marshal(msg)
		require.NoError(t, err)

		var payload map[string]any
		require.NoError(t, json.Unmarshal(raw, &payload))
		if payload["role"] != "assistant" {
			continue
		}
		toolCalls, ok := payload["tool_calls"].([]any)
		if ok && len(toolCalls) > 0 {
			return payload
		}
	}

	return nil
}
