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

func TestConverterSkipsReasoningItems(t *testing.T) {
	items := agents.InputItems{
		easyUserMessageItem("Hello"),
		{
			OfReasoning: &responses.ResponseReasoningItemParam{
				ID: "reasoning_123",
				Summary: []responses.ResponseReasoningItemSummaryParam{
					{
						Text: "User said hello",
						Type: constant.ValueOf[constant.SummaryText](),
					},
				},
				Type: constant.ValueOf[constant.Reasoning](),
			},
		},
		{
			OfOutputMessage: &responses.ResponseOutputMessageParam{
				ID:     "msg_123",
				Type:   constant.ValueOf[constant.Message](),
				Role:   constant.ValueOf[constant.Assistant](),
				Status: responses.ResponseOutputMessageStatusCompleted,
				Content: []responses.ResponseOutputMessageContentUnionParam{
					{
						OfOutputText: &responses.ResponseOutputTextParam{
							Text: "Hi there!",
							Type: constant.ValueOf[constant.OutputText](),
						},
					},
				},
			},
		},
	}

	messages, err := agents.ChatCmplConverter().ItemsToMessages(items)
	require.NoError(t, err)
	require.Len(t, messages, 2)

	assistantPayload := findAssistantMessagePayload(t, messages)
	content, ok := assistantPayload["content"]
	require.True(t, ok)
	if parts, ok := content.([]any); ok {
		for _, part := range parts {
			partMap, ok := part.(map[string]any)
			if !ok {
				continue
			}
			assert.NotEqual(t, "thinking", partMap["type"])
		}
	}
}

func TestReasoningItemsPreservedInMessageConversion(t *testing.T) {
	message := chatCompletionMessageWithThinkingBlocks(t,
		"I'll check the weather in Paris for you.",
		"I need to call the weather function for Paris",
		[]map[string]any{
			{
				"type":      "thinking",
				"thinking":  "I need to call the weather function for Paris",
				"signature": "TestSignatureOne",
			},
		},
		nil,
	)

	outputItems, err := agents.ChatCmplConverter().MessageToOutputItems(message)
	require.NoError(t, err)

	var reasoningItems []agents.TResponseOutputItem
	for _, item := range outputItems {
		if item.Type == "reasoning" {
			reasoningItems = append(reasoningItems, item)
		}
	}
	require.Len(t, reasoningItems, 1)
	assert.Equal(t, "I need to call the weather function for Paris", reasoningItems[0].Summary[0].Text)

	payload := outputItemPayload(t, reasoningItems[0])
	contentAny, ok := payload["content"].([]any)
	require.True(t, ok)
	require.Len(t, contentAny, 1)
	firstContent := contentAny[0].(map[string]any)
	assert.Equal(t, "reasoning_text", firstContent["type"])
	assert.Equal(t, "I need to call the weather function for Paris", firstContent["text"])
	assert.Equal(t, "TestSignatureOne", payload["encrypted_content"])
}

func TestAnthropicThinkingBlocksWithToolCalls(t *testing.T) {
	message := chatCompletionMessageWithThinkingBlocks(t,
		"I'll check the weather for you.",
		"The user wants weather information, I need to call the weather function",
		[]map[string]any{
			{
				"type":      "thinking",
				"thinking":  "The user is asking about weather. Let me use the weather tool to get this information.",
				"signature": "TestSignature123",
			},
			{
				"type":      "thinking",
				"thinking":  "We should use the city Tokyo as the city.",
				"signature": "TestSignature456",
			},
		},
		[]map[string]any{
			{
				"id":   "call_123",
				"type": "function",
				"function": map[string]any{
					"name":      "get_weather",
					"arguments": `{"city": "Tokyo"}`,
				},
			},
		},
	)

	outputItems, err := agents.ChatCmplConverter().MessageToOutputItems(message)
	require.NoError(t, err)

	var reasoningItems []agents.TResponseOutputItem
	var toolCallItems []agents.TResponseOutputItem
	for _, item := range outputItems {
		if item.Type == "reasoning" {
			reasoningItems = append(reasoningItems, item)
		}
		if item.Type == "function_call" {
			toolCallItems = append(toolCallItems, item)
		}
	}
	require.Len(t, reasoningItems, 1)
	require.Len(t, toolCallItems, 1)
	assert.Equal(t, "get_weather", toolCallItems[0].Name)

	reasoningPayload := outputItemPayload(t, reasoningItems[0])
	assert.Equal(t, "TestSignature123\nTestSignature456", reasoningPayload["encrypted_content"])

	inputItems := agents.ModelResponse{Output: outputItems}.ToInputItems()
	messages, err := agents.ChatCmplConverter().ItemsToMessagesWithModel(
		agents.InputItems(inputItems),
		"anthropic/claude-4-opus",
		agents.WithPreserveThinkingBlocks(),
	)
	require.NoError(t, err)

	assistantPayload := findAssistantWithToolCalls(t, messages)
	contentAny, ok := assistantPayload["content"].([]any)
	require.True(t, ok)
	require.Len(t, contentAny, 3)

	firstContent := contentAny[0].(map[string]any)
	assert.Equal(t, "thinking", firstContent["type"])
	assert.Equal(t, "The user is asking about weather. Let me use the weather tool to get this information.", firstContent["thinking"])
	assert.Equal(t, "TestSignature123", firstContent["signature"])

	secondContent := contentAny[1].(map[string]any)
	assert.Equal(t, "thinking", secondContent["type"])
	assert.Equal(t, "We should use the city Tokyo as the city.", secondContent["thinking"])
	assert.Equal(t, "TestSignature456", secondContent["signature"])

	lastContent := contentAny[2].(map[string]any)
	assert.Equal(t, "text", lastContent["type"])
	assert.Equal(t, "I'll check the weather for you.", lastContent["text"])
}

func TestAnthropicThinkingBlocksWithoutToolCalls(t *testing.T) {
	message := chatCompletionMessageWithThinkingBlocks(t,
		"The weather in Paris is sunny with a temperature of 22C.",
		"The user wants to know about the weather in Paris.",
		[]map[string]any{
			{
				"type":      "thinking",
				"thinking":  "Let me think about the weather in Paris.",
				"signature": "TestSignatureNoTools123",
			},
		},
		nil,
	)

	outputItems, err := agents.ChatCmplConverter().MessageToOutputItems(message)
	require.NoError(t, err)

	inputItems := agents.ModelResponse{Output: outputItems}.ToInputItems()
	messages, err := agents.ChatCmplConverter().ItemsToMessagesWithModel(
		agents.InputItems(inputItems),
		"anthropic/claude-4-opus",
		agents.WithPreserveThinkingBlocks(),
	)
	require.NoError(t, err)

	assistantPayload := findAssistantMessagePayload(t, messages)
	contentAny, ok := assistantPayload["content"].([]any)
	require.True(t, ok)
	require.GreaterOrEqual(t, len(contentAny), 2)

	firstContent := contentAny[0].(map[string]any)
	assert.Equal(t, "thinking", firstContent["type"])
	assert.Equal(t, "Let me think about the weather in Paris.", firstContent["thinking"])
	assert.Equal(t, "TestSignatureNoTools123", firstContent["signature"])

	secondContent := contentAny[1].(map[string]any)
	assert.Equal(t, "text", secondContent["type"])
	assert.Equal(t, "The weather in Paris is sunny with a temperature of 22C.", secondContent["text"])
}

func chatCompletionMessageWithThinkingBlocks(
	t *testing.T,
	content string,
	reasoningContent string,
	thinkingBlocks []map[string]any,
	toolCalls []map[string]any,
) openai.ChatCompletionMessage {
	t.Helper()

	rawMessage := map[string]any{
		"role":              "assistant",
		"content":           content,
		"reasoning_content": reasoningContent,
	}
	if len(thinkingBlocks) > 0 {
		rawMessage["thinking_blocks"] = thinkingBlocks
	}
	if len(toolCalls) > 0 {
		rawMessage["tool_calls"] = toolCalls
	}

	rawBytes, err := json.Marshal(rawMessage)
	require.NoError(t, err)

	var message openai.ChatCompletionMessage
	require.NoError(t, json.Unmarshal(rawBytes, &message))
	return message
}

func easyUserMessageItem(content string) agents.TResponseInputItem {
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

func findAssistantMessagePayload(t *testing.T, messages []openai.ChatCompletionMessageParamUnion) map[string]any {
	t.Helper()

	for _, msg := range messages {
		raw, err := json.Marshal(msg)
		require.NoError(t, err)

		var payload map[string]any
		require.NoError(t, json.Unmarshal(raw, &payload))
		if payload["role"] == "assistant" {
			return payload
		}
	}
	t.Fatal("assistant message not found")
	return nil
}

func outputItemPayload(t *testing.T, item agents.TResponseOutputItem) map[string]any {
	t.Helper()

	raw := item.RawJSON()
	if raw == "" {
		rawBytes, err := json.Marshal(item)
		require.NoError(t, err)
		raw = string(rawBytes)
	}

	var payload map[string]any
	require.NoError(t, json.Unmarshal([]byte(raw), &payload))
	return payload
}
