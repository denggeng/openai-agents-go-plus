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
	"strings"

	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
)

type chatCmplHelpers struct {
}

func ChatCmplHelpers() chatCmplHelpers { return chatCmplHelpers{} }

func (chatCmplHelpers) IsOpenAI(client OpenaiClient) bool {
	return strings.HasPrefix(client.BaseURL.Or(""), "https://api.openai.com")
}

func (h chatCmplHelpers) GetStoreParam(
	client OpenaiClient,
	modelSettings modelsettings.ModelSettings,
) param.Opt[bool] {
	// Match the behavior of Responses where store is True when not given
	switch {
	case modelSettings.Store.Valid():
		return modelSettings.Store
	case h.IsOpenAI(client):
		return param.NewOpt(true)
	default:
		return param.Opt[bool]{}
	}
}

func (h chatCmplHelpers) GetStreamOptionsParam(
	client OpenaiClient,
	modelSettings modelsettings.ModelSettings,
	stream bool,
) openai.ChatCompletionStreamOptionsParam {
	var options openai.ChatCompletionStreamOptionsParam
	if !stream {
		return options
	}

	if modelSettings.IncludeUsage.Valid() {
		options.IncludeUsage = param.NewOpt(modelSettings.IncludeUsage.Value)
	} else if h.IsOpenAI(client) {
		options.IncludeUsage = param.NewOpt(true)
	}
	return options
}

func (chatCmplHelpers) GetChatCompletionsHeaders(modelSettings modelsettings.ModelSettings) map[string]string {
	headers := map[string]string{
		"User-Agent": DefaultUserAgent(),
	}
	for k, v := range modelSettings.ExtraHeaders {
		headers[k] = v
	}
	if override := HeadersOverride.Get(); len(override) > 0 {
		for k, v := range override {
			headers[k] = v
		}
	}
	return headers
}

func (h chatCmplHelpers) GetLiteLLMHeaders(modelSettings modelsettings.ModelSettings) map[string]string {
	return h.GetChatCompletionsHeaders(modelSettings)
}

// CleanGeminiToolCallID removes LiteLLM's "__thought__" suffix from Gemini tool call IDs.
func (chatCmplHelpers) CleanGeminiToolCallID(toolCallID string, model string) string {
	if strings.Contains(strings.ToLower(model), "gemini") && strings.Contains(toolCallID, "__thought__") {
		parts := strings.Split(toolCallID, "__thought__")
		if len(parts) > 0 {
			return parts[0]
		}
	}
	return toolCallID
}

func shouldFixLiteLLMToolMessageOrdering(modelName string) bool {
	name := strings.ToLower(strings.TrimSpace(modelName))
	return strings.Contains(name, "anthropic") ||
		strings.Contains(name, "claude") ||
		strings.Contains(name, "gemini")
}

func fixToolMessageOrdering(
	messages []openai.ChatCompletionMessageParamUnion,
) []openai.ChatCompletionMessageParamUnion {
	if len(messages) == 0 {
		return messages
	}

	type toolCallEntry struct {
		index   int
		message openai.ChatCompletionMessageParamUnion
	}
	type toolResultEntry struct {
		index   int
		message openai.ChatCompletionMessageParamUnion
	}

	toolCallMessages := map[string]toolCallEntry{}
	toolResultMessages := map[string]toolResultEntry{}

	for i, message := range messages {
		if message.OfAssistant != nil && len(message.OfAssistant.ToolCalls) > 0 {
			for _, toolCall := range message.OfAssistant.ToolCalls {
				toolID := toolCallID(toolCall)
				if toolID == "" {
					continue
				}
				single := cloneAssistantMessageWithToolCall(*message.OfAssistant, toolCall)
				toolCallMessages[toolID] = toolCallEntry{
					index:   i,
					message: openai.ChatCompletionMessageParamUnion{OfAssistant: &single},
				}
			}
			continue
		}

		if message.OfTool != nil {
			toolID := strings.TrimSpace(message.OfTool.ToolCallID)
			if toolID == "" {
				continue
			}
			toolResultMessages[toolID] = toolResultEntry{
				index:   i,
				message: message,
			}
		}
	}

	pairedToolResultIndices := map[int]struct{}{}
	for toolID := range toolCallMessages {
		if toolResult, ok := toolResultMessages[toolID]; ok {
			pairedToolResultIndices[toolResult.index] = struct{}{}
		}
	}

	fixed := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))
	usedIndices := map[int]struct{}{}

	for i, original := range messages {
		if _, used := usedIndices[i]; used {
			continue
		}

		if original.OfAssistant != nil && len(original.OfAssistant.ToolCalls) > 0 {
			for _, toolCall := range original.OfAssistant.ToolCalls {
				toolID := toolCallID(toolCall)
				if toolID == "" {
					continue
				}
				callEntry, ok := toolCallMessages[toolID]
				if !ok {
					continue
				}
				if resultEntry, ok := toolResultMessages[toolID]; ok {
					fixed = append(fixed, callEntry.message, resultEntry.message)
					usedIndices[callEntry.index] = struct{}{}
					usedIndices[resultEntry.index] = struct{}{}
					continue
				}
				fixed = append(fixed, callEntry.message)
				usedIndices[callEntry.index] = struct{}{}
			}
			usedIndices[i] = struct{}{}
			continue
		}

		if original.OfTool != nil {
			if _, paired := pairedToolResultIndices[i]; !paired {
				fixed = append(fixed, original)
			}
			usedIndices[i] = struct{}{}
			continue
		}

		fixed = append(fixed, original)
		usedIndices[i] = struct{}{}
	}

	return fixed
}

func toolCallID(call openai.ChatCompletionMessageToolCallUnionParam) string {
	if call.OfFunction != nil {
		return strings.TrimSpace(call.OfFunction.ID)
	}
	return ""
}

func cloneAssistantMessageWithToolCall(
	message openai.ChatCompletionAssistantMessageParam,
	toolCall openai.ChatCompletionMessageToolCallUnionParam,
) openai.ChatCompletionAssistantMessageParam {
	single := message
	single.ToolCalls = []openai.ChatCompletionMessageToolCallUnionParam{toolCall}
	return single
}
