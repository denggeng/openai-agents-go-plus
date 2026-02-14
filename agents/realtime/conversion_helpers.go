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

package realtime

import (
	"encoding/base64"
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
	oairealtime "github.com/openai/openai-go/v3/realtime"
)

var supportedRawRealtimeClientEventTypes = map[string]struct{}{
	"session.update":             {},
	"response.create":            {},
	"response.cancel":            {},
	"conversation.item.create":   {},
	"conversation.item.retrieve": {},
	"conversation.item.truncate": {},
	"input_audio_buffer.append":  {},
	"input_audio_buffer.commit":  {},
}

// TryConvertRawMessage validates and converts a raw message into a map payload.
// Returns nil when the message cannot be safely converted.
func TryConvertRawMessage(message RealtimeModelSendRawMessage) map[string]any {
	eventType := strings.TrimSpace(message.Message.Type)
	if eventType == "" {
		return nil
	}
	if _, supported := supportedRawRealtimeClientEventTypes[eventType]; !supported {
		return nil
	}

	payload := map[string]any{
		"type": eventType,
	}
	for k, v := range message.Message.OtherData {
		payload[k] = v
	}

	// Minimal schema checks matching current known message types.
	switch eventType {
	case "session.update":
		if sessionValue, ok := payload["session"]; !ok || !isStringKeyMap(sessionValue) {
			return nil
		}
	case "conversation.item.create":
		if itemValue, ok := payload["item"]; !ok || !isStringKeyMap(itemValue) {
			return nil
		}
	case "conversation.item.retrieve":
		if itemID, ok := payload["item_id"].(string); !ok || strings.TrimSpace(itemID) == "" {
			return nil
		}
	}

	return payload
}

// ConvertTracingConfig converts tracing config into OpenAI realtime tracing params.
// Accepted inputs:
// 1. nil
// 2. "auto"
// 3. RealtimeModelTracingConfig or *RealtimeModelTracingConfig
func ConvertTracingConfig(tracingConfig any) *oairealtime.RealtimeTracingConfigUnionParam {
	switch v := tracingConfig.(type) {
	case nil:
		return nil
	case string:
		if strings.EqualFold(strings.TrimSpace(v), "auto") {
			auto := oairealtime.RealtimeTracingConfigParamOfAuto()
			return &auto
		}
		return nil
	case RealtimeModelTracingConfig:
		return tracingConfigToUnion(v)
	case *RealtimeModelTracingConfig:
		if v == nil {
			return nil
		}
		return tracingConfigToUnion(*v)
	default:
		if mapping, ok := toStringAnyMap(tracingConfig); ok {
			return tracingConfigToUnion(readTracingConfigFromMap(mapping))
		}
		return nil
	}
}

// ConvertUserInputToConversationItem converts user input into a conversation item payload.
func ConvertUserInputToConversationItem(event RealtimeModelSendUserInput) map[string]any {
	userInput := event.UserInput

	switch v := userInput.(type) {
	case string:
		return map[string]any{
			"type": "message",
			"role": "user",
			"content": []map[string]any{
				{
					"type": "input_text",
					"text": v,
				},
			},
		}
	case RealtimeModelUserInputMessage:
		return convertStructuredUserInput(v.Type, v.Role, toMessageContentMaps(v.Content))
	case *RealtimeModelUserInputMessage:
		if v == nil {
			return convertStructuredUserInput("message", "user", []map[string]any{})
		}
		return convertStructuredUserInput(v.Type, v.Role, toMessageContentMaps(v.Content))
	case map[string]any:
		content := make([]map[string]any, 0)
		for _, part := range extractContentParts(v["content"]) {
			partMap, ok := toStringAnyMap(part)
			if !ok {
				continue
			}
			if converted, ok := normalizeUserInputContentMap(partMap); ok {
				content = append(content, converted)
			}
		}
		typeValue, _ := v["type"].(string)
		roleValue, _ := v["role"].(string)
		return convertStructuredUserInput(typeValue, roleValue, content)
	default:
		return convertStructuredUserInput("message", "user", []map[string]any{})
	}
}

// ConvertUserInputToItemCreate wraps user input into a conversation.item.create event payload.
func ConvertUserInputToItemCreate(event RealtimeModelSendUserInput) map[string]any {
	return map[string]any{
		"type": "conversation.item.create",
		"item": ConvertUserInputToConversationItem(event),
	}
}

// ConvertAudioToInputAudioBufferAppend converts raw audio bytes to append-event payload.
func ConvertAudioToInputAudioBufferAppend(event RealtimeModelSendAudio) map[string]any {
	return map[string]any{
		"type":  "input_audio_buffer.append",
		"audio": base64.StdEncoding.EncodeToString(event.Audio),
	}
}

// ConvertToolOutput converts tool output to a function_call_output conversation item event.
func ConvertToolOutput(event RealtimeModelSendToolOutput) map[string]any {
	if strings.TrimSpace(event.ToolCall.CallID) == "" {
		return nil
	}

	return map[string]any{
		"type": "conversation.item.create",
		"item": map[string]any{
			"type":    "function_call_output",
			"output":  event.Output,
			"call_id": event.ToolCall.CallID,
		},
	}
}

// ConvertInterrupt converts interruption context into conversation.item.truncate payload.
func ConvertInterrupt(
	currentItemID string,
	currentAudioContentIndex int,
	elapsedTimeMS int,
) map[string]any {
	return map[string]any{
		"type":          "conversation.item.truncate",
		"item_id":       currentItemID,
		"content_index": currentAudioContentIndex,
		"audio_end_ms":  elapsedTimeMS,
	}
}

func tracingConfigToUnion(
	tracingConfig RealtimeModelTracingConfig,
) *oairealtime.RealtimeTracingConfigUnionParam {
	converted := &oairealtime.RealtimeTracingConfigTracingConfigurationParam{
		Metadata: tracingConfig.Metadata,
	}

	if tracingConfig.GroupID != nil {
		converted.GroupID = param.NewOpt(*tracingConfig.GroupID)
	}
	if tracingConfig.WorkflowName != nil {
		converted.WorkflowName = param.NewOpt(*tracingConfig.WorkflowName)
	}

	return &oairealtime.RealtimeTracingConfigUnionParam{
		OfTracingConfiguration: converted,
	}
}

func convertStructuredUserInput(
	typeValue string,
	roleValue string,
	content []map[string]any,
) map[string]any {
	itemType := strings.TrimSpace(typeValue)
	if itemType == "" {
		itemType = "message"
	}
	role := strings.TrimSpace(roleValue)
	if role == "" {
		role = "user"
	}

	return map[string]any{
		"type":    itemType,
		"role":    role,
		"content": content,
	}
}

func toMessageContentMaps(parts []RealtimeModelUserInputContent) []map[string]any {
	result := make([]map[string]any, 0, len(parts))
	for _, part := range parts {
		if normalized, ok := normalizeUserInputContentMap(map[string]any{
			"type":      part.Type,
			"text":      valueOrNil(part.Text),
			"image_url": valueOrNil(part.ImageURL),
			"detail":    valueOrNil(part.Detail),
		}); ok {
			result = append(result, normalized)
		}
	}
	return result
}

func normalizeUserInputContentMap(part map[string]any) (map[string]any, bool) {
	partType, _ := part["type"].(string)
	switch partType {
	case "input_text":
		text, _ := part["text"].(string)
		return map[string]any{
			"type": "input_text",
			"text": text,
		}, true
	case "input_image":
		imageURL, _ := part["image_url"].(string)
		if strings.TrimSpace(imageURL) == "" {
			return nil, false
		}
		normalized := map[string]any{
			"type":      "input_image",
			"image_url": imageURL,
		}
		if detail, _ := part["detail"].(string); detail == "auto" || detail == "low" ||
			detail == "high" {
			normalized["detail"] = detail
		}
		return normalized, true
	default:
		return nil, false
	}
}

func valueOrNil(value *string) any {
	if value == nil {
		return nil
	}
	return *value
}

func isStringKeyMap(value any) bool {
	_, ok := toStringAnyMap(value)
	return ok
}

func readTracingConfigFromMap(mapping map[string]any) RealtimeModelTracingConfig {
	cfg := RealtimeModelTracingConfig{}
	if groupID, ok := mapping["group_id"].(string); ok && strings.TrimSpace(groupID) != "" {
		cfg.GroupID = &groupID
	}
	if workflowName, ok := mapping["workflow_name"].(string); ok &&
		strings.TrimSpace(workflowName) != "" {
		cfg.WorkflowName = &workflowName
	}
	if metadata, ok := toStringAnyMap(mapping["metadata"]); ok {
		cfg.Metadata = metadata
	}
	return cfg
}

func extractContentParts(raw any) []any {
	switch v := raw.(type) {
	case []any:
		return v
	case []map[string]any:
		parts := make([]any, 0, len(v))
		for _, part := range v {
			parts = append(parts, part)
		}
		return parts
	default:
		return nil
	}
}
