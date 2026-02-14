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
	"fmt"
)

// RealtimeMessageContent represents one content part in a realtime message item.
type RealtimeMessageContent struct {
	Type       string
	Text       *string
	Audio      *string
	Transcript *string
	ImageURL   *string
	Detail     *string
}

// RealtimeMessageItem is a user/system/assistant message item.
type RealtimeMessageItem struct {
	ItemID         string
	PreviousItemID *string
	Type           string
	Role           string
	Status         *string
	Content        []RealtimeMessageContent
}

// RealtimeToolCallItem is a tool call + output item in realtime history.
type RealtimeToolCallItem struct {
	ItemID         string
	PreviousItemID *string
	CallID         string
	Type           string
	Status         string
	Arguments      string
	Name           string
	Output         *string
}

// ConversationItemToRealtimeMessageItem converts a raw conversation item payload into
// a normalized RealtimeMessageItem shape used by this SDK.
func ConversationItemToRealtimeMessageItem(
	item map[string]any,
	previousItemID *string,
) (*RealtimeMessageItem, error) {
	if item == nil {
		return nil, fmt.Errorf("unsupported conversation item type")
	}

	itemType, _ := item["type"].(string)
	role, _ := item["role"].(string)
	if itemType != "message" || !isSupportedMessageRole(role) {
		return nil, fmt.Errorf("unsupported conversation item type")
	}

	content := make([]RealtimeMessageContent, 0)
	for _, rawPart := range extractContentParts(item["content"]) {
		partMap, ok := toStringAnyMap(rawPart)
		if !ok {
			continue
		}
		if normalizedPart, ok := normalizeConversationItemContent(partMap); ok {
			content = append(content, normalizedPart)
		}
	}

	itemID, _ := item["id"].(string)
	status := "in_progress"
	return &RealtimeMessageItem{
		ItemID:         itemID,
		PreviousItemID: previousItemID,
		Type:           "message",
		Role:           role,
		Status:         &status,
		Content:        content,
	}, nil
}

func isSupportedMessageRole(role string) bool {
	return role == "system" || role == "user" || role == "assistant"
}

func normalizeConversationItemContent(part map[string]any) (RealtimeMessageContent, bool) {
	partType, _ := part["type"].(string)
	switch partType {
	case "output_text":
		partType = "text"
	case "output_audio":
		partType = "audio"
	}

	switch partType {
	case "input_text", "input_audio", "input_image", "text", "audio":
	default:
		return RealtimeMessageContent{}, false
	}

	content := RealtimeMessageContent{
		Type:       partType,
		Text:       stringValuePtr(part["text"]),
		Audio:      stringValuePtr(part["audio"]),
		Transcript: stringValuePtr(part["transcript"]),
		ImageURL:   stringValuePtr(part["image_url"]),
		Detail:     stringValuePtr(part["detail"]),
	}
	return content, true
}

func stringValuePtr(value any) *string {
	s, ok := value.(string)
	if !ok {
		return nil
	}
	return &s
}
