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

// Package tool_output_trimmer provides a call_model_input_filter that trims
// large tool outputs from older conversation turns.
package tool_output_trimmer

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/openai/openai-go/v3/packages/param"
)

// ToolOutputTrimmer trims large tool outputs from older turns before model calls.
// Use NewToolOutputTrimmer() to get default values.
type ToolOutputTrimmer struct {
	RecentTurns    int
	MaxOutputChars int
	PreviewChars   int
	TrimmableTools map[string]struct{}
}

// NewToolOutputTrimmer returns a trimmer with default settings.
func NewToolOutputTrimmer() ToolOutputTrimmer {
	return ToolOutputTrimmer{
		RecentTurns:    2,
		MaxOutputChars: 500,
		PreviewChars:   200,
	}
}

// Filter implements agents.CallModelInputFilter.
func (t ToolOutputTrimmer) Filter(_ context.Context, data agents.CallModelData) (*agents.ModelInputData, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}

	items := data.ModelData.Input
	if len(items) == 0 {
		return &agents.ModelInputData{
			Input:        nil,
			Instructions: data.ModelData.Instructions,
		}, nil
	}

	boundary := t.findRecentBoundary(items)
	if boundary == 0 {
		return &agents.ModelInputData{
			Input:        items,
			Instructions: data.ModelData.Instructions,
		}, nil
	}

	callIDToName := t.buildCallIDToName(items)
	newItems := make([]agents.TResponseInputItem, 0, len(items))
	for i, item := range items {
		if i < boundary {
			payload, ok := inputItemPayload(item)
			if ok && payload["type"] == "function_call_output" {
				outputStr := outputAsString(payload["output"])
				if len(outputStr) > t.MaxOutputChars {
					callID, _ := payload["call_id"].(string)
					toolName := callIDToName[callID]
					if t.shouldTrimTool(toolName) {
						displayName := toolName
						if displayName == "" {
							displayName = "unknown_tool"
						}
						preview := outputStr
						if t.PreviewChars < len(preview) {
							preview = preview[:t.PreviewChars]
						}
						summary := fmt.Sprintf(
							"[Trimmed: %s output — %d chars → %d char preview]\n%s...",
							displayName,
							len(outputStr),
							t.PreviewChars,
							preview,
						)
						if len(summary) < len(outputStr) {
							trimmed := mapCopy(payload)
							trimmed["output"] = summary
							if converted, ok := inputItemFromPayload(trimmed); ok {
								newItems = append(newItems, converted)
								continue
							}
						}
					}
				}
			}
		}
		newItems = append(newItems, item)
	}

	return &agents.ModelInputData{
		Input:        newItems,
		Instructions: data.ModelData.Instructions,
	}, nil
}

// Validate checks configuration values.
func (t ToolOutputTrimmer) Validate() error {
	if t.RecentTurns < 1 {
		return fmt.Errorf("recent_turns must be >= 1, got %d", t.RecentTurns)
	}
	if t.MaxOutputChars < 1 {
		return fmt.Errorf("max_output_chars must be >= 1, got %d", t.MaxOutputChars)
	}
	if t.PreviewChars < 0 {
		return fmt.Errorf("preview_chars must be >= 0, got %d", t.PreviewChars)
	}
	return nil
}

func (t ToolOutputTrimmer) shouldTrimTool(toolName string) bool {
	if t.TrimmableTools == nil {
		return true
	}
	_, ok := t.TrimmableTools[toolName]
	return ok
}

func (t ToolOutputTrimmer) findRecentBoundary(items []agents.TResponseInputItem) int {
	userCount := 0
	for i := len(items) - 1; i >= 0; i-- {
		payload, ok := inputItemPayload(items[i])
		if !ok {
			continue
		}
		if role, ok := payload["role"].(string); ok && role == "user" {
			userCount++
			if userCount >= t.RecentTurns {
				return i
			}
		}
	}
	return 0
}

func (t ToolOutputTrimmer) buildCallIDToName(items []agents.TResponseInputItem) map[string]string {
	mapping := make(map[string]string)
	for _, item := range items {
		payload, ok := inputItemPayload(item)
		if !ok || payload["type"] != "function_call" {
			continue
		}
		callID, _ := payload["call_id"].(string)
		name, _ := payload["name"].(string)
		if callID != "" && name != "" {
			mapping[callID] = name
		}
	}
	return mapping
}

func inputItemPayload(item agents.TResponseInputItem) (map[string]any, bool) {
	raw, err := item.MarshalJSON()
	if err != nil {
		return nil, false
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, false
	}
	return payload, true
}

func inputItemFromPayload(payload map[string]any) (agents.TResponseInputItem, bool) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return agents.TResponseInputItem{}, false
	}
	var item agents.TResponseInputItem
	param.SetJSON(raw, &item)
	return item, true
}

func outputAsString(output any) string {
	switch v := output.(type) {
	case string:
		return v
	default:
		return fmt.Sprint(v)
	}
}

func mapCopy(input map[string]any) map[string]any {
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}
