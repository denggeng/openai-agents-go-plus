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
	"fmt"
)

func buildResumedStreamDebugAttrs(runState *RunState, includeToolOutput bool) []any {
	if runState == nil {
		return nil
	}
	items := runState.GeneratedRunItems
	if len(items) == 0 && len(runState.GeneratedItems) > 0 {
		return []any{
			"current_turn", runState.CurrentTurn,
			"current_agent", runState.CurrentAgentName,
			"generated_items_count", len(runState.GeneratedItems),
			"generated_items_types", []string{},
			"generated_items_details", buildGeneratedInputItemDetails(runState.GeneratedItems, includeToolOutput),
			"current_step_type", runStateCurrentStepType(runState.CurrentStep),
		}
	}

	return []any{
		"current_turn", runState.CurrentTurn,
		"current_agent", runState.CurrentAgentName,
		"generated_items_count", len(items),
		"generated_items_types", generatedRunItemTypes(items),
		"generated_items_details", buildGeneratedRunItemDetails(items, includeToolOutput),
		"current_step_type", runStateCurrentStepType(runState.CurrentStep),
	}
}

func runStateCurrentStepType(step *RunStateCurrentStepState) any {
	if step == nil || step.Type == "" {
		return nil
	}
	return step.Type
}

func generatedRunItemTypes(items []RunItem) []string {
	if len(items) == 0 {
		return nil
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		_, itemType := runItemRawAndType(item)
		if itemType == "" {
			continue
		}
		out = append(out, itemType)
	}
	return out
}

func buildGeneratedRunItemDetails(items []RunItem, includeToolOutput bool) []map[string]any {
	if len(items) == 0 {
		return nil
	}
	details := make([]map[string]any, 0, len(items))
	for i, item := range items {
		rawItem, itemType := runItemRawAndType(item)
		info := map[string]any{
			"index": i,
			"type":  itemType,
		}
		addRawItemInfo(info, rawItem, itemType, includeToolOutput)
		details = append(details, info)
	}
	return details
}

func buildGeneratedInputItemDetails(items []TResponseInputItem, includeToolOutput bool) []map[string]any {
	if len(items) == 0 {
		return nil
	}
	details := make([]map[string]any, 0, len(items))
	for i, item := range items {
		info := map[string]any{
			"index": i,
			"type":  "input_item",
		}
		addRawItemInfo(info, item, "", includeToolOutput)
		details = append(details, info)
	}
	return details
}

func addRawItemInfo(info map[string]any, rawItem any, itemType string, includeToolOutput bool) {
	if rawItem == nil {
		return
	}
	rawType := stringFieldFromRaw(rawItem, "type")
	if rawType != "" {
		info["raw_type"] = rawType
	}
	if name := stringFieldFromRaw(rawItem, "name"); name != "" {
		info["name"] = name
	}
	if callID := stringFieldFromRaw(rawItem, "call_id"); callID != "" {
		info["call_id"] = callID
	}
	if itemType == "tool_call_output_item" && includeToolOutput {
		if output, ok := outputStringFromRaw(rawItem); ok {
			info["output"] = truncateRunStateString(output, 100)
		}
	}
}

func outputStringFromRaw(rawItem any) (string, bool) {
	payload, ok := coerceToMap(rawItem)
	if !ok {
		return "", false
	}
	value, ok := payload["output"]
	if !ok {
		return "", false
	}
	return fmt.Sprint(value), true
}

func truncateRunStateString(value string, maxRunes int) string {
	if maxRunes <= 0 {
		return ""
	}
	runes := []rune(value)
	if len(runes) <= maxRunes {
		return value
	}
	return string(runes[:maxRunes])
}
