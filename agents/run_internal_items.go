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
	"encoding/json"
	"fmt"
	"slices"

	"github.com/openai/openai-go/v3/packages/param"
)

type ReasoningItemIDPolicy string

const (
	ReasoningItemIDPolicyPreserve ReasoningItemIDPolicy = "preserve"
	ReasoningItemIDPolicyOmit     ReasoningItemIDPolicy = "omit"
)

var toolCallToOutputType = map[string]string{
	"function_call":    "function_call_output",
	"shell_call":       "shell_call_output",
	"apply_patch_call": "apply_patch_call_output",
	"computer_call":    "computer_call_output",
	"local_shell_call": "local_shell_call_output",
}

func dropOrphanFunctionCalls(items []any) []any {
	if len(items) == 0 {
		return nil
	}
	completed := completedCallIDsByType(items)
	filtered := make([]any, 0, len(items))
	for _, entry := range items {
		payload, ok := coerceInputItemToMap(entry)
		if !ok {
			filtered = append(filtered, entry)
			continue
		}
		itemType, _ := payload["type"].(string)
		outputType, ok := toolCallToOutputType[itemType]
		if !ok {
			filtered = append(filtered, entry)
			continue
		}
		callID, _ := payload["call_id"].(string)
		if callID == "" {
			continue
		}
		if outputCallIDs, ok := completed[outputType]; ok {
			if _, ok := outputCallIDs[callID]; ok {
				filtered = append(filtered, entry)
			}
		}
	}
	return filtered
}

func ensureInputItemFormat(item any) any {
	coerced, ok := coerceInputItemToMap(item)
	if !ok {
		return item
	}
	return coerced
}

func normalizeInputItemsForAPI(items []any) []any {
	if len(items) == 0 {
		return nil
	}
	normalized := make([]any, 0, len(items))
	for _, item := range items {
		coerced, ok := coerceInputItemToMap(item)
		if !ok {
			normalized = append(normalized, item)
			continue
		}
		normalized = append(normalized, mapCopy(coerced))
	}
	return normalized
}

func normalizeResumedInput(rawInput any) any {
	switch v := rawInput.(type) {
	case []any:
		normalized := normalizeInputItemsForAPI(v)
		return dropOrphanFunctionCalls(normalized)
	case []TResponseInputItem:
		anyItems := make([]any, len(v))
		for i, item := range v {
			anyItems[i] = item
		}
		normalized := normalizeInputItemsForAPI(anyItems)
		filtered := dropOrphanFunctionCalls(normalized)
		out := make([]TResponseInputItem, 0, len(filtered))
		for _, item := range filtered {
			if typed, ok := item.(TResponseInputItem); ok {
				out = append(out, typed)
			}
		}
		return out
	default:
		return rawInput
	}
}

func fingerprintInputItem(item any, ignoreIDsForMatching bool) (string, bool) {
	if item == nil {
		return "", false
	}
	if isModelDumpable(item) {
		payload, ok := modelDumpWithoutWarnings(item)
		if !ok {
			return "", false
		}
		if ignoreIDsForMatching {
			delete(payload, "id")
		}
		return marshalFingerprint(payload)
	}

	var payload any
	if mapping, ok := item.(map[string]any); ok {
		payload = mapCopy(mapping)
		if ignoreIDsForMatching {
			delete(payload.(map[string]any), "id")
		}
	} else {
		payload = ensureInputItemFormat(item)
		if ignoreIDsForMatching {
			if mapping, ok := payload.(map[string]any); ok {
				delete(mapping, "id")
			}
		}
	}
	return marshalFingerprint(payload)
}

func deduplicateInputItems(items []TResponseInputItem) []TResponseInputItem {
	if len(items) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(items))
	out := make([]TResponseInputItem, 0, len(items))
	for _, item := range items {
		key := dedupeKey(item)
		if key == "" {
			out = append(out, item)
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, item)
	}
	return out
}

func extractMCPRequestID(rawItem any) string {
	payload, ok := coerceInputItemToMap(rawItem)
	if !ok {
		return ""
	}
	if provider, ok := payload["provider_data"].(map[string]any); ok {
		if candidate, ok := provider["id"].(string); ok {
			return candidate
		}
	}
	if candidate, ok := payload["id"].(string); ok {
		return candidate
	}
	if candidate, ok := payload["call_id"].(string); ok {
		return candidate
	}
	return ""
}

func extractMCPRequestIDFromRun(mcpRun any) string {
	payload, ok := coerceInputItemToMap(mcpRun)
	if !ok {
		return ""
	}
	if requestItem, ok := payload["request_item"]; ok {
		if candidate := extractMCPRequestID(requestItem); candidate != "" {
			return candidate
		}
	}
	if requestItem, ok := payload["requestItem"]; ok {
		if candidate := extractMCPRequestID(requestItem); candidate != "" {
			return candidate
		}
	}
	return ""
}

func runItemToInputItem(runItem RunItem, reasoningItemIDPolicy ReasoningItemIDPolicy) (TResponseInputItem, bool) {
	if runItem == nil {
		return TResponseInputItem{}, false
	}
	switch runItem.(type) {
	case ToolApprovalItem, *ToolApprovalItem:
		return TResponseInputItem{}, false
	default:
	}
	input := runItem.ToInputItem()
	if shouldOmitReasoningItemIDs(reasoningItemIDPolicy) && isReasoningInputItem(input) {
		return withoutReasoningItemID(input), true
	}
	return input, true
}

func runItemsToInputItemsWithPolicy(items []RunItem, reasoningItemIDPolicy ReasoningItemIDPolicy) []TResponseInputItem {
	if len(items) == 0 {
		return nil
	}
	out := make([]TResponseInputItem, 0, len(items))
	for _, item := range items {
		converted, ok := runItemToInputItem(item, reasoningItemIDPolicy)
		if !ok {
			continue
		}
		out = append(out, converted)
	}
	return out
}

func completedCallIDsByType(payload []any) map[string]map[string]struct{} {
	completed := make(map[string]map[string]struct{})
	for _, outputType := range toolCallToOutputType {
		if _, ok := completed[outputType]; !ok {
			completed[outputType] = make(map[string]struct{})
		}
	}
	for _, entry := range payload {
		mapping, ok := coerceInputItemToMap(entry)
		if !ok {
			continue
		}
		itemType, _ := mapping["type"].(string)
		callID, _ := mapping["call_id"].(string)
		if itemType == "" || callID == "" {
			continue
		}
		if outputCallIDs, ok := completed[itemType]; ok {
			outputCallIDs[callID] = struct{}{}
		}
	}
	return completed
}

func dedupeKey(item TResponseInputItem) string {
	payload, ok := coerceInputItemToMap(item)
	if !ok {
		return ""
	}
	role, _ := payload["role"].(string)
	itemType, _ := payload["type"].(string)
	if itemType == "" {
		itemType = role
	}
	if role != "" || itemType == "message" {
		return ""
	}

	if itemID, ok := payload["id"].(string); ok {
		if itemID != FakeResponsesID {
			return fmt.Sprintf("id:%s:%s", itemType, itemID)
		}
	}
	if callID, ok := payload["call_id"].(string); ok {
		return fmt.Sprintf("call_id:%s:%s", itemType, callID)
	}
	if approvalID, ok := payload["approval_request_id"].(string); ok {
		return fmt.Sprintf("approval_request_id:%s:%s", itemType, approvalID)
	}
	return ""
}

func inputItemKey(item TResponseInputItem, ignoreIDs bool) string {
	if ignoreIDs {
		if key := dedupeKeyIgnoringID(item); key != "" {
			return "dedupe:" + key
		}
	} else {
		if key := dedupeKey(item); key != "" {
			return "dedupe:" + key
		}
	}
	if fp, ok := fingerprintInputItem(item, ignoreIDs); ok {
		return "fp:" + fp
	}
	data, err := json.Marshal(item)
	if err == nil {
		return "json:" + string(data)
	}
	return fmt.Sprintf("%#v", item)
}

func dedupeKeyIgnoringID(item TResponseInputItem) string {
	payload, ok := coerceInputItemToMap(item)
	if !ok {
		return ""
	}
	itemType, _ := payload["type"].(string)
	if itemType == "" {
		return ""
	}
	if callID, ok := payload["call_id"].(string); ok {
		return fmt.Sprintf("call_id:%s:%s", itemType, callID)
	}
	if approvalID, ok := payload["approval_request_id"].(string); ok {
		return fmt.Sprintf("approval_request_id:%s:%s", itemType, approvalID)
	}
	return ""
}

func shouldOmitReasoningItemIDs(policy ReasoningItemIDPolicy) bool {
	return policy == ReasoningItemIDPolicyOmit
}

func isReasoningInputItem(item TResponseInputItem) bool {
	return item.OfReasoning != nil
}

func withoutReasoningItemID(item TResponseInputItem) TResponseInputItem {
	if item.OfReasoning == nil {
		return item
	}
	copied := item
	reasoning := *item.OfReasoning
	reasoning.SetExtraFields(map[string]any{"id": param.Omit})
	copied.OfReasoning = &reasoning
	return copied
}

type modelDumpWithWarnings interface {
	ModelDump(excludeUnset bool, warnings bool) (map[string]any, error)
}

type modelDumpNoWarnings interface {
	ModelDump(excludeUnset bool) (map[string]any, error)
}

func isModelDumpable(value any) bool {
	if value == nil {
		return false
	}
	if _, ok := value.(modelDumpWithWarnings); ok {
		return true
	}
	_, ok := value.(modelDumpNoWarnings)
	return ok
}

func modelDumpWithoutWarnings(value any) (map[string]any, bool) {
	if value == nil {
		return nil, false
	}
	if v, ok := value.(modelDumpWithWarnings); ok {
		payload, err := v.ModelDump(true, false)
		if err == nil {
			return payload, true
		}
		if v2, ok := value.(modelDumpNoWarnings); ok {
			payload, err = v2.ModelDump(true)
			if err == nil {
				return payload, true
			}
		}
		return nil, false
	}
	if v, ok := value.(modelDumpNoWarnings); ok {
		payload, err := v.ModelDump(true)
		if err == nil {
			return payload, true
		}
	}
	return nil, false
}

func coerceInputItemToMap(value any) (map[string]any, bool) {
	if value == nil {
		return nil, false
	}
	if mapping, ok := value.(map[string]any); ok {
		return mapCopy(mapping), true
	}
	if isModelDumpable(value) {
		if payload, ok := modelDumpWithoutWarnings(value); ok {
			return payload, true
		}
		return nil, false
	}
	data, err := json.Marshal(value)
	if err != nil {
		return nil, false
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, false
	}
	return payload, true
}

func mapCopy(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func marshalFingerprint(payload any) (string, bool) {
	data, err := json.Marshal(payload)
	if err != nil {
		data, err = json.Marshal(fmt.Sprintf("%v", payload))
		if err != nil {
			return "", false
		}
	}
	return string(data), true
}

func deduplicateInputItemsPreferringLatest(items []TResponseInputItem) []TResponseInputItem {
	if len(items) == 0 {
		return nil
	}
	reversed := slices.Clone(items)
	slices.Reverse(reversed)
	deduped := deduplicateInputItems(reversed)
	slices.Reverse(deduped)
	return deduped
}
