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
	"bytes"
	"encoding/json"
	"reflect"
	"sort"
	"strings"

	"github.com/openai/openai-go/v3/responses"
)

// OpenAIServerConversationTracker tracks server-side conversation state for server-managed runs.
// It mirrors the behavior of the Python OpenAIServerConversationTracker used for resume/dedupe logic.
type OpenAIServerConversationTracker struct {
	ConversationID         string
	PreviousResponseID     string
	AutoPreviousResponseID bool

	sentItems            map[uintptr]struct{}
	serverItems          map[uintptr]struct{}
	serverItemIDs        map[string]struct{}
	serverToolCallIDs    map[string]struct{}
	sentItemFingerprints map[string]struct{}

	sentInitialInput      bool
	remainingInitialInput []TResponseInputItem
	primedFromState       bool
}

// NewOpenAIServerConversationTracker creates a tracker with initialized state.
func NewOpenAIServerConversationTracker(conversationID, previousResponseID string, autoPreviousResponseID bool) *OpenAIServerConversationTracker {
	tracker := &OpenAIServerConversationTracker{
		ConversationID:         conversationID,
		PreviousResponseID:     previousResponseID,
		AutoPreviousResponseID: autoPreviousResponseID,
	}
	tracker.ensureMaps()
	Logger().Debug(
		"Created OpenAIServerConversationTracker",
		"conversation_id", conversationID,
		"previous_response_id", previousResponseID,
	)
	return tracker
}

func (t *OpenAIServerConversationTracker) ensureMaps() {
	if t.sentItems == nil {
		t.sentItems = make(map[uintptr]struct{})
	}
	if t.serverItems == nil {
		t.serverItems = make(map[uintptr]struct{})
	}
	if t.serverItemIDs == nil {
		t.serverItemIDs = make(map[string]struct{})
	}
	if t.serverToolCallIDs == nil {
		t.serverToolCallIDs = make(map[string]struct{})
	}
	if t.sentItemFingerprints == nil {
		t.sentItemFingerprints = make(map[string]struct{})
	}
}

// HydrateFromState seeds the tracker from a prior run so resumed runs avoid re-sending items.
func (t *OpenAIServerConversationTracker) HydrateFromState(
	originalInput Input,
	generatedItems []RunItem,
	modelResponses []ModelResponse,
	sessionItems []TResponseInputItem,
) {
	if t == nil {
		return
	}
	if t.sentInitialInput {
		return
	}
	t.ensureMaps()

	normalizedInput := originalInput
	if items, ok := originalInput.(InputItems); ok {
		normalizedInput = items
	}

	initialItems := ItemHelpers().InputToNewInputList(normalizedInput)
	for i := range initialItems {
		item := initialItems[i]
		if id := pointerIdentity(&initialItems[i]); id != 0 {
			t.sentItems[id] = struct{}{}
		}
		if itemID := normalizedServerItemID(item); itemID != "" {
			t.serverItemIDs[itemID] = struct{}{}
		}
		if fp, ok := fingerprintForTracker(item); ok {
			t.sentItemFingerprints[fp] = struct{}{}
		}
	}

	t.sentInitialInput = true
	t.remainingInitialInput = nil

	var latestResponse *ModelResponse
	if len(modelResponses) > 0 {
		latestResponse = &modelResponses[len(modelResponses)-1]
	}

	for i := range modelResponses {
		response := modelResponses[i]
		for j := range response.Output {
			outputItem := response.Output[j]
			if id := pointerIdentity(&response.Output[j]); id != 0 {
				t.serverItems[id] = struct{}{}
			}
			if itemID := normalizedServerItemID(outputItem); itemID != "" {
				t.serverItemIDs[itemID] = struct{}{}
			}
			if callID := callIDFromRaw(outputItem); callID != "" && hasOutputPayload(outputItem) {
				t.serverToolCallIDs[callID] = struct{}{}
			}
		}
	}

	if t.ConversationID == "" && latestResponse != nil && latestResponse.ResponseID != "" {
		t.PreviousResponseID = latestResponse.ResponseID
	}

	if len(sessionItems) > 0 {
		for i := range sessionItems {
			item := sessionItems[i]
			if itemID := normalizedServerItemID(item); itemID != "" {
				t.serverItemIDs[itemID] = struct{}{}
			}
			if callID := callIDFromRaw(item); callID != "" && hasOutputPayload(item) {
				t.serverToolCallIDs[callID] = struct{}{}
			}
			if fp, ok := fingerprintForTracker(item); ok {
				t.sentItemFingerprints[fp] = struct{}{}
			}
		}
	}

	for i := range generatedItems {
		rawItem, itemType := runItemRawAndType(generatedItems[i])
		if rawItem == nil {
			continue
		}
		isToolCallItem := itemType == "tool_call_item" || itemType == "handoff_call_item"

		itemID := normalizedServerItemID(rawItem)
		callID := callIDFromRaw(rawItem)
		hasOutput := hasOutputPayload(rawItem)
		hasCallID := callID != ""
		shouldMark := itemID != "" || (hasCallID && (hasOutput || isToolCallItem))
		if !shouldMark {
			continue
		}

		if rawID := itemIdentity(rawItem); rawID != 0 {
			t.sentItems[rawID] = struct{}{}
		}
		if fp, ok := fingerprintForTracker(rawItem); ok {
			t.sentItemFingerprints[fp] = struct{}{}
		}
		if itemID != "" {
			t.serverItemIDs[itemID] = struct{}{}
		}
		if callID != "" && hasOutput {
			t.serverToolCallIDs[callID] = struct{}{}
		}
	}

	t.primedFromState = true
}

// TrackServerItems tracks server-acknowledged outputs to avoid re-sending on retries.
func (t *OpenAIServerConversationTracker) TrackServerItems(modelResponse *ModelResponse) {
	if t == nil || modelResponse == nil {
		return
	}
	t.ensureMaps()

	serverItemFingerprints := make(map[string]struct{})
	for i := range modelResponse.Output {
		outputItem := modelResponse.Output[i]
		if id := pointerIdentity(&modelResponse.Output[i]); id != 0 {
			t.serverItems[id] = struct{}{}
		}
		if itemID := normalizedServerItemID(outputItem); itemID != "" {
			t.serverItemIDs[itemID] = struct{}{}
		}
		if callID := callIDFromRaw(outputItem); callID != "" && hasOutputPayload(outputItem) {
			t.serverToolCallIDs[callID] = struct{}{}
		}
		if fp, ok := fingerprintForTracker(outputItem); ok {
			t.sentItemFingerprints[fp] = struct{}{}
			serverItemFingerprints[fp] = struct{}{}
		}
	}

	if len(t.remainingInitialInput) > 0 && len(serverItemFingerprints) > 0 {
		remaining := make([]TResponseInputItem, 0, len(t.remainingInitialInput))
		for i := range t.remainingInitialInput {
			pending := t.remainingInitialInput[i]
			if fp, ok := fingerprintForTracker(pending); ok {
				if _, exists := serverItemFingerprints[fp]; exists {
					continue
				}
			}
			remaining = append(remaining, pending)
		}
		t.remainingInitialInput = remaining
		if len(t.remainingInitialInput) == 0 {
			t.remainingInitialInput = nil
		}
	}

	if t.ConversationID == "" &&
		(t.PreviousResponseID != "" || t.AutoPreviousResponseID) &&
		modelResponse.ResponseID != "" {
		t.PreviousResponseID = modelResponse.ResponseID
	}
}

// MarkInputAsSent records delivered inputs so retries avoid duplicates.
func (t *OpenAIServerConversationTracker) MarkInputAsSent(items []TResponseInputItem) {
	if t == nil || len(items) == 0 {
		return
	}
	t.ensureMaps()

	deliveredFingerprints := make(map[string]struct{}, len(items))
	deliveredIDs := make(map[uintptr]struct{}, len(items))

	for i := range items {
		item := items[i]
		if id := pointerIdentity(&items[i]); id != 0 {
			t.sentItems[id] = struct{}{}
			deliveredIDs[id] = struct{}{}
		}
		if fp, ok := fingerprintForTracker(item); ok {
			t.sentItemFingerprints[fp] = struct{}{}
			deliveredFingerprints[fp] = struct{}{}
		}
		if callID := callIDFromRaw(item); callID != "" && hasOutputPayload(item) {
			t.serverToolCallIDs[callID] = struct{}{}
		}
	}

	if len(t.remainingInitialInput) == 0 {
		return
	}

	remaining := make([]TResponseInputItem, 0, len(t.remainingInitialInput))
	for i := range t.remainingInitialInput {
		pending := t.remainingInitialInput[i]
		if id := pointerIdentity(&t.remainingInitialInput[i]); id != 0 {
			if _, exists := deliveredIDs[id]; exists {
				continue
			}
		}
		if fp, ok := fingerprintForTracker(pending); ok {
			if _, exists := deliveredFingerprints[fp]; exists {
				continue
			}
		}
		remaining = append(remaining, pending)
	}
	t.remainingInitialInput = remaining
	if len(t.remainingInitialInput) == 0 {
		t.remainingInitialInput = nil
	}
}

// RewindInput queues previously sent items so they can be resent.
func (t *OpenAIServerConversationTracker) RewindInput(items []TResponseInputItem) {
	if t == nil || len(items) == 0 {
		return
	}
	t.ensureMaps()

	rewindItems := make([]TResponseInputItem, 0, len(items))
	for i := range items {
		item := items[i]
		rewindItems = append(rewindItems, item)
		if id := pointerIdentity(&items[i]); id != 0 {
			delete(t.sentItems, id)
		}
		if fp, ok := fingerprintForTracker(item); ok {
			delete(t.sentItemFingerprints, fp)
		}
		if callID := callIDFromRaw(item); callID != "" && hasOutputPayload(item) {
			delete(t.serverToolCallIDs, callID)
		}
	}

	if len(rewindItems) == 0 {
		return
	}

	Logger().Debug(
		"Queued items to resend after conversation retry",
		"count", len(rewindItems),
	)
	if len(t.remainingInitialInput) == 0 {
		t.remainingInitialInput = rewindItems
		return
	}
	t.remainingInitialInput = append(rewindItems, t.remainingInitialInput...)
}

// PrepareInput assembles the next model input while skipping duplicates and approvals.
func (t *OpenAIServerConversationTracker) PrepareInput(
	originalInput Input,
	generatedItems []RunItem,
) []TResponseInputItem {
	if t == nil {
		return ItemHelpers().InputToNewInputList(originalInput)
	}
	t.ensureMaps()

	var inputItems []TResponseInputItem

	if !t.sentInitialInput {
		initialItems := ItemHelpers().InputToNewInputList(originalInput)
		inputItems = append(inputItems, initialItems...)
		if len(initialItems) > 0 {
			t.remainingInitialInput = initialItems
		} else {
			t.remainingInitialInput = nil
		}
		t.sentInitialInput = true
	} else if len(t.remainingInitialInput) > 0 {
		inputItems = append(inputItems, t.remainingInitialInput...)
	}

	for i := range generatedItems {
		rawItem, itemType := runItemRawAndType(generatedItems[i])
		if itemType == "tool_approval_item" {
			continue
		}
		if rawItem == nil {
			continue
		}

		if itemID := normalizedServerItemID(rawItem); itemID != "" {
			if _, exists := t.serverItemIDs[itemID]; exists {
				continue
			}
		}

		callID := callIDFromRaw(rawItem)
		if callID != "" && hasOutputPayload(rawItem) {
			if _, exists := t.serverToolCallIDs[callID]; exists {
				continue
			}
		}

		rawID := itemIdentity(rawItem)
		if rawID != 0 {
			if _, exists := t.sentItems[rawID]; exists {
				continue
			}
			if _, exists := t.serverItems[rawID]; exists {
				continue
			}
		}

		inputItem := generatedItems[i].ToInputItem()
		if fp, ok := fingerprintForTracker(inputItem); ok {
			if _, exists := t.sentItemFingerprints[fp]; exists {
				if t.primedFromState || isOutputInputItem(inputItem) {
					continue
				}
			}
		}

		inputItems = append(inputItems, inputItem)
		if rawID != 0 {
			t.sentItems[rawID] = struct{}{}
		}
	}

	return inputItems
}

func isOutputInputItem(raw any) bool {
	if callID := callIDFromRaw(raw); callID != "" && hasOutputPayload(raw) {
		return true
	}
	itemType := stringFieldFromRaw(raw, "type")
	if itemType == "" {
		return false
	}
	if itemType == "function_call_output" {
		return true
	}
	return strings.HasSuffix(itemType, "_output")
}

func runItemRawAndType(item RunItem) (any, string) {
	switch v := item.(type) {
	case MessageOutputItem:
		return v.RawItem, v.Type
	case *MessageOutputItem:
		return v.RawItem, v.Type
	case ToolCallItem:
		return v.RawItem, v.Type
	case *ToolCallItem:
		return v.RawItem, v.Type
	case ToolCallOutputItem:
		return v.RawItem, v.Type
	case *ToolCallOutputItem:
		return v.RawItem, v.Type
	case HandoffCallItem:
		return v.RawItem, v.Type
	case *HandoffCallItem:
		return v.RawItem, v.Type
	case HandoffOutputItem:
		return v.RawItem, v.Type
	case *HandoffOutputItem:
		return v.RawItem, v.Type
	case ReasoningItem:
		return v.RawItem, v.Type
	case *ReasoningItem:
		return v.RawItem, v.Type
	case CompactionItem:
		return v.RawItem, v.Type
	case *CompactionItem:
		return v.RawItem, v.Type
	default:
		vv := reflect.ValueOf(item)
		if vv.Kind() == reflect.Ptr {
			if vv.IsNil() {
				return nil, ""
			}
			vv = vv.Elem()
		}
		if vv.Kind() != reflect.Struct {
			return nil, ""
		}
		rawField := vv.FieldByName("RawItem")
		typeField := vv.FieldByName("Type")
		var raw any
		if rawField.IsValid() {
			raw = rawField.Interface()
		}
		var itemType string
		if typeField.IsValid() && typeField.Kind() == reflect.String {
			itemType = typeField.String()
		}
		if raw == nil && itemType == "" {
			return nil, ""
		}
		return raw, itemType
	}
}

func normalizedServerItemID(raw any) string {
	id := stringFieldFromRaw(raw, "id")
	if id == "" || id == FakeResponsesID {
		return ""
	}
	return id
}

func callIDFromRaw(raw any) string {
	return stringFieldFromRaw(raw, "call_id")
}

func stringFieldFromRaw(raw any, field string) string {
	if raw == nil {
		return ""
	}
	if m, ok := raw.(map[string]any); ok {
		if value, ok := m[field]; ok {
			if s, ok := value.(string); ok {
				return s
			}
		}
	}
	if rawJSON, ok := rawJSONOverride(raw); ok {
		var payload map[string]any
		if err := json.Unmarshal(rawJSON, &payload); err == nil {
			if value, ok := payload[field]; ok {
				if s, ok := value.(string); ok {
					return s
				}
			}
		}
	}
	if payload, ok := coerceToMap(raw); ok {
		if value, ok := payload[field]; ok {
			if s, ok := value.(string); ok {
				return s
			}
		}
	}
	return ""
}

func hasOutputPayload(raw any) bool {
	if raw == nil {
		return false
	}
	if m, ok := raw.(map[string]any); ok {
		_, exists := m["output"]
		return exists
	}
	if rawJSON, ok := rawJSONOverride(raw); ok {
		var payload map[string]any
		if err := json.Unmarshal(rawJSON, &payload); err == nil {
			_, exists := payload["output"]
			return exists
		}
	}
	switch v := raw.(type) {
	case responses.ResponseOutputItemUnion:
		return responseOutputUnionHasOutput(v)
	case *responses.ResponseOutputItemUnion:
		if v == nil {
			return false
		}
		return responseOutputUnionHasOutput(*v)
	}
	v := reflect.ValueOf(raw)
	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return false
		}
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		return false
	}
	return v.FieldByName("Output").IsValid()
}

func responseOutputUnionHasOutput(item responses.ResponseOutputItemUnion) bool {
	if item.JSON.Output.Valid() {
		return true
	}
	if item.Type != "" {
		if strings.HasSuffix(item.Type, "_output") || item.Type == "function_call_output" {
			return true
		}
	}
	return !reflect.ValueOf(item.Output).IsZero()
}

func coerceToMap(raw any) (map[string]any, bool) {
	if raw == nil {
		return nil, false
	}
	if m, ok := raw.(map[string]any); ok {
		return m, true
	}
	if rawJSON, ok := rawJSONOverride(raw); ok {
		var payload map[string]any
		if err := json.Unmarshal(rawJSON, &payload); err == nil {
			return payload, true
		}
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return nil, false
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, false
	}
	return payload, true
}

func rawJSONOverride(raw any) (json.RawMessage, bool) {
	if raw == nil {
		return nil, false
	}
	overrides, ok := any(raw).(interface {
		Overrides() (any, bool)
	})
	if !ok {
		return nil, false
	}
	value, ok := overrides.Overrides()
	if !ok || value == nil {
		return nil, false
	}
	switch v := value.(type) {
	case json.RawMessage:
		return v, true
	case []byte:
		return json.RawMessage(v), true
	case string:
		return json.RawMessage(v), true
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return nil, false
		}
		return json.RawMessage(data), true
	}
}

func fingerprintForTracker(item any) (string, bool) {
	if item == nil {
		return "", false
	}
	switch v := item.(type) {
	case responses.ResponseOutputItemUnion:
		if fp, ok := fingerprintOutputUnion(v); ok {
			return fp, true
		}
	case *responses.ResponseOutputItemUnion:
		if v != nil {
			if fp, ok := fingerprintOutputUnion(*v); ok {
				return fp, true
			}
		}
	}
	if rawJSON, ok := rawJSONOverride(item); ok {
		var payload any
		if err := json.Unmarshal(rawJSON, &payload); err == nil {
			if data, err := stableJSON(payload); err == nil {
				return string(data), true
			}
		}
	}
	data, err := stableJSON(pruneEmptyValues(item))
	if err != nil {
		return "", false
	}
	return string(data), true
}

func fingerprintOutputUnion(item responses.ResponseOutputItemUnion) (string, bool) {
	if item.Type != "function_call_output" {
		return "", false
	}
	payload := make(map[string]any, 3)
	if item.Type != "" {
		payload["type"] = item.Type
	}
	if item.CallID != "" {
		payload["call_id"] = item.CallID
	}
	if item.Output.OfString != "" {
		payload["output"] = item.Output.OfString
	} else if len(item.Output.OfResponseFunctionShellToolCallOutputOutputArray) > 0 {
		payload["output"] = item.Output.OfResponseFunctionShellToolCallOutputOutputArray
	}
	data, err := stableJSON(payload)
	if err != nil {
		return "", false
	}
	return string(data), true
}

func pruneEmptyValues(v any) any {
	data, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var decoded any
	if err := json.Unmarshal(data, &decoded); err != nil {
		return v
	}
	return pruneValue(decoded)
}

func pruneValue(v any) any {
	switch val := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(val))
		for key, item := range val {
			pruned := pruneValue(item)
			if isEmptyValue(pruned) {
				continue
			}
			out[key] = pruned
		}
		return out
	case []any:
		out := make([]any, 0, len(val))
		for _, item := range val {
			out = append(out, pruneValue(item))
		}
		if len(out) == 0 {
			return []any{}
		}
		return out
	default:
		return v
	}
}

func isEmptyValue(v any) bool {
	if v == nil {
		return true
	}
	switch val := v.(type) {
	case string:
		return val == ""
	case bool:
		return !val
	case float64:
		return val == 0
	case float32:
		return val == 0
	case int:
		return val == 0
	case int64:
		return val == 0
	case int32:
		return val == 0
	case int16:
		return val == 0
	case int8:
		return val == 0
	case uint:
		return val == 0
	case uint64:
		return val == 0
	case uint32:
		return val == 0
	case uint16:
		return val == 0
	case uint8:
		return val == 0
	case []any:
		return len(val) == 0
	case map[string]any:
		return len(val) == 0
	default:
		return false
	}
}

func stableJSON(v any) ([]byte, error) {
	var buf bytes.Buffer
	if err := writeStableJSON(&buf, v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func writeStableJSON(buf *bytes.Buffer, v any) error {
	if v == nil {
		buf.WriteString("null")
		return nil
	}
	if raw, ok := v.(json.RawMessage); ok {
		var decoded any
		if err := json.Unmarshal(raw, &decoded); err == nil {
			return writeStableJSON(buf, decoded)
		}
		buf.Write(raw)
		return nil
	}

	rv := reflect.ValueOf(v)
	if rv.Kind() == reflect.Ptr {
		if rv.IsNil() {
			buf.WriteString("null")
			return nil
		}
		return writeStableJSON(buf, rv.Elem().Interface())
	}

	switch rv.Kind() {
	case reflect.Map:
		if rv.Type().Key().Kind() != reflect.String {
			data, err := json.Marshal(v)
			if err != nil {
				return err
			}
			buf.Write(data)
			return nil
		}
		keys := rv.MapKeys()
		keyStrings := make([]string, 0, len(keys))
		for _, key := range keys {
			keyStrings = append(keyStrings, key.String())
		}
		sort.Strings(keyStrings)
		buf.WriteByte('{')
		for i, key := range keyStrings {
			if i > 0 {
				buf.WriteByte(',')
			}
			keyBytes, err := json.Marshal(key)
			if err != nil {
				return err
			}
			buf.Write(keyBytes)
			buf.WriteByte(':')
			if err := writeStableJSON(buf, rv.MapIndex(reflect.ValueOf(key)).Interface()); err != nil {
				return err
			}
		}
		buf.WriteByte('}')
		return nil
	case reflect.Slice, reflect.Array:
		if rv.Kind() == reflect.Slice && rv.Type().Elem().Kind() == reflect.Uint8 {
			data, err := json.Marshal(v)
			if err != nil {
				return err
			}
			buf.Write(data)
			return nil
		}
		buf.WriteByte('[')
		for i := 0; i < rv.Len(); i++ {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := writeStableJSON(buf, rv.Index(i).Interface()); err != nil {
				return err
			}
		}
		buf.WriteByte(']')
		return nil
	case reflect.Struct:
		data, err := json.Marshal(v)
		if err != nil {
			return err
		}
		var decoded any
		if err := json.Unmarshal(data, &decoded); err != nil {
			return err
		}
		return writeStableJSON(buf, decoded)
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return err
		}
		buf.Write(data)
		return nil
	}
}

func pointerIdentity(ptr any) uintptr {
	if ptr == nil {
		return 0
	}
	v := reflect.ValueOf(ptr)
	if v.Kind() != reflect.Ptr || v.IsNil() {
		return 0
	}
	return v.Pointer()
}

func itemIdentity(raw any) uintptr {
	if raw == nil {
		return 0
	}
	v := reflect.ValueOf(raw)
	switch v.Kind() {
	case reflect.Ptr, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan, reflect.UnsafePointer:
		if v.IsNil() {
			return 0
		}
		return v.Pointer()
	default:
		return 0
	}
}
