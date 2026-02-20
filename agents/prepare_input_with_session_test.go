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

package agents

import (
	"context"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type callbackListSession struct {
	items []TResponseInputItem
}

func (s *callbackListSession) SessionID(context.Context) string {
	return "callback-session"
}

func (s *callbackListSession) GetItems(_ context.Context, limit int) ([]TResponseInputItem, error) {
	if limit <= 0 || limit >= len(s.items) {
		return slicesClone(s.items), nil
	}
	start := len(s.items) - limit
	if start < 0 {
		start = 0
	}
	return slicesClone(s.items[start:]), nil
}

func (s *callbackListSession) AddItems(_ context.Context, items []TResponseInputItem) error {
	s.items = append(s.items, items...)
	return nil
}

func (s *callbackListSession) PopItem(context.Context) (*TResponseInputItem, error) {
	if len(s.items) == 0 {
		return nil, nil
	}
	item := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return &item, nil
}

func (s *callbackListSession) ClearSession(context.Context) error {
	s.items = nil
	return nil
}

func TestPrepareInputWithSessionKeepsFunctionCallOutputs(t *testing.T) {
	historyItem := inputItemFromMap(t, map[string]any{
		"type":    "function_call_output",
		"call_id": "call_prepare",
		"output":  "ok",
	})
	session := &callbackListSession{items: []TResponseInputItem{historyItem}}

	runner := Runner{Config: RunConfig{Session: session}}
	prepared, sessionItems, err := runner.prepareInputWithSession(t.Context(), InputString("hello"), false, true, false)
	require.NoError(t, err)

	preparedItems, ok := prepared.(InputItems)
	require.True(t, ok)
	require.Len(t, preparedItems, 2)
	assert.Equal(t, "function_call_output", inputItemPayload(t, preparedItems[0])["type"])
	assert.Equal(t, "hello", inputItemContentText(t, preparedItems[1]))

	require.Len(t, sessionItems, 1)
	assert.Equal(t, "hello", inputItemContentText(t, sessionItems[0]))
}

func TestPrepareInputWithSessionPrefersLatestFunctionCallOutput(t *testing.T) {
	historyOutput := inputItemFromMap(t, map[string]any{
		"type":    "function_call_output",
		"call_id": "call_latest",
		"output":  "history-output",
	})
	latestOutput := inputItemFromMap(t, map[string]any{
		"type":    "function_call_output",
		"call_id": "call_latest",
		"output":  "new-output",
	})
	session := &callbackListSession{items: []TResponseInputItem{historyOutput}}

	runner := Runner{Config: RunConfig{Session: session}}
	prepared, sessionItems, err := runner.prepareInputWithSession(
		t.Context(),
		InputItems{latestOutput},
		false,
		true,
		false,
	)
	require.NoError(t, err)

	preparedItems, ok := prepared.(InputItems)
	require.True(t, ok)
	outputs := make([]map[string]any, 0, 1)
	for _, item := range preparedItems {
		payload := inputItemPayload(t, item)
		if payload["type"] == "function_call_output" && payload["call_id"] == "call_latest" {
			outputs = append(outputs, payload)
		}
	}
	require.Len(t, outputs, 1)
	assert.Equal(t, "new-output", outputs[0]["output"])

	require.Len(t, sessionItems, 1)
	assert.Equal(t, "new-output", inputItemPayload(t, sessionItems[0])["output"])
}

func TestPrepareInputWithSessionDropsOrphanFunctionCalls(t *testing.T) {
	orphanCall := inputItemFromMap(t, map[string]any{
		"type":      "function_call",
		"call_id":   "orphan_call",
		"name":      "tool_orphan",
		"arguments": "{}",
	})
	session := &callbackListSession{items: []TResponseInputItem{orphanCall}}

	runner := Runner{Config: RunConfig{Session: session}}
	prepared, sessionItems, err := runner.prepareInputWithSession(t.Context(), InputString("hello"), false, true, false)
	require.NoError(t, err)

	preparedItems := prepared.(InputItems)
	for _, item := range preparedItems {
		payload := inputItemPayload(t, item)
		require.NotEqual(t, "function_call", payload["type"])
	}
	assert.Len(t, sessionItems, 1)
	assert.Equal(t, "hello", inputItemContentText(t, sessionItems[0]))
}

func TestPrepareInputWithSessionUsesSyncCallback(t *testing.T) {
	historyItem := messageInputItem(string(responses.EasyInputMessageRoleUser), "hi")
	session := &callbackListSession{items: []TResponseInputItem{historyItem}}

	callback := func(history []TResponseInputItem, newInput []TResponseInputItem) ([]TResponseInputItem, error) {
		assert.Equal(t, "hi", inputItemContentText(t, history[0]))
		return append(history, newInput...), nil
	}

	runner := Runner{Config: RunConfig{Session: session, SessionInputCallback: callback}}
	prepared, sessionItems, err := runner.prepareInputWithSession(t.Context(), InputString("second"), false, true, false)
	require.NoError(t, err)

	preparedItems := prepared.(InputItems)
	require.Len(t, preparedItems, 2)
	assert.Equal(t, "second", inputItemContentText(t, preparedItems[1]))

	require.Len(t, sessionItems, 1)
	assert.Equal(t, "second", inputItemContentText(t, sessionItems[0]))
}

func TestPrepareInputWithSessionCallbackDropsNewItems(t *testing.T) {
	historyItem := messageInputItem(string(responses.EasyInputMessageRoleUser), "history")
	session := &callbackListSession{items: []TResponseInputItem{historyItem}}

	callback := func(history []TResponseInputItem, _ []TResponseInputItem) ([]TResponseInputItem, error) {
		return history, nil
	}

	runner := Runner{Config: RunConfig{Session: session, SessionInputCallback: callback}}
	prepared, sessionItems, err := runner.prepareInputWithSession(t.Context(), InputString("new"), false, true, false)
	require.NoError(t, err)

	preparedItems := prepared.(InputItems)
	require.Len(t, preparedItems, 1)
	assert.Equal(t, "history", inputItemContentText(t, preparedItems[0]))
	assert.Empty(t, sessionItems)
}

func TestPrepareInputWithSessionCallbackReordersNewItems(t *testing.T) {
	historyItem := messageInputItem(string(responses.EasyInputMessageRoleUser), "history")
	session := &callbackListSession{items: []TResponseInputItem{historyItem}}

	callback := func(history []TResponseInputItem, newInput []TResponseInputItem) ([]TResponseInputItem, error) {
		return []TResponseInputItem{newInput[1], history[0], newInput[0]}, nil
	}

	newInput := []TResponseInputItem{
		messageInputItem(string(responses.EasyInputMessageRoleUser), "first"),
		messageInputItem(string(responses.EasyInputMessageRoleUser), "second"),
	}

	runner := Runner{Config: RunConfig{Session: session, SessionInputCallback: callback}}
	prepared, sessionItems, err := runner.prepareInputWithSession(t.Context(), InputItems(newInput), false, true, false)
	require.NoError(t, err)

	preparedItems := prepared.(InputItems)
	assert.Equal(t, "second", inputItemContentText(t, preparedItems[0]))
	assert.Equal(t, "history", inputItemContentText(t, preparedItems[1]))
	assert.Equal(t, "first", inputItemContentText(t, preparedItems[2]))

	require.Len(t, sessionItems, 2)
	assert.Equal(t, []string{"second", "first"}, []string{
		inputItemContentText(t, sessionItems[0]),
		inputItemContentText(t, sessionItems[1]),
	})
}

func TestPrepareInputWithSessionCallbackAcceptsExtraItems(t *testing.T) {
	historyItem := messageInputItem(string(responses.EasyInputMessageRoleUser), "history")
	extraItem := messageInputItem(string(responses.EasyInputMessageRoleAssistant), "extra")
	session := &callbackListSession{items: []TResponseInputItem{historyItem}}

	callback := func(history []TResponseInputItem, newInput []TResponseInputItem) ([]TResponseInputItem, error) {
		return []TResponseInputItem{extraItem, history[0], newInput[0]}, nil
	}

	runner := Runner{Config: RunConfig{Session: session, SessionInputCallback: callback}}
	prepared, sessionItems, err := runner.prepareInputWithSession(t.Context(), InputString("new"), false, true, false)
	require.NoError(t, err)

	preparedItems := prepared.(InputItems)
	assert.Equal(t, []string{"extra", "history", "new"}, []string{
		inputItemContentText(t, preparedItems[0]),
		inputItemContentText(t, preparedItems[1]),
		inputItemContentText(t, preparedItems[2]),
	})
	require.Len(t, sessionItems, 2)
	assert.Equal(t, []string{"extra", "new"}, []string{
		inputItemContentText(t, sessionItems[0]),
		inputItemContentText(t, sessionItems[1]),
	})
}

func TestPrepareInputWithSessionIgnoresCallbackWithoutHistory(t *testing.T) {
	historyItem := messageInputItem(string(responses.EasyInputMessageRoleUser), "history")
	session := &callbackListSession{items: []TResponseInputItem{historyItem}}

	callback := func(_ []TResponseInputItem, _ []TResponseInputItem) ([]TResponseInputItem, error) {
		return nil, nil
	}

	runner := Runner{Config: RunConfig{Session: session, SessionInputCallback: callback}}
	prepared, sessionItems, err := runner.prepareInputWithSession(
		t.Context(),
		InputString("new"),
		false,
		false,
		true,
	)
	require.NoError(t, err)

	preparedItems := prepared.(InputItems)
	require.Len(t, preparedItems, 1)
	assert.Equal(t, "new", inputItemContentText(t, preparedItems[0]))

	require.Len(t, sessionItems, 1)
	assert.Equal(t, "new", inputItemContentText(t, sessionItems[0]))
}

func inputItemContentText(t *testing.T, item TResponseInputItem) string {
	t.Helper()
	payload := inputItemPayload(t, item)
	content := payload["content"]
	switch v := content.(type) {
	case string:
		return v
	case []any:
		if len(v) == 0 {
			return ""
		}
		entry, ok := v[0].(map[string]any)
		require.True(t, ok)
		text, _ := entry["text"].(string)
		return text
	default:
		require.FailNowf(t, "unexpected content shape", "%T", v)
		return ""
	}
}

func slicesClone(items []TResponseInputItem) []TResponseInputItem {
	if len(items) == 0 {
		return nil
	}
	out := make([]TResponseInputItem, len(items))
	copy(out, items)
	return out
}
