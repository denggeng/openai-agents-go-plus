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
	"context"
	"encoding/base64"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeRealtimeWebSocketConn struct {
	readQueue chan []byte
	closeOnce sync.Once
	closeCh   chan struct{}

	writesMu sync.Mutex
	writes   []map[string]any
}

func newFakeRealtimeWebSocketConn() *fakeRealtimeWebSocketConn {
	return &fakeRealtimeWebSocketConn{
		readQueue: make(chan []byte, 16),
		closeCh:   make(chan struct{}),
	}
}

func (f *fakeRealtimeWebSocketConn) enqueueRead(event map[string]any) {
	raw, err := json.Marshal(event)
	if err != nil {
		panic(err)
	}
	f.readQueue <- raw
}

func (f *fakeRealtimeWebSocketConn) ReadMessage() (int, []byte, error) {
	select {
	case msg := <-f.readQueue:
		return websocket.TextMessage, msg, nil
	case <-f.closeCh:
		return websocket.TextMessage, nil, &websocket.CloseError{Code: websocket.CloseNormalClosure}
	}
}

func (f *fakeRealtimeWebSocketConn) WriteJSON(v any) error {
	raw, err := json.Marshal(v)
	if err != nil {
		return err
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return err
	}
	f.writesMu.Lock()
	f.writes = append(f.writes, payload)
	f.writesMu.Unlock()
	return nil
}

func (f *fakeRealtimeWebSocketConn) Close() error {
	f.closeOnce.Do(func() {
		close(f.closeCh)
	})
	return nil
}

func (f *fakeRealtimeWebSocketConn) Writes() []map[string]any {
	f.writesMu.Lock()
	defer f.writesMu.Unlock()
	out := make([]map[string]any, 0, len(f.writes))
	for _, each := range f.writes {
		out = append(out, cloneStringAnyMap(each))
	}
	return out
}

func TestHandleWSMessageMalformedJSONEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSMessage(t.Context(), []byte("invalid json {")))
	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	_, ok = listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
}

func TestHandleWSEventInvalidAudioDeltaSchemaEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "response.output_audio.delta",
		// missing response_id/item_id/content_index/output_index/delta
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field")
}

func TestHandleWSEventSessionUpdatedMissingSessionEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "session.updated",
		// missing session
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field session")
}

func TestHandleWSEventSessionUpdatedInvalidSessionTypeEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":    "session.updated",
		"session": "not-an-object",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "invalid field session")
}

func TestHandleWSEventErrorMissingErrorPayloadEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "error",
		// missing error
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field error")
}

func TestHandleWSEventErrorInvalidErrorPayloadTypeEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":  "error",
		"error": "boom",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "invalid field error")
}

func TestHandleWSEventErrorEmitsErrorEvent(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    "invalid_request_error",
			"code":    "invalid_api_key",
			"message": "Invalid API key provided",
		},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	_, ok = errorEvent.Error.(map[string]any)
	assert.True(t, ok)
}

func TestHandleWSEventResponseCreatedMissingResponseEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "response.created",
		// missing response
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field response")
}

func TestHandleWSEventResponseDoneMissingResponseEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "response.done",
		// missing response
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field response")
}

func TestHandleWSEventResponseStatusEventsAreNoop(t *testing.T) {
	types := []string{
		"response.queued",
		"response.in_progress",
		"response.completed",
		"response.failed",
		"response.incomplete",
	}
	for _, eventType := range types {
		t.Run(eventType, func(t *testing.T) {
			model := NewOpenAIRealtimeWebSocketModel()
			listener := &captureRealtimeListener{}
			model.AddListener(listener)

			require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
				"type":            eventType,
				"sequence_number": 1,
				"response": map[string]any{
					"id": "resp_123",
				},
			}))

			require.Len(t, listener.events, 1)
			_, ok := listener.events[0].(RealtimeModelRawServerEvent)
			require.True(t, ok)
		})
	}
}

func TestHandleWSEventResponseStatusMissingSequenceEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id": "resp_123",
		},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	_, ok = listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
}

func TestHandleWSEventSessionCreatedSendsTracingUpdateByDefault(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{
		InitialSettings: RealtimeSessionModelSettings{},
	})

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "session.created",
		"session": map[string]any{
			"id": "sess_1",
		},
	}))

	require.Len(t, model.sentClientEvents, 1)
	assert.Equal(t, "session.update", model.sentClientEvents[0]["type"])
	sessionPayload, ok := toStringAnyMap(model.sentClientEvents[0]["session"])
	require.True(t, ok)
	_, hasTracing := sessionPayload["tracing"]
	assert.True(t, hasTracing)
}

func TestHandleWSEventSessionCreatedSkipsTracingUpdateWhenTracingNil(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{
		InitialSettings: RealtimeSessionModelSettings{
			"tracing": nil,
		},
	})

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "session.created",
		"session": map[string]any{
			"id": "sess_1",
		},
	}))

	assert.Empty(t, model.sentClientEvents)
}

func TestHandleWSEventAudioDelta(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{})
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.output_audio.delta",
		"response_id":   "resp_1",
		"item_id":       "item_1",
		"output_index":  0,
		"content_index": 0,
		"delta":         base64.StdEncoding.EncodeToString([]byte("test-audio")),
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	audioEvent, ok := listener.events[1].(RealtimeModelAudioEvent)
	require.True(t, ok)
	assert.Equal(t, "resp_1", audioEvent.ResponseID)
	assert.Equal(t, "item_1", audioEvent.ItemID)
	assert.Equal(t, []byte("test-audio"), audioEvent.Data)

	audioState := model.audioStateTracker.GetState("item_1", 0)
	require.NotNil(t, audioState)
	assert.Greater(t, audioState.AudioLengthMS, 0.0)
}

func TestHandleWSEventAudioDeltaInvalidBase64EmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.output_audio.delta",
		"response_id":   "resp_1",
		"item_id":       "item_1",
		"content_index": 0,
		"output_index":  0,
		"delta":         "###not-base64###",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "illegal base64")
}

func TestHandleWSEventAudioDeltaAccumulatesTimingAndTracksCurrentItem(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{})
	model.audioStateTracker.SetAudioFormat("pcm16")

	events := []map[string]any{
		{
			"type":          "response.output_audio.delta",
			"response_id":   "resp_1",
			"item_id":       "item_1",
			"output_index":  0,
			"content_index": 0,
			"delta":         base64.StdEncoding.EncodeToString([]byte("test")),
		},
		{
			"type":          "response.output_audio.delta",
			"response_id":   "resp_1",
			"item_id":       "item_1",
			"output_index":  0,
			"content_index": 0,
			"delta":         base64.StdEncoding.EncodeToString([]byte("more")),
		},
	}

	for _, event := range events {
		require.NoError(t, model.handleWSEvent(t.Context(), event))
	}

	assert.Equal(t, "item_1", model.currentItemID)
	audioState := model.audioStateTracker.GetState("item_1", 0)
	require.NotNil(t, audioState)
	expectedLength := (8.0 / (24000.0 * 2.0)) * 1000.0
	assert.InDelta(t, expectedLength, audioState.AudioLengthMS, 1e-6)

	last := model.audioStateTracker.GetLastAudioItem()
	require.NotNil(t, last)
	assert.Equal(t, "item_1", last.ItemID)
	assert.Equal(t, 0, last.ItemContentIndex)
}

func TestHandleWSEventLegacyResponseAudioDeltaAlias(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{})
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.audio.delta",
		"response_id":   "resp_legacy",
		"item_id":       "item_legacy",
		"output_index":  0,
		"content_index": 0,
		"delta":         base64.StdEncoding.EncodeToString([]byte("legacy-audio")),
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	audioEvent, ok := listener.events[1].(RealtimeModelAudioEvent)
	require.True(t, ok)
	assert.Equal(t, "resp_legacy", audioEvent.ResponseID)
	assert.Equal(t, "item_legacy", audioEvent.ItemID)
	assert.Equal(t, []byte("legacy-audio"), audioEvent.Data)
}

func TestHandleWSEventOutputItemDoneFunctionCall(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{})
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"id":        "tool_item_1",
			"type":      "function_call",
			"status":    "completed",
			"name":      "lookup_weather",
			"call_id":   "call_1",
			"arguments": `{"city":"SF"}`,
		},
	}))

	require.Len(t, listener.events, 3)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	itemUpdated, ok := listener.events[1].(RealtimeModelItemUpdatedEvent)
	require.True(t, ok)
	toolItem, ok := itemUpdated.Item.(RealtimeToolCallItem)
	require.True(t, ok)
	assert.Equal(t, "lookup_weather", toolItem.Name)
	assert.Equal(t, "call_1", toolItem.CallID)

	toolCall, ok := listener.events[2].(RealtimeModelToolCallEvent)
	require.True(t, ok)
	assert.Equal(t, "lookup_weather", toolCall.Name)
	assert.Equal(t, `{"city":"SF"}`, toolCall.Arguments)
}

func TestHandleWSEventOutputItemAddedAndDoneEmitsItemUpdated(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item": map[string]any{
			"id":   "msg_1",
			"type": "message",
			"role": "assistant",
			"content": []map[string]any{
				{"type": "output_text", "text": "hello"},
			},
		},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	itemUpdated, ok := listener.events[1].(RealtimeModelItemUpdatedEvent)
	require.True(t, ok)
	item, ok := itemUpdated.Item.(RealtimeMessageItem)
	require.True(t, ok)
	require.Len(t, item.Content, 1)
	assert.Equal(t, "text", item.Content[0].Type)
	if assert.NotNil(t, item.Content[0].Text) {
		assert.Equal(t, "hello", *item.Content[0].Text)
	}
	require.NotNil(t, item.Status)
	assert.Equal(t, "in_progress", *item.Status)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"id":   "msg_1",
			"type": "message",
			"role": "assistant",
			"content": []map[string]any{
				{"type": "output_text", "text": "bye"},
			},
		},
	}))

	require.Len(t, listener.events, 4)
	_, ok = listener.events[2].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	itemUpdated, ok = listener.events[3].(RealtimeModelItemUpdatedEvent)
	require.True(t, ok)
	item, ok = itemUpdated.Item.(RealtimeMessageItem)
	require.True(t, ok)
	require.NotNil(t, item.Status)
	assert.Equal(t, "completed", *item.Status)
}

func TestHandleWSEventOutputItemMissingOutputIndexAllowedForMessage(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "response.output_item.added",
		"item": map[string]any{
			"id":   "msg_1",
			"type": "message",
			"role": "assistant",
			"content": []map[string]any{
				{"type": "text", "text": "hello"},
			},
		},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	itemUpdated, ok := listener.events[1].(RealtimeModelItemUpdatedEvent)
	require.True(t, ok)
	item, ok := itemUpdated.Item.(RealtimeMessageItem)
	require.True(t, ok)
	assert.Equal(t, "msg_1", item.ItemID)
}

func TestHandleWSEventOutputItemMessageAudioAndTextContent(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item": map[string]any{
			"id":   "msg_audio",
			"type": "message",
			"role": "assistant",
			"content": []map[string]any{
				{"type": "text", "text": "hello"},
				{"type": "audio", "audio": "blob", "transcript": "hi"},
			},
		},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	itemUpdated, ok := listener.events[1].(RealtimeModelItemUpdatedEvent)
	require.True(t, ok)
	item, ok := itemUpdated.Item.(RealtimeMessageItem)
	require.True(t, ok)
	require.Len(t, item.Content, 2)
	assert.Equal(t, "text", item.Content[0].Type)
	assert.Equal(t, "audio", item.Content[1].Type)
	if assert.NotNil(t, item.Content[1].Audio) {
		assert.Equal(t, "blob", *item.Content[1].Audio)
	}
	if assert.NotNil(t, item.Content[1].Transcript) {
		assert.Equal(t, "hi", *item.Content[1].Transcript)
	}
}

func TestHandleWSEventOutputItemMissingItemEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		// missing item
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field item")
}

func TestHandleWSEventOutputItemFunctionCallMissingCallIDEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"id":        "tool_item_1",
			"type":      "function_call",
			"status":    "completed",
			"name":      "lookup_weather",
			"arguments": `{"city":"SF"}`,
		},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field item.call_id")
}

func TestHandleWSEventOutputAudioDoneMissingOutputIndexEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.output_audio.done",
		"response_id":   "resp_1",
		"item_id":       "item_1",
		"content_index": 0,
		// missing output_index
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field output_index")
}

func TestHandleWSEventOutputAudioDoneEmitsEvent(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.output_audio.done",
		"response_id":   "resp_1",
		"item_id":       "item_1",
		"content_index": 0,
		"output_index":  0,
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	audioDone, ok := listener.events[1].(RealtimeModelAudioDoneEvent)
	require.True(t, ok)
	assert.Equal(t, "item_1", audioDone.ItemID)
	assert.Equal(t, 0, audioDone.ContentIndex)
}

func TestHandleWSEventLegacyResponseAudioDoneAliasEmitsEvent(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.audio.done",
		"response_id":   "resp_1",
		"item_id":       "item_1",
		"content_index": 0,
		"output_index":  0,
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	audioDone, ok := listener.events[1].(RealtimeModelAudioDoneEvent)
	require.True(t, ok)
	assert.Equal(t, "item_1", audioDone.ItemID)
	assert.Equal(t, 0, audioDone.ContentIndex)
}

func TestHandleWSEventLegacyResponseAudioTranscriptDeltaAlias(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.audio.transcript.delta",
		"item_id":       "item_1",
		"response_id":   "resp_1",
		"output_index":  0,
		"content_index": 0,
		"delta":         "abc",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	transcriptEvent, ok := listener.events[1].(RealtimeModelTranscriptDeltaEvent)
	require.True(t, ok)
	assert.Equal(t, "item_1", transcriptEvent.ItemID)
	assert.Equal(t, "resp_1", transcriptEvent.ResponseID)
	assert.Equal(t, "abc", transcriptEvent.Delta)
}

func TestHandleWSEventOutputAudioDoneMissingResponseIDEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.output_audio.done",
		"item_id":       "item_1",
		"content_index": 0,
		"output_index":  0,
		// missing response_id
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field response_id")
}

func TestHandleWSEventTranscriptDeltaMissingOutputIndexEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.output_audio_transcript.delta",
		"item_id":       "item_1",
		"response_id":   "resp_1",
		"content_index": 0,
		"delta":         "abc",
		// missing output_index
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field output_index")
}

func TestHandleWSEventInputAudioTranscriptionDeltaMissingDeltaEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":    "conversation.item.input_audio_transcription.delta",
		"item_id": "item_1",
		// missing delta
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field delta")
}

func TestHandleWSEventOutputTextDeltaMissingItemIDEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.output_text.delta",
		"output_index":  0,
		"content_index": 0,
		"delta":         "abc",
		// missing item_id
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field item_id")
}

func TestHandleWSEventFunctionCallArgumentsDeltaMissingOutputIndexEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":    "response.function_call_arguments.delta",
		"item_id": "item_1",
		"delta":   "{\"x\":1}",
		// missing output_index
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field output_index")
}

func TestHandleWSEventOutputTextDeltaValidNoopEmitsOnlyRaw(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.output_text.delta",
		"item_id":       "item_1",
		"output_index":  0,
		"content_index": 0,
		"delta":         "abc",
	}))

	require.Len(t, listener.events, 1)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
}

func TestHandleWSEventOutputTextDoneMissingTextEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.output_text.done",
		"item_id":       "item_1",
		"output_index":  0,
		"content_index": 0,
		// missing text
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field text")
}

func TestHandleWSEventFunctionCallArgumentsDoneMissingNameEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":         "response.function_call_arguments.done",
		"item_id":      "item_1",
		"output_index": 0,
		"arguments":    "{}",
		// missing name
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field name")
}

func TestHandleWSEventOutputAudioTranscriptDoneMissingTranscriptEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.output_audio_transcript.done",
		"item_id":       "item_1",
		"response_id":   "resp_1",
		"output_index":  0,
		"content_index": 0,
		// missing transcript
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field transcript")
}

func TestHandleWSEventOutputAudioTranscriptDoneValidNoopEmitsOnlyRaw(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.output_audio_transcript.done",
		"item_id":       "item_1",
		"response_id":   "resp_1",
		"output_index":  0,
		"content_index": 0,
		"transcript":    "hello",
	}))

	require.Len(t, listener.events, 1)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
}

func TestHandleWSEventInputAudioBufferCommittedMissingItemIDEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "input_audio_buffer.committed",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field item_id")
}

func TestHandleWSEventInputAudioBufferCommittedInvalidPreviousItemIDTypeEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":             "input_audio_buffer.committed",
		"item_id":          "item_1",
		"previous_item_id": 123,
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "invalid field previous_item_id")
}

func TestHandleWSEventInputAudioBufferClearedNoopEmitsOnlyRaw(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "input_audio_buffer.cleared",
	}))

	require.Len(t, listener.events, 1)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
}

func TestHandleWSEventInputAudioBufferSpeechStoppedMissingAudioEndMSEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":    "input_audio_buffer.speech_stopped",
		"item_id": "item_1",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field audio_end_ms")
}

func TestHandleWSEventConversationCreatedMissingConversationEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "conversation.created",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field conversation")
}

func TestHandleWSEventConversationItemDeletedEmitsEvent(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":    "conversation.item.deleted",
		"item_id": "item_1",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	deletedEvent, ok := listener.events[1].(RealtimeModelItemDeletedEvent)
	require.True(t, ok)
	assert.Equal(t, "item_1", deletedEvent.ItemID)
}

func TestHandleWSEventInputAudioTranscriptionFailedInvalidErrorTypeEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":    "conversation.item.input_audio_transcription.failed",
		"item_id": "item_1",
		"error":   "bad",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "invalid field error")
}

func TestHandleWSEventInputAudioTranscriptionCompletedMissingTranscriptEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":    "conversation.item.input_audio_transcription.completed",
		"item_id": "item_1",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field transcript")
}

func TestHandleWSEventInputAudioTranscriptionCompletedEmitsEvent(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":       "conversation.item.input_audio_transcription.completed",
		"item_id":    "item_1",
		"transcript": "hello",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	completedEvent, ok := listener.events[1].(RealtimeModelInputAudioTranscriptionCompletedEvent)
	require.True(t, ok)
	assert.Equal(t, "item_1", completedEvent.ItemID)
	assert.Equal(t, "hello", completedEvent.Transcript)
}

func TestHandleWSEventResponseContentPartAddedMissingPartEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.content_part.added",
		"response_id":   "resp_1",
		"item_id":       "item_1",
		"output_index":  0,
		"content_index": 0,
		// missing part
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field part")
}

func TestHandleWSEventResponseContentPartAddedMissingPartTypeEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.content_part.added",
		"response_id":   "resp_1",
		"item_id":       "item_1",
		"output_index":  0,
		"content_index": 0,
		"part":          map[string]any{"text": "hello"},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field part.type")
}

func TestHandleWSEventResponseContentPartAddedValidNoopEmitsOnlyRaw(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.content_part.added",
		"response_id":   "resp_1",
		"item_id":       "item_1",
		"output_index":  0,
		"content_index": 0,
		"part": map[string]any{
			"type": "output_text",
			"text": "hello",
		},
	}))

	require.Len(t, listener.events, 1)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
}

func TestHandleWSEventOutputTextAnnotationAddedValidNoopEmitsOnlyRaw(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":             "response.output_text.annotation.added",
		"item_id":          "item_1",
		"output_index":     0,
		"content_index":    1,
		"annotation_index": 0,
		"sequence_number":  1,
		"annotation":       map[string]any{"type": "citation"},
	}))

	require.Len(t, listener.events, 1)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
}

func TestHandleWSEventOutputTextAnnotationAddedMissingAnnotationEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":             "response.output_text.annotation.added",
		"item_id":          "item_1",
		"output_index":     0,
		"content_index":    1,
		"annotation_index": 0,
		"sequence_number":  1,
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	_, ok = listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
}

func TestHandleWSEventRefusalDeltaValidNoopEmitsOnlyRaw(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":            "response.refusal.delta",
		"item_id":         "item_1",
		"output_index":    0,
		"content_index":   0,
		"sequence_number": 1,
		"delta":           "no",
	}))

	require.Len(t, listener.events, 1)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
}

func TestHandleWSEventReasoningTextDoneMissingTextEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":            "response.reasoning_text.done",
		"item_id":         "item_1",
		"output_index":    0,
		"content_index":   0,
		"sequence_number": 1,
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	_, ok = listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
}

func TestHandleWSEventReasoningSummaryPartAddedMissingTypeEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":            "response.reasoning_summary_part.added",
		"item_id":         "item_1",
		"output_index":    0,
		"summary_index":   0,
		"sequence_number": 1,
		"part":            map[string]any{"text": "summary"},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	_, ok = listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
}

func TestHandleWSEventCodeInterpreterDeltaValidNoopEmitsOnlyRaw(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":            "response.code_interpreter_call_code.delta",
		"item_id":         "item_1",
		"output_index":    0,
		"sequence_number": 1,
		"delta":           "print(1)",
	}))

	require.Len(t, listener.events, 1)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
}

func TestHandleWSEventImageGenerationPartialImageMissingB64EmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":                "response.image_generation_call.partial_image",
		"item_id":             "item_1",
		"output_index":        0,
		"sequence_number":     1,
		"partial_image_index": 0,
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	_, ok = listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
}

func TestHandleWSEventMcpCallArgumentsDoneMissingArgumentsEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":            "response.mcp_call_arguments.done",
		"item_id":         "item_1",
		"output_index":    0,
		"sequence_number": 1,
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	_, ok = listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
}

func TestHandleWSEventCustomToolCallInputDoneValidNoopEmitsOnlyRaw(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":            "response.custom_tool_call_input.done",
		"item_id":         "item_1",
		"output_index":    0,
		"sequence_number": 1,
		"input":           "{\"foo\": \"bar\"}",
	}))

	require.Len(t, listener.events, 1)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
}

func TestHandleWSEventResponseNoopEventsEmitOnlyRaw(t *testing.T) {
	cases := []struct {
		name  string
		event map[string]any
	}{
		{
			name: "refusal_done",
			event: map[string]any{
				"type":            "response.refusal.done",
				"item_id":         "item_1",
				"output_index":    0,
				"content_index":   0,
				"sequence_number": 1,
				"refusal":         "no",
			},
		},
		{
			name: "reasoning_text_delta",
			event: map[string]any{
				"type":            "response.reasoning_text.delta",
				"item_id":         "item_1",
				"output_index":    0,
				"content_index":   0,
				"sequence_number": 1,
				"delta":           "think",
			},
		},
		{
			name: "reasoning_summary_text_delta",
			event: map[string]any{
				"type":            "response.reasoning_summary_text.delta",
				"item_id":         "item_1",
				"output_index":    0,
				"summary_index":   0,
				"sequence_number": 1,
				"delta":           "summary",
			},
		},
		{
			name: "reasoning_summary_text_done",
			event: map[string]any{
				"type":            "response.reasoning_summary_text.done",
				"item_id":         "item_1",
				"output_index":    0,
				"summary_index":   0,
				"sequence_number": 1,
				"text":            "summary",
			},
		},
		{
			name: "reasoning_summary_part_done",
			event: map[string]any{
				"type":            "response.reasoning_summary_part.done",
				"item_id":         "item_1",
				"output_index":    0,
				"summary_index":   0,
				"sequence_number": 1,
				"part":            map[string]any{"type": "summary_text", "text": "summary"},
			},
		},
		{
			name: "mcp_call_arguments_delta",
			event: map[string]any{
				"type":            "response.mcp_call_arguments.delta",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
				"delta":           "{}",
			},
		},
		{
			name: "custom_tool_call_input_delta",
			event: map[string]any{
				"type":            "response.custom_tool_call_input.delta",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
				"delta":           "{}",
			},
		},
		{
			name: "code_interpreter_call_code_done",
			event: map[string]any{
				"type":            "response.code_interpreter_call_code.done",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
				"code":            "print(1)",
			},
		},
		{
			name: "content_part_done",
			event: map[string]any{
				"type":          "response.content_part.done",
				"response_id":   "resp_1",
				"item_id":       "item_1",
				"output_index":  0,
				"content_index": 0,
				"part":          map[string]any{"type": "output_text"},
			},
		},
		{
			name: "image_generation_in_progress",
			event: map[string]any{
				"type":            "response.image_generation_call.in_progress",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "image_generation_completed",
			event: map[string]any{
				"type":            "response.image_generation_call.completed",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "image_generation_generating",
			event: map[string]any{
				"type":            "response.image_generation_call.generating",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "file_search_searching",
			event: map[string]any{
				"type":            "response.file_search_call.searching",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "file_search_in_progress",
			event: map[string]any{
				"type":            "response.file_search_call.in_progress",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "file_search_completed",
			event: map[string]any{
				"type":            "response.file_search_call.completed",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "web_search_searching",
			event: map[string]any{
				"type":            "response.web_search_call.searching",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "web_search_in_progress",
			event: map[string]any{
				"type":            "response.web_search_call.in_progress",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "code_interpreter_in_progress",
			event: map[string]any{
				"type":            "response.code_interpreter_call.in_progress",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "code_interpreter_interpreting",
			event: map[string]any{
				"type":            "response.code_interpreter_call.interpreting",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "code_interpreter_completed",
			event: map[string]any{
				"type":            "response.code_interpreter_call.completed",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "mcp_call_completed",
			event: map[string]any{
				"type":            "response.mcp_call.completed",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "mcp_call_in_progress",
			event: map[string]any{
				"type":            "response.mcp_call.in_progress",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "mcp_call_failed",
			event: map[string]any{
				"type":            "response.mcp_call.failed",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "mcp_list_tools_failed",
			event: map[string]any{
				"type":            "response.mcp_list_tools.failed",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "mcp_list_tools_in_progress",
			event: map[string]any{
				"type":            "response.mcp_list_tools.in_progress",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
		{
			name: "mcp_list_tools_completed",
			event: map[string]any{
				"type":            "response.mcp_list_tools.completed",
				"item_id":         "item_1",
				"output_index":    0,
				"sequence_number": 1,
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model := NewOpenAIRealtimeWebSocketModel()
			listener := &captureRealtimeListener{}
			model.AddListener(listener)

			require.NoError(t, model.handleWSEvent(t.Context(), tc.event))

			require.Len(t, listener.events, 1)
			_, ok := listener.events[0].(RealtimeModelRawServerEvent)
			require.True(t, ok)
		})
	}
}

func TestHandleWSEventWebSearchCallCompletedValidNoopEmitsOnlyRaw(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":            "response.web_search_call.completed",
		"item_id":         "item_1",
		"output_index":    0,
		"sequence_number": 1,
	}))

	require.Len(t, listener.events, 1)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
}

func TestHandleWSEventLegacyResponseAudioTranscriptDoneAliasNoopEmitsOnlyRaw(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":          "response.audio.transcript.done",
		"item_id":       "item_1",
		"response_id":   "resp_1",
		"output_index":  0,
		"content_index": 0,
		"transcript":    "hello",
	}))

	require.Len(t, listener.events, 1)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
}

func TestHandleWSEventRateLimitsUpdatedInvalidRateLimitsTypeEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":        "rate_limits.updated",
		"rate_limits": "bad",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "invalid field rate_limits")
}

func TestHandleWSEventUnknownTypeEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "unknown.event.type",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "unsupported realtime server event type")
}

func TestHandleWSEventMissingTypeEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"item_id": "item_1",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field type")
}

func TestHandleWSEventSpeechStartedInterruptsAndCancels(t *testing.T) {
	playback := NewRealtimePlaybackTracker()
	playback.OnPlayMS("item_1", 0, 240)

	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{
		PlaybackTracker: playback,
	})
	model.automaticResponseCancellationEnabled = false
	model.ongoingResponse = true
	model.audioStateTracker.OnAudioDelta("item_1", 0, []byte("audio-bytes"))

	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":           "input_audio_buffer.speech_started",
		"item_id":        "item_1",
		"audio_start_ms": 0,
		"audio_end_ms":   120,
	}))

	require.GreaterOrEqual(t, len(listener.events), 2)
	require.IsType(t, RealtimeModelRawServerEvent{}, listener.events[0])
	containsInterrupted := false
	for _, event := range listener.events {
		if _, ok := event.(RealtimeModelAudioInterruptedEvent); ok {
			containsInterrupted = true
			break
		}
	}
	assert.True(t, containsInterrupted)

	require.Len(t, model.sentClientEvents, 2)
	assert.Equal(t, "conversation.item.truncate", model.sentClientEvents[0]["type"])
	assert.Equal(t, "response.cancel", model.sentClientEvents[1]["type"])
	assert.False(t, model.ongoingResponse)
}

func TestHandleWSEventSpeechStartedDoesNotAutoInterruptWithoutPlayback(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":           "input_audio_buffer.speech_started",
		"item_id":        "item_1",
		"audio_start_ms": 0,
	}))

	require.Len(t, listener.events, 1)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	assert.Empty(t, model.sentClientEvents)
}

func TestHandleWSEventSpeechStartedSkipsTruncateWhenAudioComplete(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{})

	model.audioStateTracker.SetAudioFormat("pcm16")
	model.audioStateTracker.OnAudioDelta("item_1", 0, make([]byte, 48000))
	key := modelAudioStateKey{itemID: "item_1", itemContentIndex: 0}
	state, ok := model.audioStateTracker.states[key]
	require.True(t, ok)
	state.InitialReceivedTime = time.Now().Add(-5 * time.Second)
	model.audioStateTracker.states[key] = state

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":           "input_audio_buffer.speech_started",
		"item_id":        "item_1",
		"audio_start_ms": 0,
		"audio_end_ms":   0,
	}))

	truncates := make([]map[string]any, 0)
	for _, event := range model.sentClientEvents {
		if eventType, _ := event["type"].(string); eventType == "conversation.item.truncate" {
			truncates = append(truncates, event)
		}
	}
	assert.Empty(t, truncates)
}

func TestHandleWSEventSpeechStartedTruncatesWhenResponseOngoing(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{})

	model.audioStateTracker.SetAudioFormat("pcm16")
	model.audioStateTracker.OnAudioDelta("item_1", 0, make([]byte, 48000))
	key := modelAudioStateKey{itemID: "item_1", itemContentIndex: 0}
	state, ok := model.audioStateTracker.states[key]
	require.True(t, ok)
	state.InitialReceivedTime = time.Now().Add(-5 * time.Second)
	model.audioStateTracker.states[key] = state
	model.ongoingResponse = true

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":           "input_audio_buffer.speech_started",
		"item_id":        "item_1",
		"audio_start_ms": 0,
		"audio_end_ms":   0,
	}))

	truncates := make([]map[string]any, 0)
	for _, event := range model.sentClientEvents {
		if eventType, _ := event["type"].(string); eventType == "conversation.item.truncate" {
			truncates = append(truncates, event)
		}
	}
	require.Len(t, truncates, 1)
	assert.Equal(t, 1000, truncates[0]["audio_end_ms"])
}

func TestHandleWSEventSpeechStartedMissingAudioStartMSEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":    "input_audio_buffer.speech_started",
		"item_id": "item_1",
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field audio_start_ms")
}

func TestHandleWSEventInputAudioTimeoutTriggeredEmitsEvent(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":           "input_audio_buffer.timeout_triggered",
		"item_id":        "item_1",
		"audio_start_ms": 0,
		"audio_end_ms":   100,
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	timeoutEvent, ok := listener.events[1].(RealtimeModelInputAudioTimeoutTriggeredEvent)
	require.True(t, ok)
	assert.Equal(t, "item_1", timeoutEvent.ItemID)
	assert.Equal(t, 0, timeoutEvent.AudioStartMS)
	assert.Equal(t, 100, timeoutEvent.AudioEndMS)
}

func TestHandleWSEventConversationItemMissingTypeEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "conversation.item.added",
		"item": map[string]any{
			"id": "item_1",
			// missing type
		},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field item.type")
}

func TestHandleWSEventConversationItemMessageMissingIDEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "conversation.item.added",
		"item": map[string]any{
			"type":    "message",
			"role":    "assistant",
			"content": []map[string]any{{"type": "output_text", "text": "hello"}},
			// missing id
		},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field item.id")
}

func TestHandleWSEventConversationItemRetrievedEmitsItemUpdated(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "conversation.item.retrieved",
		"item": map[string]any{
			"id":      "item_1",
			"type":    "message",
			"role":    "assistant",
			"content": []map[string]any{{"type": "output_text", "text": "hello"}},
		},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	itemUpdated, ok := listener.events[1].(RealtimeModelItemUpdatedEvent)
	require.True(t, ok)
	item, ok := itemUpdated.Item.(RealtimeMessageItem)
	require.True(t, ok)
	assert.Equal(t, "item_1", item.ItemID)
	assert.Equal(t, "assistant", item.Role)
	require.Len(t, item.Content, 1)
	assert.Equal(t, "text", item.Content[0].Type)
}

func TestHandleWSEventConversationItemCreatedInvalidPreviousItemIDTypeEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":             "conversation.item.created",
		"previous_item_id": 123,
		"item": map[string]any{
			"id":      "item_1",
			"type":    "message",
			"role":    "assistant",
			"content": []map[string]any{{"type": "output_text", "text": "hello"}},
		},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "invalid field previous_item_id")
}

func TestHandleWSEventConversationItemTruncatedMissingContentIndexEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":         "conversation.item.truncated",
		"item_id":      "item_1",
		"audio_end_ms": 12,
		// missing content_index
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field content_index")
}

func TestHandleWSEventOutputItemMessageMissingIDEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"type": "message",
			"role": "assistant",
			"content": []map[string]any{
				{"type": "output_text", "text": "hello"},
			},
			// missing id
		},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "missing required field item.id")
}

func TestHandleWSEventOutputItemUnknownTypeEmitsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"id":   "item_1",
			"type": "reasoning",
		},
	}))

	require.Len(t, listener.events, 2)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
	errorEvent, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
	assert.Contains(t, errorEvent.Error.(error).Error(), "unsupported output item type")
}

func TestConnectWithTransportDialerStartsListenerAndWritesSessionUpdate(t *testing.T) {
	fakeConn := newFakeRealtimeWebSocketConn()
	fakeConn.enqueueRead(map[string]any{
		"type": "response.created",
		"response": map[string]any{
			"id": "resp_1",
		},
	})

	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		EnableTransport: true,
		TransportDialer: func(context.Context, string, map[string]string, *RealtimeTransportConfig) (RealtimeWebSocketConn, error) {
			return fakeConn, nil
		},
		InitialSettings: RealtimeSessionModelSettings{},
	}))

	require.Eventually(t, func() bool {
		for _, event := range listener.events {
			if _, ok := event.(RealtimeModelTurnStartedEvent); ok {
				return true
			}
		}
		return false
	}, time.Second, 10*time.Millisecond)

	writes := fakeConn.Writes()
	require.NotEmpty(t, writes)
	assert.Equal(t, "session.update", writes[0]["type"])

	require.NoError(t, model.Close(t.Context()))
}
