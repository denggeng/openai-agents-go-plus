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

func TestHandleWSEventUnknownTypeEmitsOnlyRaw(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "unknown.event.type",
	}))

	require.Len(t, listener.events, 1)
	_, ok := listener.events[0].(RealtimeModelRawServerEvent)
	require.True(t, ok)
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
		TransportDialer: func(context.Context, string, map[string]string) (RealtimeWebSocketConn, error) {
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
