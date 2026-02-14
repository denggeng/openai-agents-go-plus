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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type captureRealtimeListener struct {
	events []RealtimeModelEvent
}

func (l *captureRealtimeListener) OnEvent(_ context.Context, event RealtimeModelEvent) error {
	l.events = append(l.events, event)
	return nil
}

func mustConnectRealtimeModel(
	t *testing.T,
	model *OpenAIRealtimeWebSocketModel,
	cfg RealtimeModelConfig,
) {
	t.Helper()
	if cfg.APIKey == "" {
		cfg.APIKey = "sk-test"
	}
	if cfg.InitialSettings == nil {
		cfg.InitialSettings = RealtimeSessionModelSettings{}
	}
	require.NoError(t, model.Connect(t.Context(), cfg))
	// Drop initial session.update emitted at connect time for test clarity.
	model.sentClientEvents = nil
}

func TestSendEventUserInput(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{})

	require.NoError(t, model.SendEvent(t.Context(), RealtimeModelSendUserInput{UserInput: "hi"}))
	require.Len(t, model.sentClientEvents, 2)
	assert.Equal(t, "conversation.item.create", model.sentClientEvents[0]["type"])
	assert.Equal(t, "response.create", model.sentClientEvents[1]["type"])
}

func TestSendEventAudioCommit(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{})

	require.NoError(t, model.SendEvent(t.Context(), RealtimeModelSendAudio{
		Audio:  []byte("abc"),
		Commit: true,
	}))
	require.Len(t, model.sentClientEvents, 2)
	assert.Equal(t, "input_audio_buffer.append", model.sentClientEvents[0]["type"])
	assert.Equal(t, "input_audio_buffer.commit", model.sentClientEvents[1]["type"])
}

func TestSendEventToolOutputEmitsItemUpdatedAndResponseCreate(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{})

	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.SendEvent(t.Context(), RealtimeModelSendToolOutput{
		ToolCall: RealtimeModelToolCallEvent{
			Name:      "echo",
			CallID:    "call_1",
			Arguments: `{"x":1}`,
		},
		Output:        "ok",
		StartResponse: true,
	}))

	require.Len(t, model.sentClientEvents, 2)
	assert.Equal(t, "conversation.item.create", model.sentClientEvents[0]["type"])
	assert.Equal(t, "response.create", model.sentClientEvents[1]["type"])

	require.Len(t, listener.events, 1)
	itemUpdated, ok := listener.events[0].(RealtimeModelItemUpdatedEvent)
	require.True(t, ok)
	toolItem, ok := itemUpdated.Item.(RealtimeToolCallItem)
	require.True(t, ok)
	assert.Equal(t, "call_1", toolItem.CallID)
	assert.Equal(t, "echo", toolItem.Name)
}

func TestSendEventSessionUpdate(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{})

	require.NoError(t, model.SendEvent(t.Context(), RealtimeModelSendSessionUpdate{
		SessionSettings: RealtimeSessionModelSettings{
			"voice": "verse",
		},
	}))

	require.Len(t, model.sentClientEvents, 1)
	assert.Equal(t, "session.update", model.sentClientEvents[0]["type"])
	assert.Equal(t, "verse", string(model.createdSession.Audio.Output.Voice))
}

func TestSendEventInterruptUsesPlaybackStateAndForceCancel(t *testing.T) {
	playbackTracker := NewRealtimePlaybackTracker()
	playbackTracker.OnPlayMS("item_1", 0, 250)

	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{
		PlaybackTracker: playbackTracker,
	})

	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.SendEvent(t.Context(), RealtimeModelSendInterrupt{}))
	require.Len(t, model.sentClientEvents, 1)
	assert.Equal(t, "conversation.item.truncate", model.sentClientEvents[0]["type"])

	require.Len(t, listener.events, 1)
	_, ok := listener.events[0].(RealtimeModelAudioInterruptedEvent)
	require.True(t, ok)

	require.NoError(t, model.SendEvent(t.Context(), RealtimeModelSendInterrupt{
		ForceResponseCancel: true,
	}))
	require.Len(t, model.sentClientEvents, 2)
	assert.Equal(t, "response.cancel", model.sentClientEvents[1]["type"])
}

func TestSendEventInvalidRawMessageReturnsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{})

	err := model.SendEvent(t.Context(), RealtimeModelSendRawMessage{
		Message: RealtimeModelRawClientMessage{Type: "invalid.type"},
	})
	require.Error(t, err)
}
