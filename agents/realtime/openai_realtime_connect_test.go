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
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConnectWithCallIDAndModelRaisesError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey: "sk-test",
		CallID: "call-123",
		InitialSettings: RealtimeSessionModelSettings{
			"model_name": "gpt-realtime-mini",
		},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot specify both `call_id` and `model_name`")
}

func TestConnectBuildsDefaultModelURL(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.NoError(t, err)
	assert.Equal(t, "wss://api.openai.com/v1/realtime?model=gpt-realtime", model.lastConnectURL)
	require.NotNil(t, model.lastConnectHeads)
	assert.Equal(t, "Bearer sk-test", model.lastConnectHeads["Authorization"])
}

func TestConnectBuildsCallIDURL(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		CallID:          "call_789",
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.NoError(t, err)
	assert.Equal(t, "wss://api.openai.com/v1/realtime?call_id=call_789", model.lastConnectURL)
}

func TestConnectUsesCustomURLAndHeaders(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey: "sk-test",
		URL:    "wss://example.test/realtime",
		Headers: map[string]string{
			"X-Test": "yes",
		},
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.NoError(t, err)
	assert.Equal(t, "wss://example.test/realtime", model.lastConnectURL)
	assert.Equal(t, map[string]string{"X-Test": "yes"}, model.lastConnectHeads)
}

func TestConnectWithTransportDialerFailurePropagates(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		EnableTransport: true,
		TransportDialer: func(_ context.Context, _ string, _ map[string]string, _ *RealtimeTransportConfig) (RealtimeWebSocketConn, error) {
			return nil, errors.New("dial failed")
		},
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to connect websocket transport")
	assert.False(t, model.connected)
	assert.Nil(t, model.websocketConn)
}

func TestConnectAlreadyConnectedReturnsError(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	require.NoError(t, model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		InitialSettings: RealtimeSessionModelSettings{},
	}))

	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already connected")
}

func TestConnectRequiresAPIKeyWithCustomHeaders(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	model := NewOpenAIRealtimeWebSocketModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		URL: "wss://example.test/realtime",
		Headers: map[string]string{
			"X-Test": "yes",
		},
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "api key is required")
}

func TestConnectRequiresAPIKeyWithoutHeaders(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	model := NewOpenAIRealtimeWebSocketModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "api key is required")
}

func TestConnectUsesModelTransportConfigWhenOptionMissing(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	model.SetTransportConfig(&RealtimeTransportConfig{})

	fakeConn := newFakeRealtimeWebSocketConn()
	var captured *RealtimeTransportConfig

	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		EnableTransport: true,
		TransportDialer: func(_ context.Context, _ string, _ map[string]string, cfg *RealtimeTransportConfig) (RealtimeWebSocketConn, error) {
			captured = cfg
			return fakeConn, nil
		},
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.NoError(t, err)
	require.NotNil(t, captured)
	assert.Nil(t, captured.PingInterval)
	assert.Nil(t, captured.PingTimeout)
	assert.Nil(t, captured.HandshakeTimeout)
	require.NoError(t, model.Close(t.Context()))
}

func TestConnectPassesTransportConfig(t *testing.T) {
	pingInterval := 100 * time.Millisecond
	pingTimeout := 250 * time.Millisecond
	handshakeTimeout := 2 * time.Second

	model := NewOpenAIRealtimeWebSocketModel()
	fakeConn := newFakeRealtimeWebSocketConn()
	var captured *RealtimeTransportConfig

	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		EnableTransport: true,
		TransportConfig: &RealtimeTransportConfig{
			PingInterval:     &pingInterval,
			PingTimeout:      &pingTimeout,
			HandshakeTimeout: &handshakeTimeout,
		},
		TransportDialer: func(_ context.Context, _ string, _ map[string]string, cfg *RealtimeTransportConfig) (RealtimeWebSocketConn, error) {
			captured = cfg
			return fakeConn, nil
		},
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.NoError(t, err)
	require.NotNil(t, captured)
	require.NotNil(t, captured.PingInterval)
	require.NotNil(t, captured.PingTimeout)
	require.NotNil(t, captured.HandshakeTimeout)
	assert.Equal(t, pingInterval, *captured.PingInterval)
	assert.Equal(t, pingTimeout, *captured.PingTimeout)
	assert.Equal(t, handshakeTimeout, *captured.HandshakeTimeout)
	require.NoError(t, model.Close(t.Context()))
}

func TestConnectUsesAPIKeyProvider(t *testing.T) {
	called := false
	model := NewOpenAIRealtimeWebSocketModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKeyProvider: func(context.Context) (string, error) {
			called = true
			return "provider-key", nil
		},
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.NoError(t, err)
	assert.True(t, called)
	require.NotNil(t, model.lastConnectHeads)
	assert.Equal(t, "Bearer provider-key", model.lastConnectHeads["Authorization"])
}

func TestSessionUpdateIncludesNoiseReduction(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey: "sk-test",
		InitialSettings: RealtimeSessionModelSettings{
			"model_name":                  "gpt-4o-realtime-preview",
			"input_audio_noise_reduction": map[string]any{"type": "near_field"},
		},
	})
	require.NoError(t, err)

	var session map[string]any
	for _, event := range model.sentClientEvents {
		if eventType, _ := event["type"].(string); eventType == "session.update" {
			session, _ = event["session"].(map[string]any)
		}
	}
	require.NotNil(t, session)

	audio, ok := session["audio"].(map[string]any)
	require.True(t, ok)
	input, ok := audio["input"].(map[string]any)
	require.True(t, ok)
	noiseReduction, ok := input["noise_reduction"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "near_field", noiseReduction["type"])
}

func TestSessionUpdateOmitsNoiseReductionWhenNotProvided(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey: "sk-test",
		InitialSettings: RealtimeSessionModelSettings{
			"model_name": "gpt-4o-realtime-preview",
		},
	})
	require.NoError(t, err)

	var session map[string]any
	for _, event := range model.sentClientEvents {
		if eventType, _ := event["type"].(string); eventType == "session.update" {
			session, _ = event["session"].(map[string]any)
		}
	}
	require.NotNil(t, session)

	audio, ok := session["audio"].(map[string]any)
	require.True(t, ok)
	input, ok := audio["input"].(map[string]any)
	require.True(t, ok)
	_, hasNoiseReduction := input["noise_reduction"]
	assert.False(t, hasNoiseReduction)
}

func TestSessionUpdateAllowsDisablingTurnDetection(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey: "sk-test",
		InitialSettings: RealtimeSessionModelSettings{
			"model_name":     "gpt-4o-realtime-preview",
			"turn_detection": nil,
		},
	})
	require.NoError(t, err)

	var session map[string]any
	for _, event := range model.sentClientEvents {
		if eventType, _ := event["type"].(string); eventType == "session.update" {
			session, _ = event["session"].(map[string]any)
		}
	}
	require.NotNil(t, session)

	audio, ok := session["audio"].(map[string]any)
	require.True(t, ok)
	input, ok := audio["input"].(map[string]any)
	require.True(t, ok)
	value, hasTurnDetection := input["turn_detection"]
	assert.True(t, hasTurnDetection)
	assert.Nil(t, value)
}

func TestSIPModelRequiresCallID(t *testing.T) {
	model := NewOpenAIRealtimeSIPModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "requires `call_id`")
}

func TestSIPModelUsesCallIDURL(t *testing.T) {
	model := NewOpenAIRealtimeSIPModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		CallID:          "call_456",
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.NoError(t, err)
	assert.Equal(
		t,
		"wss://api.openai.com/v1/realtime?call_id=call_456",
		model.lastConnectURL,
	)
}
