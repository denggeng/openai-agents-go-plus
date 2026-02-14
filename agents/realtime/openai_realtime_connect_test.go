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
	"testing"

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
		URL: "wss://example.test/realtime",
		Headers: map[string]string{
			"X-Test": "yes",
		},
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.NoError(t, err)
	assert.Equal(t, "wss://example.test/realtime", model.lastConnectURL)
	assert.Equal(t, map[string]string{"X-Test": "yes"}, model.lastConnectHeads)
}

func TestConnectRequiresAPIKeyWithoutHeaders(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "api key is required")
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
