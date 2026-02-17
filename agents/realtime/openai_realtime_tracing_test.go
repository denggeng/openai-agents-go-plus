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

func TestTracingConfigStorageAndDefaults(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	tracing := map[string]any{
		"workflow_name": "test_workflow",
		"group_id":      "group_123",
		"metadata": map[string]any{
			"version": "1.0",
		},
	}

	mustConnectRealtimeModel(t, model, RealtimeModelConfig{
		InitialSettings: RealtimeSessionModelSettings{
			"tracing": tracing,
		},
	})

	stored, ok := model.tracingConfig.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, tracing, stored)

	model2 := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model2, RealtimeModelConfig{
		InitialSettings: RealtimeSessionModelSettings{},
	})
	assert.Equal(t, "auto", model2.tracingConfig)
}

func TestSendTracingConfigOnSessionCreatedUsesConfig(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	metadata := map[string]any{
		"user_id":      "user_123",
		"session_type": "demo",
		"features":     []any{"audio", "tools"},
		"config": map[string]any{
			"mode": "fast",
		},
	}
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{
		InitialSettings: RealtimeSessionModelSettings{
			"tracing": map[string]any{
				"workflow_name": "test_workflow",
				"group_id":      "group_123",
				"metadata":      metadata,
			},
		},
	})

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "session.created",
		"session": map[string]any{
			"id": "session_456",
		},
	}))

	require.Len(t, model.sentClientEvents, 1)
	assert.Equal(t, "session.update", model.sentClientEvents[0]["type"])

	sessionPayload, ok := toStringAnyMap(model.sentClientEvents[0]["session"])
	require.True(t, ok)
	tracingPayload, ok := toStringAnyMap(sessionPayload["tracing"])
	require.True(t, ok)

	assert.Equal(t, "test_workflow", tracingPayload["workflow_name"])
	assert.Equal(t, "group_123", tracingPayload["group_id"])

	metadataPayload, ok := toStringAnyMap(tracingPayload["metadata"])
	require.True(t, ok)
	assert.Equal(t, metadata, metadataPayload)
}

func TestSendTracingConfigAutoMode(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	mustConnectRealtimeModel(t, model, RealtimeModelConfig{
		InitialSettings: RealtimeSessionModelSettings{},
	})

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "session.created",
		"session": map[string]any{
			"id": "session_auto",
		},
	}))

	require.Len(t, model.sentClientEvents, 1)
	sessionPayload, ok := toStringAnyMap(model.sentClientEvents[0]["session"])
	require.True(t, ok)
	assert.Equal(t, "auto", sessionPayload["tracing"])
}
