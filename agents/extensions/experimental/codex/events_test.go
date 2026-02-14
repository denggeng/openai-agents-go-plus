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

package codex

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCoerceThreadEventThreadStarted(t *testing.T) {
	event, err := CoerceThreadEvent(map[string]any{
		"type":      "thread.started",
		"thread_id": "thread-123",
	})
	require.NoError(t, err)

	typed, ok := event.(ThreadStartedEvent)
	require.True(t, ok)
	assert.Equal(t, "thread.started", typed.EventType())
	assert.Equal(t, "thread-123", typed.ThreadID)
}

func TestCoerceThreadEventTurnCompletedUsage(t *testing.T) {
	event, err := CoerceThreadEvent(map[string]any{
		"type": "turn.completed",
		"usage": map[string]any{
			"input_tokens":        2,
			"cached_input_tokens": 1,
			"output_tokens":       3,
		},
	})
	require.NoError(t, err)

	typed, ok := event.(TurnCompletedEvent)
	require.True(t, ok)
	require.NotNil(t, typed.Usage)
	assert.Equal(t, 2, typed.Usage.InputTokens)
	assert.Equal(t, 1, typed.Usage.CachedInputTokens)
	assert.Equal(t, 3, typed.Usage.OutputTokens)
}

func TestCoerceThreadEventItemCompletedAgentMessage(t *testing.T) {
	event, err := CoerceThreadEvent(map[string]any{
		"type": "item.completed",
		"item": map[string]any{
			"id":   "agent-1",
			"type": "agent_message",
			"text": "done",
		},
	})
	require.NoError(t, err)

	typed, ok := event.(ItemCompletedEvent)
	require.True(t, ok)
	agentMessage, ok := typed.Item.(AgentMessageItem)
	require.True(t, ok)
	assert.Equal(t, "agent-1", agentMessage.ID)
	assert.Equal(t, "done", agentMessage.Text)
}

func TestCoerceThreadEventUnknownFallback(t *testing.T) {
	event, err := CoerceThreadEvent(map[string]any{
		"type":    "unknown_event",
		"payload": "x",
	})
	require.NoError(t, err)

	typed, ok := event.(UnknownThreadEvent)
	require.True(t, ok)
	assert.Equal(t, "unknown_event", typed.EventType())
	assert.Equal(t, "x", typed.Payload["payload"])
}

func TestCoerceThreadEventUnknownTypeWhenMissingType(t *testing.T) {
	event, err := CoerceThreadEvent(map[string]any{
		"payload": "x",
	})
	require.NoError(t, err)

	typed, ok := event.(UnknownThreadEvent)
	require.True(t, ok)
	assert.Equal(t, "unknown", typed.EventType())
}

func TestCoerceThreadEventRejectsNonMapping(t *testing.T) {
	_, err := CoerceThreadEvent("not-a-map")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "thread event payload must be a mapping")
}
