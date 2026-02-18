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

package agents_test

import (
	"encoding/json"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type sampleContext struct {
	Value string `json:"value"`
}

func TestRunStateMissingSchemaVersionErrors(t *testing.T) {
	_, err := agents.RunStateFromJSONString(`{"current_turn":1}`)
	require.Error(t, err)
	assert.ErrorContains(t, err, "missing schema version")
}

func TestRunStateStrictContextRequiresSerializer(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
		Context: &agents.RunStateContextState{
			Context: sampleContext{Value: "hello"},
		},
	}

	_, err := state.ToJSONWithOptions(agents.RunStateSerializeOptions{StrictContext: true})
	require.Error(t, err)
	assert.ErrorContains(t, err, "context_serializer")
}

func TestRunStateContextSerializerAndDeserializer(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
		Context: &agents.RunStateContextState{
			Context: sampleContext{Value: "ok"},
		},
	}

	serialized, err := state.ToJSONWithOptions(agents.RunStateSerializeOptions{
		ContextSerializer: func(value any) (map[string]any, error) {
			ctx := value.(sampleContext)
			return map[string]any{"value": ctx.Value}, nil
		},
	})
	require.NoError(t, err)

	decoded, err := agents.RunStateFromJSONWithOptions(serialized, agents.RunStateDeserializeOptions{
		ContextDeserializer: func(payload map[string]any) (any, error) {
			raw, _ := payload["value"].(string)
			return sampleContext{Value: raw}, nil
		},
	})
	require.NoError(t, err)
	require.NotNil(t, decoded.Context)
	require.NotNil(t, decoded.Context.ContextMeta)
	assert.Equal(t, "context_serializer", decoded.Context.ContextMeta.SerializedVia)
	assert.True(t, decoded.Context.ContextMeta.RequiresDeserializer)
	assert.False(t, decoded.Context.ContextMeta.Omitted)
	restored, ok := decoded.Context.Context.(sampleContext)
	require.True(t, ok)
	assert.Equal(t, "ok", restored.Value)
}

func TestRunStateStrictDeserializationRequiresDeserializer(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
		Context: &agents.RunStateContextState{
			Context: sampleContext{Value: "need-deserializer"},
		},
	}

	serialized, err := state.ToJSONWithOptions(agents.RunStateSerializeOptions{})
	require.NoError(t, err)

	_, err = agents.RunStateFromJSONWithOptions(serialized, agents.RunStateDeserializeOptions{
		StrictContext: true,
	})
	require.Error(t, err)
	assert.ErrorContains(t, err, "context_deserializer")
}

func TestRunStateUsageAndToolInputRoundTrip(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
		Context: &agents.RunStateContextState{
			Context:   map[string]any{"foo": "bar"},
			ToolInput: map[string]any{"text": "hola"},
			Usage: &usage.Usage{
				Requests:     5,
				InputTokens:  100,
				OutputTokens: 50,
				TotalTokens:  150,
			},
		},
	}

	serialized, err := state.ToJSONWithOptions(agents.RunStateSerializeOptions{})
	require.NoError(t, err)

	decoded, err := agents.RunStateFromJSONWithOptions(serialized, agents.RunStateDeserializeOptions{})
	require.NoError(t, err)
	require.NotNil(t, decoded.Context)
	assert.Equal(t, uint64(5), decoded.Context.Usage.Requests)
	assert.Equal(t, uint64(100), decoded.Context.Usage.InputTokens)
	assert.Equal(t, uint64(50), decoded.Context.Usage.OutputTokens)
	assert.Equal(t, uint64(150), decoded.Context.Usage.TotalTokens)
	assert.Equal(t, map[string]any{"text": "hola"}, decoded.Context.ToolInput)
}

func TestRunStateToolUseTrackerSnapshotFilters(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
	}

	snapshot := map[any]any{
		"agent1": []any{"tool1", 123, "tool2"},
		"agent2": []string{"tool3"},
		123:      []string{"tool4"},
		nil:      []string{"tool5"},
	}

	state.SetToolUseTrackerSnapshot(snapshot)
	out := state.GetToolUseTrackerSnapshot()

	assert.Equal(t, []string{"tool1", "tool2"}, out["agent1"])
	assert.Equal(t, []string{"tool3"}, out["agent2"])
	_, ok := out["123"]
	assert.False(t, ok)
}

func TestRunStateTraceAPIKeyOptIn(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
		Trace: &agents.TraceState{
			TraceID:       "trace_123",
			WorkflowName:  "workflow",
			TracingAPIKey: "secret",
		},
	}

	withoutKey, err := state.ToJSONWithOptions(agents.RunStateSerializeOptions{})
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(withoutKey, &payload))
	trace, ok := payload["trace"].(map[string]any)
	require.True(t, ok)
	_, hasKey := trace["tracing_api_key"]
	assert.False(t, hasKey)

	withKey, err := state.ToJSONWithOptions(agents.RunStateSerializeOptions{IncludeTracingAPIKey: true})
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(withKey, &payload))
	trace, ok = payload["trace"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "secret", trace["tracing_api_key"])
}
