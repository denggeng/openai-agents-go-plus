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

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCollectEnabledHandoffsFiltersDisabled(t *testing.T) {
	parent := &RealtimeAgent[any]{Name: "parent"}

	disabled := RealtimeHandoff(&RealtimeAgent[any]{Name: "child_disabled"})
	disabled.IsEnabled = agents.HandoffDisabled()

	parent.Handoffs = []any{
		disabled,
		&RealtimeAgent[any]{Name: "child_enabled"},
	}

	enabled, err := CollectEnabledHandoffs(parent, agents.NewRunContextWrapper[any](nil))
	require.NoError(t, err)
	require.Len(t, enabled, 1)
	assert.Equal(t, "child_enabled", enabled[0].AgentName)
}

func TestBuildModelSettingsFromAgentMergesAgentFields(t *testing.T) {
	agent := &RealtimeAgent[any]{
		Name:   "root",
		Prompt: map[string]any{"id": "prompt-id"},
	}
	agent.Instructions = RealtimeInstructionsSyncFunc[any](
		func(*agents.RunContextWrapper[any], *RealtimeAgent[any]) string {
			return "sys"
		},
	)

	helper := agents.NewFunctionTool("helper", "Helper tool for testing.", func(_ context.Context, _ struct{}) (string, error) {
		return "ok", nil
	})
	agent.Tools = []agents.Tool{helper}
	agent.Handoffs = []any{&RealtimeAgent[any]{Name: "handoff-child"}}

	baseSettings := RealtimeSessionModelSettings{"model_name": "gpt-realtime"}
	startingSettings := RealtimeSessionModelSettings{"voice": "verse"}
	runConfig := RealtimeRunConfig{"tracing_disabled": true}

	merged, err := BuildModelSettingsFromAgent(
		agent,
		agents.NewRunContextWrapper[any](nil),
		baseSettings,
		startingSettings,
		runConfig,
	)
	require.NoError(t, err)

	assert.Equal(t, map[string]any{"id": "prompt-id"}, merged["prompt"])
	assert.Equal(t, "sys", merged["instructions"])
	tools, ok := merged["tools"].([]agents.Tool)
	require.True(t, ok)
	require.Len(t, tools, 1)
	assert.Equal(t, "helper", tools[0].ToolName())

	handoffs, ok := merged["handoffs"].([]agents.Handoff)
	require.True(t, ok)
	require.Len(t, handoffs, 1)
	assert.Equal(t, "handoff-child", handoffs[0].AgentName)

	assert.Equal(t, "verse", merged["voice"])
	assert.Equal(t, "gpt-realtime", merged["model_name"])
	assert.Nil(t, merged["tracing"])
	assert.Equal(t, RealtimeSessionModelSettings{"model_name": "gpt-realtime"}, baseSettings)
}

func TestBuildInitialSessionPayload(t *testing.T) {
	agent := &RealtimeAgent[map[string]string]{
		Name:   "parent",
		Prompt: map[string]any{"id": "prompt-99"},
	}
	childAgent := &RealtimeAgent[map[string]string]{Name: "child"}
	agent.Handoffs = []any{childAgent}
	agent.Instructions = RealtimeInstructionsSyncFunc[map[string]string](
		func(*agents.RunContextWrapper[map[string]string], *RealtimeAgent[map[string]string]) string {
			return "parent-system"
		},
	)

	ping := agents.NewFunctionTool("ping", "Ping tool used for session payload building.", func(_ context.Context, _ struct{}) (string, error) {
		return "pong", nil
	})
	agent.Tools = []agents.Tool{ping}

	modelConfig := RealtimeModelConfig{
		InitialSettings: RealtimeSessionModelSettings{
			"model_name": "gpt-realtime-mini",
			"voice":      "verse",
		},
	}
	runConfig := RealtimeRunConfig{
		"model_settings": RealtimeSessionModelSettings{
			"output_modalities": []string{"text"},
		},
		"tracing_disabled": true,
	}
	overrides := RealtimeSessionModelSettings{
		"audio": map[string]any{
			"input": map[string]any{
				"format": map[string]any{"type": "audio/pcmu"},
			},
		},
		"output_audio_format": "g711_ulaw",
	}

	payload, err := BuildInitialSessionPayload(
		agent,
		map[string]string{"user": "abc"},
		modelConfig,
		runConfig,
		overrides,
	)
	require.NoError(t, err)
	require.NotNil(t, payload)

	assert.Equal(t, "gpt-realtime-mini", string(payload.Model))
	assert.Equal(t, []string{"text"}, payload.OutputModalities)
	require.NotNil(t, payload.Audio.Input.Format.GetType())
	assert.Equal(t, "audio/pcmu", *payload.Audio.Input.Format.GetType())
	require.NotNil(t, payload.Audio.Output.Format.GetType())
	assert.Equal(t, "audio/pcmu", *payload.Audio.Output.Format.GetType())
	assert.Equal(t, "verse", string(payload.Audio.Output.Voice))
	assert.True(t, payload.Instructions.Valid())
	assert.Equal(t, "parent-system", payload.Instructions.Value)
	assert.Equal(t, "prompt-99", payload.Prompt.ID)

	toolNames := make(map[string]struct{}, len(payload.Tools))
	for _, tool := range payload.Tools {
		if name := tool.GetName(); name != nil && *name != "" {
			toolNames[*name] = struct{}{}
		}
	}
	_, hasPing := toolNames["ping"]
	assert.True(t, hasPing)
	_, hasHandoff := toolNames["transfer_to_child"]
	assert.True(t, hasHandoff)
}

func TestRealtimeHandoffInvokeReturnsTargetAgent(t *testing.T) {
	target := &RealtimeAgent[any]{Name: "target"}
	handoff := RealtimeHandoff(target)

	agent, err := handoff.OnInvokeHandoff(context.Background(), `{}`)
	require.NoError(t, err)
	require.NotNil(t, agent)
	assert.Equal(t, "target", agent.Name)
}
