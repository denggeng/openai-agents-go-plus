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
	"encoding/json"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSessionConfigDefaultsAudioFormatsWhenNotCall(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()

	cfg, err := model.GetSessionConfig(RealtimeSessionModelSettings{})
	require.NoError(t, err)
	require.NotNil(t, cfg)

	require.NotNil(t, cfg.Audio.Input.Format.GetType())
	assert.Equal(t, "audio/pcm", *cfg.Audio.Input.Format.GetType())
	require.NotNil(t, cfg.Audio.Output.Format.GetType())
	assert.Equal(t, "audio/pcm", *cfg.Audio.Output.Format.GetType())
}

func TestSessionConfigPreservesSIPAudioFormats(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	model.callID = "call-123"

	cfg, err := model.GetSessionConfig(RealtimeSessionModelSettings{
		"turn_detection": map[string]any{
			"type":               "semantic_vad",
			"interrupt_response": true,
		},
	})
	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Nil(t, cfg.Audio.Input.Format.GetType())
	assert.Nil(t, cfg.Audio.Output.Format.GetType())
}

func TestSessionConfigRespectsAudioBlockAndOutputModalities(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()

	cfg, err := model.GetSessionConfig(RealtimeSessionModelSettings{
		"input_audio_format":  "pcm16",
		"output_audio_format": "pcm16",
		"modalities":          []string{"audio"},
		"output_modalities":   []string{"text"},
		"audio": map[string]any{
			"input": map[string]any{
				"format": map[string]any{
					"type": "audio/pcmu",
				},
				"turn_detection": map[string]any{
					"type":              "server_vad",
					"createResponse":    true,
					"silenceDurationMs": 450,
				},
			},
			"output": map[string]any{
				"format": map[string]any{
					"type": "audio/pcma",
				},
				"voice": "synth-1",
				"speed": 1.5,
			},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, cfg)

	assert.Equal(t, []string{"text"}, cfg.OutputModalities)
	require.NotNil(t, cfg.Audio.Input.Format.GetType())
	assert.Equal(t, "audio/pcmu", *cfg.Audio.Input.Format.GetType())
	require.NotNil(t, cfg.Audio.Output.Format.GetType())
	assert.Equal(t, "audio/pcma", *cfg.Audio.Output.Format.GetType())
	assert.Equal(t, "synth-1", string(cfg.Audio.Output.Voice))
	assert.True(t, cfg.Audio.Output.Speed.Valid())
	assert.InDelta(t, 1.5, cfg.Audio.Output.Speed.Value, 1e-9)

	assert.True(t, cfg.Audio.Input.Transcription.Model != "")

	td := cfg.Audio.Input.TurnDetection
	require.NotNil(t, td.GetType())
	assert.Equal(t, "server_vad", *td.GetType())
	require.NotNil(t, td.GetCreateResponse())
	assert.Equal(t, true, *td.GetCreateResponse())
	require.NotNil(t, td.GetSilenceDurationMs())
	assert.Equal(t, int64(450), *td.GetSilenceDurationMs())
}

func TestCallIDSessionUpdateOmitsNullAudioFormats(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	model.callID = "call_123"

	cfg, err := model.GetSessionConfig(RealtimeSessionModelSettings{})
	require.NoError(t, err)

	raw, err := json.Marshal(cfg)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(raw, &payload))
	audio, ok := payload["audio"].(map[string]any)
	require.True(t, ok)
	input, ok := audio["input"].(map[string]any)
	require.True(t, ok)
	output, ok := audio["output"].(map[string]any)
	require.True(t, ok)
	_, hasInputFormat := input["format"]
	_, hasOutputFormat := output["format"]
	assert.False(t, hasInputFormat)
	assert.False(t, hasOutputFormat)
}

func TestCallIDSessionUpdateIncludesExplicitAudioFormats(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	model.callID = "call_123"

	cfg, err := model.GetSessionConfig(RealtimeSessionModelSettings{
		"input_audio_format":  "g711_ulaw",
		"output_audio_format": "g711_ulaw",
	})
	require.NoError(t, err)

	raw, err := json.Marshal(cfg)
	require.NoError(t, err)
	var payload map[string]any
	require.NoError(t, json.Unmarshal(raw, &payload))

	audio := payload["audio"].(map[string]any)
	input := audio["input"].(map[string]any)
	output := audio["output"].(map[string]any)
	inputFormat := input["format"].(map[string]any)
	outputFormat := output["format"].(map[string]any)
	assert.Equal(t, "audio/pcmu", inputFormat["type"])
	assert.Equal(t, "audio/pcmu", outputFormat["type"])
}

func TestToolsConversionRejectsNonFunctionToolsAndIncludesHandoffs(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()

	_, err := model.toolsToSessionTools([]agents.Tool{agents.FileSearchTool{}}, nil)
	require.Error(t, err)

	handoff := agents.Handoff{
		ToolName:        "transfer_to_a",
		ToolDescription: "handoff",
		InputJSONSchema: map[string]any{"type": "object"},
	}
	out, err := model.toolsToSessionTools(nil, []agents.Handoff{handoff})
	require.NoError(t, err)
	require.Len(t, out, 1)
	require.NotNil(t, out[0].GetName())
	assert.Equal(t, "transfer_to_a", *out[0].GetName())
}
