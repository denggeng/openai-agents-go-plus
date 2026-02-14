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

	oairealtime "github.com/openai/openai-go/v3/realtime"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type dummyAudioFormat struct {
	Type string
}

func sessionWithOutput(
	format oairealtime.RealtimeAudioFormatsUnionParam,
) oairealtime.RealtimeSessionCreateRequestParam {
	return oairealtime.RealtimeSessionCreateRequestParam{
		Type:  "realtime",
		Model: "gpt-realtime",
		Audio: oairealtime.RealtimeAudioConfigParam{
			Output: oairealtime.RealtimeAudioConfigOutputParam{
				Format: format,
			},
		},
	}
}

func TestNormalizeSessionPayloadVariants(t *testing.T) {
	rt := &oairealtime.RealtimeSessionCreateRequestParam{
		Type:  "realtime",
		Model: "gpt-realtime",
	}
	assert.Same(t, rt, NormalizeSessionPayload(rt))

	ts := &oairealtime.RealtimeTranscriptionSessionCreateRequestParam{
		Type: "transcription",
	}
	assert.Nil(t, NormalizeSessionPayload(ts))

	transcriptionMapping := map[string]any{"type": "transcription"}
	assert.Nil(t, NormalizeSessionPayload(transcriptionMapping))

	realtimeMapping := map[string]any{
		"type":  "realtime",
		"model": "gpt-realtime",
	}
	asModel := NormalizeSessionPayload(realtimeMapping)
	require.NotNil(t, asModel)
	assert.Equal(t, "gpt-realtime", string(asModel.Model))
	assert.Equal(t, "realtime", string(asModel.Type))

	invalidMapping := map[string]any{"type": "bogus"}
	assert.Nil(t, NormalizeSessionPayload(invalidMapping))
}

func TestExtractAudioFormatFromSessionObjects(t *testing.T) {
	sPCM := sessionWithOutput(oairealtime.RealtimeAudioFormatsUnionParam{
		OfAudioPCM: &oairealtime.RealtimeAudioFormatsAudioPCMParam{Type: "audio/pcm", Rate: 24000},
	})
	gotPCM := ExtractSessionAudioFormat(sPCM)
	require.NotNil(t, gotPCM)
	assert.Equal(t, "pcm16", *gotPCM)

	sULaw := sessionWithOutput(oairealtime.RealtimeAudioFormatsUnionParam{
		OfAudioPCMU: &oairealtime.RealtimeAudioFormatsAudioPCMUParam{Type: "audio/pcmu"},
	})
	gotULaw := ExtractSessionAudioFormat(sULaw)
	require.NotNil(t, gotULaw)
	assert.Equal(t, "g711_ulaw", *gotULaw)

	sALaw := sessionWithOutput(oairealtime.RealtimeAudioFormatsUnionParam{
		OfAudioPCMA: &oairealtime.RealtimeAudioFormatsAudioPCMAParam{Type: "audio/pcma"},
	})
	gotALaw := ExtractSessionAudioFormat(sALaw)
	require.NotNil(t, gotALaw)
	assert.Equal(t, "g711_alaw", *gotALaw)

	sNone := oairealtime.RealtimeSessionCreateRequestParam{Type: "realtime", Model: "gpt-realtime"}
	assert.Nil(t, ExtractSessionAudioFormat(sNone))
}

func TestNormalizeAudioFormatFallbacks(t *testing.T) {
	assert.Equal(t, "pcm24", NormalizeAudioFormat("pcm24"))
	assert.Equal(t, "g711_ulaw", NormalizeAudioFormat(map[string]any{"type": "g711_ulaw"}))
	assert.Equal(t, "custom", NormalizeAudioFormat(dummyAudioFormat{Type: "custom"}))
	assert.Equal(t, "weird", NormalizeAudioFormat(&dummyAudioFormat{Type: "weird"}))
}

func TestNormalizeAudioFormatKnownVariants(t *testing.T) {
	assert.Equal(
		t,
		"pcm16",
		NormalizeAudioFormat(
			oairealtime.RealtimeAudioFormatsAudioPCMParam{
				Type: "audio/pcm",
				Rate: 24000,
			},
		),
	)
	assert.Equal(
		t,
		"g711_ulaw",
		NormalizeAudioFormat(
			oairealtime.RealtimeAudioFormatsUnionParam{
				OfAudioPCMU: &oairealtime.RealtimeAudioFormatsAudioPCMUParam{
					Type: "audio/pcmu",
				},
			},
		),
	)
	assert.Equal(t, "42", NormalizeAudioFormat(42))
}

func TestNormalizeTurnDetectionConfig(t *testing.T) {
	normalizedAny := NormalizeTurnDetectionConfig(map[string]any{
		"type":              "server_vad",
		"createResponse":    true,
		"silenceDurationMs": 450,
		"modelVersion":      "default",
	})

	normalized, ok := normalizedAny.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, true, normalized["create_response"])
	assert.Equal(t, 450, normalized["silence_duration_ms"])
	assert.Equal(t, "default", normalized["model_version"])
	_, hasSilenceDurationMs := normalized["silenceDurationMs"]
	assert.False(t, hasSilenceDurationMs)
	_, hasModelVersion := normalized["modelVersion"]
	assert.False(t, hasModelVersion)
}

func TestNormalizeTurnDetectionConfigPreservesSnakeCase(t *testing.T) {
	normalizedAny := NormalizeTurnDetectionConfig(map[string]any{
		"createResponse":  false,
		"create_response": true,
	})

	normalized, ok := normalizedAny.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, true, normalized["create_response"])
	_, hasCamel := normalized["createResponse"]
	assert.False(t, hasCamel)
}

func TestNormalizeTurnDetectionConfigPassthrough(t *testing.T) {
	input := []string{"not", "a", "map"}
	assert.Equal(t, input, NormalizeTurnDetectionConfig(input))
}
