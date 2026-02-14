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

func TestToRealtimeAudioFormatFromStrings(t *testing.T) {
	testCases := []struct {
		input        string
		expectedType string
	}{
		{input: "pcm", expectedType: "audio/pcm"},
		{input: "pcm16", expectedType: "audio/pcm"},
		{input: "audio/pcm", expectedType: "audio/pcm"},
		{input: "pcmu", expectedType: "audio/pcmu"},
		{input: "audio/pcmu", expectedType: "audio/pcmu"},
		{input: "g711_ulaw", expectedType: "audio/pcmu"},
		{input: "pcma", expectedType: "audio/pcma"},
		{input: "audio/pcma", expectedType: "audio/pcma"},
		{input: "g711_alaw", expectedType: "audio/pcma"},
	}

	for _, tc := range testCases {
		format := ToRealtimeAudioFormat(tc.input)
		require.NotNil(t, format)
		require.NotNil(t, format.GetType())
		assert.Equal(t, tc.expectedType, *format.GetType())
	}
}

func TestToRealtimeAudioFormatPassthroughAndUnknown(t *testing.T) {
	format := &oairealtime.RealtimeAudioFormatsUnionParam{
		OfAudioPCM: &oairealtime.RealtimeAudioFormatsAudioPCMParam{
			Type: "audio/pcm",
			Rate: 24000,
		},
	}

	// Passing union params should keep the same pointer.
	assert.Same(t, format, ToRealtimeAudioFormat(format))

	// Unknown strings should return nil.
	assert.Nil(t, ToRealtimeAudioFormat("something_else"))
}

func TestToRealtimeAudioFormatNil(t *testing.T) {
	assert.Nil(t, ToRealtimeAudioFormat(nil))
}

func TestToRealtimeAudioFormatFromMapping(t *testing.T) {
	pcm := ToRealtimeAudioFormat(map[string]any{"type": "audio/pcm", "rate": 16000})
	require.NotNil(t, pcm)
	require.NotNil(t, pcm.GetType())
	assert.Equal(t, "audio/pcm", *pcm.GetType())
	require.NotNil(t, pcm.GetRate())
	assert.Equal(t, int64(24000), *pcm.GetRate())

	pcmDefaultRate := ToRealtimeAudioFormat(map[string]any{"type": "audio/pcm"})
	require.NotNil(t, pcmDefaultRate)
	require.NotNil(t, pcmDefaultRate.GetRate())
	assert.Equal(t, int64(24000), *pcmDefaultRate.GetRate())

	ulaw := ToRealtimeAudioFormat(map[string]any{"type": "audio/pcmu"})
	require.NotNil(t, ulaw)
	require.NotNil(t, ulaw.GetType())
	assert.Equal(t, "audio/pcmu", *ulaw.GetType())

	alaw := ToRealtimeAudioFormat(map[string]any{"type": "audio/pcma"})
	require.NotNil(t, alaw)
	require.NotNil(t, alaw.GetType())
	assert.Equal(t, "audio/pcma", *alaw.GetType())

	assert.Nil(t, ToRealtimeAudioFormat(map[string]any{"type": "audio/unknown", "rate": 8000}))
}
