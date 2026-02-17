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

func TestSpeechStartedWithoutFieldsDoesNotAutoInterrupt(t *testing.T) {
	model := NewOpenAIRealtimeWebSocketModel()
	listener := &captureRealtimeListener{}
	model.AddListener(listener)

	require.NoError(t, model.handleWSEvent(t.Context(), map[string]any{
		"type": "input_audio_buffer.speech_started",
	}))

	assert.Empty(t, model.sentClientEvents)
	require.Len(t, listener.events, 2)
	_, ok := listener.events[1].(RealtimeModelErrorEvent)
	require.True(t, ok)
}
