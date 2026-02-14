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

func TestUserMessageConversion(t *testing.T) {
	itemText := map[string]any{
		"id":   "123",
		"type": "message",
		"role": "user",
		"content": []map[string]any{
			{"type": "input_text", "text": "hello"},
		},
	}

	converted, err := ConversationItemToRealtimeMessageItem(itemText, nil)
	require.NoError(t, err)
	require.NotNil(t, converted)
	assert.Equal(t, "user", converted.Role)
	require.Len(t, converted.Content, 1)
	assert.Equal(t, "input_text", converted.Content[0].Type)

	itemAudio := map[string]any{
		"id":   "123",
		"type": "message",
		"role": "user",
		"content": []map[string]any{
			{"type": "input_audio", "audio": "base64-audio"},
		},
	}

	convertedAudio, err := ConversationItemToRealtimeMessageItem(itemAudio, nil)
	require.NoError(t, err)
	require.NotNil(t, convertedAudio)
	assert.Equal(t, "user", convertedAudio.Role)
	require.Len(t, convertedAudio.Content, 1)
	assert.Equal(t, "input_audio", convertedAudio.Content[0].Type)
}

func TestAssistantMessageConversion(t *testing.T) {
	item := map[string]any{
		"id":   "123",
		"type": "message",
		"role": "assistant",
		"content": []map[string]any{
			{"type": "output_text", "text": "assistant reply"},
		},
	}

	converted, err := ConversationItemToRealtimeMessageItem(item, nil)
	require.NoError(t, err)
	require.NotNil(t, converted)
	assert.Equal(t, "assistant", converted.Role)
	require.Len(t, converted.Content, 1)
	assert.Equal(t, "text", converted.Content[0].Type)
}

func TestSystemMessageConversion(t *testing.T) {
	item := map[string]any{
		"id":   "123",
		"type": "message",
		"role": "system",
		"content": []map[string]any{
			{"type": "input_text", "text": "system prompt"},
		},
	}

	converted, err := ConversationItemToRealtimeMessageItem(item, nil)
	require.NoError(t, err)
	require.NotNil(t, converted)
	assert.Equal(t, "system", converted.Role)
}
