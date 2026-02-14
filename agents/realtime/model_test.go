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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRealtimeModelConfigResolveAPIKeyFromString(t *testing.T) {
	cfg := RealtimeModelConfig{
		APIKey: "string-key",
		APIKeyProvider: func(context.Context) (string, error) {
			return "provider-key", nil
		},
	}
	key, err := cfg.ResolveAPIKey(t.Context())
	require.NoError(t, err)
	assert.Equal(t, "string-key", key)
}

func TestRealtimeModelConfigResolveAPIKeyFromProvider(t *testing.T) {
	cfg := RealtimeModelConfig{
		APIKeyProvider: func(context.Context) (string, error) {
			return "provider-key", nil
		},
	}
	key, err := cfg.ResolveAPIKey(t.Context())
	require.NoError(t, err)
	assert.Equal(t, "provider-key", key)
}

func TestRealtimeModelConfigResolveAPIKeyFromEnv(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-key")
	cfg := RealtimeModelConfig{}
	key, err := cfg.ResolveAPIKey(t.Context())
	require.NoError(t, err)
	assert.Equal(t, "env-key", key)
}

func TestRealtimeModelSendEventsExposeType(t *testing.T) {
	events := []RealtimeModelSendEvent{
		RealtimeModelSendRawMessage{},
		RealtimeModelSendUserInput{},
		RealtimeModelSendAudio{},
		RealtimeModelSendToolOutput{},
		RealtimeModelSendInterrupt{},
		RealtimeModelSendSessionUpdate{},
	}

	for _, event := range events {
		assert.NotEmpty(t, event.Type())
	}
}
