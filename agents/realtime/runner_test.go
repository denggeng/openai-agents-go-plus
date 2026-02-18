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

func TestNewRealtimeRunnerUsesDefaultModel(t *testing.T) {
	runner := NewRealtimeRunner(&RealtimeAgent[any]{Name: "agent"}, nil, nil)
	assert.IsType(t, &OpenAIRealtimeWebSocketModel{}, runner.model)
}

func TestRealtimeRunnerRunPassesModelConfigAndContext(t *testing.T) {
	model := &mockRealtimeModel{}
	runner := NewRealtimeRunner(
		&RealtimeAgent[any]{Name: "agent"},
		model,
		RealtimeRunConfig{
			"model_settings": RealtimeSessionModelSettings{"voice": "alloy"},
		},
	)

	modelCfg := &RealtimeModelConfig{
		CallID: "call_123",
		InitialSettings: RealtimeSessionModelSettings{
			"model_name": "gpt-realtime-mini",
		},
	}
	contextValue := map[string]any{"user_id": "u_1"}

	session := runner.Run(contextValue, modelCfg)
	require.NotNil(t, session)

	assert.Equal(t, model, session.model)
	assert.Equal(t, "call_123", session.modelConfig.CallID)
	assert.Equal(t, "gpt-realtime-mini", session.modelConfig.InitialSettings["model_name"])
	assert.Equal(t, contextValue, session.contextWrapper.Context)

	runModelSettings, ok := session.runConfig["model_settings"].(RealtimeSessionModelSettings)
	require.True(t, ok)
	assert.Equal(t, "alloy", runModelSettings["voice"])
}

func TestRealtimeRunnerRunWithoutModelConfigUsesZeroValue(t *testing.T) {
	model := &mockRealtimeModel{}
	runner := NewRealtimeRunner(&RealtimeAgent[any]{Name: "agent"}, model, nil)

	session := runner.Run(nil, nil)
	require.NotNil(t, session)
	assert.Equal(t, RealtimeModelConfig{}, session.modelConfig)
}

func TestRealtimeRunnerRunWithConfigDoesNotSetModelConfig(t *testing.T) {
	model := &mockRealtimeModel{}
	runner := NewRealtimeRunner(
		&RealtimeAgent[any]{Name: "agent"},
		model,
		RealtimeRunConfig{
			"model_settings": RealtimeSessionModelSettings{"voice": "nova"},
		},
	)

	session := runner.Run(nil, nil)
	require.NotNil(t, session)
	assert.Equal(t, RealtimeModelConfig{}, session.modelConfig)

	runModelSettings, ok := session.runConfig["model_settings"].(RealtimeSessionModelSettings)
	require.True(t, ok)
	assert.Equal(t, "nova", runModelSettings["voice"])
}
