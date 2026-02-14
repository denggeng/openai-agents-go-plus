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

// RealtimeRunner is the realtime equivalent of the standard Runner API.
//
// It creates a RealtimeSession and keeps the configured model/run settings.
type RealtimeRunner struct {
	startingAgent *RealtimeAgent[any]
	model         RealtimeModel
	config        RealtimeRunConfig
}

// NewRealtimeRunner builds a realtime runner with optional model/config overrides.
func NewRealtimeRunner(
	startingAgent *RealtimeAgent[any],
	model RealtimeModel,
	config RealtimeRunConfig,
) *RealtimeRunner {
	if model == nil {
		model = NewOpenAIRealtimeWebSocketModel()
	}
	return &RealtimeRunner{
		startingAgent: startingAgent,
		model:         model,
		config:        RealtimeRunConfig(cloneSettingsMap(RealtimeSessionModelSettings(config))),
	}
}

// Run creates a realtime session.
//
// The returned session is not connected yet; call Enter() to open the model connection.
func (r *RealtimeRunner) Run(contextValue any, modelConfig *RealtimeModelConfig) *RealtimeSession {
	cfg := RealtimeModelConfig{}
	if modelConfig != nil {
		cfg = *modelConfig
	}
	return NewRealtimeSession(
		r.model,
		r.startingAgent,
		contextValue,
		cfg,
		RealtimeRunConfig(cloneSettingsMap(RealtimeSessionModelSettings(r.config))),
	)
}
