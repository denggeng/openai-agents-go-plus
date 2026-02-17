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
	"fmt"

	"github.com/denggeng/openai-agents-go-plus/agents"
	oairealtime "github.com/openai/openai-go/v3/realtime"
)

// CollectEnabledHandoffs converts realtime handoff declarations into enabled handoff tools.
func CollectEnabledHandoffs[T any](
	agent *RealtimeAgent[T],
	contextWrapper *agents.RunContextWrapper[T],
) ([]agents.Handoff, error) {
	if agent == nil || len(agent.Handoffs) == 0 {
		return nil, nil
	}

	enabled := make([]agents.Handoff, 0, len(agent.Handoffs))
	for _, item := range agent.Handoffs {
		switch v := item.(type) {
		case agents.Handoff:
			if isHandoffEnabled(v, agent, contextWrapper) {
				enabled = append(enabled, v)
			}
		case *RealtimeAgent[T]:
			if v != nil {
				enabled = append(enabled, RealtimeHandoff(v))
			}
		case RealtimeAgent[T]:
			clone := v
			enabled = append(enabled, RealtimeHandoff(&clone))
		default:
			return nil, fmt.Errorf("unsupported handoff item type %T", item)
		}
	}

	return enabled, nil
}

// BuildModelSettingsFromAgent merges agent-level data into realtime model settings.
func BuildModelSettingsFromAgent[T any](
	agent *RealtimeAgent[T],
	contextWrapper *agents.RunContextWrapper[T],
	baseSettings RealtimeSessionModelSettings,
	startingSettings RealtimeSessionModelSettings,
	runConfig RealtimeRunConfig,
) (RealtimeSessionModelSettings, error) {
	updated := cloneSettingsMap(baseSettings)
	if updated == nil {
		updated = RealtimeSessionModelSettings{}
	}

	if agent != nil {
		if agent.Prompt != nil {
			updated["prompt"] = cloneStringAnyMap(agent.Prompt)
		}

		instructions, err := agent.GetSystemPrompt(contextWrapper)
		if err != nil {
			return nil, err
		}
		updated["instructions"] = instructions

		tools, err := agent.GetAllTools(contextWrapper)
		if err != nil {
			return nil, err
		}
		updated["tools"] = tools

		handoffs, err := CollectEnabledHandoffs(agent, contextWrapper)
		if err != nil {
			return nil, err
		}
		updated["handoffs"] = handoffs
	}

	for key, value := range startingSettings {
		updated[key] = value
	}

	if tracingDisabled, _ := runConfig["tracing_disabled"].(bool); tracingDisabled {
		updated["tracing"] = nil
	}

	return updated, nil
}

func cloneSettingsMap(input RealtimeSessionModelSettings) RealtimeSessionModelSettings {
	if input == nil {
		return nil
	}
	out := make(RealtimeSessionModelSettings, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func isHandoffEnabled[T any](
	handoff agents.Handoff,
	agent *RealtimeAgent[T],
	contextWrapper *agents.RunContextWrapper[T],
) bool {
	if handoff.IsEnabled == nil {
		return true
	}

	ctx := context.Background()
	if contextWrapper != nil {
		anyContext := agents.NewRunContextWrapper[any](contextWrapper.Context)
		anyContext.Usage = contextWrapper.Usage
		anyContext.TurnInput = contextWrapper.TurnInput
		anyContext.ToolInput = contextWrapper.ToolInput
		ctx = agents.ContextWithRunContextValue(ctx, anyContext)
	}

	enabled, err := handoff.IsEnabled.IsEnabled(ctx, &agents.Agent{Name: agent.Name})
	if err != nil {
		return false
	}
	return enabled
}

// BuildInitialSessionPayload builds SIP-ready session config with agent/run overrides.
func BuildInitialSessionPayload[T any](
	agent *RealtimeAgent[T],
	contextValue T,
	modelConfig RealtimeModelConfig,
	runConfig RealtimeRunConfig,
	overrides RealtimeSessionModelSettings,
) (*oairealtime.RealtimeSessionCreateRequestParam, error) {
	runConfigSettings, _ := toRealtimeSettings(runConfig["model_settings"])
	initialSettings := cloneSettingsMap(modelConfig.InitialSettings)
	baseSettings := cloneSettingsMap(runConfigSettings)
	if baseSettings == nil {
		baseSettings = RealtimeSessionModelSettings{}
	}
	for key, value := range initialSettings {
		baseSettings[key] = value
	}

	contextWrapper := agents.NewRunContextWrapper[T](contextValue)
	mergedSettings, err := BuildModelSettingsFromAgent(
		agent,
		contextWrapper,
		baseSettings,
		initialSettings,
		runConfig,
	)
	if err != nil {
		return nil, err
	}

	for key, value := range overrides {
		mergedSettings[key] = value
	}

	baseModel := NewOpenAIRealtimeWebSocketModel()
	return baseModel.GetSessionConfig(mergedSettings)
}

func toRealtimeSettings(input any) (RealtimeSessionModelSettings, bool) {
	switch v := input.(type) {
	case RealtimeSessionModelSettings:
		return cloneSettingsMap(v), true
	case map[string]any:
		return RealtimeSessionModelSettings(v), true
	default:
		return nil, false
	}
}
