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

package agents

import (
	"context"
	"maps"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/denggeng/openai-agents-go-plus/tracing"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func assertPrepareRequestExtraArgs(
	t *testing.T,
	model OpenAIChatCompletionsModel,
	settings modelsettings.ModelSettings,
	expectedExtras map[string]any,
	expectedTemperature param.Opt[float64],
) {
	t.Helper()

	origExtraArgs := maps.Clone(settings.ExtraArgs)
	settings.CustomizeChatCompletionsRequest = func(
		_ context.Context,
		params *openai.ChatCompletionNewParams,
		opts []option.RequestOption,
	) (*openai.ChatCompletionNewParams, []option.RequestOption, error) {
		assert.Equal(t, expectedTemperature, params.Temperature)
		expectedLen := len(expectedExtras) + 1 // User-Agent header
		assert.Len(t, opts, expectedLen)
		return params, opts, nil
	}

	err := tracing.GenerationSpan(
		t.Context(),
		tracing.GenerationSpanParams{Disabled: true},
		func(ctx context.Context, span tracing.Span) error {
			_, _, err := model.prepareRequest(
				ctx,
				param.Opt[string]{},
				InputString("test input"),
				settings,
				nil,
				nil,
				nil,
				span,
				ModelTracingDisabled,
				false,
			)
			return err
		},
	)
	require.NoError(t, err)
	merged := mergedModelExtraJSON(settings)
	if len(expectedExtras) == 0 {
		assert.True(t, merged == nil || len(merged) == 0)
	} else {
		assert.Equal(t, expectedExtras, merged)
	}
	assert.Equal(t, origExtraArgs, settings.ExtraArgs)
}

func TestLiteLLMExtraArgsForwarded(t *testing.T) {
	provider := NewLiteLLMProvider(LiteLLMProviderParams{})
	model, err := provider.GetModel("test-model")
	require.NoError(t, err)

	chatModel, ok := model.(OpenAIChatCompletionsModel)
	require.True(t, ok)

	settings := modelsettings.ModelSettings{
		Temperature: param.NewOpt(0.5),
		ExtraArgs: map[string]any{
			"custom_param": "custom_value",
			"seed":         42,
			"stop":         []string{"END"},
			"logit_bias":   map[string]int{"123": -100},
		},
	}

	expected := map[string]any{
		"custom_param": "custom_value",
		"seed":         42,
		"stop":         []string{"END"},
		"logit_bias":   map[string]int{"123": -100},
	}

	assertPrepareRequestExtraArgs(t, chatModel, settings, expected, param.NewOpt(0.5))
}

func TestOpenAIChatCompletionsExtraArgsForwarded(t *testing.T) {
	model := NewOpenAIChatCompletionsModel("gpt-4", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))

	settings := modelsettings.ModelSettings{
		Temperature: param.NewOpt(0.7),
		ExtraArgs: map[string]any{
			"seed":       123,
			"logit_bias": map[string]int{"456": 10},
			"stop":       []string{"STOP", "END"},
			"user":       "test-user",
		},
	}

	expected := map[string]any{
		"seed":       123,
		"logit_bias": map[string]int{"456": 10},
		"stop":       []string{"STOP", "END"},
		"user":       "test-user",
	}

	assertPrepareRequestExtraArgs(t, model, settings, expected, param.NewOpt(0.7))
}

func TestEmptyExtraArgsHandling(t *testing.T) {
	model := NewOpenAIChatCompletionsModel("gpt-4", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))

	settingsNone := modelsettings.ModelSettings{Temperature: param.NewOpt(0.5)}
	assertPrepareRequestExtraArgs(t, model, settingsNone, map[string]any{}, param.NewOpt(0.5))

	settingsEmpty := modelsettings.ModelSettings{
		Temperature: param.NewOpt(0.3),
		ExtraArgs:   map[string]any{},
	}
	assertPrepareRequestExtraArgs(t, model, settingsEmpty, map[string]any{}, param.NewOpt(0.3))
}

func TestReasoningEffortFallsBackToExtraArgs(t *testing.T) {
	settings := modelsettings.ModelSettings{
		ExtraArgs: map[string]any{
			"reasoning_effort": "none",
			"custom_param":     "custom_value",
		},
	}

	expected := map[string]any{
		"reasoning_effort": "none",
		"custom_param":     "custom_value",
	}

	assert.Equal(t, expected, mergedModelExtraJSON(settings))
	assert.Equal(t, expected, settings.ExtraArgs)
}
