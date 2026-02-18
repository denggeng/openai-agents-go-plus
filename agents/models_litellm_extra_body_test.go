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
	"testing"

	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/denggeng/openai-agents-go-plus/tracing"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func litellmPrepareRequest(
	t *testing.T,
	settings modelsettings.ModelSettings,
	assertFn func(*openai.ChatCompletionNewParams, []option.RequestOption),
) {
	t.Helper()

	m := NewOpenAIChatCompletionsModelWithImpl(
		"model-name",
		NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}),
		"litellm",
	)

	err := tracing.GenerationSpan(
		t.Context(), tracing.GenerationSpanParams{Disabled: true},
		func(ctx context.Context, span tracing.Span) error {
			settings.CustomizeChatCompletionsRequest = func(
				_ context.Context,
				params *openai.ChatCompletionNewParams,
				opts []option.RequestOption,
			) (*openai.ChatCompletionNewParams, []option.RequestOption, error) {
				assertFn(params, opts)
				return params, opts, nil
			}

			_, _, err := m.prepareRequest(
				ctx,
				param.Opt[string]{},
				InputString("input"),
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
}

func TestLiteLLMExtraBodyIsForwarded(t *testing.T) {
	settings := modelsettings.ModelSettings{
		Temperature: param.NewOpt(0.1),
		ExtraBody: map[string]any{
			"cached_content": "some_cache",
			"foo":            123,
		},
	}

	expectedExtras := map[string]any{
		"cached_content": "some_cache",
		"foo":            123,
	}
	assert.Equal(t, expectedExtras, litellmExtraJSON(settings))

	litellmPrepareRequest(t, settings, func(params *openai.ChatCompletionNewParams, opts []option.RequestOption) {
		assert.Equal(t, param.NewOpt(0.1), params.Temperature)
		assert.Len(t, opts, 3)
	})
}

func TestLiteLLMExtraBodyReasoningEffortIsPromoted(t *testing.T) {
	settings := modelsettings.ModelSettings{
		ExtraBody: map[string]any{
			"reasoning_effort": "none",
			"cached_content":   "some_cache",
		},
	}

	expectedExtras := map[string]any{
		"reasoning_effort": "none",
		"cached_content":   "some_cache",
	}
	assert.Equal(t, expectedExtras, litellmExtraJSON(settings))
	assert.Equal(t, expectedExtras, settings.ExtraBody)

	litellmPrepareRequest(t, settings, func(params *openai.ChatCompletionNewParams, opts []option.RequestOption) {
		assert.Equal(t, openai.ReasoningEffort(""), params.ReasoningEffort)
		assert.Len(t, opts, 3)
	})
}

func TestLiteLLMReasoningEffortPrefersModelSettings(t *testing.T) {
	settings := modelsettings.ModelSettings{
		Reasoning: openai.ReasoningParam{Effort: openai.ReasoningEffortLow},
		ExtraBody: map[string]any{
			"reasoning_effort": "high",
		},
	}

	assert.Nil(t, litellmExtraJSON(settings))
	assert.Equal(t, map[string]any{"reasoning_effort": "high"}, settings.ExtraBody)

	litellmPrepareRequest(t, settings, func(params *openai.ChatCompletionNewParams, opts []option.RequestOption) {
		assert.Equal(t, openai.ReasoningEffortLow, params.ReasoningEffort)
		assert.Len(t, opts, 1)
	})
}

func TestLiteLLMExtraBodyReasoningEffortOverridesExtraArgs(t *testing.T) {
	settings := modelsettings.ModelSettings{
		ExtraBody: map[string]any{
			"reasoning_effort": "none",
		},
		ExtraArgs: map[string]any{
			"reasoning_effort": "low",
			"custom_param":     "custom",
		},
	}

	expectedExtras := map[string]any{
		"reasoning_effort": "none",
		"custom_param":     "custom",
	}
	assert.Equal(t, expectedExtras, litellmExtraJSON(settings))
	assert.Equal(t, map[string]any{"reasoning_effort": "low", "custom_param": "custom"}, settings.ExtraArgs)

	litellmPrepareRequest(t, settings, func(params *openai.ChatCompletionNewParams, opts []option.RequestOption) {
		assert.Equal(t, openai.ReasoningEffort(""), params.ReasoningEffort)
		assert.Len(t, opts, 3)
	})
}

func TestLiteLLMReasoningSummaryIsPreserved(t *testing.T) {
	settings := modelsettings.ModelSettings{
		Reasoning: openai.ReasoningParam{
			Effort:  openai.ReasoningEffortMedium,
			Summary: openai.ReasoningSummaryAuto,
		},
	}

	expectedExtras := map[string]any{
		"reasoning_effort": map[string]any{
			"effort":  "medium",
			"summary": "auto",
		},
	}
	assert.Equal(t, expectedExtras, litellmExtraJSON(settings))

	litellmPrepareRequest(t, settings, func(params *openai.ChatCompletionNewParams, opts []option.RequestOption) {
		assert.Equal(t, openai.ReasoningEffort(""), params.ReasoningEffort)
		assert.Len(t, opts, 2)
	})
}
