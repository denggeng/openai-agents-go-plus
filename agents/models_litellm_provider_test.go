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
	"log/slog"
	"strings"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLiteLLMProvider_GetModelUsesDefaultModelWhenEmpty(t *testing.T) {
	t.Setenv("OPENAI_DEFAULT_MODEL", "GPT-4.1-MINI")

	provider := NewLiteLLMProvider(LiteLLMProviderParams{})
	model, err := provider.GetModel("")
	require.NoError(t, err)

	chatModel, ok := model.(OpenAIChatCompletionsModel)
	require.True(t, ok)
	assert.Equal(t, openai.ChatModel("gpt-4.1-mini"), chatModel.Model)
	assert.Equal(t, "litellm", chatModel.modelImpl)
}

func TestLiteLLMProvider_UsesDefaultsWhenNoParamsProvided(t *testing.T) {
	provider := NewLiteLLMProvider(LiteLLMProviderParams{})

	assert.Equal(t, DefaultLiteLLMBaseURL, provider.provider.params.BaseURL.Or(""))
	assert.Equal(t, defaultLiteLLMAPIKey, provider.provider.params.APIKey.Or(""))
	assert.False(t, provider.provider.useResponses)
}

func TestLiteLLMProvider_UsesEnvWhenParamsMissing(t *testing.T) {
	t.Setenv("LITELLM_BASE_URL", "https://litellm.example.com")
	t.Setenv("LITELLM_API_KEY", "litellm-key")

	provider := NewLiteLLMProvider(LiteLLMProviderParams{})

	assert.Equal(t, "https://litellm.example.com", provider.provider.params.BaseURL.Or(""))
	assert.Equal(t, "litellm-key", provider.provider.params.APIKey.Or(""))
}

func TestLiteLLMProvider_ParamsOverrideEnv(t *testing.T) {
	t.Setenv("LITELLM_BASE_URL", "https://litellm.example.com")
	t.Setenv("LITELLM_API_KEY", "litellm-key")
	t.Setenv("OPENAI_DEFAULT_MODEL", "gpt-4.1")

	provider := NewLiteLLMProvider(LiteLLMProviderParams{
		BaseURL:      param.NewOpt("https://override.example.com"),
		APIKey:       param.NewOpt("override-key"),
		DefaultModel: param.NewOpt("claude-3-5-sonnet-20241022"),
	})

	assert.Equal(t, "https://override.example.com", provider.provider.params.BaseURL.Or(""))
	assert.Equal(t, "override-key", provider.provider.params.APIKey.Or(""))
	assert.Equal(t, "claude-3-5-sonnet-20241022", provider.defaultModel)
}

func TestLiteLLMProvider_FallsBackToOpenAIAPIKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "openai-fallback-key")

	provider := NewLiteLLMProvider(LiteLLMProviderParams{})

	assert.Equal(t, "openai-fallback-key", provider.provider.params.APIKey.Or(""))
}

func TestLiteLLMProvider_GenerationSpanIncludesModelImpl(t *testing.T) {
	provider := NewLiteLLMProvider(LiteLLMProviderParams{
		BaseURL: param.NewOpt("https://litellm.example.com"),
		APIKey:  param.NewOpt("litellm-key"),
	})

	model, err := provider.GetModel("gpt-4.1-mini")
	require.NoError(t, err)

	chatModel, ok := model.(OpenAIChatCompletionsModel)
	require.True(t, ok)

	spanParams, err := chatModel.generationSpanParams(ModelResponseParams{
		ModelSettings: modelsettings.ModelSettings{},
		Tracing:       ModelTracingDisabled,
	})
	require.NoError(t, err)
	assert.Equal(t, "litellm", spanParams.ModelConfig["model_impl"])
	assert.Equal(t, "https://litellm.example.com", spanParams.ModelConfig["base_url"])
}

func TestLiteLLMProvider_LiteLLMSerializerPatchEnvLogsNoopMessage(t *testing.T) {
	t.Setenv("OPENAI_AGENTS_ENABLE_LITELLM_SERIALIZER_PATCH", "true")

	var sb strings.Builder
	prevLogger := Logger()
	SetLogger(slog.New(slog.NewTextHandler(&sb, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() {
		SetLogger(prevLogger)
	})

	_ = NewLiteLLMProvider(LiteLLMProviderParams{})
	logOutput := sb.String()
	assert.Contains(t, logOutput, "LiteLLM serializer patch env is ignored in Go runtime")
	assert.Contains(t, logOutput, "OPENAI_AGENTS_ENABLE_LITELLM_SERIALIZER_PATCH")
}
