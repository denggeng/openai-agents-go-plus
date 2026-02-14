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
	"os"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
)

const (
	DefaultLiteLLMBaseURL     = "http://localhost:4000"
	defaultLiteLLMAPIKey      = "dummy-key"
	litellmSerializerPatchEnv = "OPENAI_AGENTS_ENABLE_LITELLM_SERIALIZER_PATCH"
)

type LiteLLMProviderParams struct {
	// API key used to call LiteLLM's OpenAI-compatible endpoint.
	// When omitted, this resolves from LITELLM_API_KEY, then OPENAI_API_KEY,
	// and finally falls back to "dummy-key".
	APIKey param.Opt[string]

	// Base URL for LiteLLM's OpenAI-compatible endpoint.
	// When omitted, this resolves from LITELLM_BASE_URL and then defaults to
	// "http://localhost:4000".
	BaseURL param.Opt[string]

	// Optional OpenAI client override. When set, APIKey/BaseURL are ignored.
	OpenaiClient *OpenaiClient

	// Default model name used when GetModel receives an empty model name.
	// When omitted, this resolves from OPENAI_DEFAULT_MODEL and then defaults
	// to "gpt-4.1".
	DefaultModel param.Opt[string]
}

type LiteLLMProvider struct {
	provider     *OpenAIProvider
	defaultModel string
}

func NewLiteLLMProvider(params LiteLLMProviderParams) *LiteLLMProvider {
	maybeLogLiteLLMSerializerPatchNoop()

	providerParams := OpenAIProviderParams{
		OpenaiClient: params.OpenaiClient,
		UseResponses: param.NewOpt(false),
	}
	if params.OpenaiClient == nil {
		providerParams.BaseURL = resolveLiteLLMBaseURL(params.BaseURL)
		providerParams.APIKey = resolveLiteLLMAPIKey(params.APIKey)
	}

	return &LiteLLMProvider{
		provider:     NewOpenAIProvider(providerParams),
		defaultModel: resolveLiteLLMDefaultModel(params.DefaultModel),
	}
}

func maybeLogLiteLLMSerializerPatchNoop() {
	raw := strings.TrimSpace(os.Getenv(litellmSerializerPatchEnv))
	if raw == "" {
		return
	}

	switch strings.ToLower(raw) {
	case "1", "true":
		Logger().Debug(
			"LiteLLM serializer patch env is ignored in Go runtime",
			slog.String("env_var", litellmSerializerPatchEnv),
		)
	}
}

func (provider *LiteLLMProvider) GetModel(modelName string) (Model, error) {
	if strings.TrimSpace(modelName) == "" {
		modelName = provider.defaultModel
	}

	client := provider.provider.getClient()
	return NewOpenAIChatCompletionsModelWithImpl(openai.ChatModel(modelName), client, "litellm"), nil
}

func resolveLiteLLMBaseURL(explicitBaseURL param.Opt[string]) param.Opt[string] {
	if explicitBaseURL.Valid() {
		return explicitBaseURL
	}
	if envBaseURL := strings.TrimSpace(os.Getenv("LITELLM_BASE_URL")); envBaseURL != "" {
		return param.NewOpt(envBaseURL)
	}
	return param.NewOpt(DefaultLiteLLMBaseURL)
}

func resolveLiteLLMAPIKey(explicitAPIKey param.Opt[string]) param.Opt[string] {
	if explicitAPIKey.Valid() {
		return explicitAPIKey
	}
	if envAPIKey := strings.TrimSpace(os.Getenv("LITELLM_API_KEY")); envAPIKey != "" {
		return param.NewOpt(envAPIKey)
	}
	if envOpenAIAPIKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")); envOpenAIAPIKey != "" {
		return param.NewOpt(envOpenAIAPIKey)
	}
	return param.NewOpt(defaultLiteLLMAPIKey)
}

func resolveLiteLLMDefaultModel(explicitDefaultModel param.Opt[string]) string {
	if explicitDefaultModel.Valid() {
		modelName := strings.ToLower(strings.TrimSpace(explicitDefaultModel.Value))
		if modelName != "" {
			return modelName
		}
	}
	if envDefaultModel := strings.TrimSpace(os.Getenv("OPENAI_DEFAULT_MODEL")); envDefaultModel != "" {
		return strings.ToLower(envDefaultModel)
	}
	return "gpt-4.1"
}
