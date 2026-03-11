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
	"errors"
	"maps"
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
)

type MultiProviderOpenAIPrefixMode string

const (
	MultiProviderOpenAIPrefixModeAlias   MultiProviderOpenAIPrefixMode = "alias"
	MultiProviderOpenAIPrefixModeModelID MultiProviderOpenAIPrefixMode = "model_id"
)

type MultiProviderUnknownPrefixMode string

const (
	MultiProviderUnknownPrefixModeError   MultiProviderUnknownPrefixMode = "error"
	MultiProviderUnknownPrefixModeModelID MultiProviderUnknownPrefixMode = "model_id"
)

// MultiProvider is a ModelProvider that maps to a Model based on the prefix of the model name.
// By default, the mapping is:
// - "openai/" prefix or no prefix -> OpenAIProvider. e.g. "openai/gpt-4.1", "gpt-4.1"
//
//	You can override or customize this mapping.
type MultiProvider struct {
	// Optional provider map.
	ProviderMap       *MultiProviderMap
	OpenAIProvider    *OpenAIProvider
	fallbackProviders map[string]ModelProvider
	openaiPrefixMode  MultiProviderOpenAIPrefixMode
	unknownPrefixMode MultiProviderUnknownPrefixMode
}

type NewMultiProviderParams struct {
	// Optional MultiProviderMap that maps prefixes to ModelProviders. If not provided,
	// we will use a default mapping. See the documentation for MultiProvider to see the
	// default mapping.
	ProviderMap *MultiProviderMap

	// The API key to use for the OpenAI provider. If not provided, we will use
	// the default API key.
	OpenaiAPIKey param.Opt[string]

	// The base URL to use for the OpenAI provider. If not provided, we will
	// use the default base URL.
	OpenaiBaseURL param.Opt[string]

	// Optional websocket base URL for OpenAI Responses websocket transport.
	OpenaiWebsocketBaseURL param.Opt[string]

	// Optional OpenAI client to use. If not provided, we will create a new
	// OpenAI client using the OpenaiAPIKey and OpenaiBaseURL.
	OpenaiClient *OpenaiClient

	// The organization to use for the OpenAI provider.
	OpenaiOrganization param.Opt[string]

	// The project to use for the OpenAI provider.
	OpenaiProject param.Opt[string]

	// Whether to use the OpenAI responses API.
	OpenaiUseResponses param.Opt[bool]

	// Whether to use websocket transport for OpenAI responses API.
	OpenaiUseResponsesWebsocket param.Opt[bool]

	// Controls how `openai/<model>` strings are interpreted.
	// `alias` strips the prefix before delegating to OpenAIProvider.
	// `model_id` preserves the full model id for OpenAI-compatible endpoints.
	OpenaiPrefixMode MultiProviderOpenAIPrefixMode

	// Controls how unknown `<prefix>/<model>` strings are handled.
	// `error` preserves fail-fast behavior.
	// `model_id` preserves the full model id and delegates to OpenAIProvider.
	UnknownPrefixMode MultiProviderUnknownPrefixMode
}

// NewMultiProvider creates a new OpenAI provider.
func NewMultiProvider(params NewMultiProviderParams) *MultiProvider {
	openaiPrefixMode, err := validateOpenAIPrefixMode(params.OpenaiPrefixMode)
	if err != nil {
		panic(err)
	}
	unknownPrefixMode, err := validateUnknownPrefixMode(params.UnknownPrefixMode)
	if err != nil {
		panic(err)
	}

	return &MultiProvider{
		ProviderMap: params.ProviderMap,
		OpenAIProvider: NewOpenAIProvider(OpenAIProviderParams{
			APIKey:                params.OpenaiAPIKey,
			BaseURL:               params.OpenaiBaseURL,
			OpenaiClient:          params.OpenaiClient,
			Organization:          params.OpenaiOrganization,
			Project:               params.OpenaiProject,
			UseResponses:          params.OpenaiUseResponses,
			WebsocketBaseURL:      params.OpenaiWebsocketBaseURL,
			UseResponsesWebsocket: params.OpenaiUseResponsesWebsocket,
		}),
		fallbackProviders: make(map[string]ModelProvider),
		openaiPrefixMode:  openaiPrefixMode,
		unknownPrefixMode: unknownPrefixMode,
	}
}

func (mp *MultiProvider) getPrefixAndModelName(modelName string) (_, _ string) {
	if modelName == "" {
		return "", ""
	}
	if prefix, name, ok := strings.Cut(modelName, "/"); ok {
		return prefix, name
	}
	return "", modelName
}

func (mp *MultiProvider) createFallbackProvider(prefix string) (ModelProvider, error) {
	switch prefix {
	case "litellm":
		return NewLiteLLMProvider(LiteLLMProviderParams{}), nil
	default:
		// We didn't implement any fallback provider, so here we always return an error.
		return nil, UserErrorf("unknown prefix %q", prefix)
	}
}

func validateOpenAIPrefixMode(mode MultiProviderOpenAIPrefixMode) (MultiProviderOpenAIPrefixMode, error) {
	if mode == "" {
		return MultiProviderOpenAIPrefixModeAlias, nil
	}
	switch mode {
	case MultiProviderOpenAIPrefixModeAlias, MultiProviderOpenAIPrefixModeModelID:
		return mode, nil
	default:
		return "", UserErrorf(
			"MultiProvider openai_prefix_mode must be one of: 'alias', 'model_id'.",
		)
	}
}

func validateUnknownPrefixMode(mode MultiProviderUnknownPrefixMode) (MultiProviderUnknownPrefixMode, error) {
	if mode == "" {
		return MultiProviderUnknownPrefixModeError, nil
	}
	switch mode {
	case MultiProviderUnknownPrefixModeError, MultiProviderUnknownPrefixModeModelID:
		return mode, nil
	default:
		return "", UserErrorf(
			"MultiProvider unknown_prefix_mode must be one of: 'error', 'model_id'.",
		)
	}
}

func (mp *MultiProvider) getFallbackProvider(prefix string) (ModelProvider, error) {
	if prefix == "" || prefix == "openai" {
		return mp.OpenAIProvider, nil
	}
	if fp, ok := mp.fallbackProviders[prefix]; ok {
		return fp, nil
	}

	fp, err := mp.createFallbackProvider(prefix)
	if err != nil {
		return nil, err
	}
	mp.fallbackProviders[prefix] = fp
	return fp, nil
}

func (mp *MultiProvider) resolvePrefixedModel(
	originalModelName string,
	prefix string,
	strippedModelName string,
) (ModelProvider, string, error) {
	// Explicit provider_map entries always win and receive the stripped model name.
	if mp.ProviderMap != nil {
		if provider, ok := mp.ProviderMap.GetProvider(prefix); ok {
			return provider, strippedModelName, nil
		}
	}

	if prefix == "litellm" {
		provider, err := mp.getFallbackProvider(prefix)
		return provider, strippedModelName, err
	}

	if prefix == "openai" {
		if mp.openaiPrefixMode == MultiProviderOpenAIPrefixModeAlias {
			return mp.OpenAIProvider, strippedModelName, nil
		}
		return mp.OpenAIProvider, originalModelName, nil
	}

	if mp.unknownPrefixMode == MultiProviderUnknownPrefixModeModelID {
		return mp.OpenAIProvider, originalModelName, nil
	}

	return nil, "", UserErrorf("Unknown prefix: %s", prefix)
}

// GetModel returns a Model based on the model name. The model name can have a prefix, ending with
// a "/", which will be used to look up the ModelProvider. If there is no prefix, we will use
// the OpenAI provider.
func (mp *MultiProvider) GetModel(modelName string) (Model, error) {
	prefix, name := mp.getPrefixAndModelName(modelName)

	if prefix == "" {
		return mp.OpenAIProvider.GetModel(name)
	}

	provider, resolvedModelName, err := mp.resolvePrefixedModel(modelName, prefix, name)
	if err != nil {
		return nil, err
	}
	return provider.GetModel(resolvedModelName)
}

// Aclose releases resources owned by underlying providers that expose lifecycle hooks.
func (mp *MultiProvider) Aclose(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	var closeErr error

	if mp.OpenAIProvider != nil {
		closeErr = errors.Join(closeErr, mp.OpenAIProvider.Aclose(ctx))
	}

	for _, provider := range mp.fallbackProviders {
		closer, ok := provider.(ModelProviderCloser)
		if !ok || closer == nil {
			continue
		}
		if err := closer.Aclose(ctx); err != nil {
			closeErr = errors.Join(closeErr, err)
		}
	}
	if mp.ProviderMap != nil {
		for _, provider := range mp.ProviderMap.GetMapping() {
			closer, ok := provider.(ModelProviderCloser)
			if !ok || closer == nil {
				continue
			}
			if err := closer.Aclose(ctx); err != nil {
				closeErr = errors.Join(closeErr, err)
			}
		}
	}

	return closeErr
}

// MultiProviderMap is a map of model name prefixes to ModelProvider objects.
type MultiProviderMap struct {
	m map[string]ModelProvider
}

func NewMultiProviderMap() *MultiProviderMap {
	return &MultiProviderMap{
		m: make(map[string]ModelProvider),
	}
}

// HasPrefix returns true if the given prefix is in the mapping.
func (m *MultiProviderMap) HasPrefix(prefix string) bool {
	_, ok := m.m[prefix]
	return ok
}

// GetMapping returns a copy of the current prefix -> ModelProvider mapping.
func (m *MultiProviderMap) GetMapping() map[string]ModelProvider {
	return maps.Clone(m.m)
}

// SetMapping overwrites the current mapping with a new one.
func (m *MultiProviderMap) SetMapping(mapping map[string]ModelProvider) {
	m.m = mapping
}

// GetProvider returns the ModelProvider for the given prefix.
func (m *MultiProviderMap) GetProvider(prefix string) (ModelProvider, bool) {
	v, ok := m.m[prefix]
	return v, ok
}

// AddProvider adds a new prefix -> ModelProvider mapping.
func (m *MultiProviderMap) AddProvider(prefix string, provider ModelProvider) {
	m.m[prefix] = provider
}

// RemoveProvider removes the mapping for the given prefix.
func (m *MultiProviderMap) RemoveProvider(prefix string) {
	delete(m.m, prefix)
}
