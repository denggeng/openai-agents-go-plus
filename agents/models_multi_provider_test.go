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
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type recordingProvider struct {
	model     Model
	requested []string
}

func (p *recordingProvider) GetModel(modelName string) (Model, error) {
	p.requested = append(p.requested, modelName)
	if p.model != nil {
		return p.model, nil
	}
	return OpenAIResponsesModel{Model: openai.ChatModel(modelName)}, nil
}

func TestMultiProvider_GetModel_UsesLiteLLMFallbackProvider(t *testing.T) {
	t.Setenv("LITELLM_BASE_URL", "https://litellm.example.com")
	t.Setenv("LITELLM_API_KEY", "litellm-key")

	mp := NewMultiProvider(NewMultiProviderParams{})
	model, err := mp.GetModel("litellm/anthropic/claude-3-5-sonnet-20241022")
	require.NoError(t, err)

	chatModel, ok := model.(OpenAIChatCompletionsModel)
	require.True(t, ok)
	assert.Equal(t, openai.ChatModel("anthropic/claude-3-5-sonnet-20241022"), chatModel.Model)
	assert.Equal(t, "https://litellm.example.com", chatModel.client.BaseURL.Or(""))
}

func TestMultiProvider_GetModel_UsesLiteLLMFallbackDefaultModel(t *testing.T) {
	t.Setenv("OPENAI_DEFAULT_MODEL", "GPT-4.1")

	mp := NewMultiProvider(NewMultiProviderParams{})
	model, err := mp.GetModel("litellm/")
	require.NoError(t, err)

	chatModel, ok := model.(OpenAIChatCompletionsModel)
	require.True(t, ok)
	assert.Equal(t, openai.ChatModel("gpt-4.1"), chatModel.Model)
}

func TestMultiProvider_GetModel_UsesResponsesWebsocketWithoutPrefix(t *testing.T) {
	mp := NewMultiProvider(NewMultiProviderParams{
		OpenaiUseResponses:          param.NewOpt(true),
		OpenaiUseResponsesWebsocket: param.NewOpt(true),
	})

	model, err := mp.GetModel("gpt-4o")
	require.NoError(t, err)
	assert.IsType(t, &OpenAIResponsesWSModel{}, model)
}

func TestMultiProvider_GetModel_UsesResponsesWebsocketWithOpenAIPrefix(t *testing.T) {
	mp := NewMultiProvider(NewMultiProviderParams{
		OpenaiUseResponses:          param.NewOpt(true),
		OpenaiUseResponsesWebsocket: param.NewOpt(true),
	})

	model, err := mp.GetModel("openai/gpt-4o")
	require.NoError(t, err)
	assert.IsType(t, &OpenAIResponsesWSModel{}, model)
}

func TestMultiProvider_PassesWebsocketBaseURLToOpenAIProvider(t *testing.T) {
	mp := NewMultiProvider(NewMultiProviderParams{
		OpenaiWebsocketBaseURL: param.NewOpt("wss://proxy.example.test/v1"),
	})

	require.True(t, mp.OpenAIProvider.params.WebsocketBaseURL.Valid())
	assert.Equal(t, "wss://proxy.example.test/v1", mp.OpenAIProvider.params.WebsocketBaseURL.Value)
}

func TestMultiProvider_OpenAIPrefixDefaultsToAliasMode(t *testing.T) {
	mp := NewMultiProvider(NewMultiProviderParams{
		OpenaiUseResponses: param.NewOpt(true),
	})

	model, err := mp.GetModel("openai/gpt-4o")
	require.NoError(t, err)

	responseModel, ok := model.(OpenAIResponsesModel)
	require.True(t, ok)
	assert.Equal(t, openai.ChatModel("gpt-4o"), responseModel.Model)
}

func TestMultiProvider_OpenAIPrefixCanBePreservedAsLiteralModelID(t *testing.T) {
	mp := NewMultiProvider(NewMultiProviderParams{
		OpenaiUseResponses: param.NewOpt(true),
		OpenaiPrefixMode:   MultiProviderOpenAIPrefixModeModelID,
	})

	model, err := mp.GetModel("openai/gpt-4o")
	require.NoError(t, err)

	responseModel, ok := model.(OpenAIResponsesModel)
	require.True(t, ok)
	assert.Equal(t, openai.ChatModel("openai/gpt-4o"), responseModel.Model)
}

func TestMultiProvider_UnknownPrefixDefaultsToError(t *testing.T) {
	mp := NewMultiProvider(NewMultiProviderParams{})

	_, err := mp.GetModel("openrouter/openai/gpt-4o")
	require.Error(t, err)
	assert.EqualError(t, err, "Unknown prefix: openrouter")

	var userErr UserError
	assert.ErrorAs(t, err, &userErr)
}

func TestMultiProvider_UnknownPrefixCanBePreservedForOpenAICompatibleModelIDs(t *testing.T) {
	mp := NewMultiProvider(NewMultiProviderParams{
		OpenaiUseResponses: param.NewOpt(true),
		UnknownPrefixMode:  MultiProviderUnknownPrefixModeModelID,
	})

	model, err := mp.GetModel("openrouter/openai/gpt-4o")
	require.NoError(t, err)

	responseModel, ok := model.(OpenAIResponsesModel)
	require.True(t, ok)
	assert.Equal(t, openai.ChatModel("openrouter/openai/gpt-4o"), responseModel.Model)
}

func TestMultiProvider_ProviderMapEntriesOverrideOpenAIPrefixMode(t *testing.T) {
	customProvider := &recordingProvider{}
	providerMap := NewMultiProviderMap()
	providerMap.AddProvider("openai", customProvider)

	mp := NewMultiProvider(NewMultiProviderParams{
		ProviderMap:      providerMap,
		OpenaiPrefixMode: MultiProviderOpenAIPrefixModeModelID,
	})

	_, err := mp.GetModel("openai/gpt-4o")
	require.NoError(t, err)
	require.Len(t, customProvider.requested, 1)
	assert.Equal(t, "gpt-4o", customProvider.requested[0])
}

func TestMultiProvider_InvalidPrefixModesPanic(t *testing.T) {
	assert.PanicsWithError(
		t,
		"MultiProvider openai_prefix_mode must be one of: 'alias', 'model_id'.",
		func() {
			NewMultiProvider(NewMultiProviderParams{
				OpenaiPrefixMode: MultiProviderOpenAIPrefixMode("invalid"),
			})
		},
	)

	assert.PanicsWithError(
		t,
		"MultiProvider unknown_prefix_mode must be one of: 'error', 'model_id'.",
		func() {
			NewMultiProvider(NewMultiProviderParams{
				UnknownPrefixMode: MultiProviderUnknownPrefixMode("invalid"),
			})
		},
	)
}
