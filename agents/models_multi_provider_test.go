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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
