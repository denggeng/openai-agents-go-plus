// Copyright 2026 The NLP Odyssey Authors
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

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultOpenAIResponsesTransportSettings(t *testing.T) {
	ClearOpenaiSettings()
	assert.Equal(t, OpenAIResponsesTransportHTTP, GetDefaultOpenAIResponsesTransport())
	assert.False(t, GetUseResponsesWebsocketByDefault())

	SetDefaultOpenAIResponsesTransport(OpenAIResponsesTransportWebsocket)
	assert.Equal(t, OpenAIResponsesTransportWebsocket, GetDefaultOpenAIResponsesTransport())
	assert.True(t, GetUseResponsesWebsocketByDefault())

	SetUseResponsesWebsocketByDefault(false)
	assert.Equal(t, OpenAIResponsesTransportHTTP, GetDefaultOpenAIResponsesTransport())
	assert.False(t, GetUseResponsesWebsocketByDefault())
}

func TestOpenAIProviderReturnsCachedResponsesWSModel(t *testing.T) {
	client := NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{})
	provider := NewOpenAIProvider(OpenAIProviderParams{
		OpenaiClient:          &client,
		UseResponses:          param.NewOpt(true),
		UseResponsesWebsocket: param.NewOpt(true),
	})

	model1, err := provider.GetModel("gpt-4.1")
	require.NoError(t, err)
	model2, err := provider.GetModel("gpt-4.1")
	require.NoError(t, err)
	model3, err := provider.GetModel("gpt-4.1-mini")
	require.NoError(t, err)

	ws1, ok := model1.(*OpenAIResponsesWSModel)
	require.True(t, ok)
	ws2, ok := model2.(*OpenAIResponsesWSModel)
	require.True(t, ok)
	ws3, ok := model3.(*OpenAIResponsesWSModel)
	require.True(t, ok)

	assert.Same(t, ws1, ws2)
	assert.NotSame(t, ws1, ws3)

	require.NoError(t, provider.Aclose(context.Background()))
	assert.True(t, ws1.closed.Load())
	assert.True(t, ws3.closed.Load())
}

func TestOpenAIResponsesWSModelGetResponseReturnsExplicitUnsupportedError(t *testing.T) {
	client := NewOpenaiClient(param.NewOpt("http://127.0.0.1:1"), param.NewOpt("FAKE_KEY"))
	provider := NewOpenAIProvider(OpenAIProviderParams{
		OpenaiClient:          &client,
		UseResponses:          param.NewOpt(true),
		UseResponsesWebsocket: param.NewOpt(true),
	})

	model, err := provider.GetModel("gpt-4.1")
	require.NoError(t, err)

	_, err = model.GetResponse(t.Context(), ModelResponseParams{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not implemented")
	assert.Contains(t, err.Error(), "websocket")
}

func TestOpenAIResponsesWSModelStreamResponseReturnsExplicitUnsupportedError(t *testing.T) {
	client := NewOpenaiClient(param.NewOpt("http://127.0.0.1:1"), param.NewOpt("FAKE_KEY"))
	model := NewOpenAIResponsesWSModel("gpt-4.1", client, "")

	err := model.StreamResponse(t.Context(), ModelResponseParams{}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not implemented")
	assert.Contains(t, err.Error(), "websocket")
}
