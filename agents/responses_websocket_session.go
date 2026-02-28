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

	"github.com/openai/openai-go/v3/packages/param"
)

// ResponsesWebSocketSessionParams configures a shared websocket-capable run session.
type ResponsesWebSocketSessionParams struct {
	APIKey           param.Opt[string]
	BaseURL          param.Opt[string]
	WebsocketBaseURL param.Opt[string]
	Organization     param.Opt[string]
	Project          param.Opt[string]
	OpenaiClient     *OpenaiClient
}

// ResponsesWebSocketSession pins runs to a shared websocket-enabled provider.
type ResponsesWebSocketSession struct {
	Provider  *OpenAIProvider
	RunConfig RunConfig
}

func NewResponsesWebSocketSession(params ResponsesWebSocketSessionParams) *ResponsesWebSocketSession {
	provider := NewOpenAIProvider(OpenAIProviderParams{
		APIKey:                params.APIKey,
		BaseURL:               params.BaseURL,
		WebsocketBaseURL:      params.WebsocketBaseURL,
		OpenaiClient:          params.OpenaiClient,
		Organization:          params.Organization,
		Project:               params.Project,
		UseResponses:          param.NewOpt(true),
		UseResponsesWebsocket: param.NewOpt(true),
	})
	modelProvider := NewMultiProvider(NewMultiProviderParams{})
	modelProvider.OpenAIProvider = provider

	return &ResponsesWebSocketSession{
		Provider: provider,
		RunConfig: RunConfig{
			ModelProvider: modelProvider,
		},
	}
}

func (s *ResponsesWebSocketSession) Runner() Runner {
	return Runner{Config: s.RunConfig}
}

func (s *ResponsesWebSocketSession) Run(
	ctx context.Context,
	startingAgent *Agent,
	input string,
) (*RunResult, error) {
	return s.Runner().Run(ctx, startingAgent, input)
}

func (s *ResponsesWebSocketSession) RunInputs(
	ctx context.Context,
	startingAgent *Agent,
	input []TResponseInputItem,
) (*RunResult, error) {
	return s.Runner().RunInputs(ctx, startingAgent, input)
}

func (s *ResponsesWebSocketSession) RunFromState(
	ctx context.Context,
	startingAgent *Agent,
	state RunState,
) (*RunResult, error) {
	return s.Runner().RunFromState(ctx, startingAgent, state)
}

func (s *ResponsesWebSocketSession) RunStreamed(
	ctx context.Context,
	startingAgent *Agent,
	input string,
) (*RunResultStreaming, error) {
	return s.Runner().RunStreamed(ctx, startingAgent, input)
}

func (s *ResponsesWebSocketSession) RunInputsStreamed(
	ctx context.Context,
	startingAgent *Agent,
	input []TResponseInputItem,
) (*RunResultStreaming, error) {
	return s.Runner().RunInputsStreamed(ctx, startingAgent, input)
}

func (s *ResponsesWebSocketSession) RunFromStateStreamed(
	ctx context.Context,
	startingAgent *Agent,
	state RunState,
) (*RunResultStreaming, error) {
	return s.Runner().RunFromStateStreamed(ctx, startingAgent, state)
}

func (s *ResponsesWebSocketSession) Close(ctx context.Context) error {
	if s == nil || s.Provider == nil {
		return nil
	}
	return s.Provider.Aclose(ctx)
}
