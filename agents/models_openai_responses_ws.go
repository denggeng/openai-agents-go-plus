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
	"sync/atomic"

	"github.com/openai/openai-go/v3"
)

// OpenAIResponsesWSModel is the websocket-transport wrapper for the Responses API.
//
// The current implementation preserves provider-level lifecycle and caching semantics
// while delegating request execution to the shared Responses model implementation.
type OpenAIResponsesWSModel struct {
	delegate         OpenAIResponsesModel
	websocketBaseURL string
	closed           atomic.Bool
}

func NewOpenAIResponsesWSModel(
	model openai.ChatModel,
	client OpenaiClient,
	websocketBaseURL string,
) *OpenAIResponsesWSModel {
	if websocketBaseURL == "" {
		websocketBaseURL = client.WebsocketBaseURL.Or("")
	}
	return &OpenAIResponsesWSModel{
		delegate:         NewOpenAIResponsesModel(model, client),
		websocketBaseURL: websocketBaseURL,
	}
}

func (m *OpenAIResponsesWSModel) GetResponse(
	ctx context.Context,
	params ModelResponseParams,
) (*ModelResponse, error) {
	if m.closed.Load() {
		return nil, UserErrorf("responses websocket model is closed")
	}
	return m.delegate.GetResponse(ctx, params)
}

func (m *OpenAIResponsesWSModel) StreamResponse(
	ctx context.Context,
	params ModelResponseParams,
	yield ModelStreamResponseCallback,
) error {
	if m.closed.Load() {
		return UserErrorf("responses websocket model is closed")
	}
	return m.delegate.StreamResponse(ctx, params, yield)
}

func (m *OpenAIResponsesWSModel) WebsocketBaseURL() string {
	return m.websocketBaseURL
}

func (m *OpenAIResponsesWSModel) Close() error {
	m.closed.Store(true)
	return nil
}
