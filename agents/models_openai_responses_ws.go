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
// Websocket execution is intentionally not implemented yet. The model returns
// explicit errors instead of silently falling back to HTTP transport.
type OpenAIResponsesWSModel struct {
	websocketBaseURL string
	closed           atomic.Bool
}

func NewOpenAIResponsesWSModel(
	_ openai.ChatModel,
	client OpenaiClient,
	websocketBaseURL string,
) *OpenAIResponsesWSModel {
	if websocketBaseURL == "" {
		websocketBaseURL = client.WebsocketBaseURL.Or("")
	}
	return &OpenAIResponsesWSModel{
		websocketBaseURL: websocketBaseURL,
	}
}

func (m *OpenAIResponsesWSModel) GetResponse(
	_ context.Context,
	_ ModelResponseParams,
) (*ModelResponse, error) {
	if m.closed.Load() {
		return nil, UserErrorf("responses websocket model is closed")
	}
	return nil, UserErrorf("responses websocket transport is not implemented; use HTTP transport")
}

func (m *OpenAIResponsesWSModel) StreamResponse(
	_ context.Context,
	_ ModelResponseParams,
	_ ModelStreamResponseCallback,
) error {
	if m.closed.Load() {
		return UserErrorf("responses websocket model is closed")
	}
	return UserErrorf("responses websocket transport is not implemented; use HTTP transport")
}

func (m *OpenAIResponsesWSModel) WebsocketBaseURL() string {
	return m.websocketBaseURL
}

func (m *OpenAIResponsesWSModel) Close() error {
	m.closed.Store(true)
	return nil
}
