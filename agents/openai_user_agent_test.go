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

package agents_test

import (
	"bytes"
	"io"
	"net/http"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func makeOpenaiClientWithBody(t *testing.T, reqHeader *http.Header, body string) agents.OpenaiClient {
	t.Helper()

	return agents.OpenaiClient{
		Client: openai.NewClient(
			option.WithMiddleware(func(req *http.Request, _ option.MiddlewareNext) (*http.Response, error) {
				*reqHeader = req.Header.Clone()
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(bytes.NewBufferString(body)),
					Header: http.Header{
						"Content-Type": []string{"application/json"},
					},
				}, nil
			}),
		),
	}
}

func TestUserAgentHeaderChatCompletions(t *testing.T) {
	for _, override := range []string{"", "test_user_agent"} {
		t.Run(override, func(t *testing.T) {
			var reqHeader http.Header
			body := `{"id":"resp","object":"chat.completion","created":0,"model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
			dummyClient := makeOpenaiClientWithBody(t, &reqHeader, body)

			model := agents.NewOpenAIChatCompletionsModel("gpt-4", dummyClient)

			var token agents.HeadersOverrideToken
			if override != "" {
				token = agents.HeadersOverride.Set(map[string]string{"User-Agent": override})
				defer agents.HeadersOverride.Reset(token)
			}

			_, err := model.GetResponse(t.Context(), agents.ModelResponseParams{
				SystemInstructions: param.Opt[string]{},
				Input:              agents.InputString("hi"),
				ModelSettings:      modelsettings.ModelSettings{},
				Tools:              nil,
				OutputType:         nil,
				Handoffs:           nil,
				Tracing:            agents.ModelTracingDisabled,
				PreviousResponseID: "",
				Prompt:             responses.ResponsePromptParam{},
			})
			require.NoError(t, err)

			expectedUA := agents.DefaultUserAgent()
			if override != "" {
				expectedUA = override
			}
			assert.Equal(t, expectedUA, reqHeader.Get("User-Agent"))
		})
	}
}

func TestUserAgentHeaderResponses(t *testing.T) {
	for _, override := range []string{"", "test_user_agent"} {
		t.Run(override, func(t *testing.T) {
			var reqHeader http.Header
			body := `{"id":"resp","object":"response","created_at":0,"model":"gpt-4","status":"completed","output":[]}`
			dummyClient := makeOpenaiClientWithBody(t, &reqHeader, body)

			model := agents.NewOpenAIResponsesModel("gpt-4", dummyClient)

			var token agents.HeadersOverrideToken
			if override != "" {
				token = agents.ResponsesHeadersOverride.Set(map[string]string{"User-Agent": override})
				defer agents.ResponsesHeadersOverride.Reset(token)
			}

			_, err := model.GetResponse(t.Context(), agents.ModelResponseParams{
				SystemInstructions: param.Opt[string]{},
				Input:              agents.InputString("hi"),
				ModelSettings:      modelsettings.ModelSettings{},
				Tools:              nil,
				OutputType:         nil,
				Handoffs:           nil,
				Tracing:            agents.ModelTracingDisabled,
				PreviousResponseID: "",
				Prompt:             responses.ResponsePromptParam{},
			})
			require.NoError(t, err)

			expectedUA := agents.DefaultUserAgent()
			if override != "" {
				expectedUA = override
			}
			assert.Equal(t, expectedUA, reqHeader.Get("User-Agent"))
		})
	}
}
