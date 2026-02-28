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
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newResponsesTestClient(body string, headers map[string]string) OpenaiClient {
	header := http.Header{"Content-Type": []string{"application/json"}}
	for key, value := range headers {
		header.Set(key, value)
	}
	return OpenaiClient{
		Client: openai.NewClient(
			option.WithMiddleware(func(_ *http.Request, _ option.MiddlewareNext) (*http.Response, error) {
				return &http.Response{
					StatusCode:    http.StatusOK,
					Body:          io.NopCloser(strings.NewReader(body)),
					ContentLength: int64(len(body)),
					Header:        header,
				}, nil
			}),
		),
	}
}

func TestOpenAIResponsesModel_prepareRequest(t *testing.T) {
	t.Run("with ModelSettings.CustomizeResponsesRequest nil", func(t *testing.T) {
		m := NewOpenAIResponsesModel("model-name", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))
		params, opts, err := m.prepareRequest(
			t.Context(),
			param.Opt[string]{},
			InputString("input"),
			modelsettings.ModelSettings{
				CustomizeResponsesRequest: nil,
			},
			nil,
			nil,
			nil,
			"",
			"",
			false,
			responses.ResponsePromptParam{},
		)
		require.NoError(t, err)
		assert.Equal(t, &responses.ResponseNewParams{
			Input: responses.ResponseNewParamsInputUnion{
				OfInputItemList: responses.ResponseInputParam{{
					OfMessage: &responses.EasyInputMessageParam{
						Content: responses.EasyInputMessageContentUnionParam{
							OfString: param.NewOpt("input"),
						},
						Role: responses.EasyInputMessageRoleUser,
						Type: responses.EasyInputMessageTypeMessage,
					},
				}},
			},
			Model: "model-name",
		}, params)
		assert.Len(t, opts, 1)
	})

	t.Run("with ModelSettings.CustomizeResponsesRequest returning values", func(t *testing.T) {
		customParams := &responses.ResponseNewParams{
			Model: "foo",
		}
		customOpts := []option.RequestOption{
			option.WithHeader("bar", "baz"),
		}

		m := NewOpenAIResponsesModel("model-name", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))
		params, opts, err := m.prepareRequest(
			t.Context(),
			param.Opt[string]{},
			InputString("input"),
			modelsettings.ModelSettings{
				CustomizeResponsesRequest: func(ctx context.Context, params *responses.ResponseNewParams, opts []option.RequestOption) (*responses.ResponseNewParams, []option.RequestOption, error) {
					assert.Equal(t, &responses.ResponseNewParams{
						Input: responses.ResponseNewParamsInputUnion{
							OfInputItemList: responses.ResponseInputParam{{
								OfMessage: &responses.EasyInputMessageParam{
									Content: responses.EasyInputMessageContentUnionParam{
										OfString: param.NewOpt("input"),
									},
									Role: responses.EasyInputMessageRoleUser,
									Type: responses.EasyInputMessageTypeMessage,
								},
							}},
						},
						Model: "model-name",
					}, params)
					assert.Len(t, opts, 1)
					return customParams, customOpts, nil
				},
			},
			nil,
			nil,
			nil,
			"",
			"",
			false,
			responses.ResponsePromptParam{},
		)
		require.NoError(t, err)
		assert.Same(t, customParams, params)
		assert.Equal(t, customOpts, opts)
	})

	t.Run("with extra body and args", func(t *testing.T) {
		m := NewOpenAIResponsesModel("model-name", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))
		_, _, err := m.prepareRequest(
			t.Context(),
			param.Opt[string]{},
			InputString("input"),
			modelsettings.ModelSettings{
				ExtraHeaders: map[string]string{"x-header": "v"},
				ExtraQuery:   map[string]string{"q": "1"},
				ExtraBody: map[string]any{
					"cached_content":   "cache",
					"reasoning_effort": "none",
				},
				ExtraArgs: map[string]any{
					"custom_param":     "custom",
					"reasoning_effort": "low",
				},
				CustomizeResponsesRequest: func(ctx context.Context, params *responses.ResponseNewParams, opts []option.RequestOption) (*responses.ResponseNewParams, []option.RequestOption, error) {
					// user-agent + header + query + cached_content + custom_param + reasoning_effort
					assert.Len(t, opts, 6)
					return params, opts, nil
				},
			},
			nil,
			nil,
			nil,
			"",
			"",
			false,
			responses.ResponsePromptParam{},
		)
		require.NoError(t, err)
	})

	t.Run("explicit reasoning effort wins over extra reasoning_effort", func(t *testing.T) {
		m := NewOpenAIResponsesModel("model-name", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))
		_, _, err := m.prepareRequest(
			t.Context(),
			param.Opt[string]{},
			InputString("input"),
			modelsettings.ModelSettings{
				Reasoning: openai.ReasoningParam{Effort: openai.ReasoningEffortLow},
				ExtraBody: map[string]any{
					"reasoning_effort": "none",
					"cached_content":   "cache",
				},
				ExtraArgs: map[string]any{
					"reasoning_effort": "high",
					"custom_param":     "custom",
				},
				CustomizeResponsesRequest: func(ctx context.Context, params *responses.ResponseNewParams, opts []option.RequestOption) (*responses.ResponseNewParams, []option.RequestOption, error) {
					assert.Equal(t, openai.ReasoningEffortLow, params.Reasoning.Effort)
					// user-agent + cached_content + custom_param (reasoning_effort removed from extras)
					assert.Len(t, opts, 3)
					return params, opts, nil
				},
			},
			nil,
			nil,
			nil,
			"",
			"",
			false,
			responses.ResponsePromptParam{},
		)
		require.NoError(t, err)
	})

	t.Run("with ModelSettings.CustomizeResponsesRequest returning error", func(t *testing.T) {
		customError := errors.New("error")
		m := NewOpenAIResponsesModel("model-name", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))
		_, _, err := m.prepareRequest(
			t.Context(),
			param.Opt[string]{},
			InputString("input"),
			modelsettings.ModelSettings{
				CustomizeResponsesRequest: func(ctx context.Context, params *responses.ResponseNewParams, opts []option.RequestOption) (*responses.ResponseNewParams, []option.RequestOption, error) {
					return nil, nil, customError
				},
			},
			nil,
			nil,
			nil,
			"",
			"",
			false,
			responses.ResponsePromptParam{},
		)
		require.ErrorIs(t, err, customError)
	})

	t.Run("with top_logprobs includes logprobs in include list", func(t *testing.T) {
		m := NewOpenAIResponsesModel("model-name", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))
		params, _, err := m.prepareRequest(
			t.Context(),
			param.Opt[string]{},
			InputString("hi"),
			modelsettings.ModelSettings{
				TopLogprobs: param.NewOpt(int64(2)),
			},
			nil,
			nil,
			nil,
			"",
			"",
			false,
			responses.ResponsePromptParam{},
		)
		require.NoError(t, err)
		require.True(t, params.TopLogprobs.Valid())
		assert.Equal(t, int64(2), params.TopLogprobs.Value)
		assert.Contains(t, params.Include, responses.ResponseIncludableMessageOutputTextLogprobs)
	})
}

func TestOpenAIResponsesModelGetResponseCapturesRequestID(t *testing.T) {
	model := NewOpenAIResponsesModel(
		"model-name",
		newResponsesTestClient(`{"id":"resp-1","output":[]}`, map[string]string{
			"X-Request-ID": "req_123",
		}),
	)

	response, err := model.GetResponse(t.Context(), ModelResponseParams{
		Input: InputString("hello"),
	})
	require.NoError(t, err)
	require.NotNil(t, response)
	assert.Equal(t, "req_123", response.RequestID)
	assert.Equal(t, "resp-1", response.ResponseID)
}

func TestOpenAIResponsesModelStreamResponsePropagatesRequestIDInContext(t *testing.T) {
	model := NewOpenAIResponsesModel(
		"model-name",
		newResponsesTestClient(`event: response.completed
data: {"type":"response.completed","response":{"id":"resp-2","output":[]}}

`, map[string]string{
			"x-request-id": "req_stream_456",
		}),
	)

	var callbackRequestID string
	err := model.StreamResponse(t.Context(), ModelResponseParams{
		Input: InputString("hello"),
	}, func(ctx context.Context, event TResponseStreamEvent) error {
		if event.Type == "response.completed" {
			callbackRequestID = modelRequestIDFromContext(ctx)
		}
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, "req_stream_456", callbackRequestID)
}
