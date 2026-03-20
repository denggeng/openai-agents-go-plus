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
	"errors"
	"net/http"
	"net/url"
	"testing"
	"time"

	"github.com/denggeng/openai-agents-go-plus/retry"
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetResponseWithRetryRetriesAndAugmentsUsage(t *testing.T) {
	originalSleep := sleepForModelRetry
	defer func() { sleepForModelRetry = originalSleep }()

	var sleeps []float64
	sleepForModelRetry = func(_ context.Context, delay time.Duration) error {
		sleeps = append(sleeps, delay.Seconds())
		return nil
	}

	calls := 0
	rewinds := 0
	result, err := getResponseWithRetry(
		t.Context(),
		func(context.Context) (*ModelResponse, error) {
			calls++
			if calls == 1 {
				return nil, transientNetworkError()
			}
			return &ModelResponse{
				Output:     nil,
				Usage:      &usage.Usage{Requests: 1},
				ResponseID: "resp-retry-usage",
			}, nil
		},
		func() error {
			rewinds++
			return nil
		},
		&retry.ModelRetrySettings{
			MaxRetries: retry.Int(1),
			Backoff: &retry.ModelRetryBackoffSettings{
				InitialDelay: retry.Float64(0.5),
				Jitter:       retry.Bool(false),
			},
			Policy: retry.RetryPolicies.NetworkError(),
		},
		nil,
		"",
		"",
	)
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, 2, calls)
	assert.Equal(t, 1, rewinds)
	assert.Equal(t, []float64{0.5}, sleeps)
	assert.Equal(t, uint64(2), result.Usage.Requests)
}

func TestGetResponseWithRetryKeepsProviderRetriesOnFirstAttempt(t *testing.T) {
	originalSleep := sleepForModelRetry
	defer func() { sleepForModelRetry = originalSleep }()
	sleepForModelRetry = func(context.Context, time.Duration) error { return nil }

	var flags []bool
	calls := 0
	_, err := getResponseWithRetry(
		t.Context(),
		func(ctx context.Context) (*ModelResponse, error) {
			flags = append(flags, providerManagedRetriesDisabledFromContext(ctx))
			calls++
			if calls == 1 {
				return nil, transientNetworkError()
			}
			return &ModelResponse{Usage: &usage.Usage{Requests: 1}, ResponseID: "resp-provider-flags"}, nil
		},
		func() error { return nil },
		&retry.ModelRetrySettings{
			MaxRetries: retry.Int(1),
			Policy:     retry.RetryPolicies.NetworkError(),
		},
		nil,
		"",
		"",
	)
	require.NoError(t, err)
	assert.Equal(t, []bool{false, true}, flags)
}

func TestGetResponseWithRetryDisablesProviderRetriesOnFirstStatefulProviderHint(t *testing.T) {
	originalSleep := sleepForModelRetry
	defer func() { sleepForModelRetry = originalSleep }()
	sleepForModelRetry = func(context.Context, time.Duration) error { return nil }

	var flags []bool
	calls := 0
	_, err := getResponseWithRetry(
		t.Context(),
		func(ctx context.Context) (*ModelResponse, error) {
			flags = append(flags, providerManagedRetriesDisabledFromContext(ctx))
			calls++
			if calls == 1 {
				return nil, transientNetworkError()
			}
			return &ModelResponse{Usage: &usage.Usage{Requests: 1}, ResponseID: "resp-stateful-provider-hint"}, nil
		},
		func() error { return nil },
		&retry.ModelRetrySettings{
			MaxRetries: retry.Int(1),
			Policy:     retry.RetryPolicies.ProviderSuggested(),
		},
		func(retry.ModelRetryAdviceRequest) *retry.ModelRetryAdvice {
			return &retry.ModelRetryAdvice{
				Suggested:    retry.Bool(true),
				ReplaySafety: "safe",
			}
		},
		"resp-prev",
		"",
	)
	require.NoError(t, err)
	assert.Equal(t, []bool{true, true}, flags)
}

func TestGetResponseWithRetryDisablesWebsocketPreEventRetriesWhenRunnerManaged(t *testing.T) {
	originalSleep := sleepForModelRetry
	defer func() { sleepForModelRetry = originalSleep }()
	sleepForModelRetry = func(context.Context, time.Duration) error { return nil }

	var flags []bool
	calls := 0
	_, err := getResponseWithRetry(
		t.Context(),
		func(ctx context.Context) (*ModelResponse, error) {
			flags = append(flags, websocketPreEventRetriesDisabledFromContext(ctx))
			calls++
			if calls == 1 {
				return nil, transientNetworkError()
			}
			return &ModelResponse{Usage: &usage.Usage{Requests: 1}, ResponseID: "resp-disable-ws-pre-event"}, nil
		},
		func() error { return nil },
		&retry.ModelRetrySettings{
			MaxRetries: retry.Int(1),
			Policy:     retry.RetryPolicies.NetworkError(),
		},
		nil,
		"",
		"",
	)
	require.NoError(t, err)
	assert.Equal(t, []bool{true, true}, flags)
}

func TestGetResponseWithRetryHonorsExplicitNoneRetryAfterOverride(t *testing.T) {
	calls := 0
	_, err := getResponseWithRetry(
		t.Context(),
		func(context.Context) (*ModelResponse, error) {
			calls++
			return nil, openAIStatusError(t, 429, "rate_limit", http.Header{
				"Retry-After-Ms": []string{"1250"},
			})
		},
		func() error {
			t.Fatalf("explicit retry_after=nil override should suppress retries")
			return nil
		},
		&retry.ModelRetrySettings{
			MaxRetries: retry.Int(1),
			Backoff: &retry.ModelRetryBackoffSettings{
				Jitter: retry.Bool(false),
			},
			Policy: retry.RetryPolicies.RetryAfter(),
		},
		func(retry.ModelRetryAdviceRequest) *retry.ModelRetryAdvice {
			return &retry.ModelRetryAdvice{
				Normalized: &retry.ModelRetryNormalizedError{
					RetryAfterSet: true,
				},
			}
		},
		"",
		"",
	)
	require.Error(t, err)

	var statusErr *openai.Error
	require.ErrorAs(t, err, &statusErr)
	assert.Equal(t, 429, statusErr.StatusCode)
	assert.Equal(t, 1, calls)
}

func TestStreamResponseWithRetryPropagatesFailedAttemptCount(t *testing.T) {
	originalSleep := sleepForModelRetry
	defer func() { sleepForModelRetry = originalSleep }()
	sleepForModelRetry = func(context.Context, time.Duration) error { return nil }

	attempts := 0
	var failedAttemptCounts []int
	err := streamResponseWithRetry(
		t.Context(),
		func(ctx context.Context, yield ModelStreamResponseCallback) error {
			attempts++
			if attempts == 1 {
				return transientNetworkError()
			}
			return yield(ctx, TResponseStreamEvent{
				Type: "response.completed",
				Response: responses.Response{
					ID: "resp-stream-retry",
					Usage: responses.ResponseUsage{
						InputTokens:  5,
						OutputTokens: 2,
						TotalTokens:  7,
					},
				},
			})
		},
		func() error { return nil },
		&retry.ModelRetrySettings{
			MaxRetries: retry.Int(1),
			Policy:     retry.RetryPolicies.NetworkError(),
		},
		nil,
		"",
		"",
		func(ctx context.Context, event TResponseStreamEvent) error {
			if event.Type != "response.completed" {
				return nil
			}
			failedAttemptCounts = append(failedAttemptCounts, modelRetryAttemptsFromContext(ctx))
			u := usage.NewUsage()
			if event.Response.Usage.TotalTokens > 0 {
				u.InputTokens = uint64(event.Response.Usage.InputTokens)
				u.OutputTokens = uint64(event.Response.Usage.OutputTokens)
				u.TotalTokens = uint64(event.Response.Usage.TotalTokens)
			}
			u = applyRetryAttemptUsage(u, modelRetryAttemptsFromContext(ctx))
			assert.Equal(t, uint64(2), u.Requests)
			assert.Equal(t, uint64(7), u.TotalTokens)
			return nil
		},
	)
	require.NoError(t, err)
	assert.Equal(t, []int{1}, failedAttemptCounts)
}

func TestStreamResponseWithRetryDoesNotLeakRetryDisableFlagsToConsumer(t *testing.T) {
	originalSleep := sleepForModelRetry
	defer func() { sleepForModelRetry = originalSleep }()
	sleepForModelRetry = func(context.Context, time.Duration) error { return nil }

	attempts := 0
	var providerRetryFlags []bool
	var consumerProviderRetryFlags []bool
	var consumerWebsocketRetryFlags []bool
	var consumerRequestIDs []string

	err := streamResponseWithRetry(
		t.Context(),
		func(ctx context.Context, yield ModelStreamResponseCallback) error {
			providerRetryFlags = append(providerRetryFlags, providerManagedRetriesDisabledFromContext(ctx))
			attempts++
			if attempts == 1 {
				return transientNetworkError()
			}

			eventCtx := contextWithModelRequestID(ctx, "req-stream-123")
			return yield(eventCtx, TResponseStreamEvent{Type: "response.created"})
		},
		func() error { return nil },
		&retry.ModelRetrySettings{
			MaxRetries: retry.Int(1),
			Policy:     retry.RetryPolicies.NetworkError(),
		},
		nil,
		"",
		"",
		func(ctx context.Context, _ TResponseStreamEvent) error {
			consumerProviderRetryFlags = append(
				consumerProviderRetryFlags,
				providerManagedRetriesDisabledFromContext(ctx),
			)
			consumerWebsocketRetryFlags = append(
				consumerWebsocketRetryFlags,
				websocketPreEventRetriesDisabledFromContext(ctx),
			)
			consumerRequestIDs = append(consumerRequestIDs, modelRequestIDFromContext(ctx))
			return nil
		},
	)
	require.NoError(t, err)
	assert.Equal(t, []bool{false, true}, providerRetryFlags)
	assert.Equal(t, []bool{false}, consumerProviderRetryFlags)
	assert.Equal(t, []bool{false}, consumerWebsocketRetryFlags)
	assert.Equal(t, []string{"req-stream-123"}, consumerRequestIDs)
}

func TestGetOpenAIRetryAdviceKeepsStatefulTransportFailuresAmbiguous(t *testing.T) {
	advice := getOpenAIRetryAdvice(retry.ModelRetryAdviceRequest{
		Error:              transientNetworkError(),
		Attempt:            1,
		Stream:             false,
		PreviousResponseID: "resp-prev",
	})
	require.NotNil(t, advice)
	require.NotNil(t, advice.Suggested)
	assert.True(t, *advice.Suggested)
	assert.Empty(t, advice.ReplaySafety)
}

func TestGetOpenAIRetryAdviceMarksStatefulHTTPFailuresReplaySafe(t *testing.T) {
	advice := getOpenAIRetryAdvice(retry.ModelRetryAdviceRequest{
		Error:              openAIStatusError(t, 429, "rate_limit", http.Header{"Retry-After-Ms": []string{"1250"}}),
		Attempt:            1,
		Stream:             false,
		PreviousResponseID: "resp-prev",
	})
	require.NotNil(t, advice)
	require.NotNil(t, advice.Suggested)
	assert.True(t, *advice.Suggested)
	assert.Equal(t, "safe", advice.ReplaySafety)
	require.NotNil(t, advice.RetryAfter)
	assert.InDelta(t, 1.25, *advice.RetryAfter, 0.001)
}

func TestGetOpenAIRetryAdviceRejectsStatefulWebsocketReceiveTimeout(t *testing.T) {
	advice := getOpenAIRetryAdvice(retry.ModelRetryAdviceRequest{
		Error:              errors.New("Responses websocket receive timed out after 0.1 seconds"),
		Attempt:            1,
		Stream:             false,
		PreviousResponseID: "resp-prev",
	})
	require.NotNil(t, advice)
	require.NotNil(t, advice.Suggested)
	assert.False(t, *advice.Suggested)
	assert.Equal(t, "unsafe", advice.ReplaySafety)
}

func transientNetworkError() error {
	return &url.Error{
		Op:  "POST",
		URL: "https://example.com",
		Err: errors.New("connection error"),
	}
}

func openAIStatusError(t *testing.T, statusCode int, code string, headers http.Header) error {
	t.Helper()

	req, err := http.NewRequest(http.MethodPost, "https://example.com", nil)
	require.NoError(t, err)

	resp := &http.Response{
		StatusCode: statusCode,
		Header:     headers.Clone(),
		Request:    req,
	}

	return &openai.Error{
		Code:       code,
		Message:    code,
		Type:       "invalid_request_error",
		StatusCode: statusCode,
		Request:    req,
		Response:   resp,
	}
}
