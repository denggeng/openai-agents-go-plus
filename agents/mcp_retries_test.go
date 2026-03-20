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
	"sync"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type dummyMCPSession struct {
	failCallTool  int
	failListTools int
	callAttempts  int
	listAttempts  int
}

func (d *dummyMCPSession) CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	d.callAttempts++
	if d.callAttempts <= d.failCallTool {
		return nil, errors.New("call_tool failure")
	}
	return &mcp.CallToolResult{Content: []mcp.Content{}}, nil
}

func (d *dummyMCPSession) ListTools(context.Context, *mcp.ListToolsParams) (*mcp.ListToolsResult, error) {
	d.listAttempts++
	if d.listAttempts <= d.failListTools {
		return nil, errors.New("list_tools failure")
	}
	return &mcp.ListToolsResult{Tools: []*mcp.Tool{{Name: "tool", InputSchema: &jsonschema.Schema{}}}}, nil
}

func (d *dummyMCPSession) ListPrompts(context.Context, *mcp.ListPromptsParams) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}

func (d *dummyMCPSession) GetPrompt(context.Context, *mcp.GetPromptParams) (*mcp.GetPromptResult, error) {
	return &mcp.GetPromptResult{}, nil
}

func (d *dummyMCPSession) Close() error { return nil }

type timeoutMCPSession struct {
	message      string
	callAttempts int
}

func (s *timeoutMCPSession) CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	s.callAttempts++
	return nil, errors.New(s.message)
}

func (s *timeoutMCPSession) ListTools(context.Context, *mcp.ListToolsParams) (*mcp.ListToolsResult, error) {
	return &mcp.ListToolsResult{}, nil
}

func (s *timeoutMCPSession) ListPrompts(context.Context, *mcp.ListPromptsParams) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}

func (s *timeoutMCPSession) GetPrompt(context.Context, *mcp.GetPromptParams) (*mcp.GetPromptResult, error) {
	return &mcp.GetPromptResult{}, nil
}

func (s *timeoutMCPSession) Close() error { return nil }

type cancelledMCPSession struct {
	callAttempts int
}

func (s *cancelledMCPSession) CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	s.callAttempts++
	return nil, context.Canceled
}

func (s *cancelledMCPSession) ListTools(context.Context, *mcp.ListToolsParams) (*mcp.ListToolsResult, error) {
	return &mcp.ListToolsResult{}, nil
}

func (s *cancelledMCPSession) ListPrompts(context.Context, *mcp.ListPromptsParams) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}

func (s *cancelledMCPSession) GetPrompt(context.Context, *mcp.GetPromptParams) (*mcp.GetPromptResult, error) {
	return &mcp.GetPromptResult{}, nil
}

func (s *cancelledMCPSession) Close() error { return nil }

type isolatedRetrySession struct {
	callAttempts int
}

func (s *isolatedRetrySession) CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	s.callAttempts++
	return &mcp.CallToolResult{Content: []mcp.Content{}}, nil
}

func (s *isolatedRetrySession) ListTools(context.Context, *mcp.ListToolsParams) (*mcp.ListToolsResult, error) {
	return &mcp.ListToolsResult{}, nil
}

func (s *isolatedRetrySession) ListPrompts(context.Context, *mcp.ListPromptsParams) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}

func (s *isolatedRetrySession) GetPrompt(context.Context, *mcp.GetPromptParams) (*mcp.GetPromptResult, error) {
	return &mcp.GetPromptResult{}, nil
}

func (s *isolatedRetrySession) Close() error { return nil }

type mcpHTTPStatusError struct {
	Response *http.Response
}

func (e *mcpHTTPStatusError) Error() string {
	if e == nil || e.Response == nil {
		return "HTTP error"
	}
	return "HTTP error " + e.Response.Status
}

type httpStatusMCPSession struct {
	statusCode   int
	callAttempts int
}

func (s *httpStatusMCPSession) CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	s.callAttempts++
	resp := &http.Response{StatusCode: s.statusCode, Status: http.StatusText(s.statusCode)}
	return nil, &mcpHTTPStatusError{Response: resp}
}

func (s *httpStatusMCPSession) ListTools(context.Context, *mcp.ListToolsParams) (*mcp.ListToolsResult, error) {
	return &mcp.ListToolsResult{}, nil
}

func (s *httpStatusMCPSession) ListPrompts(context.Context, *mcp.ListPromptsParams) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}

func (s *httpStatusMCPSession) GetPrompt(context.Context, *mcp.GetPromptParams) (*mcp.GetPromptResult, error) {
	return &mcp.GetPromptResult{}, nil
}

func (s *httpStatusMCPSession) Close() error { return nil }

type mixedErrorMCPSession struct{}

func (s *mixedErrorMCPSession) CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	resp := &http.Response{StatusCode: http.StatusUnauthorized, Status: http.StatusText(http.StatusUnauthorized)}
	return nil, errors.Join(context.Canceled, &mcpHTTPStatusError{Response: resp})
}

func (s *mixedErrorMCPSession) ListTools(context.Context, *mcp.ListToolsParams) (*mcp.ListToolsResult, error) {
	return &mcp.ListToolsResult{}, nil
}

func (s *mixedErrorMCPSession) ListPrompts(context.Context, *mcp.ListPromptsParams) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}

func (s *mixedErrorMCPSession) GetPrompt(context.Context, *mcp.GetPromptParams) (*mcp.GetPromptResult, error) {
	return &mcp.GetPromptResult{}, nil
}

func (s *mixedErrorMCPSession) Close() error { return nil }

type mcpRequestTimeoutError struct {
	Code    int
	Message string
}

func (e *mcpRequestTimeoutError) Error() string {
	return e.Message
}

type mcpRequestTimeoutSession struct {
	message      string
	callAttempts int
}

func (s *mcpRequestTimeoutSession) CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	s.callAttempts++
	return nil, &mcpRequestTimeoutError{
		Code:    http.StatusRequestTimeout,
		Message: s.message,
	}
}

func (s *mcpRequestTimeoutSession) ListTools(context.Context, *mcp.ListToolsParams) (*mcp.ListToolsResult, error) {
	return &mcp.ListToolsResult{}, nil
}

func (s *mcpRequestTimeoutSession) ListPrompts(context.Context, *mcp.ListPromptsParams) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}

func (s *mcpRequestTimeoutSession) GetPrompt(context.Context, *mcp.GetPromptParams) (*mcp.GetPromptResult, error) {
	return &mcp.GetPromptResult{}, nil
}

func (s *mcpRequestTimeoutSession) Close() error { return nil }

type overlapTrackingSession struct {
	mu          sync.Mutex
	inFlight    int
	maxInFlight int
}

func (s *overlapTrackingSession) withRequestDelay() func() {
	s.mu.Lock()
	s.inFlight++
	if s.inFlight > s.maxInFlight {
		s.maxInFlight = s.inFlight
	}
	s.mu.Unlock()

	time.Sleep(20 * time.Millisecond)

	return func() {
		s.mu.Lock()
		s.inFlight--
		s.mu.Unlock()
	}
}

func (s *overlapTrackingSession) CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	done := s.withRequestDelay()
	defer done()
	return &mcp.CallToolResult{Content: []mcp.Content{}}, nil
}

func (s *overlapTrackingSession) ListTools(context.Context, *mcp.ListToolsParams) (*mcp.ListToolsResult, error) {
	done := s.withRequestDelay()
	defer done()
	return &mcp.ListToolsResult{}, nil
}

func (s *overlapTrackingSession) ListPrompts(context.Context, *mcp.ListPromptsParams) (*mcp.ListPromptsResult, error) {
	done := s.withRequestDelay()
	defer done()
	return &mcp.ListPromptsResult{}, nil
}

func (s *overlapTrackingSession) GetPrompt(context.Context, *mcp.GetPromptParams) (*mcp.GetPromptResult, error) {
	done := s.withRequestDelay()
	defer done()
	return &mcp.GetPromptResult{}, nil
}

func (s *overlapTrackingSession) Close() error { return nil }

func newTestStreamableHTTPServer(
	shared mcpClientSession,
	retries int,
	isolatedFactory func(context.Context) (mcpClientSession, error),
) *MCPServerStreamableHTTP {
	server := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL:                  "https://example.test/mcp",
		MaxRetryAttempts:     retries,
		RetryBackoffBase:     0,
		CacheToolsList:       false,
		TransportOpts:        &mcp.StreamableClientTransport{},
		ClientOptions:        &mcp.ClientOptions{},
		ClientSessionTimeout: 0,
	})
	server.session = shared
	server.isolatedSessionFactory = isolatedFactory
	return server
}

func TestMCPServerCallToolRetriesUntilSuccess(t *testing.T) {
	session := &dummyMCPSession{failCallTool: 2}
	server := NewMCPServerWithClientSession(MCPServerWithClientSessionParams{
		MaxRetryAttempts: 2,
		RetryBackoffBase: 0,
	})
	server.session = session

	_, err := server.CallTool(t.Context(), "tool", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, 3, session.callAttempts)
}

func TestMCPServerListToolsUnlimitedRetries(t *testing.T) {
	session := &dummyMCPSession{failListTools: 3}
	server := NewMCPServerWithClientSession(MCPServerWithClientSessionParams{
		MaxRetryAttempts: -1,
		RetryBackoffBase: 0,
	})
	server.session = session

	tools, err := server.ListTools(t.Context(), nil)
	require.NoError(t, err)
	require.Len(t, tools, 1)
	assert.Equal(t, 4, session.listAttempts)
}

func TestMCPServerCallToolValidatesRequiredParameters(t *testing.T) {
	session := &dummyMCPSession{}
	server := NewMCPServerWithClientSession(MCPServerWithClientSessionParams{})
	server.session = session
	server.toolsList = []*mcp.Tool{{
		Name: "tool",
		InputSchema: &jsonschema.Schema{
			Type:     "object",
			Required: []string{"param_a"},
		},
	}}

	_, err := server.CallTool(t.Context(), "tool", map[string]any{}, nil)
	require.Error(t, err)
	assert.ErrorContains(t, err, "missing required parameters")
	assert.Equal(t, 0, session.callAttempts)
}

func TestMCPServerCallToolWithRequiredParametersCallsRemote(t *testing.T) {
	session := &dummyMCPSession{}
	server := NewMCPServerWithClientSession(MCPServerWithClientSessionParams{})
	server.session = session
	server.toolsList = []*mcp.Tool{{
		Name: "tool",
		InputSchema: &jsonschema.Schema{
			Type:     "object",
			Required: []string{"param_a"},
		},
	}}

	_, err := server.CallTool(t.Context(), "tool", map[string]any{"param_a": "value"}, nil)
	require.NoError(t, err)
	assert.Equal(t, 1, session.callAttempts)
}

func TestMCPServerCallToolSkipsValidationWhenToolMissing(t *testing.T) {
	session := &dummyMCPSession{}
	server := NewMCPServerWithClientSession(MCPServerWithClientSessionParams{})
	server.session = session
	server.toolsList = []*mcp.Tool{{Name: "other_tool"}}

	_, err := server.CallTool(t.Context(), "tool", map[string]any{}, nil)
	require.NoError(t, err)
	assert.Equal(t, 1, session.callAttempts)
}

func TestMCPServerCallToolSkipsValidationWhenRequiredAbsent(t *testing.T) {
	session := &dummyMCPSession{}
	server := NewMCPServerWithClientSession(MCPServerWithClientSessionParams{})
	server.session = session
	server.toolsList = []*mcp.Tool{{
		Name:        "tool",
		InputSchema: &jsonschema.Schema{Type: "object"},
	}}

	_, err := server.CallTool(t.Context(), "tool", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, 1, session.callAttempts)
}

func TestMCPServerCallToolValidatesRequiredWhenArgsNil(t *testing.T) {
	session := &dummyMCPSession{}
	server := NewMCPServerWithClientSession(MCPServerWithClientSessionParams{})
	server.session = session
	server.toolsList = []*mcp.Tool{{
		Name: "tool",
		InputSchema: &jsonschema.Schema{
			Type:     "object",
			Required: []string{"param_a"},
		},
	}}

	_, err := server.CallTool(t.Context(), "tool", nil, nil)
	require.Error(t, err)
	assert.ErrorContains(t, err, "missing required parameters")
	assert.Equal(t, 0, session.callAttempts)
}

func TestMCPServerStreamableHTTPRetriesTransientSharedFailuresOnIsolatedSession(t *testing.T) {
	tests := []struct {
		name   string
		shared mcpClientSession
	}{
		{
			name:   "cancelled",
			shared: &cancelledMCPSession{},
		},
		{
			name:   "http_5xx",
			shared: &httpStatusMCPSession{statusCode: http.StatusGatewayTimeout},
		},
		{
			name:   "closed_resource",
			shared: &timeoutMCPSession{message: "closed resource"},
		},
		{
			name:   "mcp_408",
			shared: &mcpRequestTimeoutSession{message: "Timed out while waiting for response to ClientRequest."},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			isolated := &isolatedRetrySession{}
			server := newTestStreamableHTTPServer(tt.shared, 1, func(context.Context) (mcpClientSession, error) {
				return isolated, nil
			})

			result, err := server.CallTool(t.Context(), "tool", nil, nil)
			require.NoError(t, err)
			require.NotNil(t, result)
			assert.Equal(t, 1, isolated.callAttempts)
		})
	}
}

func TestMCPServerStreamableHTTPDoesNotRetry4xxOnIsolatedSession(t *testing.T) {
	isolated := &isolatedRetrySession{}
	server := newTestStreamableHTTPServer(
		&httpStatusMCPSession{statusCode: http.StatusUnauthorized},
		1,
		func(context.Context) (mcpClientSession, error) {
			return isolated, nil
		},
	)

	_, err := server.CallTool(t.Context(), "tool", nil, nil)
	require.Error(t, err)
	assert.ErrorContains(t, err, http.StatusText(http.StatusUnauthorized))
	assert.Equal(t, 0, isolated.callAttempts)
}

func TestMCPServerStreamableHTTPDoesNotRetryMixedErrorsOnIsolatedSession(t *testing.T) {
	isolated := &isolatedRetrySession{}
	server := newTestStreamableHTTPServer(&mixedErrorMCPSession{}, 1, func(context.Context) (mcpClientSession, error) {
		return isolated, nil
	})

	_, err := server.CallTool(t.Context(), "tool", nil, nil)
	require.Error(t, err)
	assert.ErrorContains(t, err, http.StatusText(http.StatusUnauthorized))
	assert.Equal(t, 0, isolated.callAttempts)
}

func TestMCPServerStreamableHTTPDoesNotIsolatedRetryWithoutRetryBudget(t *testing.T) {
	shared := &cancelledMCPSession{}
	isolated := &isolatedRetrySession{}
	server := newTestStreamableHTTPServer(shared, 0, func(context.Context) (mcpClientSession, error) {
		return isolated, nil
	})

	_, err := server.CallTool(t.Context(), "tool", nil, nil)
	require.ErrorIs(t, err, context.Canceled)
	assert.Equal(t, 1, shared.callAttempts)
	assert.Equal(t, 0, isolated.callAttempts)
}

func TestMCPServerStreamableHTTPCountsIsolatedRetryAgainstRetryBudget(t *testing.T) {
	shared := &timeoutMCPSession{message: "shared timed out"}
	isolated := &timeoutMCPSession{message: "isolated timed out"}
	server := newTestStreamableHTTPServer(shared, 2, func(context.Context) (mcpClientSession, error) {
		return isolated, nil
	})

	_, err := server.CallTool(t.Context(), "tool", nil, nil)
	require.Error(t, err)
	assert.ErrorContains(t, err, "shared timed out")
	assert.Equal(t, 2, shared.callAttempts)
	assert.Equal(t, 1, isolated.callAttempts)
}

func TestMCPServerStreamableHTTPCountsIsolatedSessionSetupFailureAgainstRetryBudget(t *testing.T) {
	shared := &timeoutMCPSession{message: "shared timed out"}
	isolatedAttempts := 0
	server := newTestStreamableHTTPServer(shared, 2, func(context.Context) (mcpClientSession, error) {
		isolatedAttempts++
		return nil, errors.New("isolated setup timed out")
	})

	_, err := server.CallTool(t.Context(), "tool", nil, nil)
	require.Error(t, err)
	assert.ErrorContains(t, err, "shared timed out")
	assert.Equal(t, 2, shared.callAttempts)
	assert.Equal(t, 1, isolatedAttempts)
}

func TestMCPServerStreamableHTTPSerializesCallToolWithPromptRequests(t *testing.T) {
	tests := []struct {
		name string
		run  func(context.Context, *MCPServerStreamableHTTP) error
	}{
		{
			name: "list_prompts",
			run: func(ctx context.Context, server *MCPServerStreamableHTTP) error {
				var wg sync.WaitGroup
				errs := make(chan error, 2)

				wg.Add(2)
				go func() {
					defer wg.Done()
					_, err := server.CallTool(ctx, "slow", nil, nil)
					errs <- err
				}()
				go func() {
					defer wg.Done()
					_, err := server.ListPrompts(ctx)
					errs <- err
				}()
				wg.Wait()
				close(errs)

				for err := range errs {
					if err != nil {
						return err
					}
				}
				return nil
			},
		},
		{
			name: "get_prompt",
			run: func(ctx context.Context, server *MCPServerStreamableHTTP) error {
				var wg sync.WaitGroup
				errs := make(chan error, 2)

				wg.Add(2)
				go func() {
					defer wg.Done()
					_, err := server.CallTool(ctx, "slow", nil, nil)
					errs <- err
				}()
				go func() {
					defer wg.Done()
					_, err := server.GetPrompt(ctx, "prompt", nil)
					errs <- err
				}()
				wg.Wait()
				close(errs)

				for err := range errs {
					if err != nil {
						return err
					}
				}
				return nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shared := &overlapTrackingSession{}
			isolated := &isolatedRetrySession{}
			server := newTestStreamableHTTPServer(shared, 1, func(context.Context) (mcpClientSession, error) {
				return isolated, nil
			})

			require.NoError(t, tt.run(t.Context(), server))
			assert.Equal(t, 1, shared.maxInFlight)
			assert.Equal(t, 0, isolated.callAttempts)
		})
	}
}
