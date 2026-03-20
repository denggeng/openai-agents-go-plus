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
	"io"
	"net/http"
	"os/exec"
	"sync"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMCPServerStreamableHTTPTransportOptionsApplied(t *testing.T) {
	httpClient := &http.Client{Timeout: 42 * time.Second}
	server := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL: "http://example.com",
		TransportOpts: &mcp.StreamableClientTransport{
			HTTPClient: httpClient,
			MaxRetries: 7,
		},
	})

	transport, ok := server.transport.(*mcp.StreamableClientTransport)
	require.True(t, ok)
	assert.Equal(t, httpClient.Timeout, transport.HTTPClient.Timeout)
	assert.Equal(t, 7, transport.MaxRetries)
}

func TestMCPServerStreamableHTTPHTTPClientFactoryApplied(t *testing.T) {
	called := false
	client := &http.Client{Timeout: 13 * time.Second}

	server := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL: "http://example.com",
		HTTPClientFactory: func() *http.Client {
			called = true
			return client
		},
	})

	transport, ok := server.transport.(*mcp.StreamableClientTransport)
	require.True(t, ok)
	assert.True(t, called)
	assert.Equal(t, client.Timeout, transport.HTTPClient.Timeout)
}

func TestMCPServerStreamableHTTPHTTPClientFactoryDoesNotOverrideTransportOpts(t *testing.T) {
	called := false
	transportClient := &http.Client{Timeout: 11 * time.Second}

	server := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL: "http://example.com",
		TransportOpts: &mcp.StreamableClientTransport{
			HTTPClient: transportClient,
		},
		HTTPClientFactory: func() *http.Client {
			called = true
			return &http.Client{}
		},
	})

	transport, ok := server.transport.(*mcp.StreamableClientTransport)
	require.True(t, ok)
	assert.False(t, called)
	assert.Equal(t, transportClient.Timeout, transport.HTTPClient.Timeout)
}

func TestMCPServerSSEHTTPClientFactoryApplied(t *testing.T) {
	called := false
	client := &http.Client{Timeout: 9 * time.Second}

	server := NewMCPServerSSE(MCPServerSSEParams{
		BaseURL: "http://example.com/sse",
		HTTPClientFactory: func() *http.Client {
			called = true
			return client
		},
	})

	transport, ok := server.transport.(*mcp.SSEClientTransport)
	require.True(t, ok)
	assert.True(t, called)
	assert.Equal(t, client.Timeout, transport.HTTPClient.Timeout)
}

func TestMCPServerClientOptionsPropagate(t *testing.T) {
	opts := &mcp.ClientOptions{
		LoggingMessageHandler: func(context.Context, *mcp.LoggingMessageRequest) {},
	}
	server := NewMCPServerStdio(MCPServerStdioParams{
		Command:       exec.Command("true"),
		ClientOptions: opts,
	})

	require.Equal(t, opts, server.clientOptions)
}

func TestMCPServerStreamableHTTPDefaultConfigValues(t *testing.T) {
	server := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL: "http://example.com",
	})

	assert.Equal(t, 5*time.Second, server.Timeout())
	assert.Equal(t, 300*time.Second, server.SSEReadTimeout())
	assert.True(t, server.TerminateOnClose())
	assert.Empty(t, server.Headers())
}

func TestMCPServerSSEDefaultConfigValues(t *testing.T) {
	server := NewMCPServerSSE(MCPServerSSEParams{
		BaseURL: "http://example.com/sse",
	})

	assert.Equal(t, 5*time.Second, server.Timeout())
	assert.Equal(t, 300*time.Second, server.SSEReadTimeout())
	assert.Empty(t, server.Headers())
}

func TestMCPServerStreamableHTTPFactoryWithConfigApplied(t *testing.T) {
	var capturedHeaders map[string]string
	var capturedTimeout time.Duration
	client := &http.Client{Timeout: 17 * time.Second}

	server := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL:     "http://example.com",
		Headers: map[string]string{"Authorization": "Bearer token"},
		Timeout: 12 * time.Second,
		HTTPClientFactoryWithConfig: func(headers map[string]string, timeout time.Duration) *http.Client {
			capturedHeaders = headers
			capturedTimeout = timeout
			return client
		},
	})

	transport, ok := server.transport.(*mcp.StreamableClientTransport)
	require.True(t, ok)
	assert.Equal(t, map[string]string{"Authorization": "Bearer token"}, capturedHeaders)
	assert.Equal(t, 12*time.Second, capturedTimeout)
	assert.Equal(t, client.Timeout, transport.HTTPClient.Timeout)
}

func TestMCPServerStreamableHTTPFactoryWithConfigSkipsHeaderInjection(t *testing.T) {
	baseTransport := &http.Transport{}
	client := &http.Client{Transport: baseTransport}

	server := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL:     "http://example.com",
		Headers: map[string]string{"Authorization": "Bearer token"},
		HTTPClientFactoryWithConfig: func(headers map[string]string, timeout time.Duration) *http.Client {
			return client
		},
	})

	transport, ok := server.transport.(*mcp.StreamableClientTransport)
	require.True(t, ok)
	require.NotNil(t, transport.HTTPClient)
	wrapped, ok := transport.HTTPClient.Transport.(*mcpHeaderTransport)
	require.True(t, ok)
	assert.Empty(t, wrapped.headers)
	nextTransport, ok := wrapped.next.(*http.Transport)
	require.True(t, ok)
	assert.Equal(t, 5*time.Second, nextTransport.ResponseHeaderTimeout)
}

func TestMCPServerStreamableHTTPHeadersInjected(t *testing.T) {
	seenHeaders := make(http.Header)
	baseTransport := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		seenHeaders = req.Header.Clone()
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(http.NoBody),
			Header:     make(http.Header),
			Request:    req,
		}, nil
	})
	client := &http.Client{Transport: baseTransport}

	server := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL:               "http://example.com",
		Headers:           map[string]string{"Authorization": "Bearer token"},
		HTTPClientFactory: func() *http.Client { return client },
	})
	transport := server.transport.(*mcp.StreamableClientTransport)

	req, err := http.NewRequest(http.MethodGet, "http://example.com/mcp", http.NoBody)
	require.NoError(t, err)
	_, err = transport.HTTPClient.Transport.RoundTrip(req)
	require.NoError(t, err)
	assert.Equal(t, "Bearer token", seenHeaders.Get("Authorization"))
}

func TestMCPServerSSEHeadersInjected(t *testing.T) {
	seenHeaders := make(http.Header)
	baseTransport := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		seenHeaders = req.Header.Clone()
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(http.NoBody),
			Header:     make(http.Header),
			Request:    req,
		}, nil
	})
	client := &http.Client{Transport: baseTransport}

	server := NewMCPServerSSE(MCPServerSSEParams{
		BaseURL:           "http://example.com/sse",
		Headers:           map[string]string{"Authorization": "Bearer token"},
		HTTPClientFactory: func() *http.Client { return client },
	})
	transport := server.transport.(*mcp.SSEClientTransport)

	req, err := http.NewRequest(http.MethodGet, "http://example.com/sse", http.NoBody)
	require.NoError(t, err)
	_, err = transport.HTTPClient.Transport.RoundTrip(req)
	require.NoError(t, err)
	assert.Equal(t, "Bearer token", seenHeaders.Get("Authorization"))
}

func TestMCPServerSSEAuthInjected(t *testing.T) {
	seenHeaders := make(http.Header)
	baseTransport := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		seenHeaders = req.Header.Clone()
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(http.NoBody),
			Header:     make(http.Header),
			Request:    req,
		}, nil
	})
	client := &http.Client{Transport: baseTransport}

	server := NewMCPServerSSE(MCPServerSSEParams{
		BaseURL: "http://example.com/sse",
		Auth: func(req *http.Request) error {
			req.SetBasicAuth("user", "pass")
			return nil
		},
		HTTPClientFactory: func() *http.Client { return client },
	})
	transport := server.transport.(*mcp.SSEClientTransport)

	req, err := http.NewRequest(http.MethodGet, "http://example.com/sse", http.NoBody)
	require.NoError(t, err)
	_, err = transport.HTTPClient.Transport.RoundTrip(req)
	require.NoError(t, err)
	assert.Equal(t, "Basic dXNlcjpwYXNz", seenHeaders.Get("Authorization"))
}

func TestMCPServerStreamableHTTPAuthInjected(t *testing.T) {
	seenHeaders := make(http.Header)
	baseTransport := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		seenHeaders = req.Header.Clone()
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(http.NoBody),
			Header:     make(http.Header),
			Request:    req,
		}, nil
	})
	client := &http.Client{Transport: baseTransport}

	server := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL: "http://example.com",
		Auth: func(req *http.Request) error {
			req.SetBasicAuth("user", "pass")
			return nil
		},
		HTTPClientFactory: func() *http.Client { return client },
	})
	transport := server.transport.(*mcp.StreamableClientTransport)

	req, err := http.NewRequest(http.MethodGet, "http://example.com/mcp", http.NoBody)
	require.NoError(t, err)
	_, err = transport.HTTPClient.Transport.RoundTrip(req)
	require.NoError(t, err)
	assert.Equal(t, "Basic dXNlcjpwYXNz", seenHeaders.Get("Authorization"))
}

func TestMCPServerStreamableHTTPAuthInjectedWithConfigFactory(t *testing.T) {
	seenHeaders := make(http.Header)
	baseTransport := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		seenHeaders = req.Header.Clone()
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(http.NoBody),
			Header:     make(http.Header),
			Request:    req,
		}, nil
	})
	client := &http.Client{Transport: baseTransport}

	server := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL: "http://example.com",
		Auth: func(req *http.Request) error {
			req.SetBasicAuth("user", "pass")
			return nil
		},
		HTTPClientFactoryWithConfig: func(headers map[string]string, timeout time.Duration) *http.Client {
			return client
		},
	})
	transport := server.transport.(*mcp.StreamableClientTransport)

	req, err := http.NewRequest(http.MethodGet, "http://example.com/mcp", http.NoBody)
	require.NoError(t, err)
	_, err = transport.HTTPClient.Transport.RoundTrip(req)
	require.NoError(t, err)
	assert.Equal(t, "Basic dXNlcjpwYXNz", seenHeaders.Get("Authorization"))
}

func TestMCPServerSSETransportTimeoutApplied(t *testing.T) {
	server := NewMCPServerSSE(MCPServerSSEParams{
		BaseURL: "http://example.com/sse",
		Timeout: 12 * time.Second,
	})

	transport, ok := server.transport.(*mcp.SSEClientTransport)
	require.True(t, ok)
	wrapped, ok := transport.HTTPClient.Transport.(*mcpHeaderTransport)
	require.True(t, ok)
	nextTransport, ok := wrapped.next.(*http.Transport)
	require.True(t, ok)
	assert.Equal(t, 12*time.Second, nextTransport.ResponseHeaderTimeout)
}

func TestMCPServerStreamableHTTPTransportTimeoutApplied(t *testing.T) {
	server := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL:     "http://example.com",
		Timeout: 12 * time.Second,
	})

	transport, ok := server.transport.(*mcp.StreamableClientTransport)
	require.True(t, ok)
	assert.Zero(t, transport.HTTPClient.Timeout)
	wrapped, ok := transport.HTTPClient.Transport.(*mcpHeaderTransport)
	require.True(t, ok)
	nextTransport, ok := wrapped.next.(*http.Transport)
	require.True(t, ok)
	assert.Equal(t, 12*time.Second, nextTransport.ResponseHeaderTimeout)
}

func TestMCPServerStreamableHTTPSSEReadTimeoutApplied(t *testing.T) {
	body := newBlockingReadCloser()
	baseTransport := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Body:       body,
			Header:     make(http.Header),
			Request:    req,
		}, nil
	})
	client := &http.Client{Transport: baseTransport}

	server := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL:               "http://example.com",
		SSEReadTimeout:    20 * time.Millisecond,
		HTTPClientFactory: func() *http.Client { return client },
	})
	transport := server.transport.(*mcp.StreamableClientTransport)

	req, err := http.NewRequest(http.MethodGet, "http://example.com/mcp", http.NoBody)
	require.NoError(t, err)
	req.Header.Set("Accept", "text/event-stream")

	resp, err := transport.HTTPClient.Transport.RoundTrip(req)
	require.NoError(t, err)

	start := time.Now()
	_, err = resp.Body.Read(make([]byte, 1))
	require.Error(t, err)
	assert.GreaterOrEqual(t, time.Since(start), 20*time.Millisecond)
	var timeoutErr interface{ Timeout() bool }
	require.True(t, errors.As(err, &timeoutErr))
	assert.True(t, timeoutErr.Timeout())
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type blockingReadCloser struct {
	closed chan struct{}
	once   sync.Once
}

func newBlockingReadCloser() *blockingReadCloser {
	return &blockingReadCloser{closed: make(chan struct{})}
}

func (b *blockingReadCloser) Read([]byte) (int, error) {
	<-b.closed
	return 0, io.EOF
}

func (b *blockingReadCloser) Close() error {
	b.once.Do(func() {
		close(b.closed)
	})
	return nil
}
