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
	"net/http"
	"os/exec"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMCPServerStreamableHTTPTransportOptionsApplied(t *testing.T) {
	httpClient := &http.Client{}
	server := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL: "http://example.com",
		TransportOpts: &mcp.StreamableClientTransport{
			HTTPClient: httpClient,
			MaxRetries: 7,
		},
	})

	transport, ok := server.transport.(*mcp.StreamableClientTransport)
	require.True(t, ok)
	assert.Equal(t, httpClient, transport.HTTPClient)
	assert.Equal(t, 7, transport.MaxRetries)
}

func TestMCPServerStreamableHTTPHTTPClientFactoryApplied(t *testing.T) {
	called := false
	client := &http.Client{}

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
	assert.Equal(t, client, transport.HTTPClient)
}

func TestMCPServerStreamableHTTPHTTPClientFactoryDoesNotOverrideTransportOpts(t *testing.T) {
	called := false
	transportClient := &http.Client{}

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
	assert.Equal(t, transportClient, transport.HTTPClient)
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
