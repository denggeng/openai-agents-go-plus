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
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
)

func TestMCPServerClientOptionsPassedToClient(t *testing.T) {
	opts := &mcp.ClientOptions{
		ToolListChangedHandler: func(ctx context.Context, _ *mcp.ToolListChangedRequest) {},
	}
	server := NewMCPServerWithClientSession(MCPServerWithClientSessionParams{
		Name:          "test",
		Transport:     failingMCPTransport{err: errors.New("boom")},
		ClientOptions: opts,
	})

	var captured *mcp.ClientOptions
	orig := newMCPClient
	newMCPClient = func(impl *mcp.Implementation, provided *mcp.ClientOptions) *mcp.Client {
		captured = provided
		return orig(impl, provided)
	}
	t.Cleanup(func() { newMCPClient = orig })

	_ = server.Connect(t.Context())

	assert.Same(t, opts, captured)
}

func TestMCPServerClientOptionsPropagateToServers(t *testing.T) {
	opts := &mcp.ClientOptions{}

	stdio := NewMCPServerStdio(MCPServerStdioParams{
		Command:       createMCPServerCommand(t),
		ClientOptions: opts,
	})
	assert.Same(t, opts, stdio.clientOptions)

	sse := NewMCPServerSSE(MCPServerSSEParams{
		BaseURL:       "https://example.com",
		ClientOptions: opts,
	})
	assert.Same(t, opts, sse.clientOptions)

	streamable := NewMCPServerStreamableHTTP(MCPServerStreamableHTTPParams{
		URL:           "https://example.com",
		ClientOptions: opts,
	})
	assert.Same(t, opts, streamable.clientOptions)
}
