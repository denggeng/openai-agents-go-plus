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
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type listToolsOnlySession struct {
	tools []*mcp.Tool
}

func (s *listToolsOnlySession) ListTools(context.Context, *mcp.ListToolsParams) (*mcp.ListToolsResult, error) {
	return &mcp.ListToolsResult{Tools: s.tools}, nil
}

func (*listToolsOnlySession) CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	return &mcp.CallToolResult{}, nil
}

func (*listToolsOnlySession) ListPrompts(context.Context, *mcp.ListPromptsParams) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}

func (*listToolsOnlySession) GetPrompt(context.Context, *mcp.GetPromptParams) (*mcp.GetPromptResult, error) {
	return &mcp.GetPromptResult{}, nil
}

func (*listToolsOnlySession) Close() error { return nil }

type blockingSession struct{}

func (*blockingSession) ListTools(ctx context.Context, _ *mcp.ListToolsParams) (*mcp.ListToolsResult, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func (*blockingSession) CallTool(ctx context.Context, _ *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func (*blockingSession) ListPrompts(ctx context.Context, _ *mcp.ListPromptsParams) (*mcp.ListPromptsResult, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func (*blockingSession) GetPrompt(ctx context.Context, _ *mcp.GetPromptParams) (*mcp.GetPromptResult, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func (*blockingSession) Close() error { return nil }

func TestMCPServerWithClientSessionCachedToolsAccessor(t *testing.T) {
	session := &listToolsOnlySession{
		tools: []*mcp.Tool{{Name: "tool_a"}},
	}
	server := NewMCPServerWithClientSession(MCPServerWithClientSessionParams{
		Name:           "test",
		CacheToolsList: true,
	})
	server.session = session

	tools, err := server.ListTools(t.Context(), New("agent"))
	require.NoError(t, err)
	require.Len(t, tools, 1)
	require.Equal(t, "tool_a", tools[0].Name)

	cached := server.CachedTools()
	require.Len(t, cached, 1)
	assert.Equal(t, "tool_a", cached[0].Name)
	cached[0] = nil
	require.NotNil(t, server.toolsList[0])
}

func TestMCPServerWithClientSessionTimeoutAppliesToCallTool(t *testing.T) {
	server := NewMCPServerWithClientSession(MCPServerWithClientSessionParams{
		Name:                 "test",
		ClientSessionTimeout: 10 * time.Millisecond,
	})
	server.session = &blockingSession{}

	_, err := server.CallTool(context.Background(), "tool", map[string]any{}, nil)
	require.Error(t, err)
	assert.ErrorIs(t, err, context.DeadlineExceeded)
}
