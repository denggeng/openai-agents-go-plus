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

func TestMCPServerCallToolRetriesUntilSuccess(t *testing.T) {
	session := &dummyMCPSession{failCallTool: 2}
	server := NewMCPServerWithClientSession(MCPServerWithClientSessionParams{
		MaxRetryAttempts: 2,
		RetryBackoffBase: 0,
	})
	server.session = session

	_, err := server.CallTool(t.Context(), "tool", nil)
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

	_, err := server.CallTool(t.Context(), "tool", map[string]any{})
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

	_, err := server.CallTool(t.Context(), "tool", map[string]any{"param_a": "value"})
	require.NoError(t, err)
	assert.Equal(t, 1, session.callAttempts)
}

func TestMCPServerCallToolSkipsValidationWhenToolMissing(t *testing.T) {
	session := &dummyMCPSession{}
	server := NewMCPServerWithClientSession(MCPServerWithClientSessionParams{})
	server.session = session
	server.toolsList = []*mcp.Tool{{Name: "other_tool"}}

	_, err := server.CallTool(t.Context(), "tool", map[string]any{})
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

	_, err := server.CallTool(t.Context(), "tool", nil)
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

	_, err := server.CallTool(t.Context(), "tool", nil)
	require.Error(t, err)
	assert.ErrorContains(t, err, "missing required parameters")
	assert.Equal(t, 0, session.callAttempts)
}
