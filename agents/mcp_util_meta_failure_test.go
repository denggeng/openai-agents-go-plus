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

package agents_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMCPMetaResolverMergesAndPasses(t *testing.T) {
	var captured agents.MCPToolMetaContext
	server := agentstesting.NewFakeMCPServer(nil, nil, "fake_mcp_server")
	server.ToolMetaResolver = func(_ context.Context, metaCtx agents.MCPToolMetaContext) (map[string]any, error) {
		captured = metaCtx
		return map[string]any{"request_id": "req-123", "locale": "ja"}, nil
	}
	server.AddTool("test_tool_1", nil)

	ctx := agents.ContextWithRunContextValue(t.Context(), map[string]any{"request_id": "req-123"})
	tool := &mcp.Tool{Name: "test_tool_1"}

	_, err := agents.MCPUtil().InvokeMCPTool(
		ctx,
		server,
		tool,
		"{}",
		map[string]any{"locale": "en", "extra": "value"},
	)
	require.NoError(t, err)

	require.NotEmpty(t, server.ToolMetas)
	assert.Equal(t,
		map[string]any{"request_id": "req-123", "locale": "en", "extra": "value"},
		server.ToolMetas[len(server.ToolMetas)-1],
	)
	require.NotNil(t, captured.RunContext)
	assert.Equal(t, "fake_mcp_server", captured.ServerName)
	assert.Equal(t, "test_tool_1", captured.ToolName)
	assert.Equal(t, map[string]any{}, captured.Arguments)
	assert.Equal(t, map[string]any{"request_id": "req-123"}, captured.RunContext.Context)
}

func TestMCPMetaResolverDoesNotMutateArguments(t *testing.T) {
	server := agentstesting.NewFakeMCPServer(nil, nil, "fake_mcp_server")
	server.ToolMetaResolver = func(_ context.Context, metaCtx agents.MCPToolMetaContext) (map[string]any, error) {
		if metaCtx.Arguments != nil {
			metaCtx.Arguments["mutated"] = "yes"
		}
		return map[string]any{"meta": "ok"}, nil
	}
	server.AddTool("test_tool_1", nil)

	tool := &mcp.Tool{Name: "test_tool_1"}
	_, err := agents.MCPUtil().InvokeMCPTool(t.Context(), server, tool, `{"foo":"bar"}`)
	require.NoError(t, err)

	require.NotEmpty(t, server.ToolResults)
	result := server.ToolResults[len(server.ToolResults)-1]
	prefix := "result_test_tool_1_"
	require.True(t, strings.HasPrefix(result, prefix))

	var parsed map[string]any
	require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(result, prefix)), &parsed))
	assert.Equal(t, map[string]any{"foo": "bar"}, parsed)
}

func TestMCPToolFailureErrorFunctionAgentDefault(t *testing.T) {
	server := &CrashingFakeMCPServer{
		FakeMCPServer: agentstesting.NewFakeMCPServer(nil, nil, "fake_mcp_server"),
		err:           errors.New("Crash!"),
	}
	server.AddTool("crashing_tool", nil)

	handler := agents.ToolErrorFunction(func(context.Context, error) (any, error) {
		return "custom_mcp_failure", nil
	})
	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("test-agent").
		WithModelInstance(model).
		AddMCPServer(server).
		WithMCPConfig(agents.MCPConfig{
			FailureErrorFunction:    &handler,
			FailureErrorFunctionSet: true,
		})

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("crashing_tool", "{}")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("done")}},
	})

	result, err := agents.Run(t.Context(), agent, "call crashing tool")
	require.NoError(t, err)
	assert.Equal(t, "done", result.FinalOutput)
	assert.Contains(t, collectMCPToolOutputs(result.NewItems), "custom_mcp_failure")
}

func TestMCPToolFailureErrorFunctionServerOverride(t *testing.T) {
	server := &CrashingFakeMCPServer{
		FakeMCPServer: agentstesting.NewFakeMCPServer(nil, nil, "fake_mcp_server"),
		err:           errors.New("Crash!"),
	}
	server.AddTool("crashing_tool", nil)

	agentHandler := agents.ToolErrorFunction(func(context.Context, error) (any, error) {
		return "agent_failure", nil
	})
	serverHandler := agents.ToolErrorFunction(func(context.Context, error) (any, error) {
		return "server_failure", nil
	})
	server.FailureErrorFunctionSet = true
	server.FailureErrorFunction = &serverHandler

	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("test-agent").
		WithModelInstance(model).
		AddMCPServer(server).
		WithMCPConfig(agents.MCPConfig{
			FailureErrorFunction:    &agentHandler,
			FailureErrorFunctionSet: true,
		})

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("crashing_tool", "{}")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("done")}},
	})

	result, err := agents.Run(t.Context(), agent, "call crashing tool")
	require.NoError(t, err)
	assert.Contains(t, collectMCPToolOutputs(result.NewItems), "server_failure")
	assert.NotContains(t, collectMCPToolOutputs(result.NewItems), "agent_failure")
}

func TestMCPToolFailureErrorFunctionServerNilRaises(t *testing.T) {
	server := &CrashingFakeMCPServer{
		FakeMCPServer: agentstesting.NewFakeMCPServer(nil, nil, "fake_mcp_server"),
		err:           errors.New("Crash!"),
	}
	server.AddTool("crashing_tool", nil)
	server.FailureErrorFunctionSet = true
	server.FailureErrorFunction = nil

	agentHandler := agents.ToolErrorFunction(func(context.Context, error) (any, error) {
		return "agent_failure", nil
	})

	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("test-agent").
		WithModelInstance(model).
		AddMCPServer(server).
		WithMCPConfig(agents.MCPConfig{
			FailureErrorFunction:    &agentHandler,
			FailureErrorFunctionSet: true,
		})
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("crashing_tool", "{}")}},
	})

	_, err := agents.Run(t.Context(), agent, "call crashing tool")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "error running tool")
	assert.Contains(t, err.Error(), "Crash!")
}

func collectMCPToolOutputs(items []agents.RunItem) []string {
	outputs := make([]string, 0)
	for _, item := range items {
		toolOutput, ok := item.(agents.ToolCallOutputItem)
		if !ok {
			continue
		}
		outputs = append(outputs, fmt.Sprint(toolOutput.Output))
	}
	return outputs
}
