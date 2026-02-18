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
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/require"
)

func TestToolContextRequiresFields(t *testing.T) {
	ctx := agents.NewRunContextWrapper(map[string]any{})
	_, err := agents.ToolContextFromAgentContext[map[string]any](ctx, "call-1", nil, nil, nil)
	require.Error(t, err)
}

func TestToolContextMissingDefaultsRaise(t *testing.T) {
	_, err := agents.NewToolContext(map[string]any{}, "name", "call-1", "")
	require.Error(t, err)

	_, err = agents.NewToolContext(map[string]any{}, "", "call-1", "{}")
	require.Error(t, err)

	_, err = agents.NewToolContext(map[string]any{}, "name", "", "{}")
	require.Error(t, err)
}

func TestToolContextFromAgentContextPopulatesFields(t *testing.T) {
	toolCall := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "test_tool",
		CallID:    "call-123",
		Arguments: `{"a": 1}`,
	}
	ctx := agents.NewRunContextWrapper(map[string]any{})
	agent := &agents.Agent{Name: "agent"}

	toolCtx, err := agents.ToolContextFromAgentContext[map[string]any](
		ctx,
		"call-123",
		&toolCall,
		agent,
		nil,
	)
	require.NoError(t, err)
	require.Equal(t, "test_tool", toolCtx.ToolName)
	require.Equal(t, "call-123", toolCtx.ToolCallID)
	require.Equal(t, `{"a": 1}`, toolCtx.ToolArguments)
	require.Equal(t, agent, toolCtx.Agent)
}

func TestToolContextAgentNoneByDefault(t *testing.T) {
	toolCall := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "test_tool",
		CallID:    "call-1",
		Arguments: `{}`,
	}
	ctx := agents.NewRunContextWrapper(map[string]any{})

	toolCtx, err := agents.ToolContextFromAgentContext[map[string]any](ctx, "call-1", &toolCall, nil, nil)
	require.NoError(t, err)
	require.Nil(t, toolCtx.Agent)
}

func TestToolContextConstructorAcceptsAgentKeyword(t *testing.T) {
	agent := &agents.Agent{Name: "direct-agent"}
	toolCtx, err := agents.NewToolContext(
		map[string]any{},
		"my_tool",
		"call-2",
		"{}",
		agents.ToolContextWithAgent[map[string]any](agent),
	)
	require.NoError(t, err)
	require.Equal(t, agent, toolCtx.Agent)
}

func TestToolContextFromToolContextInheritsAgent(t *testing.T) {
	originalCall := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "test_tool",
		CallID:    "call-3",
		Arguments: `{}`,
	}
	derivedCall := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "test_tool",
		CallID:    "call-4",
		Arguments: `{}`,
	}
	agent := &agents.Agent{Name: "origin-agent"}
	parentContext, err := agents.NewToolContext(
		map[string]any{},
		"test_tool",
		"call-3",
		"{}",
		agents.ToolContextWithToolCall[map[string]any](&originalCall),
		agents.ToolContextWithAgent[map[string]any](agent),
	)
	require.NoError(t, err)

	derivedContext, err := agents.ToolContextFromAgentContext[map[string]any](
		parentContext,
		"call-4",
		&derivedCall,
		nil,
		nil,
	)
	require.NoError(t, err)
	require.Equal(t, agent, derivedContext.Agent)
}

func TestToolContextFromToolContextInheritsRunConfig(t *testing.T) {
	originalCall := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "test_tool",
		CallID:    "call-3",
		Arguments: `{}`,
	}
	derivedCall := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "test_tool",
		CallID:    "call-4",
		Arguments: `{}`,
	}
	runConfig := &agents.RunConfig{WorkflowName: "parent"}
	parentContext, err := agents.NewToolContext(
		map[string]any{},
		"test_tool",
		"call-3",
		"{}",
		agents.ToolContextWithToolCall[map[string]any](&originalCall),
		agents.ToolContextWithRunConfig[map[string]any](runConfig),
	)
	require.NoError(t, err)

	derivedContext, err := agents.ToolContextFromAgentContext[map[string]any](
		parentContext,
		"call-4",
		&derivedCall,
		nil,
		nil,
	)
	require.NoError(t, err)
	require.Equal(t, runConfig, derivedContext.RunConfig)
}

func TestToolContextFromAgentContextPrefersExplicitRunConfig(t *testing.T) {
	toolCall := responses.ResponseFunctionToolCall{
		Type:      constant.ValueOf[constant.FunctionCall](),
		Name:      "test_tool",
		CallID:    "call-1",
		Arguments: `{}`,
	}
	ctx := agents.NewRunContextWrapper(map[string]any{})
	explicitRunConfig := &agents.RunConfig{WorkflowName: "explicit"}

	toolCtx, err := agents.ToolContextFromAgentContext[map[string]any](
		ctx,
		"call-1",
		&toolCall,
		nil,
		explicitRunConfig,
	)
	require.NoError(t, err)
	require.Equal(t, explicitRunConfig, toolCtx.RunConfig)
}
