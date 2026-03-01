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
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMCPRequireApprovalPausesAndResumes(t *testing.T) {
	server := agentstesting.NewFakeMCPServer(nil, nil, "fake_mcp_server")
	server.RequireApproval = "always"
	server.AddTool("add", nil)

	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("TestAgent").WithModelInstance(model).AddMCPServer(server)

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("add", "{}")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("done")}},
	})

	first, err := agents.Run(t.Context(), agent, "call add")
	require.NoError(t, err)
	require.NotEmpty(t, first.Interruptions)
	assert.Equal(t, "add", first.Interruptions[0].ToolName)
	assert.Empty(t, server.ToolCalls)

	state := agents.NewRunStateFromResult(*first, 1, agents.DefaultMaxTurns)
	require.NoError(t, state.ApproveTool(first.Interruptions[0]))
	resumed, err := agents.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	assert.Empty(t, resumed.Interruptions)
	assert.Equal(t, "done", resumed.FinalOutput)
	assert.Equal(t, []string{"add"}, server.ToolCalls)
}

func TestMCPRequireApprovalToolListPolicy(t *testing.T) {
	server := agentstesting.NewFakeMCPServer(nil, nil, "fake_mcp_server")
	server.RequireApproval = map[string]any{
		"always": map[string]any{"tool_names": []any{"add"}},
		"never":  map[string]any{"tool_names": []any{"noop"}},
	}
	server.AddTool("add", nil)

	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("TestAgent").WithModelInstance(model).AddMCPServer(server)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("add", "{}")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("done")}},
	})

	first, err := agents.Run(t.Context(), agent, "call add")
	require.NoError(t, err)
	require.NotEmpty(t, first.Interruptions)
	assert.Equal(t, "add", first.Interruptions[0].ToolName)
}

func TestMCPRequireApprovalMappingPolicy(t *testing.T) {
	server := agentstesting.NewFakeMCPServer(nil, nil, "fake_mcp_server")
	server.RequireApproval = map[string]string{"add": "always", "noop": "never"}
	server.AddTool("add", nil)

	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("TestAgent").WithModelInstance(model).AddMCPServer(server)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("add", "{}")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("done")}},
	})

	first, err := agents.Run(t.Context(), agent, "call add")
	require.NoError(t, err)
	require.NotEmpty(t, first.Interruptions)
	assert.Equal(t, "add", first.Interruptions[0].ToolName)
}

func TestMCPRequireApprovalMappingAllowsLiteralPolicyNamesAsTools(t *testing.T) {
	server := agentstesting.NewFakeMCPServer(nil, nil, "fake_mcp_server")
	server.RequireApproval = map[string]string{"always": "always", "never": "never"}
	server.AddTool("always", nil)
	server.AddTool("never", nil)

	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("TestAgent").WithModelInstance(model).AddMCPServer(server)

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("always", "{}")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("done")}},
	})
	first, err := agents.Run(t.Context(), agent, "call always")
	require.NoError(t, err)
	require.NotEmpty(t, first.Interruptions)
	assert.Equal(t, "always", first.Interruptions[0].ToolName)

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("never", "{}")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("done")}},
	})
	second, err := agents.Run(t.Context(), agent, "call never")
	require.NoError(t, err)
	assert.Empty(t, second.Interruptions)
}
