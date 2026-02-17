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
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunReturnsInterruptionsForHostedMCPApprovalWithoutCallback(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			{
				ID:          "approval_1",
				Type:        "mcp_approval_request",
				ServerLabel: "mcp_server",
				Name:        "add",
				Arguments:   `{"a":1}`,
			},
		},
	})

	agent := agents.New("test").WithModelInstance(model).WithTools(
		agents.HostedMCPTool{
			ToolConfig: responses.ToolMcpParam{
				ServerLabel: "mcp_server",
				Type:        constant.ValueOf[constant.Mcp](),
			},
		},
	)

	result, err := agents.Runner{}.Run(t.Context(), agent, "hello")
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Nil(t, result.FinalOutput)
	require.Len(t, result.Interruptions, 1)
	assert.Equal(t, "add", result.Interruptions[0].ToolName)
}

func TestRunStreamedReturnsInterruptionsForHostedMCPApprovalWithoutCallback(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			{
				ID:          "approval_1",
				Type:        "mcp_approval_request",
				ServerLabel: "mcp_server",
				Name:        "add",
				Arguments:   `{"a":1}`,
			},
		},
	})

	agent := agents.New("test").WithModelInstance(model).WithTools(
		agents.HostedMCPTool{
			ToolConfig: responses.ToolMcpParam{
				ServerLabel: "mcp_server",
				Type:        constant.ValueOf[constant.Mcp](),
			},
		},
	)

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "hello")
	require.NoError(t, err)

	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)
	assert.Nil(t, result.FinalOutput())
	require.Len(t, result.Interruptions(), 1)
	assert.Equal(t, "add", result.Interruptions()[0].ToolName)
}
