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
	"slices"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAgentCloneShallowCopy(t *testing.T) {
	target := agents.New("Target")
	greet := agentstesting.GetFunctionTool("greet", "Hello")
	tool := &greet
	handoff := agents.HandoffFromAgent(agents.HandoffFromAgentParams{Agent: target})

	original := &agents.Agent{
		Name:         "Original",
		Instructions: agents.InstructionsStr("Testing clone shallow copy"),
		Tools:        []agents.Tool{tool},
		Handoffs:     []agents.Handoff{handoff},
	}

	cloned := original.Clone(func(agent *agents.Agent) {
		agent.Name = "Cloned"
		agent.Tools = slices.Clone(original.Tools)
		agent.Handoffs = slices.Clone(original.Handoffs)
	})

	require.NotSame(t, original, cloned)
	assert.Equal(t, "Cloned", cloned.Name)
	assert.Equal(t, original.Instructions, cloned.Instructions)

	require.Len(t, cloned.Tools, 1)
	require.Len(t, original.Tools, 1)
	assert.False(t, &cloned.Tools[0] == &original.Tools[0], "Tools should be different list")
	clonedTool, ok := cloned.Tools[0].(*agents.FunctionTool)
	require.True(t, ok)
	originalTool, ok := original.Tools[0].(*agents.FunctionTool)
	require.True(t, ok)
	assert.Same(t, originalTool, clonedTool)

	require.Len(t, cloned.Handoffs, 1)
	require.Len(t, original.Handoffs, 1)
	assert.False(t, &cloned.Handoffs[0] == &original.Handoffs[0], "Handoffs should be different list")
	assert.Equal(t, original.Handoffs[0].ToolName, cloned.Handoffs[0].ToolName)
	assert.Equal(t, original.Handoffs[0].AgentName, cloned.Handoffs[0].AgentName)
}
