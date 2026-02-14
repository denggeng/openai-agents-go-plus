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

package realtime

import (
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCanInitializeRealtimeAgent(t *testing.T) {
	agent := &RealtimeAgent[any]{
		Name:         "test",
		Instructions: "Hello",
	}
	assert.Equal(t, "test", agent.Name)
	assert.Equal(t, "Hello", agent.Instructions)
}

func TestDynamicInstructions(t *testing.T) {
	agent := &RealtimeAgent[any]{Name: "test"}
	assert.Nil(t, agent.Instructions)

	dynamicAgent := &RealtimeAgent[any]{Name: "test"}
	dynamicAgent.Instructions = RealtimeInstructionsSyncFunc[any](
		func(ctx *agents.RunContextWrapper[any], agt *RealtimeAgent[any]) string {
			assert.Nil(t, ctx.Context)
			assert.Same(t, dynamicAgent, agt)
			return "Dynamic"
		},
	)

	instructions, err := dynamicAgent.GetSystemPrompt(agents.NewRunContextWrapper[any](nil))
	require.NoError(t, err)
	assert.Equal(t, "Dynamic", instructions)
}

func TestRealtimeAgentClone(t *testing.T) {
	agent := &RealtimeAgent[any]{
		Name:         "test",
		Instructions: "Hello",
	}

	cloned := agent.Clone()
	require.NotNil(t, cloned)
	assert.NotSame(t, agent, cloned)
	assert.Equal(t, agent.Name, cloned.Name)
	assert.Equal(t, agent.Instructions, cloned.Instructions)
}
