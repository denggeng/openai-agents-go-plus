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
	"context"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type AgentLLMHooksForTests struct {
	Events map[string]int
}

func NewAgentLLMHooksForTests() *AgentLLMHooksForTests {
	return &AgentLLMHooksForTests{Events: make(map[string]int)}
}

func (h *AgentLLMHooksForTests) OnStart(context.Context, *agents.Agent) error {
	h.Events["OnStart"]++
	return nil
}

func (h *AgentLLMHooksForTests) OnEnd(context.Context, *agents.Agent, any) error {
	h.Events["OnEnd"]++
	return nil
}

func (h *AgentLLMHooksForTests) OnHandoff(context.Context, *agents.Agent, *agents.Agent) error {
	return nil
}

func (h *AgentLLMHooksForTests) OnToolStart(context.Context, *agents.Agent, agents.Tool, any) error {
	return nil
}

func (h *AgentLLMHooksForTests) OnToolEnd(context.Context, *agents.Agent, agents.Tool, any) error {
	return nil
}

func (h *AgentLLMHooksForTests) OnLLMStart(context.Context, *agents.Agent, param.Opt[string], []agents.TResponseInputItem) error {
	h.Events["OnLLMStart"]++
	return nil
}

func (h *AgentLLMHooksForTests) OnLLMEnd(context.Context, *agents.Agent, agents.ModelResponse) error {
	h.Events["OnLLMEnd"]++
	return nil
}

func TestAgentHooksWithLLMRun(t *testing.T) {
	hooks := NewAgentLLMHooksForTests()
	model := agentstesting.NewFakeModel(false, nil)
	agent := &agents.Agent{
		Name:  "A",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Hooks: hooks,
	}

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("hello")},
	})

	_, err := agents.Runner{}.Run(t.Context(), agent, "hello")
	require.NoError(t, err)
	assert.Equal(t, map[string]int{
		"OnStart":    1,
		"OnLLMStart": 1,
		"OnLLMEnd":   1,
		"OnEnd":      1,
	}, hooks.Events)
}

func TestAgentHooksWithLLMRunStreamed(t *testing.T) {
	hooks := NewAgentLLMHooksForTests()
	model := agentstesting.NewFakeModel(false, nil)
	agent := &agents.Agent{
		Name:  "A",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Hooks: hooks,
	}

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("hello")},
	})

	output, err := agents.Runner{}.RunStreamed(t.Context(), agent, "hello")
	require.NoError(t, err)
	require.NoError(t, output.StreamEvents(func(agents.StreamEvent) error { return nil }))
	assert.Equal(t, map[string]int{
		"OnStart":    1,
		"OnLLMStart": 1,
		"OnLLMEnd":   1,
		"OnEnd":      1,
	}, hooks.Events)
}
