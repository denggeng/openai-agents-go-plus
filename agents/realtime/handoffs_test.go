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

package realtime

import (
	"context"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRealtimeHandoffCreation(t *testing.T) {
	rt := &RealtimeAgent[any]{Name: "test_agent"}
	handoff := RealtimeHandoff(rt)

	assert.Equal(t, "test_agent", handoff.AgentName)
	assert.Equal(t, "transfer_to_test_agent", handoff.ToolName)
	assert.Nil(t, handoff.InputFilter)
	require.NotNil(t, handoff.IsEnabled)

	enabled, err := handoff.IsEnabled.IsEnabled(context.Background(), &agents.Agent{Name: "caller"})
	require.NoError(t, err)
	assert.True(t, enabled)
}

func TestRealtimeHandoffWithCustomParams(t *testing.T) {
	rt := &RealtimeAgent[any]{Name: "helper_agent"}
	handoff, err := SafeRealtimeHandoff(rt, RealtimeHandoffParams{
		ToolNameOverride:        "custom_handoff",
		ToolDescriptionOverride: "Custom handoff description",
		IsEnabled:               false,
	})
	require.NoError(t, err)
	assert.Equal(t, "custom_handoff", handoff.ToolName)
	assert.Equal(t, "Custom handoff description", handoff.ToolDescription)
	assert.Equal(t, "helper_agent", handoff.AgentName)

	enabled, err := handoff.IsEnabled.IsEnabled(context.Background(), &agents.Agent{Name: "caller"})
	require.NoError(t, err)
	assert.False(t, enabled)
}

func TestRealtimeHandoffInvokeReturnsAgent(t *testing.T) {
	rt := &RealtimeAgent[any]{Name: "target_agent"}
	handoff := RealtimeHandoff(rt)

	agent, err := handoff.OnInvokeHandoff(context.Background(), "")
	require.NoError(t, err)
	require.NotNil(t, agent)
	assert.Equal(t, "target_agent", agent.Name)
}

func TestRealtimeHandoffOnHandoffCallbackRuns(t *testing.T) {
	rt := &RealtimeAgent[any]{Name: "callback_agent"}
	runContext := agents.NewRunContextWrapper[any](map[string]any{"k": "v"})
	ctx := agents.ContextWithRunContextValue(context.Background(), runContext)

	called := false
	handoff, err := SafeRealtimeHandoff(rt, RealtimeHandoffParams{
		OnHandoff: func(ctx *agents.RunContextWrapper[any]) error {
			called = true
			assert.Same(t, runContext, ctx)
			return nil
		},
	})
	require.NoError(t, err)

	_, err = handoff.OnInvokeHandoff(ctx, "")
	require.NoError(t, err)
	assert.True(t, called)
}

func TestRealtimeHandoffInvalidParamCountsRaise(t *testing.T) {
	rt := &RealtimeAgent[any]{Name: "x"}

	_, err := SafeRealtimeHandoff(rt, RealtimeHandoffParams{
		OnHandoff: func(ctx *agents.RunContextWrapper[any]) {},
		InputType: 0,
	})
	require.Error(t, err)
	var userErr agents.UserError
	assert.ErrorAs(t, err, &userErr)

	_, err = SafeRealtimeHandoff(rt, RealtimeHandoffParams{
		OnHandoff: func(ctx *agents.RunContextWrapper[any], value int) {},
	})
	require.Error(t, err)
	assert.ErrorAs(t, err, &userErr)
}

func TestRealtimeHandoffMissingInputJSONRaisesModelError(t *testing.T) {
	rt := &RealtimeAgent[any]{Name: "x"}
	handoff, err := SafeRealtimeHandoff(rt, RealtimeHandoffParams{
		OnHandoff: func(ctx *agents.RunContextWrapper[any], value int) {},
		InputType: 0,
	})
	require.NoError(t, err)

	_, err = handoff.OnInvokeHandoff(context.Background(), "null")
	require.Error(t, err)
	var modelErr agents.ModelBehaviorError
	assert.ErrorAs(t, err, &modelErr)
}

func TestRealtimeHandoffRejectsEmptyInput(t *testing.T) {
	rt := &RealtimeAgent[any]{Name: "x"}
	handoff, err := SafeRealtimeHandoff(rt, RealtimeHandoffParams{
		OnHandoff: func(ctx *agents.RunContextWrapper[any], value int) {},
		InputType: 0,
	})
	require.NoError(t, err)

	_, err = handoff.OnInvokeHandoff(context.Background(), "")
	require.Error(t, err)
	var modelErr agents.ModelBehaviorError
	assert.ErrorAs(t, err, &modelErr)
}

func TestRealtimeHandoffIsEnabledCallable(t *testing.T) {
	rt := &RealtimeAgent[any]{Name: "x"}
	calls := 0
	handoff, err := SafeRealtimeHandoff(rt, RealtimeHandoffParams{
		IsEnabled: func(ctx *agents.RunContextWrapper[any], agent *RealtimeAgent[any]) bool {
			calls++
			assert.Equal(t, "x", agent.Name)
			return true
		},
	})
	require.NoError(t, err)

	ctx := agents.ContextWithRunContextValue(context.Background(), agents.NewRunContextWrapper[any](nil))
	enabled, err := handoff.IsEnabled.IsEnabled(ctx, &agents.Agent{Name: "caller"})
	require.NoError(t, err)
	assert.True(t, enabled)
	assert.Equal(t, 1, calls)
}

func TestRealtimeHandoffOnHandoffWithInputRuns(t *testing.T) {
	rt := &RealtimeAgent[any]{Name: "sync"}
	called := []int{}
	handoff, err := SafeRealtimeHandoff(rt, RealtimeHandoffParams{
		OnHandoff: func(ctx *agents.RunContextWrapper[any], value int) {
			called = append(called, value)
		},
		InputType: 0,
	})
	require.NoError(t, err)

	agent, err := handoff.OnInvokeHandoff(context.Background(), "5")
	require.NoError(t, err)
	require.NotNil(t, agent)
	assert.Equal(t, "sync", agent.Name)
	assert.Equal(t, []int{5}, called)
}

func TestRealtimeHandoffOnHandoffWithoutInputRuns(t *testing.T) {
	rt := &RealtimeAgent[any]{Name: "no_input"}
	called := []bool{}
	handoff, err := SafeRealtimeHandoff(rt, RealtimeHandoffParams{
		OnHandoff: func(ctx *agents.RunContextWrapper[any]) {
			called = append(called, true)
		},
	})
	require.NoError(t, err)

	agent, err := handoff.OnInvokeHandoff(context.Background(), "")
	require.NoError(t, err)
	require.NotNil(t, agent)
	assert.Equal(t, "no_input", agent.Name)
	assert.Equal(t, []bool{true}, called)
}
