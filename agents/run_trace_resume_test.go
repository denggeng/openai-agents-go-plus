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
	"github.com/denggeng/openai-agents-go-plus/tracing/tracingtesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunStateCapturesTraceAndResumeReattaches(t *testing.T) {
	tracingtesting.Setup(t)

	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("second")}},
	})
	agent := &agents.Agent{
		Name:  "trace-agent",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	first, err := agents.Run(t.Context(), agent, "hello")
	require.NoError(t, err)
	require.NotNil(t, first.Trace)
	require.NotEmpty(t, first.Trace.TraceID)

	state := agents.NewRunStateFromResult(*first, 1, agents.DefaultMaxTurns)
	require.NotNil(t, state.Trace)
	assert.Equal(t, first.Trace.TraceID, state.Trace.TraceID)

	resumed, err := agents.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	assert.Equal(t, "second", resumed.FinalOutput)

	startEvents := 0
	for _, event := range tracingtesting.FetchEvents() {
		if event == tracingtesting.TraceStart {
			startEvents++
		}
	}
	assert.Equal(t, 1, startEvents, "resumed run should reattach trace instead of starting a new one")
}
