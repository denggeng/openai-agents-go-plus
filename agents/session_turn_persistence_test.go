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
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSessionPersistsTurnItemsDuringRun(t *testing.T) {
	session := &recordingSession{}
	model := agentstesting.NewFakeModel(false, nil)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("foo", "tool_result"),
		},
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetFunctionToolCall("foo", `{}`),
			},
		},
		{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetTextMessage("done"),
			},
		},
	})

	runner := agents.Runner{
		Config: agents.RunConfig{
			Session: session,
		},
	}
	result, err := runner.Run(t.Context(), agent, "user_message")
	require.NoError(t, err)
	require.Equal(t, "done", result.FinalOutput)

	// Expect at least three persistence batches:
	// 1) initial user input, 2) first turn tool items, 3) second turn output item.
	require.GreaterOrEqual(t, len(session.savedBatches), 3)
	require.Len(t, session.savedBatches[0], 1)

	firstSaved := agents.ItemHelpers().InputToNewInputList(agents.InputItems(session.savedBatches[0]))
	require.Len(t, firstSaved, 1)
	assert.NotNil(t, firstSaved[0].OfMessage)
}
