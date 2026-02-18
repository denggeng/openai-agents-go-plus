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

package memory_test

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/denggeng/openai-agents-go-plus/memory"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAsyncSQLiteSessionRunnerIntegration(t *testing.T) {
	ctx := t.Context()
	session, err := memory.NewAsyncSQLiteSession(ctx, memory.AsyncSQLiteSessionParams{
		SessionID:        "runner_integration",
		DBDataSourceName: filepath.Join(t.TempDir(), "async_runner.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	model := agentstesting.NewFakeModel(false, nil)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	runner := agents.Runner{Config: agents.RunConfig{Session: session}}

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("San Francisco")},
	})
	result1, err := runner.Run(ctx, agent, "What city is the Golden Gate Bridge in?")
	require.NoError(t, err)
	assert.Equal(t, "San Francisco", result1.FinalOutput)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("California")},
	})
	result2, err := runner.Run(ctx, agent, "What state is it in?")
	require.NoError(t, err)
	assert.Equal(t, "California", result2.FinalOutput)

	lastInput, ok := model.LastTurnArgs.Input.(agents.InputItems)
	require.True(t, ok)
	assert.Greater(t, len(lastInput), 1)
	assert.True(t, containsInputText(lastInput, "Golden Gate Bridge"))
}

func containsInputText(items []agents.TResponseInputItem, needle string) bool {
	for _, item := range items {
		raw, err := json.Marshal(item)
		if err != nil {
			continue
		}
		if strings.Contains(string(raw), needle) {
			return true
		}
	}
	return false
}
