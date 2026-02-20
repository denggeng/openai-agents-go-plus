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
	"encoding/json"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/denggeng/openai-agents-go-plus/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSessionLimitWithRunner(t *testing.T) {
	runAgent := func(t *testing.T, streaming bool, session memory.Session, agent *agents.Agent, input string, limit int) any {
		t.Helper()

		runner := agents.Runner{
			Config: agents.RunConfig{
				Session:     session,
				LimitMemory: limit,
			},
		}

		if streaming {
			result, err := runner.RunStreamed(t.Context(), agent, input)
			require.NoError(t, err)
			require.NoError(t, result.StreamEvents(func(agents.StreamEvent) error { return nil }))
			return result.FinalOutput()
		}
		result, err := runner.Run(t.Context(), agent, input)
		require.NoError(t, err)
		return result.FinalOutput
	}

	for _, streaming := range []bool{false, true} {
		t.Run("streaming="+boolLabel(streaming), func(t *testing.T) {
			dbPath := filepath.Join(t.TempDir(), "session_limit.db")
			session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
				SessionID:        "limit_test",
				DBDataSourceName: dbPath,
			})
			require.NoError(t, err)
			t.Cleanup(func() { assert.NoError(t, session.Close()) })

			model := agentstesting.NewFakeModel(false, nil)
			agent := agents.New("test").WithModelInstance(model)

			model.SetNextOutput(agentstesting.FakeModelTurnOutput{
				Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 1")},
			})
			_ = runAgent(t, streaming, session, agent, "Message 1", 0)

			model.SetNextOutput(agentstesting.FakeModelTurnOutput{
				Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 2")},
			})
			_ = runAgent(t, streaming, session, agent, "Message 2", 0)

			model.SetNextOutput(agentstesting.FakeModelTurnOutput{
				Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 3")},
			})
			_ = runAgent(t, streaming, session, agent, "Message 3", 0)

			allItems, err := session.GetItems(t.Context(), 0)
			require.NoError(t, err)
			assert.Len(t, allItems, 6)

			model.SetNextOutput(agentstesting.FakeModelTurnOutput{
				Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 4")},
			})
			_ = runAgent(t, streaming, session, agent, "Message 4", 2)

			lastInput := model.LastTurnArgs.Input
			require.IsType(t, agents.InputItems{}, lastInput)
			inputItems := lastInput.(agents.InputItems)
			require.Len(t, inputItems, 3)

			assert.Equal(t, "Message 3", inputItemText(t, inputItems[0]))
			assert.Equal(t, "Reply 3", inputItemText(t, inputItems[1]))
			assert.Equal(t, "Message 4", inputItemText(t, inputItems[2]))
		})
	}
}

func TestSessionSettingsLimitWithRunner(t *testing.T) {
	ptrInt := func(v int) *int { return &v }

	runAgent := func(t *testing.T, streaming bool, session memory.Session, agent *agents.Agent, input string, settings *memory.SessionSettings) any {
		t.Helper()

		runner := agents.Runner{
			Config: agents.RunConfig{
				Session:         session,
				SessionSettings: settings,
			},
		}

		if streaming {
			result, err := runner.RunStreamed(t.Context(), agent, input)
			require.NoError(t, err)
			require.NoError(t, result.StreamEvents(func(agents.StreamEvent) error { return nil }))
			return result.FinalOutput()
		}
		result, err := runner.Run(t.Context(), agent, input)
		require.NoError(t, err)
		return result.FinalOutput
	}

	for _, streaming := range []bool{false, true} {
		t.Run("streaming="+boolLabel(streaming), func(t *testing.T) {
			dbPath := filepath.Join(t.TempDir(), "session_limit_settings.db")
			session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
				SessionID:        "limit_settings_test",
				DBDataSourceName: dbPath,
			})
			require.NoError(t, err)
			t.Cleanup(func() { assert.NoError(t, session.Close()) })

			model := agentstesting.NewFakeModel(false, nil)
			agent := agents.New("test").WithModelInstance(model)

			model.SetNextOutput(agentstesting.FakeModelTurnOutput{
				Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 1")},
			})
			_ = runAgent(t, streaming, session, agent, "Message 1", nil)

			model.SetNextOutput(agentstesting.FakeModelTurnOutput{
				Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 2")},
			})
			_ = runAgent(t, streaming, session, agent, "Message 2", nil)

			model.SetNextOutput(agentstesting.FakeModelTurnOutput{
				Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 3")},
			})
			_ = runAgent(t, streaming, session, agent, "Message 3", nil)

			allItems, err := session.GetItems(t.Context(), 0)
			require.NoError(t, err)
			assert.Len(t, allItems, 6)

			model.SetNextOutput(agentstesting.FakeModelTurnOutput{
				Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 4")},
			})
			_ = runAgent(t, streaming, session, agent, "Message 4", &memory.SessionSettings{Limit: ptrInt(2)})

			lastInput := model.LastTurnArgs.Input
			require.IsType(t, agents.InputItems{}, lastInput)
			inputItems := lastInput.(agents.InputItems)
			require.Len(t, inputItems, 3)

			assert.Equal(t, "Message 3", inputItemText(t, inputItems[0]))
			assert.Equal(t, "Reply 3", inputItemText(t, inputItems[1]))
			assert.Equal(t, "Message 4", inputItemText(t, inputItems[2]))
		})
	}
}

func TestSessionSettingsLimitZero(t *testing.T) {
	ptrInt := func(v int) *int { return &v }

	dbPath := filepath.Join(t.TempDir(), "session_settings_limit_zero.db")
	session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
		SessionID:        "limit_zero_test",
		DBDataSourceName: dbPath,
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("test").WithModelInstance(model)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 1")},
	})
	_, err = agents.Runner{Config: agents.RunConfig{Session: session}}.Run(t.Context(), agent, "Message 1")
	require.NoError(t, err)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 2")},
	})
	_, err = agents.Runner{Config: agents.RunConfig{Session: session}}.Run(t.Context(), agent, "Message 2")
	require.NoError(t, err)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 3")},
	})
	_, err = agents.Runner{Config: agents.RunConfig{
		Session:         session,
		SessionSettings: &memory.SessionSettings{Limit: ptrInt(0)},
	}}.Run(t.Context(), agent, "Message 3")
	require.NoError(t, err)

	lastInput := model.LastTurnArgs.Input
	require.IsType(t, agents.InputItems{}, lastInput)
	inputItems := lastInput.(agents.InputItems)
	require.Len(t, inputItems, 1)
	assert.Equal(t, "Message 3", inputItemText(t, inputItems[0]))
}

func TestSessionSettingsLimitNilGetsAllHistory(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "session_settings_limit_none.db")
	session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
		SessionID:        "limit_none_test",
		DBDataSourceName: dbPath,
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("test").WithModelInstance(model)

	for i := 1; i <= 5; i++ {
		model.SetNextOutput(agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply " + strconv.Itoa(i))},
		})
		_, err := agents.Runner{Config: agents.RunConfig{Session: session}}.Run(t.Context(), agent, "Message "+strconv.Itoa(i))
		require.NoError(t, err)
	}

	allItems, err := session.GetItems(t.Context(), 0)
	require.NoError(t, err)
	assert.Len(t, allItems, 10)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 6")},
	})
	_, err = agents.Runner{Config: agents.RunConfig{
		Session:         session,
		SessionSettings: &memory.SessionSettings{Limit: nil},
	}}.Run(t.Context(), agent, "Message 6")
	require.NoError(t, err)

	lastInput := model.LastTurnArgs.Input
	require.IsType(t, agents.InputItems{}, lastInput)
	inputItems := lastInput.(agents.InputItems)
	require.Len(t, inputItems, 11)
	assert.Equal(t, "Message 1", inputItemText(t, inputItems[0]))
	assert.Equal(t, "Message 6", inputItemText(t, inputItems[10]))
}

func TestSessionSettingsLimitLargerThanHistory(t *testing.T) {
	ptrInt := func(v int) *int { return &v }

	dbPath := filepath.Join(t.TempDir(), "session_settings_limit_large.db")
	session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
		SessionID:        "limit_large_test",
		DBDataSourceName: dbPath,
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("test").WithModelInstance(model)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 1")},
	})
	_, err = agents.Runner{Config: agents.RunConfig{Session: session}}.Run(t.Context(), agent, "Message 1")
	require.NoError(t, err)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 2")},
	})
	_, err = agents.Runner{Config: agents.RunConfig{
		Session:         session,
		SessionSettings: &memory.SessionSettings{Limit: ptrInt(100)},
	}}.Run(t.Context(), agent, "Message 2")
	require.NoError(t, err)

	lastInput := model.LastTurnArgs.Input
	require.IsType(t, agents.InputItems{}, lastInput)
	inputItems := lastInput.(agents.InputItems)
	require.Len(t, inputItems, 3)
	assert.Equal(t, "Message 1", inputItemText(t, inputItems[0]))
	assert.Equal(t, "Reply 1", inputItemText(t, inputItems[1]))
	assert.Equal(t, "Message 2", inputItemText(t, inputItems[2]))
}

func TestSessionLimitZeroReturnsAllHistory(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "session_limit_zero.db")
	session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
		SessionID:        "limit_zero_test",
		DBDataSourceName: dbPath,
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("test").WithModelInstance(model)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 1")},
	})
	_, err = agents.Runner{Config: agents.RunConfig{Session: session}}.Run(t.Context(), agent, "Message 1")
	require.NoError(t, err)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 2")},
	})
	_, err = agents.Runner{Config: agents.RunConfig{Session: session}}.Run(t.Context(), agent, "Message 2")
	require.NoError(t, err)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 3")},
	})
	_, err = agents.Runner{Config: agents.RunConfig{Session: session, LimitMemory: 0}}.Run(t.Context(), agent, "Message 3")
	require.NoError(t, err)

	lastInput := model.LastTurnArgs.Input
	require.IsType(t, agents.InputItems{}, lastInput)
	inputItems := lastInput.(agents.InputItems)
	require.Len(t, inputItems, 5)
	assert.Equal(t, "Message 1", inputItemText(t, inputItems[0]))
	assert.Equal(t, "Message 3", inputItemText(t, inputItems[4]))
}

func TestSessionLimitLargerThanHistoryReturnsAllItems(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "session_limit_large.db")
	session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
		SessionID:        "limit_large_test",
		DBDataSourceName: dbPath,
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("test").WithModelInstance(model)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 1")},
	})
	_, err = agents.Runner{Config: agents.RunConfig{Session: session}}.Run(t.Context(), agent, "Message 1")
	require.NoError(t, err)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Reply 2")},
	})
	_, err = agents.Runner{Config: agents.RunConfig{Session: session, LimitMemory: 100}}.Run(t.Context(), agent, "Message 2")
	require.NoError(t, err)

	lastInput := model.LastTurnArgs.Input
	require.IsType(t, agents.InputItems{}, lastInput)
	inputItems := lastInput.(agents.InputItems)
	require.Len(t, inputItems, 3)
	assert.Equal(t, "Message 1", inputItemText(t, inputItems[0]))
	assert.Equal(t, "Reply 1", inputItemText(t, inputItems[1]))
	assert.Equal(t, "Message 2", inputItemText(t, inputItems[2]))
}

func inputItemText(t *testing.T, item agents.TResponseInputItem) string {
	t.Helper()

	raw, err := item.MarshalJSON()
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(raw, &payload))

	content, ok := payload["content"]
	require.True(t, ok)

	switch v := content.(type) {
	case string:
		return v
	case []any:
		if len(v) == 0 {
			return ""
		}
		entry, ok := v[0].(map[string]any)
		require.True(t, ok)
		text, _ := entry["text"].(string)
		return text
	default:
		require.FailNowf(t, "unexpected content shape", "%T", v)
		return ""
	}
}

func boolLabel(v bool) string {
	if v {
		return "true"
	}
	return "false"
}
