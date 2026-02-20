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
	"fmt"
	"path/filepath"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/denggeng/openai-agents-go-plus/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAgentSession(t *testing.T) {
	runAgent := func(t *testing.T, streaming bool, session memory.Session, agent *agents.Agent, input string) any {
		t.Helper()

		runner := agents.Runner{
			Config: agents.RunConfig{
				Session: session,
			},
		}

		var finalOutput any
		if streaming {
			result, err := runner.RunStreamed(t.Context(), agent, input)
			require.NoError(t, err)
			err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
			require.NoError(t, err)
			finalOutput = result.FinalOutput()
		} else {
			result, err := runner.Run(t.Context(), agent, input)
			require.NoError(t, err)
			finalOutput = result.FinalOutput
		}
		return finalOutput
	}

	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming %v", streaming), func(t *testing.T) {
			t.Run("basic functionality", func(t *testing.T) {
				// Test basic session memory functionality with SQLite backend.
				session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
					SessionID:        "test",
					DBDataSourceName: filepath.Join(t.TempDir(), "test.db"),
				})
				require.NoError(t, err)
				t.Cleanup(func() { assert.NoError(t, session.Close()) })

				model := agentstesting.NewFakeModel(false, nil)
				agent := agents.New("test").WithModelInstance(model)

				// First turn
				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("San Francisco")},
				})
				result1 := runAgent(t, streaming, session, agent, "What city is the Golden Gate Bridge in?")
				assert.Equal(t, "San Francisco", result1)

				// Second turn - should have conversation history
				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("California")},
				})
				result2 := runAgent(t, streaming, session, agent, "What state is it in?")
				assert.Equal(t, "California", result2)

				// Verify that the input to the second turn includes the previous conversation
				// The model should have received the full conversation history
				lastInput := model.LastTurnArgs.Input
				require.IsType(t, agents.InputItems{}, lastInput)
				assert.Len(t, lastInput.(agents.InputItems), 3)
			})

			t.Run("no session", func(t *testing.T) {
				// Test that session memory is disabled when Session is nil.
				model := agentstesting.NewFakeModel(false, nil)
				agent := agents.New("test").WithModelInstance(model)

				// First turn (no session parameters = disabled)
				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Hello")},
				})
				result1 := runAgent(t, streaming, nil, agent, "Hi there")
				assert.Equal(t, "Hello", result1)

				// Second turn - should NOT have conversation history
				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("I don't remember")},
				})
				result2 := runAgent(t, streaming, nil, agent, "Do you remember what I said?")
				assert.Equal(t, "I don't remember", result2)

				// Verify that the input to the second turn is just the current message
				lastInput := model.LastTurnArgs.Input
				require.IsType(t, agents.InputItems{}, lastInput)
				assert.Len(t, lastInput.(agents.InputItems), 1)
			})

			t.Run("different sessions", func(t *testing.T) {
				// Test that different session IDs maintain separate conversation histories.
				dbPath := filepath.Join(t.TempDir(), "test.db")

				model := agentstesting.NewFakeModel(false, nil)
				agent := agents.New("test").WithModelInstance(model)

				// Session 1
				session1, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
					SessionID:        "session_1",
					DBDataSourceName: dbPath,
				})
				require.NoError(t, err)
				t.Cleanup(func() { assert.NoError(t, session1.Close()) })

				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("I like cats")},
				})
				result1 := runAgent(t, streaming, session1, agent, "I like cats")
				assert.Equal(t, "I like cats", result1)

				// Session 2 - different session
				session2, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
					SessionID:        "session_2",
					DBDataSourceName: dbPath,
				})
				require.NoError(t, err)
				t.Cleanup(func() { assert.NoError(t, session2.Close()) })

				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("I like dogs")},
				})
				result2 := runAgent(t, streaming, session2, agent, "I like dogs")
				assert.Equal(t, "I like dogs", result2)

				// Back to Session 1 - should remember cats, not dogs
				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Yes, you mentioned cats")},
				})
				result3 := runAgent(t, streaming, session1, agent, "What did I say I like?")
				assert.Equal(t, "Yes, you mentioned cats", result3)

				lastInput := model.LastTurnArgs.Input
				require.IsType(t, agents.InputItems{}, lastInput)
				require.Len(t, lastInput.(agents.InputItems), 3)
				assert.Equal(t, "I like cats", lastInput.(agents.InputItems)[0].OfMessage.Content.OfString.Value)

				// Back to Session 2 - should remember dogs, not cats
				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Yes, you mentioned dogs")},
				})
				result4 := runAgent(t, streaming, session2, agent, "What did I say I like?")
				assert.Equal(t, "Yes, you mentioned dogs", result4)

				lastInput = model.LastTurnArgs.Input
				require.IsType(t, agents.InputItems{}, lastInput)
				require.Len(t, lastInput.(agents.InputItems), 3)
				assert.Equal(t, "I like dogs", lastInput.(agents.InputItems)[0].OfMessage.Content.OfString.Value)
			})

		})
	}
}

func TestSessionMemoryAppendsListInputByDefault(t *testing.T) {
	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming %v", streaming), func(t *testing.T) {
			session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
				SessionID:        "test_validation_parametrized",
				DBDataSourceName: filepath.Join(t.TempDir(), "test_validation.db"),
			})
			require.NoError(t, err)
			t.Cleanup(func() { assert.NoError(t, session.Close()) })

			model := agentstesting.NewFakeModel(false, nil)
			agent := agents.New("test").WithModelInstance(model)

			initialHistory := []agents.TResponseInputItem{
				agentstesting.GetTextInputItem("Earlier message"),
				inputItemFromPayload(t, map[string]any{
					"type":    "message",
					"role":    "assistant",
					"content": "Saved reply",
				}),
			}
			require.NoError(t, session.AddItems(t.Context(), initialHistory))

			listInput := []agents.TResponseInputItem{
				agentstesting.GetTextInputItem("Test message"),
			}

			model.SetNextOutput(agentstesting.FakeModelTurnOutput{
				Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("This should run")},
			})

			if streaming {
				result, err := agents.Runner{Config: agents.RunConfig{Session: session}}.RunInputsStreamed(t.Context(), agent, listInput)
				require.NoError(t, err)
				require.NoError(t, result.StreamEvents(func(agents.StreamEvent) error { return nil }))
			} else {
				_, err := agents.Runner{Config: agents.RunConfig{Session: session}}.RunInputs(t.Context(), agent, listInput)
				require.NoError(t, err)
			}

			lastInput := agents.ItemHelpers().InputToNewInputList(model.LastTurnArgs.Input)
			require.Len(t, lastInput, 3)
			assert.Equal(t, "Earlier message", inputItemContentText(t, lastInput[0]))
			assert.Equal(t, "Saved reply", inputItemContentText(t, lastInput[1]))
			assert.Equal(t, "Test message", inputItemContentText(t, lastInput[2]))
		})
	}
}

func TestSessionCallbackPreparedInput(t *testing.T) {
	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming %v", streaming), func(t *testing.T) {
			session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
				SessionID:        "session_callback_test",
				DBDataSourceName: filepath.Join(t.TempDir(), "session_callback.db"),
			})
			require.NoError(t, err)
			t.Cleanup(func() { assert.NoError(t, session.Close()) })

			model := agentstesting.NewFakeModel(false, nil)
			agent := agents.New("test").WithModelInstance(model)

			initialHistory := []agents.TResponseInputItem{
				agentstesting.GetTextInputItem("Hello there."),
				inputItemFromPayload(t, map[string]any{
					"type":    "message",
					"role":    "assistant",
					"content": "Hi, I'm here to assist you.",
				}),
			}
			require.NoError(t, session.AddItems(t.Context(), initialHistory))

			callback := func(history []agents.TResponseInputItem, newInput []agents.TResponseInputItem) ([]agents.TResponseInputItem, error) {
				filtered := make([]agents.TResponseInputItem, 0, len(history)+len(newInput))
				for _, item := range history {
					if inputItemRole(t, item) == "user" {
						filtered = append(filtered, item)
					}
				}
				filtered = append(filtered, newInput...)
				return filtered, nil
			}

			newTurnInput := []agents.TResponseInputItem{
				agentstesting.GetTextInputItem("What your name?"),
			}

			model.SetNextOutput(agentstesting.FakeModelTurnOutput{
				Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("I'm gpt-4o")},
			})

			runner := agents.Runner{Config: agents.RunConfig{
				Session:              session,
				SessionInputCallback: callback,
			}}

			if streaming {
				result, err := runner.RunInputsStreamed(t.Context(), agent, newTurnInput)
				require.NoError(t, err)
				require.NoError(t, result.StreamEvents(func(agents.StreamEvent) error { return nil }))
			} else {
				_, err := runner.RunInputs(t.Context(), agent, newTurnInput)
				require.NoError(t, err)
			}

			lastInput := agents.ItemHelpers().InputToNewInputList(model.LastTurnArgs.Input)
			require.Len(t, lastInput, 2)
			assert.Equal(t, "Hello there.", inputItemContentText(t, lastInput[0]))
			assert.Equal(t, "What your name?", inputItemContentText(t, lastInput[1]))
		})
	}
}

func inputItemContentText(t *testing.T, item agents.TResponseInputItem) string {
	t.Helper()
	raw, err := item.MarshalJSON()
	require.NoError(t, err)
	var payload map[string]any
	require.NoError(t, json.Unmarshal(raw, &payload))
	content := payload["content"]
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

func inputItemRole(t *testing.T, item agents.TResponseInputItem) string {
	t.Helper()
	raw, err := item.MarshalJSON()
	require.NoError(t, err)
	var payload map[string]any
	require.NoError(t, json.Unmarshal(raw, &payload))
	role, _ := payload["role"].(string)
	return role
}
