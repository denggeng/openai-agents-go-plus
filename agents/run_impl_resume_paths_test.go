package agents_test

import (
	"context"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type simpleListSession struct {
	items []agents.TResponseInputItem
}

func (s *simpleListSession) SessionID(context.Context) string {
	return "simple-session"
}

func (s *simpleListSession) GetItems(_ context.Context, limit int) ([]agents.TResponseInputItem, error) {
	if limit <= 0 || limit >= len(s.items) {
		return append([]agents.TResponseInputItem(nil), s.items...), nil
	}
	start := len(s.items) - limit
	if start < 0 {
		start = 0
	}
	return append([]agents.TResponseInputItem(nil), s.items[start:]...), nil
}

func (s *simpleListSession) AddItems(_ context.Context, items []agents.TResponseInputItem) error {
	s.items = append(s.items, items...)
	return nil
}

func (s *simpleListSession) PopItem(context.Context) (*agents.TResponseInputItem, error) {
	if len(s.items) == 0 {
		return nil, nil
	}
	item := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return &item, nil
}

func (s *simpleListSession) ClearSession(context.Context) error {
	s.items = nil
	return nil
}

func TestResumedSessionPersistenceUsesSavedCount(t *testing.T) {
	session := &simpleListSession{}
	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{
			Value: []agents.TResponseOutputItem{
				{
					ID:        "call-1",
					CallID:    "call-1",
					Name:      "test_tool",
					Type:      "function_call",
					Arguments: "{}",
				},
				{
					ID:        "call-1",
					CallID:    "call-1",
					Name:      "test_tool",
					Type:      "function_call",
					Arguments: "{}",
				},
			},
		},
		{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetTextMessage("done"),
			},
		},
	})

	tool := agentstesting.GetFunctionTool("test_tool", "ok")
	agent := agents.New("resume-agent").WithModelInstance(model).WithTools(tool)
	state := agents.RunState{
		SchemaVersion:    agents.CurrentRunStateSchemaVersion,
		CurrentTurn:      0,
		MaxTurns:         3,
		CurrentAgentName: agent.Name,
		OriginalInput: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem("input"),
		},
	}

	runner := agents.Runner{Config: agents.RunConfig{Session: session}}
	result, err := runner.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	require.NotNil(t, result)

	items, err := session.GetItems(t.Context(), 0)
	require.NoError(t, err)
	assert.Equal(t, 1, countSessionItems(items, "function_call", "call-1"))
	assert.Equal(t, 1, countSessionItems(items, "function_call_output", "call-1"))
}

func TestResumedRunAgainResetsPersistedCount(t *testing.T) {
	session := &simpleListSession{}
	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{
			Value: []agents.TResponseOutputItem{
				{
					ID:        "call-1",
					CallID:    "call-1",
					Name:      "test_tool",
					Type:      "function_call",
					Arguments: "{}",
				},
			},
		},
		{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetTextMessage("done"),
			},
		},
	})

	tool := agentstesting.GetFunctionTool("test_tool", "tool_output")
	agent := agents.New("resume-agent").WithModelInstance(model).WithTools(tool)

	state := agents.RunState{
		SchemaVersion:                 agents.CurrentRunStateSchemaVersion,
		CurrentTurn:                   0,
		MaxTurns:                      3,
		CurrentAgentName:              agent.Name,
		CurrentTurnPersistedItemCount: 1,
		OriginalInput: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem("input"),
		},
	}

	runner := agents.Runner{Config: agents.RunConfig{Session: session}}
	result, err := runner.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "done", result.FinalOutput)

	items, err := session.GetItems(t.Context(), 0)
	require.NoError(t, err)
	assert.Equal(t, 1, countSessionItems(items, "function_call", "call-1"))
}

func TestResumedApprovalDoesNotDuplicateSessionItems(t *testing.T) {
	session := &simpleListSession{}
	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{
			Value: []agents.TResponseOutputItem{
				{
					ID:        "call-resume",
					CallID:    "call-resume",
					Name:      "test_tool",
					Type:      "function_call",
					Arguments: "{}",
				},
			},
		},
		{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetTextMessage("done"),
			},
		},
	})

	tool := agentstesting.GetFunctionTool("test_tool", "tool_result")
	tool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()

	agent := agents.New("resume-agent").WithModelInstance(model).WithTools(tool)
	runner := agents.Runner{Config: agents.RunConfig{Session: session}}

	first, err := runner.Run(t.Context(), agent, "Use test_tool")
	require.NoError(t, err)
	require.NotNil(t, first)
	require.NotEmpty(t, first.Interruptions)

	state := agents.NewRunStateFromResult(*first, 1, 2)
	require.NoError(t, state.ApproveTool(first.Interruptions[0]))

	resumed, err := runner.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	require.NotNil(t, resumed)
	assert.Equal(t, "done", resumed.FinalOutput)

	items, err := session.GetItems(t.Context(), 0)
	require.NoError(t, err)
	assert.Equal(t, 1, countSessionItems(items, "function_call", "call-resume"))
	assert.Equal(t, 1, countSessionItems(items, "function_call_output", "call-resume"))
}

func countSessionItems(items []agents.TResponseInputItem, itemType, callID string) int {
	count := 0
	for _, item := range items {
		itemTypeValue := item.GetType()
		if itemTypeValue == nil || *itemTypeValue != itemType {
			continue
		}
		if callID != "" {
			itemCallID := item.GetCallID()
			if itemCallID == nil || *itemCallID != callID {
				continue
			}
		}
		count++
	}
	return count
}
