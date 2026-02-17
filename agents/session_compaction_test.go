package agents

import (
	"context"
	"errors"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/memory"
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockCompactionAwareSession struct {
	items []memory.TResponseInputItem

	addErr   error
	runErr   error
	usageErr error

	addCalls           int
	runCompactionCalls []memory.OpenAIResponsesCompactionArgs
	storedUsageCalls   []*usage.Usage
}

func (m *mockCompactionAwareSession) SessionID(context.Context) string {
	return "session-1"
}

func (m *mockCompactionAwareSession) GetItems(context.Context, int) ([]memory.TResponseInputItem, error) {
	return append([]memory.TResponseInputItem(nil), m.items...), nil
}

func (m *mockCompactionAwareSession) AddItems(_ context.Context, items []memory.TResponseInputItem) error {
	m.addCalls++
	if m.addErr != nil {
		return m.addErr
	}
	m.items = append(m.items, items...)
	return nil
}

func (m *mockCompactionAwareSession) PopItem(context.Context) (*memory.TResponseInputItem, error) {
	if len(m.items) == 0 {
		return nil, nil
	}
	v := m.items[len(m.items)-1]
	m.items = m.items[:len(m.items)-1]
	return &v, nil
}

func (m *mockCompactionAwareSession) ClearSession(context.Context) error {
	m.items = nil
	return nil
}

func (m *mockCompactionAwareSession) RunCompaction(
	_ context.Context,
	args *memory.OpenAIResponsesCompactionArgs,
) error {
	if m.runErr != nil {
		return m.runErr
	}
	if args != nil {
		m.runCompactionCalls = append(m.runCompactionCalls, *args)
	} else {
		m.runCompactionCalls = append(m.runCompactionCalls, memory.OpenAIResponsesCompactionArgs{})
	}
	return nil
}

func (m *mockCompactionAwareSession) StoreRunUsage(_ context.Context, runUsage *usage.Usage) error {
	if m.usageErr != nil {
		return m.usageErr
	}
	m.storedUsageCalls = append(m.storedUsageCalls, runUsage)
	return nil
}

func TestSaveResultToSessionRunsCompactionForCompactionAwareSession(t *testing.T) {
	session := &mockCompactionAwareSession{}
	runner := Runner{
		Config: RunConfig{
			Session: session,
		},
	}

	result := &RunResult{
		NewItems: nil,
		RawResponses: []ModelResponse{
			{ResponseID: "resp-123"},
		},
	}

	err := runner.saveResultToSession(t.Context(), InputString("hello"), result, nil)
	require.NoError(t, err)
	assert.Equal(t, 1, session.addCalls)
	require.Len(t, session.runCompactionCalls, 1)
	assert.Equal(t, "resp-123", session.runCompactionCalls[0].ResponseID)
}

func TestSaveResultToSessionSkipsCompactionWithoutResponseID(t *testing.T) {
	session := &mockCompactionAwareSession{}
	runner := Runner{
		Config: RunConfig{
			Session: session,
		},
	}

	result := &RunResult{
		NewItems: nil,
		RawResponses: []ModelResponse{
			{ResponseID: ""},
		},
	}

	err := runner.saveResultToSession(t.Context(), InputString("hello"), result, nil)
	require.NoError(t, err)
	assert.Equal(t, 1, session.addCalls)
	assert.Empty(t, session.runCompactionCalls)
}

func TestSaveResultToSessionPropagatesCompactionError(t *testing.T) {
	session := &mockCompactionAwareSession{
		runErr: errors.New("compact failed"),
	}
	runner := Runner{
		Config: RunConfig{
			Session: session,
		},
	}

	result := &RunResult{
		NewItems: []RunItem{
			MessageOutputItem{
				RawItem: responses.ResponseOutputMessage{
					ID:     "msg-1",
					Role:   constant.ValueOf[constant.Assistant](),
					Type:   constant.ValueOf[constant.Message](),
					Status: responses.ResponseOutputMessageStatusCompleted,
					Content: []responses.ResponseOutputMessageContentUnion{
						{Type: "output_text", Text: "hello"},
					},
				},
				Type: "message_output_item",
			},
		},
		RawResponses: []ModelResponse{
			{ResponseID: "resp-123"},
		},
	}

	err := runner.saveResultToSession(t.Context(), InputString("hello"), result, nil)
	require.Error(t, err)
	assert.ErrorContains(t, err, "failed to run session compaction")
	assert.ErrorContains(t, err, "compact failed")
}

func TestSaveResultToSessionStoresRunUsageForUsageTrackingSession(t *testing.T) {
	session := &mockCompactionAwareSession{}
	runner := Runner{
		Config: RunConfig{
			Session: session,
		},
	}

	result := &RunResult{
		RawResponses: []ModelResponse{
			{Usage: &usage.Usage{
				Requests:     1,
				InputTokens:  5,
				OutputTokens: 3,
				TotalTokens:  8,
			}},
			{Usage: &usage.Usage{
				Requests:     2,
				InputTokens:  4,
				OutputTokens: 6,
				TotalTokens:  10,
			}},
		},
	}

	err := runner.saveResultToSession(t.Context(), InputString("hello"), result, nil)
	require.NoError(t, err)
	require.Len(t, session.storedUsageCalls, 1)
	require.NotNil(t, session.storedUsageCalls[0])
	assert.Equal(t, uint64(3), session.storedUsageCalls[0].Requests)
	assert.Equal(t, uint64(9), session.storedUsageCalls[0].InputTokens)
	assert.Equal(t, uint64(9), session.storedUsageCalls[0].OutputTokens)
	assert.Equal(t, uint64(18), session.storedUsageCalls[0].TotalTokens)
}

func TestSaveResultToSessionPropagatesStoreRunUsageError(t *testing.T) {
	session := &mockCompactionAwareSession{
		usageErr: errors.New("store usage failed"),
	}
	runner := Runner{
		Config: RunConfig{
			Session: session,
		},
	}

	result := &RunResult{
		RawResponses: []ModelResponse{
			{Usage: &usage.Usage{
				Requests:     1,
				InputTokens:  5,
				OutputTokens: 3,
				TotalTokens:  8,
			}},
		},
	}

	err := runner.saveResultToSession(t.Context(), InputString("hello"), result, nil)
	require.Error(t, err)
	assert.ErrorContains(t, err, "failed to store run usage")
	assert.ErrorContains(t, err, "store usage failed")
}
