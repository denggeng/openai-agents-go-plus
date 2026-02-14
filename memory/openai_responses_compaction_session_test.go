package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockCompactionService struct {
	compactErr  error
	compactResp *responses.CompactedResponse
	calls       []responses.ResponseCompactParams
}

func (m *mockCompactionService) Compact(
	_ context.Context,
	body responses.ResponseCompactParams,
	_ ...option.RequestOption,
) (*responses.CompactedResponse, error) {
	m.calls = append(m.calls, body)
	if m.compactErr != nil {
		return nil, m.compactErr
	}
	return m.compactResp, nil
}

type mockSessionForCompaction struct {
	sessionID string
	items     []TResponseInputItem

	getErr   error
	addErr   error
	popErr   error
	clearErr error

	getCalls   []int
	addCalls   [][]TResponseInputItem
	popCalls   int
	clearCalls int
}

func (m *mockSessionForCompaction) SessionID(context.Context) string {
	return m.sessionID
}

func (m *mockSessionForCompaction) GetItems(_ context.Context, limit int) ([]TResponseInputItem, error) {
	m.getCalls = append(m.getCalls, limit)
	if m.getErr != nil {
		return nil, m.getErr
	}
	if limit <= 0 || limit >= len(m.items) {
		return append([]TResponseInputItem(nil), m.items...), nil
	}
	return append([]TResponseInputItem(nil), m.items[len(m.items)-limit:]...), nil
}

func (m *mockSessionForCompaction) AddItems(_ context.Context, items []TResponseInputItem) error {
	m.addCalls = append(m.addCalls, append([]TResponseInputItem(nil), items...))
	if m.addErr != nil {
		return m.addErr
	}
	m.items = append(m.items, items...)
	return nil
}

func (m *mockSessionForCompaction) PopItem(context.Context) (*TResponseInputItem, error) {
	m.popCalls++
	if m.popErr != nil {
		return nil, m.popErr
	}
	if len(m.items) == 0 {
		return nil, nil
	}
	v := m.items[len(m.items)-1]
	m.items = m.items[:len(m.items)-1]
	return &v, nil
}

func (m *mockSessionForCompaction) ClearSession(context.Context) error {
	m.clearCalls++
	if m.clearErr != nil {
		return m.clearErr
	}
	m.items = nil
	return nil
}

func makeUserMessageForCompactionTests(text string) TResponseInputItem {
	return responses.ResponseInputItemParamOfMessage(text, responses.EasyInputMessageRoleUser)
}

func makeAssistantMessageForCompactionTests(text string) TResponseInputItem {
	return responses.ResponseInputItemParamOfMessage(text, responses.EasyInputMessageRoleAssistant)
}

func makeCompactionItemForCompactionTests(encrypted string) TResponseInputItem {
	return responses.ResponseInputItemUnionParam{
		OfCompaction: &responses.ResponseCompactionItemParam{
			EncryptedContent: encrypted,
			Type:             constant.ValueOf[constant.Compaction](),
		},
	}
}

func mustOutputItemFromJSON(t *testing.T, raw string) responses.ResponseOutputItemUnion {
	t.Helper()
	var item responses.ResponseOutputItemUnion
	require.NoError(t, json.Unmarshal([]byte(raw), &item))
	return item
}

func TestIsOpenAIModelName(t *testing.T) {
	assert.True(t, IsOpenAIModelName("gpt-4.1"))
	assert.True(t, IsOpenAIModelName("gpt-5"))
	assert.True(t, IsOpenAIModelName("o3"))
	assert.True(t, IsOpenAIModelName("ft:gpt-4.1:org:proj:suffix"))
	assert.False(t, IsOpenAIModelName(""))
	assert.False(t, IsOpenAIModelName("claude-3"))
}

func TestSelectCompactionCandidateItems(t *testing.T) {
	items := []TResponseInputItem{
		makeUserMessageForCompactionTests("hello"),
		makeAssistantMessageForCompactionTests("hi"),
		makeCompactionItemForCompactionTests("encrypted"),
	}

	got := SelectCompactionCandidateItems(items)
	require.Len(t, got, 1)
	role := got[0].GetRole()
	require.NotNil(t, role)
	assert.Equal(t, "assistant", *role)
}

func TestNewOpenAIResponsesCompactionSessionValidatesModel(t *testing.T) {
	_, err := NewOpenAIResponsesCompactionSession(OpenAIResponsesCompactionSessionParams{
		SessionID:         "test",
		UnderlyingSession: &mockSessionForCompaction{},
		Model:             "claude-3",
		CompactionService: &mockCompactionService{},
	})
	require.Error(t, err)
	assert.ErrorContains(t, err, "unsupported model")
}

func TestRunCompactionRequiresResponseIDForPreviousMode(t *testing.T) {
	session, err := NewOpenAIResponsesCompactionSession(OpenAIResponsesCompactionSessionParams{
		SessionID:         "test",
		UnderlyingSession: &mockSessionForCompaction{},
		Model:             "gpt-4.1",
		CompactionMode:    OpenAIResponsesCompactionModePreviousResponseID,
		CompactionService: &mockCompactionService{},
	})
	require.NoError(t, err)

	err = session.RunCompaction(t.Context(), nil)
	require.Error(t, err)
	assert.ErrorContains(t, err, "requires a response_id")
}

func TestRunCompactionInputModeWithoutResponseID(t *testing.T) {
	underlying := &mockSessionForCompaction{
		items: []TResponseInputItem{
			makeUserMessageForCompactionTests("hello"),
			makeAssistantMessageForCompactionTests("world"),
		},
	}
	mockCompact := &mockCompactionService{
		compactResp: &responses.CompactedResponse{
			Output: []responses.ResponseOutputItemUnion{},
		},
	}
	session, err := NewOpenAIResponsesCompactionSession(OpenAIResponsesCompactionSessionParams{
		SessionID:         "test",
		UnderlyingSession: underlying,
		Model:             "gpt-4.1",
		CompactionMode:    OpenAIResponsesCompactionModeInput,
		CompactionService: mockCompact,
		ShouldTriggerCompaction: func(CompactionDecisionContext) bool {
			return true
		},
	})
	require.NoError(t, err)

	err = session.RunCompaction(t.Context(), &OpenAIResponsesCompactionArgs{
		Force: true,
	})
	require.NoError(t, err)
	require.Len(t, mockCompact.calls, 1)
	call := mockCompact.calls[0]
	assert.Equal(t, responses.ResponseCompactParamsModel("gpt-4.1"), call.Model)
	assert.False(t, call.PreviousResponseID.Valid())
	assert.Len(t, call.Input.OfResponseInputItemArray, 2)
}

func TestRunCompactionAutoUsesInputWhenStoreFalse(t *testing.T) {
	underlying := &mockSessionForCompaction{
		items: []TResponseInputItem{
			makeUserMessageForCompactionTests("hello"),
		},
	}
	mockCompact := &mockCompactionService{
		compactResp: &responses.CompactedResponse{
			Output: []responses.ResponseOutputItemUnion{},
		},
	}
	session, err := NewOpenAIResponsesCompactionSession(OpenAIResponsesCompactionSessionParams{
		SessionID:         "test",
		UnderlyingSession: underlying,
		Model:             "gpt-4.1",
		CompactionMode:    OpenAIResponsesCompactionModeAuto,
		CompactionService: mockCompact,
		ShouldTriggerCompaction: func(CompactionDecisionContext) bool {
			return true
		},
	})
	require.NoError(t, err)

	store := false
	err = session.RunCompaction(t.Context(), &OpenAIResponsesCompactionArgs{
		ResponseID: "resp-auto",
		Store:      &store,
		Force:      true,
	})
	require.NoError(t, err)
	require.Len(t, mockCompact.calls, 1)
	call := mockCompact.calls[0]
	assert.False(t, call.PreviousResponseID.Valid())
	assert.Len(t, call.Input.OfResponseInputItemArray, 1)
}

func TestRunCompactionSkipsWhenBelowThreshold(t *testing.T) {
	underlying := &mockSessionForCompaction{
		items: []TResponseInputItem{
			makeAssistantMessageForCompactionTests("a"),
			makeAssistantMessageForCompactionTests("b"),
		},
	}
	mockCompact := &mockCompactionService{}
	session, err := NewOpenAIResponsesCompactionSession(OpenAIResponsesCompactionSessionParams{
		SessionID:         "test",
		UnderlyingSession: underlying,
		Model:             "gpt-4.1",
		CompactionService: mockCompact,
	})
	require.NoError(t, err)

	err = session.RunCompaction(t.Context(), &OpenAIResponsesCompactionArgs{
		ResponseID: "resp-123",
	})
	require.NoError(t, err)
	assert.Empty(t, mockCompact.calls)
}

func TestRunCompactionExecutesWhenThresholdMet(t *testing.T) {
	underlying := &mockSessionForCompaction{}
	for i := 0; i < DefaultCompactionThreshold; i++ {
		underlying.items = append(underlying.items, makeAssistantMessageForCompactionTests(fmt.Sprintf("msg%d", i)))
	}
	mockCompact := &mockCompactionService{
		compactResp: &responses.CompactedResponse{
			Output: []responses.ResponseOutputItemUnion{
				mustOutputItemFromJSON(t, `{"id":"msg_1","type":"message","role":"user","status":"completed","content":[{"type":"input_text","text":"hello"}]}`),
				mustOutputItemFromJSON(t, `{"id":"cmp_1","type":"compaction","encrypted_content":"enc123"}`),
			},
		},
	}
	session, err := NewOpenAIResponsesCompactionSession(OpenAIResponsesCompactionSessionParams{
		SessionID:         "test",
		UnderlyingSession: underlying,
		Model:             "gpt-4.1",
		CompactionService: mockCompact,
	})
	require.NoError(t, err)

	err = session.RunCompaction(t.Context(), &OpenAIResponsesCompactionArgs{
		ResponseID: "resp-123",
	})
	require.NoError(t, err)

	require.Len(t, mockCompact.calls, 1)
	assert.True(t, mockCompact.calls[0].PreviousResponseID.Valid())
	assert.Equal(t, "resp-123", mockCompact.calls[0].PreviousResponseID.Value)
	assert.Equal(t, 1, underlying.clearCalls)
	require.Len(t, underlying.addCalls, 1)
	require.Len(t, underlying.addCalls[0], 2)
	typ := underlying.addCalls[0][1].GetType()
	require.NotNil(t, typ)
	assert.Equal(t, "compaction", *typ)
}

func TestRunCompactionPropagatesCompactError(t *testing.T) {
	underlying := &mockSessionForCompaction{
		items: []TResponseInputItem{
			makeAssistantMessageForCompactionTests("a"),
		},
	}
	mockCompact := &mockCompactionService{
		compactErr: errors.New("compact failed"),
	}
	session, err := NewOpenAIResponsesCompactionSession(OpenAIResponsesCompactionSessionParams{
		SessionID:         "test",
		UnderlyingSession: underlying,
		Model:             "gpt-4.1",
		CompactionService: mockCompact,
		ShouldTriggerCompaction: func(CompactionDecisionContext) bool {
			return true
		},
	})
	require.NoError(t, err)

	err = session.RunCompaction(t.Context(), &OpenAIResponsesCompactionArgs{
		ResponseID: "resp-123",
	})
	require.Error(t, err)
	assert.ErrorContains(t, err, "compact failed")
}
