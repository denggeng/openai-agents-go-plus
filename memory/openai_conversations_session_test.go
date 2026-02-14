package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/openai/openai-go/v3/conversations"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockConversationsService struct {
	newID string

	newErr                error
	newCalls              int
	newBodies             []conversations.ConversationNewParams
	deleteErr             error
	deleteCalls           int
	deleteConversationIDs []string
}

func (m *mockConversationsService) New(
	_ context.Context,
	body conversations.ConversationNewParams,
	_ ...option.RequestOption,
) (*conversations.Conversation, error) {
	m.newCalls++
	m.newBodies = append(m.newBodies, body)
	if m.newErr != nil {
		return nil, m.newErr
	}
	return &conversations.Conversation{ID: m.newID}, nil
}

func (m *mockConversationsService) Delete(
	_ context.Context,
	conversationID string,
	_ ...option.RequestOption,
) (*conversations.ConversationDeletedResource, error) {
	m.deleteCalls++
	m.deleteConversationIDs = append(m.deleteConversationIDs, conversationID)
	if m.deleteErr != nil {
		return nil, m.deleteErr
	}
	return &conversations.ConversationDeletedResource{
		ID:      conversationID,
		Deleted: true,
	}, nil
}

type mockConversationPager struct {
	items []conversations.ConversationItemUnion
	index int
	cur   conversations.ConversationItemUnion
	err   error
}

func (m *mockConversationPager) Next() bool {
	if m.index >= len(m.items) {
		return false
	}
	m.cur = m.items[m.index]
	m.index++
	return true
}

func (m *mockConversationPager) Current() conversations.ConversationItemUnion {
	return m.cur
}

func (m *mockConversationPager) Err() error {
	return m.err
}

type mockConversationItemsService struct {
	newErr    error
	newCalls  int
	newInputs []struct {
		conversationID string
		params         conversations.ItemNewParams
	}

	listCalls  int
	listInputs []struct {
		conversationID string
		query          conversations.ItemListParams
	}
	listItems []conversations.ConversationItemUnion
	listErr   error

	deleteErr    error
	deleteCalls  int
	deleteInputs []struct {
		conversationID string
		itemID         string
	}
}

func (m *mockConversationItemsService) New(
	_ context.Context,
	conversationID string,
	params conversations.ItemNewParams,
	_ ...option.RequestOption,
) (*conversations.ConversationItemList, error) {
	m.newCalls++
	m.newInputs = append(m.newInputs, struct {
		conversationID string
		params         conversations.ItemNewParams
	}{
		conversationID: conversationID,
		params:         params,
	})
	if m.newErr != nil {
		return nil, m.newErr
	}
	return &conversations.ConversationItemList{}, nil
}

func (m *mockConversationItemsService) ListAutoPaging(
	_ context.Context,
	conversationID string,
	query conversations.ItemListParams,
	_ ...option.RequestOption,
) conversationItemPager {
	m.listCalls++
	m.listInputs = append(m.listInputs, struct {
		conversationID string
		query          conversations.ItemListParams
	}{
		conversationID: conversationID,
		query:          query,
	})
	return &mockConversationPager{
		items: m.listItems,
		err:   m.listErr,
	}
}

func (m *mockConversationItemsService) Delete(
	_ context.Context,
	conversationID string,
	itemID string,
	_ ...option.RequestOption,
) (*conversations.Conversation, error) {
	m.deleteCalls++
	m.deleteInputs = append(m.deleteInputs, struct {
		conversationID string
		itemID         string
	}{
		conversationID: conversationID,
		itemID:         itemID,
	})
	if m.deleteErr != nil {
		return nil, m.deleteErr
	}
	return &conversations.Conversation{ID: conversationID}, nil
}

func makeInputMessage(text string) TResponseInputItem {
	return TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Role:    responses.EasyInputMessageRoleUser,
			Type:    responses.EasyInputMessageTypeMessage,
			Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(text)},
		},
	}
}

func mustConversationItem(t *testing.T, id string, role string, text string) conversations.ConversationItemUnion {
	t.Helper()
	raw := fmt.Sprintf(
		`{"id":%q,"type":"message","role":%q,"content":[{"type":"input_text","text":%q}]}`,
		id, role, text,
	)
	var item conversations.ConversationItemUnion
	require.NoError(t, json.Unmarshal([]byte(raw), &item))
	return item
}

func requireItemText(t *testing.T, item TResponseInputItem) string {
	t.Helper()
	data, err := item.MarshalJSON()
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(data, &payload))

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

func TestStartOpenAIConversationsSession(t *testing.T) {
	ctx := t.Context()
	conversationsSvc := &mockConversationsService{newID: "test_conversation_id"}

	id, err := startOpenAIConversationsSession(ctx, conversationsSvc)
	require.NoError(t, err)
	assert.Equal(t, "test_conversation_id", id)
	require.Len(t, conversationsSvc.newBodies, 1)
	assert.Empty(t, conversationsSvc.newBodies[0].Items)
}

func TestOpenAIConversationsSessionAddItemsUsesExistingConversationID(t *testing.T) {
	ctx := t.Context()
	conversationsSvc := &mockConversationsService{newID: "unused"}
	itemsSvc := &mockConversationItemsService{}
	session := NewOpenAIConversationsSession(OpenAIConversationsSessionParams{
		ConversationID:   "existing_id",
		conversationsSvc: conversationsSvc,
		itemsSvc:         itemsSvc,
	})

	items := []TResponseInputItem{
		makeInputMessage("hello"),
		makeInputMessage("world"),
	}

	err := session.AddItems(ctx, items)
	require.NoError(t, err)
	assert.Equal(t, 0, conversationsSvc.newCalls)
	require.Len(t, itemsSvc.newInputs, 1)
	assert.Equal(t, "existing_id", itemsSvc.newInputs[0].conversationID)
	assert.Equal(t, items, itemsSvc.newInputs[0].params.Items)
}

func TestOpenAIConversationsSessionAddItemsCreatesSessionIDLazily(t *testing.T) {
	ctx := t.Context()
	conversationsSvc := &mockConversationsService{newID: "created_id"}
	itemsSvc := &mockConversationItemsService{}
	session := NewOpenAIConversationsSession(OpenAIConversationsSessionParams{
		conversationsSvc: conversationsSvc,
		itemsSvc:         itemsSvc,
	})

	err := session.AddItems(ctx, []TResponseInputItem{makeInputMessage("hello")})
	require.NoError(t, err)
	assert.Equal(t, 1, conversationsSvc.newCalls)
	require.Len(t, itemsSvc.newInputs, 1)
	assert.Equal(t, "created_id", itemsSvc.newInputs[0].conversationID)
	assert.Equal(t, "created_id", session.SessionID(ctx))
}

func TestOpenAIConversationsSessionGetItemsNoLimitAscending(t *testing.T) {
	ctx := t.Context()
	itemsSvc := &mockConversationItemsService{
		listItems: []conversations.ConversationItemUnion{
			mustConversationItem(t, "item_1", "user", "hello"),
			mustConversationItem(t, "item_2", "assistant", "hi"),
		},
	}
	session := NewOpenAIConversationsSession(OpenAIConversationsSessionParams{
		ConversationID:   "test_id",
		conversationsSvc: &mockConversationsService{},
		itemsSvc:         itemsSvc,
	})

	items, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	require.Len(t, items, 2)
	assert.Equal(t, "hello", requireItemText(t, items[0]))
	assert.Equal(t, "hi", requireItemText(t, items[1]))
	require.Len(t, itemsSvc.listInputs, 1)
	assert.Equal(t, "test_id", itemsSvc.listInputs[0].conversationID)
	assert.Equal(t, conversations.ItemListParamsOrderAsc, itemsSvc.listInputs[0].query.Order)
	assert.False(t, itemsSvc.listInputs[0].query.Limit.Valid())
}

func TestOpenAIConversationsSessionGetItemsWithLimitUsesDescAndReverses(t *testing.T) {
	ctx := t.Context()
	itemsSvc := &mockConversationItemsService{
		// Simulate API descending order: newest first.
		listItems: []conversations.ConversationItemUnion{
			mustConversationItem(t, "item_3", "assistant", "three"),
			mustConversationItem(t, "item_2", "user", "two"),
		},
	}
	session := NewOpenAIConversationsSession(OpenAIConversationsSessionParams{
		ConversationID:   "test_id",
		conversationsSvc: &mockConversationsService{},
		itemsSvc:         itemsSvc,
	})

	items, err := session.GetItems(ctx, 2)
	require.NoError(t, err)
	require.Len(t, items, 2)
	assert.Equal(t, "two", requireItemText(t, items[0]))
	assert.Equal(t, "three", requireItemText(t, items[1]))
	require.Len(t, itemsSvc.listInputs, 1)
	assert.Equal(t, conversations.ItemListParamsOrderDesc, itemsSvc.listInputs[0].query.Order)
	assert.True(t, itemsSvc.listInputs[0].query.Limit.Valid())
	assert.EqualValues(t, 2, itemsSvc.listInputs[0].query.Limit.Value)
}

func TestOpenAIConversationsSessionPopItemWithItems(t *testing.T) {
	ctx := t.Context()
	itemsSvc := &mockConversationItemsService{
		listItems: []conversations.ConversationItemUnion{
			mustConversationItem(t, "item_123", "assistant", "latest"),
		},
	}
	session := NewOpenAIConversationsSession(OpenAIConversationsSessionParams{
		ConversationID:   "test_id",
		conversationsSvc: &mockConversationsService{},
		itemsSvc:         itemsSvc,
	})

	popped, err := session.PopItem(ctx)
	require.NoError(t, err)
	require.NotNil(t, popped)
	assert.Equal(t, "latest", requireItemText(t, *popped))
	require.Len(t, itemsSvc.deleteInputs, 1)
	assert.Equal(t, "test_id", itemsSvc.deleteInputs[0].conversationID)
	assert.Equal(t, "item_123", itemsSvc.deleteInputs[0].itemID)
}

func TestOpenAIConversationsSessionPopItemEmptySession(t *testing.T) {
	ctx := t.Context()
	itemsSvc := &mockConversationItemsService{
		listItems: nil,
	}
	session := NewOpenAIConversationsSession(OpenAIConversationsSessionParams{
		ConversationID:   "test_id",
		conversationsSvc: &mockConversationsService{},
		itemsSvc:         itemsSvc,
	})

	popped, err := session.PopItem(ctx)
	require.NoError(t, err)
	assert.Nil(t, popped)
	assert.Empty(t, itemsSvc.deleteInputs)
}

func TestOpenAIConversationsSessionClearSession(t *testing.T) {
	ctx := t.Context()
	conversationsSvc := &mockConversationsService{}
	session := NewOpenAIConversationsSession(OpenAIConversationsSessionParams{
		ConversationID:   "test_id",
		conversationsSvc: conversationsSvc,
		itemsSvc:         &mockConversationItemsService{},
	})

	err := session.ClearSession(ctx)
	require.NoError(t, err)
	require.Len(t, conversationsSvc.deleteConversationIDs, 1)
	assert.Equal(t, "test_id", conversationsSvc.deleteConversationIDs[0])
	assert.Equal(t, "", session.SessionID(ctx))
}

func TestOpenAIConversationsSessionSessionIDLazyCreationConsistency(t *testing.T) {
	ctx := t.Context()
	conversationsSvc := &mockConversationsService{newID: "lazy_id"}
	session := NewOpenAIConversationsSession(OpenAIConversationsSessionParams{
		conversationsSvc: conversationsSvc,
		itemsSvc:         &mockConversationItemsService{},
	})

	id1, err := session.ensureSessionID(ctx)
	require.NoError(t, err)
	id2, err := session.ensureSessionID(ctx)
	require.NoError(t, err)
	id3, err := session.ensureSessionID(ctx)
	require.NoError(t, err)

	assert.Equal(t, "lazy_id", id1)
	assert.Equal(t, id1, id2)
	assert.Equal(t, id2, id3)
	assert.Equal(t, 1, conversationsSvc.newCalls)
}

func TestOpenAIConversationsSessionCreateFailure(t *testing.T) {
	ctx := t.Context()
	conversationsSvc := &mockConversationsService{newErr: errors.New("API error")}
	session := NewOpenAIConversationsSession(OpenAIConversationsSessionParams{
		conversationsSvc: conversationsSvc,
		itemsSvc:         &mockConversationItemsService{},
	})

	_, err := session.GetItems(ctx, 0)
	require.Error(t, err)
	assert.ErrorContains(t, err, "API error")
}
