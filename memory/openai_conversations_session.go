package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/conversations"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
)

type openAIConversationsService interface {
	New(
		ctx context.Context,
		body conversations.ConversationNewParams,
		opts ...option.RequestOption,
	) (*conversations.Conversation, error)

	Delete(
		ctx context.Context,
		conversationID string,
		opts ...option.RequestOption,
	) (*conversations.ConversationDeletedResource, error)
}

type conversationItemPager interface {
	Next() bool
	Current() conversations.ConversationItemUnion
	Err() error
}

type openAIConversationItemsService interface {
	New(
		ctx context.Context,
		conversationID string,
		params conversations.ItemNewParams,
		opts ...option.RequestOption,
	) (*conversations.ConversationItemList, error)

	ListAutoPaging(
		ctx context.Context,
		conversationID string,
		query conversations.ItemListParams,
		opts ...option.RequestOption,
	) conversationItemPager

	Delete(
		ctx context.Context,
		conversationID string,
		itemID string,
		opts ...option.RequestOption,
	) (*conversations.Conversation, error)
}

type openAIConversationItemsServiceAdapter struct {
	svc *conversations.ItemService
}

func (a openAIConversationItemsServiceAdapter) New(
	ctx context.Context,
	conversationID string,
	params conversations.ItemNewParams,
	opts ...option.RequestOption,
) (*conversations.ConversationItemList, error) {
	return a.svc.New(ctx, conversationID, params, opts...)
}

func (a openAIConversationItemsServiceAdapter) ListAutoPaging(
	ctx context.Context,
	conversationID string,
	query conversations.ItemListParams,
	opts ...option.RequestOption,
) conversationItemPager {
	return a.svc.ListAutoPaging(ctx, conversationID, query, opts...)
}

func (a openAIConversationItemsServiceAdapter) Delete(
	ctx context.Context,
	conversationID string,
	itemID string,
	opts ...option.RequestOption,
) (*conversations.Conversation, error) {
	return a.svc.Delete(ctx, conversationID, itemID, opts...)
}

// OpenAIConversationsSessionParams configures a conversation-backed Session.
type OpenAIConversationsSessionParams struct {
	ConversationID string
	Client         *openai.Client

	// Optional session settings (e.g., default history limit).
	SessionSettings *SessionSettings

	// Internal dependency injection hooks used in tests.
	conversationsSvc openAIConversationsService
	itemsSvc         openAIConversationItemsService
}

// OpenAIConversationsSession stores chat history directly in OpenAI Conversations.
type OpenAIConversationsSession struct {
	sessionID string

	conversationsSvc openAIConversationsService
	itemsSvc         openAIConversationItemsService
	sessionSettings  *SessionSettings
	mu               sync.Mutex
}

var newDefaultOpenAIClient = func() openai.Client {
	return openai.NewClient()
}

// NewOpenAIConversationsSession creates a Session backed by OpenAI Conversations.
func NewOpenAIConversationsSession(params OpenAIConversationsSessionParams) *OpenAIConversationsSession {
	conversationsSvc := params.conversationsSvc
	itemsSvc := params.itemsSvc

	if conversationsSvc == nil || itemsSvc == nil {
		client := params.Client
		if client == nil {
			c := newDefaultOpenAIClient()
			client = &c
		}

		if conversationsSvc == nil {
			conversationsSvc = &client.Conversations
		}
		if itemsSvc == nil {
			itemsSvc = openAIConversationItemsServiceAdapter{
				svc: &client.Conversations.Items,
			}
		}
	}

	settings := params.SessionSettings
	if settings == nil {
		settings = &SessionSettings{}
	}
	return &OpenAIConversationsSession{
		sessionID:        params.ConversationID,
		conversationsSvc: conversationsSvc,
		itemsSvc:         itemsSvc,
		sessionSettings:  settings,
	}
}

// StartOpenAIConversationsSession creates a conversation and returns its ID.
// When client is nil, it uses the default client.
func StartOpenAIConversationsSession(ctx context.Context, client *openai.Client) (string, error) {
	if client == nil {
		c := newDefaultOpenAIClient()
		client = &c
	}
	return startOpenAIConversationsSession(ctx, &client.Conversations)
}

func startOpenAIConversationsSession(
	ctx context.Context,
	conversationsSvc openAIConversationsService,
) (string, error) {
	if conversationsSvc == nil {
		return "", errors.New("openai conversations service is nil")
	}
	conversation, err := conversationsSvc.New(ctx, conversations.ConversationNewParams{
		Items: []TResponseInputItem{},
	})
	if err != nil {
		return "", err
	}
	if conversation == nil || conversation.ID == "" {
		return "", errors.New("openai conversations create returned empty ID")
	}
	return conversation.ID, nil
}

func (s *OpenAIConversationsSession) SessionID(context.Context) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sessionID
}

func (s *OpenAIConversationsSession) SessionSettings() *SessionSettings {
	return s.sessionSettings
}

func (s *OpenAIConversationsSession) IgnoreIDsForMatching() bool {
	return true
}

func (s *OpenAIConversationsSession) SanitizeInputItemsForPersistence(items []TResponseInputItem) []TResponseInputItem {
	if len(items) == 0 {
		return nil
	}
	sanitized := make([]TResponseInputItem, 0, len(items))
	for _, item := range items {
		sanitized = append(sanitized, sanitizeConversationItem(item))
	}
	return sanitized
}

func sanitizeConversationItem(item TResponseInputItem) TResponseInputItem {
	raw, err := item.MarshalJSON()
	if err != nil {
		return item
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return item
	}
	delete(payload, "id")
	delete(payload, "provider_data")
	updated, err := json.Marshal(payload)
	if err != nil {
		return item
	}
	var out TResponseInputItem
	if err := json.Unmarshal(updated, &out); err != nil {
		return item
	}
	return out
}

func (s *OpenAIConversationsSession) ensureSessionID(ctx context.Context) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.sessionID != "" {
		return s.sessionID, nil
	}

	sessionID, err := startOpenAIConversationsSession(ctx, s.conversationsSvc)
	if err != nil {
		return "", err
	}
	s.sessionID = sessionID
	return s.sessionID, nil
}

func (s *OpenAIConversationsSession) clearSessionID() {
	s.mu.Lock()
	s.sessionID = ""
	s.mu.Unlock()
}

func (s *OpenAIConversationsSession) GetItems(
	ctx context.Context,
	limit int,
) ([]TResponseInputItem, error) {
	sessionID, err := s.ensureSessionID(ctx)
	if err != nil {
		return nil, err
	}

	rawItems, err := s.listConversationItems(ctx, sessionID, limit)
	if err != nil {
		return nil, err
	}

	allItems := make([]TResponseInputItem, 0, len(rawItems))
	for _, rawItem := range rawItems {
		item, err := conversationItemToResponseInput(rawItem)
		if err != nil {
			return nil, err
		}
		allItems = append(allItems, item)
	}
	return allItems, nil
}

func (s *OpenAIConversationsSession) AddItems(ctx context.Context, items []TResponseInputItem) error {
	if len(items) == 0 {
		return nil
	}

	sessionID, err := s.ensureSessionID(ctx)
	if err != nil {
		return err
	}

	_, err = s.itemsSvc.New(ctx, sessionID, conversations.ItemNewParams{
		Items: items,
	})
	return err
}

func (s *OpenAIConversationsSession) PopItem(ctx context.Context) (*TResponseInputItem, error) {
	sessionID, err := s.ensureSessionID(ctx)
	if err != nil {
		return nil, err
	}

	items, err := s.listConversationItems(ctx, sessionID, 1)
	if err != nil {
		return nil, err
	}
	if len(items) == 0 {
		return nil, nil
	}

	itemID := items[0].ID
	if itemID == "" {
		return nil, errors.New("conversation item is missing id")
	}
	if _, err := s.itemsSvc.Delete(ctx, sessionID, itemID); err != nil {
		return nil, err
	}

	out, err := conversationItemToResponseInput(items[0])
	if err != nil {
		return nil, err
	}
	return &out, nil
}

func (s *OpenAIConversationsSession) ClearSession(ctx context.Context) error {
	sessionID, err := s.ensureSessionID(ctx)
	if err != nil {
		return err
	}

	if _, err := s.conversationsSvc.Delete(ctx, sessionID); err != nil {
		return err
	}
	s.clearSessionID()
	return nil
}

func conversationItemToResponseInput(item conversations.ConversationItemUnion) (TResponseInputItem, error) {
	raw := item.RawJSON()
	if raw == "" {
		jsonBytes, err := json.Marshal(item)
		if err != nil {
			return TResponseInputItem{}, fmt.Errorf("failed to marshal conversation item: %w", err)
		}
		raw = string(jsonBytes)
	}

	var out TResponseInputItem
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return TResponseInputItem{}, fmt.Errorf("failed to convert conversation item: %w", err)
	}
	return out, nil
}

func (s *OpenAIConversationsSession) listConversationItems(
	ctx context.Context,
	sessionID string,
	limit int,
) ([]conversations.ConversationItemUnion, error) {
	query := conversations.ItemListParams{
		Order: conversations.ItemListParamsOrderAsc,
	}
	if limit > 0 {
		query.Limit = param.NewOpt(int64(limit))
		query.Order = conversations.ItemListParamsOrderDesc
	}

	pager := s.itemsSvc.ListAutoPaging(ctx, sessionID, query)
	allItems := make([]conversations.ConversationItemUnion, 0)
	for pager.Next() {
		allItems = append(allItems, pager.Current())
		if limit > 0 && len(allItems) >= limit {
			break
		}
	}
	if err := pager.Err(); err != nil {
		return nil, err
	}

	if limit > 0 {
		slices.Reverse(allItems)
	}
	return allItems, nil
}

func responseInputItemID(item TResponseInputItem) (string, error) {
	data, err := item.MarshalJSON()
	if err != nil {
		return "", fmt.Errorf("failed to marshal response input item: %w", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return "", fmt.Errorf("failed to decode response input item: %w", err)
	}

	rawID, ok := payload["id"]
	if !ok {
		return "", errors.New("response input item is missing id")
	}
	itemID, ok := rawID.(string)
	if !ok || itemID == "" {
		return "", errors.New("response input item has invalid id")
	}
	return itemID, nil
}
