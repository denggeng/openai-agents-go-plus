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

package memory

import (
	"context"
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type SimpleListSession struct {
	sessionID            string
	items                []TResponseInputItem
	SavedItems           []TResponseInputItem
	IgnoreIDsForMatching bool
}

func NewSimpleListSession(sessionID string, history []TResponseInputItem) *SimpleListSession {
	items := append([]TResponseInputItem(nil), history...)
	return &SimpleListSession{
		sessionID:  sessionID,
		items:      items,
		SavedItems: items,
	}
}

func (s *SimpleListSession) SessionID(context.Context) string {
	return s.sessionID
}

func (s *SimpleListSession) GetItems(_ context.Context, limit int) ([]TResponseInputItem, error) {
	if limit <= 0 || limit >= len(s.items) {
		return append([]TResponseInputItem(nil), s.items...), nil
	}
	return append([]TResponseInputItem(nil), s.items[len(s.items)-limit:]...), nil
}

func (s *SimpleListSession) AddItems(_ context.Context, items []TResponseInputItem) error {
	s.items = append(s.items, items...)
	s.SavedItems = s.items
	return nil
}

func (s *SimpleListSession) PopItem(context.Context) (*TResponseInputItem, error) {
	if len(s.items) == 0 {
		return nil, nil
	}
	item := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	s.SavedItems = s.items
	return &item, nil
}

func (s *SimpleListSession) ClearSession(context.Context) error {
	s.items = nil
	s.SavedItems = s.items
	return nil
}

type CountingSession struct {
	*SimpleListSession
	PopCalls int
}

func NewCountingSession(sessionID string, history []TResponseInputItem) *CountingSession {
	return &CountingSession{
		SimpleListSession: NewSimpleListSession(sessionID, history),
	}
}

func (s *CountingSession) PopItem(ctx context.Context) (*TResponseInputItem, error) {
	s.PopCalls += 1
	return s.SimpleListSession.PopItem(ctx)
}

type IdStrippingSession struct {
	*CountingSession
}

func NewIdStrippingSession(sessionID string, history []TResponseInputItem) *IdStrippingSession {
	base := NewCountingSession(sessionID, history)
	base.IgnoreIDsForMatching = true
	return &IdStrippingSession{CountingSession: base}
}

func (s *IdStrippingSession) AddItems(ctx context.Context, items []TResponseInputItem) error {
	sanitized := make([]TResponseInputItem, 0, len(items))
	for _, item := range items {
		sanitized = append(sanitized, stripItemID(item))
	}
	return s.CountingSession.AddItems(ctx, sanitized)
}

func stripItemID(item TResponseInputItem) TResponseInputItem {
	if item.OfFunctionCall != nil {
		copied := *item.OfFunctionCall
		copied.ID = param.Opt[string]{}
		item.OfFunctionCall = &copied
	}
	if item.OfFunctionCallOutput != nil {
		copied := *item.OfFunctionCallOutput
		copied.ID = param.Opt[string]{}
		item.OfFunctionCallOutput = &copied
	}
	return item
}

func TestSimpleListSessionPreservesHistoryAndSavedItems(t *testing.T) {
	ctx := t.Context()
	history := []TResponseInputItem{
		inputMessage("hi", responses.EasyInputMessageRoleUser),
		inputMessage("hello", responses.EasyInputMessageRoleAssistant),
	}
	session := NewSimpleListSession("test", history)

	items, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, history, items)
	items[0] = inputMessage("changed", responses.EasyInputMessageRoleUser)

	items2, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, history, items2)
	assert.Equal(t, history, session.SavedItems)
}

func TestCountingSessionTracksPopCalls(t *testing.T) {
	ctx := t.Context()
	session := NewCountingSession("test", []TResponseInputItem{
		inputMessage("hi", responses.EasyInputMessageRoleUser),
	})

	assert.Equal(t, 0, session.PopCalls)
	_, err := session.PopItem(ctx)
	require.NoError(t, err)
	assert.Equal(t, 1, session.PopCalls)
	_, err = session.PopItem(ctx)
	require.NoError(t, err)
	assert.Equal(t, 2, session.PopCalls)
}

func TestIdStrippingSessionRemovesIDsOnAdd(t *testing.T) {
	ctx := t.Context()
	session := NewIdStrippingSession("test", nil)

	items := []TResponseInputItem{
		{OfFunctionCall: &responses.ResponseFunctionToolCallParam{
			Arguments: "{}",
			CallID:    "call-1",
			Name:      "test_tool",
			ID:        param.NewOpt("keep-removed"),
			Type:      constant.ValueOf[constant.FunctionCall](),
		}},
		inputMessage("no-id", responses.EasyInputMessageRoleAssistant),
	}

	require.NoError(t, session.AddItems(ctx, items))
	stored, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	require.Len(t, stored, 2)
	require.NotNil(t, stored[0].OfFunctionCall)
	assert.False(t, stored[0].OfFunctionCall.ID.Valid())

	_, err = session.PopItem(ctx)
	require.NoError(t, err)
	assert.Equal(t, 1, session.PopCalls)
}

func inputMessage(text string, role responses.EasyInputMessageRole) TResponseInputItem {
	return TResponseInputItem{OfMessage: &responses.EasyInputMessageParam{
		Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(text)},
		Role:    role,
		Type:    responses.EasyInputMessageTypeMessage,
	}}
}
