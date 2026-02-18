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
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAsyncSQLiteSessionBasicFlow(t *testing.T) {
	ctx := t.Context()
	session, err := NewAsyncSQLiteSession(ctx, AsyncSQLiteSessionParams{
		SessionID:        "async_basic",
		DBDataSourceName: filepath.Join(t.TempDir(), "async_basic.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	items := []TResponseInputItem{
		makeInputMessageItem(responses.EasyInputMessageRoleUser, "Hello"),
		makeInputMessageItem(responses.EasyInputMessageRoleAssistant, "Hi there!"),
	}

	require.NoError(t, session.AddItems(ctx, items))

	retrieved, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, items, retrieved)

	require.NoError(t, session.ClearSession(ctx))
	retrieved, err = session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Empty(t, retrieved)
}

func TestAsyncSQLiteSessionPopItem(t *testing.T) {
	ctx := t.Context()
	session, err := NewAsyncSQLiteSession(ctx, AsyncSQLiteSessionParams{
		SessionID:        "async_pop",
		DBDataSourceName: filepath.Join(t.TempDir(), "async_pop.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	items := []TResponseInputItem{
		makeInputMessageItem(responses.EasyInputMessageRoleUser, "One"),
		makeInputMessageItem(responses.EasyInputMessageRoleAssistant, "Two"),
	}
	require.NoError(t, session.AddItems(ctx, items))

	popped, err := session.PopItem(ctx)
	require.NoError(t, err)
	require.NotNil(t, popped)
	assert.Equal(t, items[1], *popped)

	remaining, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, items[:1], remaining)
}

func TestAsyncSQLiteSessionGetItemsLimit(t *testing.T) {
	ctx := t.Context()
	session, err := NewAsyncSQLiteSession(ctx, AsyncSQLiteSessionParams{
		SessionID:        "async_limit",
		DBDataSourceName: filepath.Join(t.TempDir(), "async_limit.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	items := []TResponseInputItem{
		makeInputMessageItem(responses.EasyInputMessageRoleUser, "Message 1"),
		makeInputMessageItem(responses.EasyInputMessageRoleAssistant, "Response 1"),
		makeInputMessageItem(responses.EasyInputMessageRoleUser, "Message 2"),
	}
	require.NoError(t, session.AddItems(ctx, items))

	latest, err := session.GetItems(ctx, 2)
	require.NoError(t, err)
	assert.Equal(t, items[1:], latest)

	allItems, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, items, allItems)
}

func TestAsyncSQLiteSessionUnicodeContent(t *testing.T) {
	ctx := t.Context()
	session, err := NewAsyncSQLiteSession(ctx, AsyncSQLiteSessionParams{
		SessionID:        "async_unicode",
		DBDataSourceName: filepath.Join(t.TempDir(), "async_unicode.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	items := []TResponseInputItem{
		makeInputMessageItem(responses.EasyInputMessageRoleUser, "こんにちは"),
		makeInputMessageItem(responses.EasyInputMessageRoleAssistant, "Привет"),
	}
	require.NoError(t, session.AddItems(ctx, items))

	retrieved, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, items, retrieved)
}

func TestAsyncSQLiteSessionSessionIsolation(t *testing.T) {
	ctx := t.Context()
	dbPath := filepath.Join(t.TempDir(), "async_isolation.db")

	session1, err := NewAsyncSQLiteSession(ctx, AsyncSQLiteSessionParams{
		SessionID:        "session_1",
		DBDataSourceName: dbPath,
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session1.Close()) })

	session2, err := NewAsyncSQLiteSession(ctx, AsyncSQLiteSessionParams{
		SessionID:        "session_2",
		DBDataSourceName: dbPath,
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session2.Close()) })

	require.NoError(t, session1.AddItems(ctx, []TResponseInputItem{
		makeInputMessageItem(responses.EasyInputMessageRoleUser, "I like cats."),
	}))
	require.NoError(t, session2.AddItems(ctx, []TResponseInputItem{
		makeInputMessageItem(responses.EasyInputMessageRoleUser, "I like dogs."),
	}))

	items1, err := session1.GetItems(ctx, 0)
	require.NoError(t, err)
	items2, err := session2.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.NotEqual(t, items1, items2)
}

func TestAsyncSQLiteSessionAddEmptyItemsList(t *testing.T) {
	ctx := t.Context()
	session, err := NewAsyncSQLiteSession(ctx, AsyncSQLiteSessionParams{
		SessionID:        "add_empty",
		DBDataSourceName: filepath.Join(t.TempDir(), "async_add_empty.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	require.NoError(t, session.AddItems(ctx, nil))
	items, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Empty(t, items)
}

func TestAsyncSQLiteSessionPopFromEmptySession(t *testing.T) {
	ctx := t.Context()
	session, err := NewAsyncSQLiteSession(ctx, AsyncSQLiteSessionParams{
		SessionID:        "empty_session",
		DBDataSourceName: filepath.Join(t.TempDir(), "async_pop_empty.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	popped, err := session.PopItem(ctx)
	require.NoError(t, err)
	assert.Nil(t, popped)
}

func TestAsyncSQLiteSessionGetItemsLimitMoreThanAvailable(t *testing.T) {
	ctx := t.Context()
	session, err := NewAsyncSQLiteSession(ctx, AsyncSQLiteSessionParams{
		SessionID:        "limit_more_test",
		DBDataSourceName: filepath.Join(t.TempDir(), "async_limit_more.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	items := []TResponseInputItem{
		makeInputMessageItem(responses.EasyInputMessageRoleUser, "1"),
		makeInputMessageItem(responses.EasyInputMessageRoleAssistant, "2"),
		makeInputMessageItem(responses.EasyInputMessageRoleUser, "3"),
		makeInputMessageItem(responses.EasyInputMessageRoleAssistant, "4"),
	}
	require.NoError(t, session.AddItems(ctx, items))

	retrieved, err := session.GetItems(ctx, 10)
	require.NoError(t, err)
	assert.Equal(t, items, retrieved)
}

func TestAsyncSQLiteSessionGetItemsSameTimestampConsistentOrder(t *testing.T) {
	ctx := t.Context()
	session, err := NewAsyncSQLiteSession(ctx, AsyncSQLiteSessionParams{
		SessionID:        "same_timestamp_test",
		DBDataSourceName: filepath.Join(t.TempDir(), "async_same_timestamp.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	items := []TResponseInputItem{
		makeReasoningItem("older_same_ts"),
		makeReasoningItem("rs_same_ts"),
		makeOutputMessageItem("msg_same_ts"),
	}
	require.NoError(t, session.AddItems(ctx, items))

	shared := time.Date(2025, 10, 15, 17, 26, 39, 132483000, time.UTC)
	_, err = session.db.ExecContext(ctx, `UPDATE "`+session.messagesTable+`" SET created_at = ? WHERE session_id = ?`,
		shared.Format("2006-01-02 15:04:05.000000"), session.sessionID,
	)
	require.NoError(t, err)

	retrieved, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, []string{"older_same_ts", "rs_same_ts", "msg_same_ts"}, itemIDs(retrieved))

	latestTwo, err := session.GetItems(ctx, 2)
	require.NoError(t, err)
	assert.Equal(t, []string{"rs_same_ts", "msg_same_ts"}, itemIDs(latestTwo))
}

func TestAsyncSQLiteSessionPopItemSameTimestampReturnsLatest(t *testing.T) {
	ctx := t.Context()
	session, err := NewAsyncSQLiteSession(ctx, AsyncSQLiteSessionParams{
		SessionID:        "same_timestamp_pop",
		DBDataSourceName: filepath.Join(t.TempDir(), "async_same_timestamp_pop.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	items := []TResponseInputItem{
		makeReasoningItem("rs_pop_same_ts"),
		makeOutputMessageItem("msg_pop_same_ts"),
	}
	require.NoError(t, session.AddItems(ctx, items))

	shared := time.Date(2025, 10, 15, 17, 26, 39, 132483000, time.UTC)
	_, err = session.db.ExecContext(ctx, `UPDATE "`+session.messagesTable+`" SET created_at = ? WHERE session_id = ?`,
		shared.Format("2006-01-02 15:04:05.000000"), session.sessionID,
	)
	require.NoError(t, err)

	popped, err := session.PopItem(ctx)
	require.NoError(t, err)
	require.NotNil(t, popped)
	assert.Equal(t, "msg_pop_same_ts", itemID(*popped))

	remaining, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, []string{"rs_pop_same_ts"}, itemIDs(remaining))
}

func makeInputMessageItem(role responses.EasyInputMessageRole, content string) TResponseInputItem {
	return TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{
				OfString: param.NewOpt(content),
			},
			Role: role,
			Type: responses.EasyInputMessageTypeMessage,
		},
	}
}

func makeReasoningItem(id string) TResponseInputItem {
	return TResponseInputItem{
		OfReasoning: &responses.ResponseReasoningItemParam{
			ID:      id,
			Summary: []responses.ResponseReasoningItemSummaryParam{},
			Type:    constant.ValueOf[constant.Reasoning](),
		},
	}
}

func makeOutputMessageItem(id string) TResponseInputItem {
	return TResponseInputItem{
		OfOutputMessage: &responses.ResponseOutputMessageParam{
			ID: id,
			Content: []responses.ResponseOutputMessageContentUnionParam{
				{OfOutputText: &responses.ResponseOutputTextParam{
					Text: "ok",
					Type: constant.ValueOf[constant.OutputText](),
				}},
			},
			Status: responses.ResponseOutputMessageStatusCompleted,
			Role:   constant.ValueOf[constant.Assistant](),
			Type:   constant.ValueOf[constant.Message](),
		},
	}
}

func itemIDs(items []TResponseInputItem) []string {
	result := make([]string, 0, len(items))
	for _, item := range items {
		result = append(result, itemID(item))
	}
	return result
}

func itemID(item TResponseInputItem) string {
	switch {
	case item.OfReasoning != nil:
		return item.OfReasoning.ID
	case item.OfOutputMessage != nil:
		return item.OfOutputMessage.ID
	}
	raw, err := json.Marshal(item)
	if err != nil {
		return ""
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return ""
	}
	if id, ok := payload["id"].(string); ok {
		return id
	}
	return ""
}
