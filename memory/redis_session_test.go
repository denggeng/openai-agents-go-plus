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

package memory

import (
	"fmt"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func redisTestMessage(role responses.EasyInputMessageRole, text string) TResponseInputItem {
	return TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(text)},
			Role:    role,
			Type:    responses.EasyInputMessageTypeMessage,
		},
	}
}

func newMiniRedisSession(t *testing.T, sessionID string) (*miniredis.Miniredis, *redis.Client, *RedisSession) {
	t.Helper()

	server := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: server.Addr()})
	t.Cleanup(func() { _ = client.Close() })

	session, err := NewRedisSession(t.Context(), RedisSessionParams{
		SessionID: sessionID,
		Client:    client,
		KeyPrefix: "test:agents:session",
	})
	require.NoError(t, err)
	return server, client, session
}

func TestRedisSession_DirectOps(t *testing.T) {
	_, _, session := newMiniRedisSession(t, "redis-basic")
	ctx := t.Context()

	items := []TResponseInputItem{
		redisTestMessage(responses.EasyInputMessageRoleUser, "Hello"),
		redisTestMessage(responses.EasyInputMessageRoleAssistant, "Hi there!"),
	}

	require.NoError(t, session.AddItems(ctx, items))

	retrieved, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, items, retrieved)

	popped, err := session.PopItem(ctx)
	require.NoError(t, err)
	require.NotNil(t, popped)
	assert.Equal(t, items[1], *popped)

	retrievedAfterPop, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, items[:1], retrievedAfterPop)

	require.NoError(t, session.ClearSession(ctx))
	retrievedAfterClear, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Empty(t, retrievedAfterClear)
}

func TestRedisSession_GetItemsWithLimit(t *testing.T) {
	_, _, session := newMiniRedisSession(t, "redis-limit")
	ctx := t.Context()

	items := []TResponseInputItem{
		redisTestMessage(responses.EasyInputMessageRoleUser, "1"),
		redisTestMessage(responses.EasyInputMessageRoleAssistant, "2"),
		redisTestMessage(responses.EasyInputMessageRoleUser, "3"),
		redisTestMessage(responses.EasyInputMessageRoleAssistant, "4"),
	}
	require.NoError(t, session.AddItems(ctx, items))

	latest2, err := session.GetItems(ctx, 2)
	require.NoError(t, err)
	assert.Equal(t, items[2:], latest2)

	allItems, err := session.GetItems(ctx, -1)
	require.NoError(t, err)
	assert.Equal(t, items, allItems)
}

func TestRedisSession_SessionIsolation(t *testing.T) {
	server := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: server.Addr()})
	t.Cleanup(func() { _ = client.Close() })

	s1, err := NewRedisSession(t.Context(), RedisSessionParams{
		SessionID: "s1",
		Client:    client,
		KeyPrefix: "test:agents:session",
	})
	require.NoError(t, err)
	s2, err := NewRedisSession(t.Context(), RedisSessionParams{
		SessionID: "s2",
		Client:    client,
		KeyPrefix: "test:agents:session",
	})
	require.NoError(t, err)

	ctx := t.Context()
	require.NoError(t, s1.AddItems(ctx, []TResponseInputItem{redisTestMessage(responses.EasyInputMessageRoleUser, "cats")}))
	require.NoError(t, s2.AddItems(ctx, []TResponseInputItem{redisTestMessage(responses.EasyInputMessageRoleUser, "dogs")}))

	items1, err := s1.GetItems(ctx, 0)
	require.NoError(t, err)
	items2, err := s2.GetItems(ctx, 0)
	require.NoError(t, err)

	assert.Equal(t, "cats", items1[0].OfMessage.Content.OfString.Value)
	assert.Equal(t, "dogs", items2[0].OfMessage.Content.OfString.Value)
}

func TestRedisSession_SkipsCorruptedItems(t *testing.T) {
	_, client, session := newMiniRedisSession(t, "redis-corrupted")
	ctx := t.Context()

	good1 := redisTestMessage(responses.EasyInputMessageRoleUser, "good-1")
	good2 := redisTestMessage(responses.EasyInputMessageRoleAssistant, "good-2")
	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{good1}))
	require.NoError(t, client.RPush(ctx, session.messagesKey, "{not-json").Err())
	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{good2}))

	retrieved, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, []TResponseInputItem{good1, good2}, retrieved)
}

func TestRedisSession_FromURLOwnsClientAndClose(t *testing.T) {
	server := miniredis.RunT(t)
	ctx := t.Context()

	session, err := NewRedisSession(ctx, RedisSessionParams{
		SessionID: "redis-url",
		URL:       fmt.Sprintf("redis://%s/0", server.Addr()),
		KeyPrefix: "test:agents:session",
		TTL:       5 * time.Second,
	})
	require.NoError(t, err)
	require.True(t, session.ownsClient)

	require.True(t, session.Ping(ctx))
	require.NoError(t, session.Close())
	assert.False(t, session.Ping(ctx))
}

func TestNewRedisSession_Validation(t *testing.T) {
	ctx := t.Context()

	_, err := NewRedisSession(ctx, RedisSessionParams{})
	require.ErrorContains(t, err, "session id")

	_, err = NewRedisSession(ctx, RedisSessionParams{SessionID: "redis"})
	require.ErrorContains(t, err, "redis client or url")
}
