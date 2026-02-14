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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeDaprStateClient struct {
	mu          sync.Mutex
	state       map[string][]byte
	etags       map[string]string
	etagCounter int
	closed      bool
	failGet     bool
	lastSave    map[string]savedRequest
}

type savedRequest struct {
	metadata map[string]string
	options  DaprStateOptions
}

func newFakeDaprStateClient() *fakeDaprStateClient {
	return &fakeDaprStateClient{
		state:    make(map[string][]byte),
		etags:    make(map[string]string),
		lastSave: make(map[string]savedRequest),
	}
}

func (c *fakeDaprStateClient) GetState(_ context.Context, _ string, key string, _ map[string]string) (DaprStateItem, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.failGet {
		return DaprStateItem{}, errors.New("connection failed")
	}
	item := DaprStateItem{ETag: c.etags[key]}
	if data, ok := c.state[key]; ok {
		item.Data = append([]byte(nil), data...)
	}
	return item, nil
}

func (c *fakeDaprStateClient) SaveState(_ context.Context, _ string, key string, value []byte, etag string, metadata map[string]string, options DaprStateOptions) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	currentETag := c.etags[key]
	expectsMatch := options.Concurrency == DaprConcurrencyFirstWrite
	if expectsMatch {
		if currentETag == "" {
			if etag != "" {
				return errors.New("etag mismatch: key does not exist")
			}
		} else if etag != currentETag {
			return errors.New("etag mismatch: stale data")
		}
	}

	c.state[key] = append([]byte(nil), value...)
	c.etagCounter++
	c.etags[key] = fmt.Sprintf("%d", c.etagCounter)

	metaCopy := make(map[string]string, len(metadata))
	for k, v := range metadata {
		metaCopy[k] = v
	}
	c.lastSave[key] = savedRequest{metadata: metaCopy, options: options}
	return nil
}

func (c *fakeDaprStateClient) DeleteState(_ context.Context, _ string, key string, _ map[string]string, _ DaprStateOptions) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.state, key)
	delete(c.etags, key)
	delete(c.lastSave, key)
	return nil
}

func (c *fakeDaprStateClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	return nil
}

type conflictFakeDaprStateClient struct {
	*fakeDaprStateClient
	conflictedKeys map[string]bool
}

func newConflictFakeDaprStateClient() *conflictFakeDaprStateClient {
	return &conflictFakeDaprStateClient{
		fakeDaprStateClient: newFakeDaprStateClient(),
		conflictedKeys:      make(map[string]bool),
	}
}

func (c *conflictFakeDaprStateClient) SaveState(ctx context.Context, storeName, key string, value []byte, etag string, metadata map[string]string, options DaprStateOptions) error {
	c.mu.Lock()
	currentETag := c.etags[key]
	shouldConflict := options.Concurrency == DaprConcurrencyFirstWrite && strings.HasSuffix(key, ":messages") && currentETag != "" && !c.conflictedKeys[key]
	if shouldConflict {
		c.conflictedKeys[key] = true
		decoded := c.decodeMessagesLocked(key)
		competitorMsg := messageJSONString(daprTestMessage(responses.EasyInputMessageRoleAssistant, "from-concurrent-writer"))
		decoded = append(decoded, competitorMsg)
		payload, _ := json.Marshal(decoded)
		c.state[key] = payload
		c.etagCounter++
		c.etags[key] = fmt.Sprintf("%d", c.etagCounter)
		c.mu.Unlock()
		return errors.New("etag mismatch: concurrent writer")
	}
	c.mu.Unlock()

	return c.fakeDaprStateClient.SaveState(ctx, storeName, key, value, etag, metadata, options)
}

func (c *conflictFakeDaprStateClient) decodeMessagesLocked(key string) []any {
	data := c.state[key]
	if len(data) == 0 {
		return nil
	}
	var messages []any
	if err := json.Unmarshal(data, &messages); err != nil {
		return nil
	}
	return messages
}

func daprTestMessage(role responses.EasyInputMessageRole, text string) TResponseInputItem {
	return TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(text)},
			Role:    role,
			Type:    responses.EasyInputMessageTypeMessage,
		},
	}
}

func messageJSONString(item TResponseInputItem) string {
	b, _ := item.MarshalJSON()
	return string(b)
}

func newDaprSessionForTest(t *testing.T, client DaprStateClient, sessionID string, opts ...func(*DaprSessionParams)) *DaprSession {
	t.Helper()
	params := DaprSessionParams{
		SessionID:      sessionID,
		StateStoreName: "statestore",
		Client:         client,
	}
	for _, opt := range opts {
		opt(&params)
	}
	s, err := NewDaprSession(params)
	require.NoError(t, err)
	s.randFloat64 = func() float64 { return 0 }
	s.sleep = func(context.Context, time.Duration) error { return nil }
	return s
}

func TestDaprSession_DirectOps(t *testing.T) {
	ctx := t.Context()
	client := newFakeDaprStateClient()
	session := newDaprSessionForTest(t, client, "dapr-direct")

	items := []TResponseInputItem{
		daprTestMessage(responses.EasyInputMessageRoleUser, "Hello"),
		daprTestMessage(responses.EasyInputMessageRoleAssistant, "Hi there!"),
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

func TestDaprSession_GetItemsWithLimit(t *testing.T) {
	ctx := t.Context()
	client := newFakeDaprStateClient()
	session := newDaprSessionForTest(t, client, "dapr-limit")

	items := []TResponseInputItem{
		daprTestMessage(responses.EasyInputMessageRoleUser, "1"),
		daprTestMessage(responses.EasyInputMessageRoleAssistant, "2"),
		daprTestMessage(responses.EasyInputMessageRoleUser, "3"),
		daprTestMessage(responses.EasyInputMessageRoleAssistant, "4"),
	}
	require.NoError(t, session.AddItems(ctx, items))

	latest2, err := session.GetItems(ctx, 2)
	require.NoError(t, err)
	assert.Equal(t, items[2:], latest2)

	allItems, err := session.GetItems(ctx, 10)
	require.NoError(t, err)
	assert.Equal(t, items, allItems)
}

func TestDaprSession_SessionIsolation(t *testing.T) {
	ctx := t.Context()
	client := newFakeDaprStateClient()
	s1 := newDaprSessionForTest(t, client, "session_1")
	s2 := newDaprSessionForTest(t, client, "session_2")

	require.NoError(t, s1.AddItems(ctx, []TResponseInputItem{daprTestMessage(responses.EasyInputMessageRoleUser, "cats")}))
	require.NoError(t, s2.AddItems(ctx, []TResponseInputItem{daprTestMessage(responses.EasyInputMessageRoleUser, "dogs")}))

	items1, err := s1.GetItems(ctx, 0)
	require.NoError(t, err)
	items2, err := s2.GetItems(ctx, 0)
	require.NoError(t, err)

	assert.Equal(t, "cats", items1[0].OfMessage.Content.OfString.Value)
	assert.Equal(t, "dogs", items2[0].OfMessage.Content.OfString.Value)
}

func TestDaprSession_AddItemsRetriesOnConcurrency(t *testing.T) {
	ctx := t.Context()
	client := newConflictFakeDaprStateClient()
	session := newDaprSessionForTest(t, client, "concurrency_add")

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		daprTestMessage(responses.EasyInputMessageRoleUser, "seed"),
	}))
	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		daprTestMessage(responses.EasyInputMessageRoleAssistant, "new message"),
	}))

	items, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	contents := []string{
		items[0].OfMessage.Content.OfString.Value,
		items[1].OfMessage.Content.OfString.Value,
		items[2].OfMessage.Content.OfString.Value,
	}
	assert.Equal(t, []string{"seed", "from-concurrent-writer", "new message"}, contents)
	assert.True(t, client.conflictedKeys[session.messagesKey])
}

func TestDaprSession_PopItemRetriesOnConcurrency(t *testing.T) {
	ctx := t.Context()
	client := newConflictFakeDaprStateClient()
	session := newDaprSessionForTest(t, client, "concurrency_pop")

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		daprTestMessage(responses.EasyInputMessageRoleUser, "first"),
		daprTestMessage(responses.EasyInputMessageRoleAssistant, "second"),
	}))

	popped, err := session.PopItem(ctx)
	require.NoError(t, err)
	require.NotNil(t, popped)
	assert.Equal(t, "from-concurrent-writer", popped.OfMessage.Content.OfString.Value)

	items, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	contents := []string{
		items[0].OfMessage.Content.OfString.Value,
		items[1].OfMessage.Content.OfString.Value,
	}
	assert.Equal(t, []string{"first", "second"}, contents)
	assert.True(t, client.conflictedKeys[session.messagesKey])
}

func TestDaprSession_CorruptedDataHandling(t *testing.T) {
	ctx := t.Context()
	client := newFakeDaprStateClient()
	session := newDaprSessionForTest(t, client, "corruption_test")

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		daprTestMessage(responses.EasyInputMessageRoleUser, "valid"),
	}))

	client.mu.Lock()
	client.state[session.messagesKey] = []byte("invalid json data")
	client.mu.Unlock()

	items, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Empty(t, items)

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		daprTestMessage(responses.EasyInputMessageRoleUser, "valid after corruption"),
	}))

	items, err = session.GetItems(ctx, 0)
	require.NoError(t, err)
	require.Len(t, items, 1)
	assert.Equal(t, "valid after corruption", items[0].OfMessage.Content.OfString.Value)
}

func TestDaprSession_ConsistencyAndTTLMetadata(t *testing.T) {
	ctx := t.Context()
	client := newFakeDaprStateClient()
	session := newDaprSessionForTest(t, client, "ttl_consistency", func(p *DaprSessionParams) {
		p.Consistency = DaprConsistencyStrong
		p.TTL = time.Hour
	})

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		daprTestMessage(responses.EasyInputMessageRoleUser, "hello"),
	}))

	client.mu.Lock()
	lastMessagesSave := client.lastSave[session.messagesKey]
	client.mu.Unlock()

	assert.Equal(t, "3600", lastMessagesSave.metadata["ttlInSeconds"])
	assert.Equal(t, DaprConsistencyStrong, lastMessagesSave.options.Consistency)
	assert.Equal(t, DaprConcurrencyFirstWrite, lastMessagesSave.options.Concurrency)
}

func TestDaprSession_PingAndCloseOwnership(t *testing.T) {
	ctx := t.Context()
	client := newFakeDaprStateClient()
	session := newDaprSessionForTest(t, client, "ping_close")

	assert.True(t, session.Ping(ctx))

	client.failGet = true
	assert.False(t, session.Ping(ctx))
	client.failGet = false

	require.NoError(t, session.Close())
	assert.False(t, client.closed)

	sessionOwned := newDaprSessionForTest(t, client, "ping_close_owned", func(p *DaprSessionParams) {
		p.OwnsClient = true
	})
	require.NoError(t, sessionOwned.Close())
	assert.True(t, client.closed)
}

func TestNewDaprSession_Validation(t *testing.T) {
	client := newFakeDaprStateClient()

	_, err := NewDaprSession(DaprSessionParams{Client: client, StateStoreName: "statestore"})
	require.ErrorContains(t, err, "session id")

	_, err = NewDaprSession(DaprSessionParams{Client: client, SessionID: "s"})
	require.ErrorContains(t, err, "state store name")

	_, err = NewDaprSession(DaprSessionParams{SessionID: "s", StateStoreName: "statestore"})
	require.ErrorContains(t, err, "dapr state client")

	_, err = NewDaprSession(DaprSessionParams{SessionID: "s", StateStoreName: "statestore", Client: client, Consistency: DaprConsistencyLevel("weird")})
	require.ErrorContains(t, err, "invalid consistency")
}
