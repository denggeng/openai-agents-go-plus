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
	"cmp"
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// RedisSession is a Redis-backed implementation of Session storage.
type RedisSession struct {
	sessionID   string
	client      redis.UniversalClient
	ownsClient  bool
	keyPrefix   string
	ttl         time.Duration
	sessionSettings *SessionSettings
	sessionKey  string
	messagesKey string
	counterKey  string
	mu          sync.Mutex
}

type RedisSessionParams struct {
	// Unique identifier for the conversation session.
	SessionID string

	// Existing Redis client. When provided, this session does not close the client.
	Client redis.UniversalClient

	// Redis URL used to create a dedicated client when Client is nil.
	// Example: redis://localhost:6379/0
	URL string

	// Optional key prefix for all session keys.
	// Defaults to "agents:session".
	KeyPrefix string

	// Optional TTL for session keys. Zero means no expiration.
	TTL time.Duration

	// Optional session settings (e.g., default history limit).
	SessionSettings *SessionSettings
}

// NewRedisSession initializes a Redis session.
func NewRedisSession(ctx context.Context, params RedisSessionParams) (*RedisSession, error) {
	if params.SessionID == "" {
		return nil, fmt.Errorf("session id is required")
	}

	client := params.Client
	ownsClient := false
	if client == nil {
		if params.URL == "" {
			return nil, fmt.Errorf("redis client or url is required")
		}
		opts, err := redis.ParseURL(params.URL)
		if err != nil {
			return nil, fmt.Errorf("parse redis url: %w", err)
		}
		client = redis.NewClient(opts)
		ownsClient = true
	}

	settings := params.SessionSettings
	if settings == nil {
		settings = &SessionSettings{}
	}
	s := &RedisSession{
		sessionID:  params.SessionID,
		client:     client,
		ownsClient: ownsClient,
		keyPrefix:  cmp.Or(params.KeyPrefix, "agents:session"),
		ttl:        params.TTL,
		sessionSettings: settings,
	}
	s.sessionKey = fmt.Sprintf("%s:%s", s.keyPrefix, s.sessionID)
	s.messagesKey = fmt.Sprintf("%s:messages", s.sessionKey)
	s.counterKey = fmt.Sprintf("%s:counter", s.sessionKey)

	if !s.Ping(ctx) {
		_ = s.Close()
		return nil, fmt.Errorf("redis is not reachable")
	}
	return s, nil
}

func (s *RedisSession) SessionID(context.Context) string {
	return s.sessionID
}

func (s *RedisSession) SessionSettings() *SessionSettings {
	return s.sessionSettings
}

func (s *RedisSession) GetItems(ctx context.Context, limit int) ([]TResponseInputItem, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var (
		raw []string
		err error
	)
	if limit <= 0 {
		raw, err = s.client.LRange(ctx, s.messagesKey, 0, -1).Result()
	} else {
		raw, err = s.client.LRange(ctx, s.messagesKey, -int64(limit), -1).Result()
	}
	if err != nil {
		return nil, fmt.Errorf("get redis items: %w", err)
	}

	items := make([]TResponseInputItem, 0, len(raw))
	for _, payload := range raw {
		item, err := unmarshalMessageData(payload)
		if err != nil {
			continue
		}
		items = append(items, item)
	}

	if limit > 0 {
		items = trimOrphanedCallOutputAtHead(items)
	}
	return items, nil
}

func (s *RedisSession) AddItems(ctx context.Context, items []TResponseInputItem) error {
	if len(items) == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	now := strconv.FormatInt(time.Now().Unix(), 10)
	pipe := s.client.TxPipeline()

	pipe.HSet(ctx, s.sessionKey, map[string]any{
		"session_id": s.sessionID,
		"created_at": now,
		"updated_at": now,
	})

	serialized := make([]any, 0, len(items))
	for _, item := range items {
		payload, err := item.MarshalJSON()
		if err != nil {
			return fmt.Errorf("marshal input item: %w", err)
		}
		serialized = append(serialized, string(payload))
	}
	if len(serialized) > 0 {
		pipe.RPush(ctx, s.messagesKey, serialized...)
	}
	pipe.HSet(ctx, s.sessionKey, "updated_at", now)

	if s.ttl > 0 {
		pipe.Expire(ctx, s.sessionKey, s.ttl)
		pipe.Expire(ctx, s.messagesKey, s.ttl)
		pipe.Expire(ctx, s.counterKey, s.ttl)
	}

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("add redis items: %w", err)
	}
	return nil
}

func (s *RedisSession) PopItem(ctx context.Context) (*TResponseInputItem, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	payload, err := s.client.RPop(ctx, s.messagesKey).Result()
	if err == redis.Nil {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("pop redis item: %w", err)
	}

	item, err := unmarshalMessageData(payload)
	if err != nil {
		// Corrupted entry has already been removed.
		return nil, nil
	}
	return &item, nil
}

func (s *RedisSession) ClearSession(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.client.Del(ctx, s.sessionKey, s.messagesKey, s.counterKey).Err(); err != nil {
		return fmt.Errorf("clear redis session: %w", err)
	}
	return nil
}

// Ping checks Redis connectivity.
func (s *RedisSession) Ping(ctx context.Context) bool {
	return s.client.Ping(ctx).Err() == nil
}

// Close closes the Redis client if this session owns it.
func (s *RedisSession) Close() error {
	if !s.ownsClient {
		return nil
	}
	return s.client.Close()
}

var _ Session = (*RedisSession)(nil)
