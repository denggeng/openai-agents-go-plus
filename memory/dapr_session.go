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
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
	"time"
)

type DaprConsistencyLevel string

const (
	DaprConsistencyEventual DaprConsistencyLevel = "eventual"
	DaprConsistencyStrong   DaprConsistencyLevel = "strong"
)

type DaprConcurrencyMode string

const (
	DaprConcurrencyFirstWrite DaprConcurrencyMode = "first_write"
)

const (
	daprMaxWriteAttempts = 5
	daprRetryBaseDelay   = 50 * time.Millisecond
	daprRetryMaxDelay    = time.Second
)

// DaprStateOptions represents write/read options for a Dapr state operation.
type DaprStateOptions struct {
	Consistency DaprConsistencyLevel
	Concurrency DaprConcurrencyMode
}

// DaprStateItem represents a state record returned from a Dapr state store.
type DaprStateItem struct {
	Data []byte
	ETag string
}

// DaprStateClient is the minimal state operations interface required by DaprSession.
type DaprStateClient interface {
	GetState(ctx context.Context, storeName, key string, metadata map[string]string) (DaprStateItem, error)
	SaveState(ctx context.Context, storeName, key string, value []byte, etag string, metadata map[string]string, options DaprStateOptions) error
	DeleteState(ctx context.Context, storeName, key string, metadata map[string]string, options DaprStateOptions) error
	Close() error
}

// DaprSession is a Dapr State Store-backed implementation of Session.
type DaprSession struct {
	sessionID      string
	stateStoreName string
	client         DaprStateClient
	ttl            time.Duration
	consistency    DaprConsistencyLevel
	ownsClient     bool
	sessionSettings *SessionSettings
	messagesKey    string
	metadataKey    string
	mu             sync.Mutex
	now            func() time.Time
	randFloat64    func() float64
	sleep          func(context.Context, time.Duration) error
}

type DaprSessionParams struct {
	SessionID      string
	StateStoreName string
	Client         DaprStateClient
	TTL            time.Duration
	Consistency    DaprConsistencyLevel
	OwnsClient     bool
	SessionSettings *SessionSettings
}

func NewDaprSession(params DaprSessionParams) (*DaprSession, error) {
	if strings.TrimSpace(params.SessionID) == "" {
		return nil, fmt.Errorf("session id is required")
	}
	if strings.TrimSpace(params.StateStoreName) == "" {
		return nil, fmt.Errorf("state store name is required")
	}
	if params.Client == nil {
		return nil, fmt.Errorf("dapr state client is required")
	}
	consistency := params.Consistency
	if consistency == "" {
		consistency = DaprConsistencyEventual
	}
	if consistency != DaprConsistencyEventual && consistency != DaprConsistencyStrong {
		return nil, fmt.Errorf("invalid consistency level %q", consistency)
	}

	settings := params.SessionSettings
	if settings == nil {
		settings = &SessionSettings{}
	}
	sessionID := strings.TrimSpace(params.SessionID)
	return &DaprSession{
		sessionID:      sessionID,
		stateStoreName: strings.TrimSpace(params.StateStoreName),
		client:         params.Client,
		ttl:            params.TTL,
		consistency:    consistency,
		ownsClient:     params.OwnsClient,
		sessionSettings: settings,
		messagesKey:    sessionID + ":messages",
		metadataKey:    sessionID + ":metadata",
		now:            time.Now,
		randFloat64:    rand.Float64,
		sleep: func(ctx context.Context, d time.Duration) error {
			t := time.NewTimer(d)
			defer t.Stop()
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-t.C:
				return nil
			}
		},
	}, nil
}

func (s *DaprSession) SessionID(context.Context) string {
	return s.sessionID
}

func (s *DaprSession) SessionSettings() *SessionSettings {
	return s.sessionSettings
}

func (s *DaprSession) GetItems(ctx context.Context, limit int) ([]TResponseInputItem, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	state, err := s.client.GetState(ctx, s.stateStoreName, s.messagesKey, s.readMetadata())
	if err != nil {
		return nil, fmt.Errorf("dapr get state: %w", err)
	}

	messages := s.decodeMessages(state.Data)
	if len(messages) == 0 {
		return nil, nil
	}
	if limit > 0 {
		if limit >= len(messages) {
			// keep all
		} else {
			messages = messages[len(messages)-limit:]
		}
	} else if limit == 0 {
		// 0 means no limit in Go Session API
	}

	items := make([]TResponseInputItem, 0, len(messages))
	for _, msg := range messages {
		item, ok := s.decodeMessageItem(msg)
		if !ok {
			continue
		}
		items = append(items, item)
	}
	if limit > 0 {
		items = trimOrphanedCallOutputAtHead(items)
	}
	return items, nil
}

func (s *DaprSession) AddItems(ctx context.Context, items []TResponseInputItem) error {
	if len(items) == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	serialized := make([]string, 0, len(items))
	for _, item := range items {
		payload, err := item.MarshalJSON()
		if err != nil {
			return fmt.Errorf("marshal input item: %w", err)
		}
		serialized = append(serialized, string(payload))
	}

	for attempt := 1; ; attempt++ {
		state, err := s.client.GetState(ctx, s.stateStoreName, s.messagesKey, s.readMetadata())
		if err != nil {
			return fmt.Errorf("dapr get state: %w", err)
		}

		existingMessages := s.decodeMessages(state.Data)
		updated := make([]any, 0, len(existingMessages)+len(serialized))
		updated = append(updated, existingMessages...)
		for _, message := range serialized {
			updated = append(updated, message)
		}

		payload, err := json.Marshal(updated)
		if err != nil {
			return fmt.Errorf("marshal messages payload: %w", err)
		}

		err = s.client.SaveState(
			ctx,
			s.stateStoreName,
			s.messagesKey,
			payload,
			state.ETag,
			s.writeMetadata(),
			DaprStateOptions{Consistency: s.consistency, Concurrency: DaprConcurrencyFirstWrite},
		)
		if err == nil {
			break
		}
		if !s.shouldRetryOnConcurrencyConflict(err, attempt) {
			return fmt.Errorf("dapr save state: %w", err)
		}
		if err := s.sleep(ctx, s.retryDelay(attempt)); err != nil {
			return err
		}
	}

	metadata := map[string]any{
		"session_id": s.sessionID,
		"created_at": s.now().Unix(),
		"updated_at": s.now().Unix(),
	}
	metaBytes, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("marshal session metadata: %w", err)
	}

	if err := s.client.SaveState(
		ctx,
		s.stateStoreName,
		s.metadataKey,
		metaBytes,
		"",
		s.writeMetadata(),
		DaprStateOptions{Consistency: s.consistency},
	); err != nil {
		return fmt.Errorf("dapr save metadata: %w", err)
	}

	return nil
}

func (s *DaprSession) PopItem(ctx context.Context) (*TResponseInputItem, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for attempt := 1; ; attempt++ {
		state, err := s.client.GetState(ctx, s.stateStoreName, s.messagesKey, s.readMetadata())
		if err != nil {
			return nil, fmt.Errorf("dapr get state: %w", err)
		}

		messages := s.decodeMessages(state.Data)
		if len(messages) == 0 {
			return nil, nil
		}

		last := messages[len(messages)-1]
		messages = messages[:len(messages)-1]

		payload, err := json.Marshal(messages)
		if err != nil {
			return nil, fmt.Errorf("marshal messages payload: %w", err)
		}

		err = s.client.SaveState(
			ctx,
			s.stateStoreName,
			s.messagesKey,
			payload,
			state.ETag,
			s.writeMetadata(),
			DaprStateOptions{Consistency: s.consistency, Concurrency: DaprConcurrencyFirstWrite},
		)
		if err != nil {
			if s.shouldRetryOnConcurrencyConflict(err, attempt) {
				if err := s.sleep(ctx, s.retryDelay(attempt)); err != nil {
					return nil, err
				}
				continue
			}
			return nil, fmt.Errorf("dapr save state: %w", err)
		}

		item, ok := s.decodeMessageItem(last)
		if !ok {
			return nil, nil
		}
		return &item, nil
	}
}

func (s *DaprSession) ClearSession(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.client.DeleteState(ctx, s.stateStoreName, s.messagesKey, s.writeMetadata(), DaprStateOptions{Consistency: s.consistency}); err != nil {
		return fmt.Errorf("delete messages state: %w", err)
	}
	if err := s.client.DeleteState(ctx, s.stateStoreName, s.metadataKey, s.writeMetadata(), DaprStateOptions{Consistency: s.consistency}); err != nil {
		return fmt.Errorf("delete metadata state: %w", err)
	}
	return nil
}

func (s *DaprSession) Ping(ctx context.Context) bool {
	_, err := s.client.GetState(ctx, s.stateStoreName, s.metadataKey, s.readMetadata())
	return err == nil
}

func (s *DaprSession) Close() error {
	if s.ownsClient {
		return s.client.Close()
	}
	return nil
}

func (s *DaprSession) readMetadata() map[string]string {
	metadata := make(map[string]string)
	if s.consistency != "" {
		metadata["consistency"] = string(s.consistency)
	}
	return metadata
}

func (s *DaprSession) writeMetadata() map[string]string {
	metadata := make(map[string]string)
	if s.ttl > 0 {
		metadata["ttlInSeconds"] = fmt.Sprintf("%d", int64(s.ttl/time.Second))
	}
	return metadata
}

func (s *DaprSession) decodeMessages(data []byte) []any {
	if len(data) == 0 {
		return nil
	}
	var messages []any
	if err := json.Unmarshal(data, &messages); err != nil {
		return nil
	}
	return messages
}

func (s *DaprSession) decodeMessageItem(v any) (TResponseInputItem, bool) {
	switch message := v.(type) {
	case string:
		item, err := unmarshalMessageData(message)
		if err != nil {
			return TResponseInputItem{}, false
		}
		return item, true
	case map[string]any:
		payload, err := json.Marshal(message)
		if err != nil {
			return TResponseInputItem{}, false
		}
		item, err := unmarshalMessageData(string(payload))
		if err != nil {
			return TResponseInputItem{}, false
		}
		return item, true
	default:
		return TResponseInputItem{}, false
	}
}

func (s *DaprSession) shouldRetryOnConcurrencyConflict(err error, attempt int) bool {
	if attempt >= daprMaxWriteAttempts {
		return false
	}
	message := strings.ToLower(err.Error())
	for _, marker := range []string{
		"etag mismatch",
		"etag does not match",
		"precondition failed",
		"concurrency conflict",
		"invalid etag",
		"failed to set key",
		"user_script",
	} {
		if strings.Contains(message, marker) {
			return true
		}
	}
	return false
}

func (s *DaprSession) retryDelay(attempt int) time.Duration {
	exponent := math.Pow(2, float64(max(0, attempt-1)))
	baseDelay := time.Duration(float64(daprRetryBaseDelay) * exponent)
	if baseDelay > daprRetryMaxDelay {
		baseDelay = daprRetryMaxDelay
	}
	jitter := time.Duration(float64(baseDelay) * 0.1 * s.randFloat64())
	return baseDelay + jitter
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

var _ Session = (*DaprSession)(nil)
