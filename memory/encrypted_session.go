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
	"crypto/aes"
	"crypto/cipher"
	"crypto/hkdf"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

const (
	encryptedSessionInfoLabel = "agents.session-store.hkdf.v1"
	encryptedSessionKeyID     = "hkdf-v1"
	encryptedSessionVersion   = 1
)

// EncryptedSession wraps any Session and transparently encrypts stored items.
//
// Items are persisted as `compaction` input items containing encrypted payloads.
// During reads, valid encrypted items are decrypted back into their original
// input items. Expired or corrupted encrypted items are skipped.
type EncryptedSession struct {
	sessionID  string
	underlying Session
	aead       cipher.AEAD
	ttl        time.Duration
	now        func() time.Time
}

type EncryptedSessionParams struct {
	// Optional session id. If omitted, it is derived from UnderlyingSession.SessionID.
	SessionID string

	// Underlying session implementation used for storage.
	UnderlyingSession Session

	// Master encryption key. Accepts either:
	// - a base64-url encoded 32-byte key
	// - any raw secret string
	EncryptionKey string

	// Token TTL. Expired encrypted entries are skipped when reading.
	// Defaults to 10 minutes when <= 0.
	TTL time.Duration

	// Optional clock source for deterministic testing.
	Now func() time.Time
}

type encryptedEnvelope struct {
	Enc     int    `json:"__enc__"`
	Version int    `json:"v"`
	KeyID   string `json:"kid"`
	TS      int64  `json:"ts"`
	Nonce   string `json:"nonce"`
	Payload string `json:"payload"`
}

// NewEncryptedSession creates a Session wrapper that encrypts/decrypts
// items around an underlying session backend.
func NewEncryptedSession(params EncryptedSessionParams) (*EncryptedSession, error) {
	if params.UnderlyingSession == nil {
		return nil, fmt.Errorf("underlying session is required")
	}

	masterBytes, err := ensureMasterKeyBytes(params.EncryptionKey)
	if err != nil {
		return nil, err
	}

	sessionID := strings.TrimSpace(params.SessionID)
	if sessionID == "" {
		sessionID = params.UnderlyingSession.SessionID(context.Background())
	}
	if sessionID == "" {
		return nil, fmt.Errorf("session id is required")
	}

	derivedKey, err := hkdf.Key(sha256.New, masterBytes, []byte(sessionID), encryptedSessionInfoLabel, 32)
	if err != nil {
		return nil, fmt.Errorf("derive key: %w", err)
	}
	block, err := aes.NewCipher(derivedKey)
	if err != nil {
		return nil, fmt.Errorf("create cipher: %w", err)
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("create AEAD: %w", err)
	}

	ttl := params.TTL
	if ttl <= 0 {
		ttl = 10 * time.Minute
	}

	nowFn := params.Now
	if nowFn == nil {
		nowFn = time.Now
	}

	return &EncryptedSession{
		sessionID:  sessionID,
		underlying: params.UnderlyingSession,
		aead:       aead,
		ttl:        ttl,
		now:        nowFn,
	}, nil
}

func (s *EncryptedSession) SessionID(context.Context) string {
	return s.sessionID
}

func (s *EncryptedSession) GetItems(ctx context.Context, limit int) ([]TResponseInputItem, error) {
	items, err := s.underlying.GetItems(ctx, limit)
	if err != nil {
		return nil, err
	}

	decrypted := make([]TResponseInputItem, 0, len(items))
	for _, item := range items {
		decoded, ok := s.unwrapItem(item)
		if ok {
			decrypted = append(decrypted, decoded)
		}
	}

	if limit > 0 {
		decrypted = trimOrphanedCallOutputAtHead(decrypted)
	}
	return decrypted, nil
}

func (s *EncryptedSession) AddItems(ctx context.Context, items []TResponseInputItem) error {
	if len(items) == 0 {
		return nil
	}

	wrapped := make([]TResponseInputItem, 0, len(items))
	for _, item := range items {
		encItem, err := s.wrapItem(item)
		if err != nil {
			return err
		}
		wrapped = append(wrapped, encItem)
	}
	return s.underlying.AddItems(ctx, wrapped)
}

func (s *EncryptedSession) PopItem(ctx context.Context) (*TResponseInputItem, error) {
	for {
		item, err := s.underlying.PopItem(ctx)
		if err != nil {
			return nil, err
		}
		if item == nil {
			return nil, nil
		}

		decoded, ok := s.unwrapItem(*item)
		if ok {
			return &decoded, nil
		}
	}
}

func (s *EncryptedSession) ClearSession(ctx context.Context) error {
	return s.underlying.ClearSession(ctx)
}

// Close delegates to the underlying session when it supports Close().
func (s *EncryptedSession) Close() error {
	if closer, ok := s.underlying.(interface{ Close() error }); ok {
		return closer.Close()
	}
	return nil
}

func (s *EncryptedSession) wrapItem(item TResponseInputItem) (TResponseInputItem, error) {
	plain, err := item.MarshalJSON()
	if err != nil {
		return TResponseInputItem{}, fmt.Errorf("marshal input item: %w", err)
	}

	nonce := make([]byte, s.aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return TResponseInputItem{}, fmt.Errorf("generate nonce: %w", err)
	}

	ciphertext := s.aead.Seal(nil, nonce, plain, []byte(s.sessionID))
	env := encryptedEnvelope{
		Enc:     1,
		Version: encryptedSessionVersion,
		KeyID:   encryptedSessionKeyID,
		TS:      s.now().Unix(),
		Nonce:   base64.RawURLEncoding.EncodeToString(nonce),
		Payload: base64.RawURLEncoding.EncodeToString(ciphertext),
	}
	envBytes, err := json.Marshal(env)
	if err != nil {
		return TResponseInputItem{}, fmt.Errorf("marshal encrypted envelope: %w", err)
	}

	return responses.ResponseInputItemParamOfCompaction(string(envBytes)), nil
}

func (s *EncryptedSession) unwrapItem(item TResponseInputItem) (TResponseInputItem, bool) {
	if item.OfCompaction == nil {
		return item, true
	}

	content := strings.TrimSpace(item.OfCompaction.EncryptedContent)
	if content == "" {
		return item, true
	}

	var env encryptedEnvelope
	if err := json.Unmarshal([]byte(content), &env); err != nil || env.Enc != 1 {
		// Not one of our encrypted envelopes, pass it through unchanged.
		return item, true
	}

	if env.TS <= 0 || env.Version != encryptedSessionVersion || env.KeyID != encryptedSessionKeyID {
		return TResponseInputItem{}, false
	}

	if s.ttl > 0 {
		messageTime := time.Unix(env.TS, 0)
		if s.now().After(messageTime.Add(s.ttl)) {
			return TResponseInputItem{}, false
		}
	}

	nonce, err := base64.RawURLEncoding.DecodeString(env.Nonce)
	if err != nil {
		return TResponseInputItem{}, false
	}
	ciphertext, err := base64.RawURLEncoding.DecodeString(env.Payload)
	if err != nil {
		return TResponseInputItem{}, false
	}
	plaintext, err := s.aead.Open(nil, nonce, ciphertext, []byte(s.sessionID))
	if err != nil {
		return TResponseInputItem{}, false
	}

	var decoded TResponseInputItem
	if err := json.Unmarshal(plaintext, &decoded); err != nil {
		return TResponseInputItem{}, false
	}
	return decoded, true
}

func ensureMasterKeyBytes(masterKey string) ([]byte, error) {
	masterKey = strings.TrimSpace(masterKey)
	if masterKey == "" {
		return nil, fmt.Errorf("encryption key is required")
	}

	for _, encoding := range []*base64.Encoding{
		base64.RawURLEncoding,
		base64.URLEncoding,
		base64.RawStdEncoding,
		base64.StdEncoding,
	} {
		decoded, err := encoding.DecodeString(masterKey)
		if err == nil && len(decoded) == 32 {
			return decoded, nil
		}
	}

	return []byte(masterKey), nil
}

func trimOrphanedCallOutputAtHead(items []TResponseInputItem) []TResponseInputItem {
	if len(items) == 0 {
		return items
	}
	itemType := items[0].GetType()
	if itemType == nil {
		return items
	}
	switch *itemType {
	case string(constant.ValueOf[constant.FunctionCallOutput]()):
		return slices.Delete(items, 0, 1)
	case string(constant.ValueOf[constant.ComputerCallOutput]()):
		return slices.Delete(items, 0, 1)
	case string(constant.ValueOf[constant.LocalShellCallOutput]()):
		return slices.Delete(items, 0, 1)
	case string(constant.ValueOf[constant.CustomToolCallOutput]()):
		return slices.Delete(items, 0, 1)
	default:
		return items
	}
}

// compile-time interface check.
var _ Session = (*EncryptedSession)(nil)
