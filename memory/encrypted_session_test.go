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
	"crypto/rand"
	"encoding/base64"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testEncryptionKey(t *testing.T) string {
	t.Helper()
	key := make([]byte, 32)
	_, err := rand.Read(key)
	require.NoError(t, err)
	return base64.RawURLEncoding.EncodeToString(key)
}

func testMessage(role responses.EasyInputMessageRole, text string) TResponseInputItem {
	return TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{
				OfString: param.NewOpt(text),
			},
			Role: role,
			Type: responses.EasyInputMessageTypeMessage,
		},
	}
}

func TestEncryptedSession_BasicRoundTrip(t *testing.T) {
	ctx := t.Context()
	underlying, err := NewSQLiteSession(ctx, SQLiteSessionParams{
		SessionID:        "enc-basic",
		DBDataSourceName: filepath.Join(t.TempDir(), "enc-basic.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, underlying.Close()) })

	session, err := NewEncryptedSession(EncryptedSessionParams{
		SessionID:         "enc-basic",
		UnderlyingSession: underlying,
		EncryptionKey:     testEncryptionKey(t),
		TTL:               10 * time.Minute,
	})
	require.NoError(t, err)

	items := []TResponseInputItem{
		testMessage(responses.EasyInputMessageRoleUser, "Hello"),
		testMessage(responses.EasyInputMessageRoleAssistant, "Hi there!"),
	}
	require.NoError(t, session.AddItems(ctx, items))

	retrieved, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, items, retrieved)

	underlyingItems, err := underlying.GetItems(ctx, 0)
	require.NoError(t, err)
	require.Len(t, underlyingItems, 2)
	for _, item := range underlyingItems {
		require.NotNil(t, item.OfCompaction)
		assert.NotContains(t, item.OfCompaction.EncryptedContent, "Hello")
		assert.NotContains(t, item.OfCompaction.EncryptedContent, "Hi there!")
	}
}

func TestEncryptedSession_TTLExpirationSkipsItems(t *testing.T) {
	ctx := t.Context()
	underlying, err := NewSQLiteSession(ctx, SQLiteSessionParams{
		SessionID:        "enc-ttl",
		DBDataSourceName: filepath.Join(t.TempDir(), "enc-ttl.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, underlying.Close()) })

	now := time.Unix(1_700_000_000, 0)
	session, err := NewEncryptedSession(EncryptedSessionParams{
		SessionID:         "enc-ttl",
		UnderlyingSession: underlying,
		EncryptionKey:     testEncryptionKey(t),
		TTL:               time.Second,
		Now: func() time.Time {
			return now
		},
	})
	require.NoError(t, err)

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		testMessage(responses.EasyInputMessageRoleUser, "Expires soon"),
	}))

	now = now.Add(2 * time.Second)

	retrieved, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Empty(t, retrieved)

	underlyingItems, err := underlying.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Len(t, underlyingItems, 1)
}

func TestEncryptedSession_PopSkipsExpiredThenContinues(t *testing.T) {
	ctx := t.Context()
	underlying, err := NewSQLiteSession(ctx, SQLiteSessionParams{
		SessionID:        "enc-pop",
		DBDataSourceName: filepath.Join(t.TempDir(), "enc-pop.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, underlying.Close()) })

	now := time.Unix(1_700_000_000, 0)
	session, err := NewEncryptedSession(EncryptedSessionParams{
		SessionID:         "enc-pop",
		UnderlyingSession: underlying,
		EncryptionKey:     testEncryptionKey(t),
		TTL:               2 * time.Second,
		Now: func() time.Time {
			return now
		},
	})
	require.NoError(t, err)

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		testMessage(responses.EasyInputMessageRoleUser, "Old message"),
	}))
	now = now.Add(3 * time.Second)
	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		testMessage(responses.EasyInputMessageRoleAssistant, "Fresh message"),
	}))

	first, err := session.PopItem(ctx)
	require.NoError(t, err)
	require.NotNil(t, first)
	assert.Equal(t, testMessage(responses.EasyInputMessageRoleAssistant, "Fresh message"), *first)

	second, err := session.PopItem(ctx)
	require.NoError(t, err)
	assert.Nil(t, second)
}

func TestEncryptedSession_RawStringMasterKey(t *testing.T) {
	ctx := t.Context()
	underlying, err := NewSQLiteSession(ctx, SQLiteSessionParams{
		SessionID:        "enc-raw-key",
		DBDataSourceName: filepath.Join(t.TempDir(), "enc-raw-key.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, underlying.Close()) })

	session, err := NewEncryptedSession(EncryptedSessionParams{
		SessionID:         "enc-raw-key",
		UnderlyingSession: underlying,
		EncryptionKey:     "my-secret-password",
	})
	require.NoError(t, err)

	items := []TResponseInputItem{testMessage(responses.EasyInputMessageRoleUser, "Test")}
	require.NoError(t, session.AddItems(ctx, items))

	retrieved, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, items, retrieved)
}

func TestEncryptedSession_PassthroughForUnencryptedEntries(t *testing.T) {
	ctx := t.Context()
	underlying, err := NewSQLiteSession(ctx, SQLiteSessionParams{
		SessionID:        "enc-pass",
		DBDataSourceName: filepath.Join(t.TempDir(), "enc-pass.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, underlying.Close()) })

	plainItems := []TResponseInputItem{testMessage(responses.EasyInputMessageRoleUser, "Plain text")}
	require.NoError(t, underlying.AddItems(ctx, plainItems))

	session, err := NewEncryptedSession(EncryptedSessionParams{
		SessionID:         "enc-pass",
		UnderlyingSession: underlying,
		EncryptionKey:     testEncryptionKey(t),
	})
	require.NoError(t, err)

	retrieved, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, plainItems, retrieved)
}

func TestNewEncryptedSession_Validation(t *testing.T) {
	ctx := t.Context()
	underlying, err := NewSQLiteSession(ctx, SQLiteSessionParams{
		SessionID:        "enc-validation",
		DBDataSourceName: filepath.Join(t.TempDir(), "enc-validation.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, underlying.Close()) })

	_, err = NewEncryptedSession(EncryptedSessionParams{
		SessionID:         "enc-validation",
		UnderlyingSession: nil,
		EncryptionKey:     testEncryptionKey(t),
	})
	require.ErrorContains(t, err, "underlying session")

	_, err = NewEncryptedSession(EncryptedSessionParams{
		SessionID:         "enc-validation",
		UnderlyingSession: underlying,
		EncryptionKey:     "",
	})
	require.ErrorContains(t, err, "encryption key")

	_, err = NewEncryptedSession(EncryptedSessionParams{
		SessionID:         "",
		UnderlyingSession: underlying,
		EncryptionKey:     testEncryptionKey(t),
	})
	require.NoError(t, err)
}

func TestEncryptedSession_CorruptedEnvelopeIsSkipped(t *testing.T) {
	ctx := t.Context()
	underlying, err := NewSQLiteSession(ctx, SQLiteSessionParams{
		SessionID:        "enc-corrupt",
		DBDataSourceName: filepath.Join(t.TempDir(), "enc-corrupt.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, underlying.Close()) })

	badEnvelope := responses.ResponseInputItemParamOfCompaction(`{"__enc__":1,"v":1,"kid":"hkdf-v1","payload":"bad","nonce":"bad","ts":1700000000}`)
	require.NoError(t, underlying.AddItems(ctx, []TResponseInputItem{badEnvelope}))

	session, err := NewEncryptedSession(EncryptedSessionParams{
		SessionID:         "enc-corrupt",
		UnderlyingSession: underlying,
		EncryptionKey:     testEncryptionKey(t),
	})
	require.NoError(t, err)

	retrieved, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Empty(t, retrieved)

	popped, err := session.PopItem(ctx)
	require.NoError(t, err)
	assert.Nil(t, popped)
}

func TestEncryptedSession_CloseDelegatesWhenSupported(t *testing.T) {
	ctx := t.Context()
	type closeableSession interface {
		Session
		Close() error
	}

	sqliteSession, err := NewSQLiteSession(ctx, SQLiteSessionParams{
		SessionID:        "enc-close",
		DBDataSourceName: filepath.Join(t.TempDir(), "enc-close.db"),
	})
	require.NoError(t, err)

	session, err := NewEncryptedSession(EncryptedSessionParams{
		SessionID:         "enc-close",
		UnderlyingSession: closeableSession(sqliteSession),
		EncryptionKey:     testEncryptionKey(t),
	})
	require.NoError(t, err)

	require.NoError(t, session.Close())

	_, err = sqliteSession.GetItems(ctx, 0)
	require.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), "closed") || strings.Contains(err.Error(), "database is closed"))
}
