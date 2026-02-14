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
	"path/filepath"
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func sqlalchemyTestMessage(role responses.EasyInputMessageRole, text string) TResponseInputItem {
	return TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(text)},
			Role:    role,
			Type:    responses.EasyInputMessageTypeMessage,
		},
	}
}

func TestSQLAlchemySession_SQLiteURL_DirectOps(t *testing.T) {
	ctx := t.Context()
	dbPath := filepath.Join(t.TempDir(), "sqlalchemy_test.db")

	session, err := NewSQLAlchemySession(ctx, SQLAlchemySessionParams{
		SessionID: "sqlalchemy-sqlite",
		URL:       "sqlite:///" + dbPath,
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close(ctx)) })

	items := []TResponseInputItem{
		sqlalchemyTestMessage(responses.EasyInputMessageRoleUser, "Hello"),
		sqlalchemyTestMessage(responses.EasyInputMessageRoleAssistant, "Hi there!"),
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

func TestSQLAlchemySession_InferCloseFromInjectedSession(t *testing.T) {
	ctx := t.Context()
	dbPath := filepath.Join(t.TempDir(), "sqlalchemy_injected.db")

	sqliteSession, err := NewSQLiteSession(ctx, SQLiteSessionParams{
		SessionID:        "sqlalchemy-injected",
		DBDataSourceName: dbPath,
	})
	require.NoError(t, err)

	session, err := NewSQLAlchemySession(ctx, SQLAlchemySessionParams{Session: sqliteSession})
	require.NoError(t, err)
	require.NoError(t, session.Close(ctx))

	_, err = sqliteSession.GetItems(ctx, 0)
	require.Error(t, err)
}

func TestNormalizeSQLAlchemyURL(t *testing.T) {
	tests := []struct {
		name       string
		url        string
		wantDriver string
		wantDSN    string
		wantErr    string
	}{
		{
			name:       "postgres asyncpg",
			url:        "postgresql+asyncpg://app:secret@db.example.com/agents",
			wantDriver: "postgres",
			wantDSN:    "postgres://app:secret@db.example.com/agents",
		},
		{
			name:       "postgres direct",
			url:        "postgres://app:secret@db.example.com/agents",
			wantDriver: "postgres",
			wantDSN:    "postgres://app:secret@db.example.com/agents",
		},
		{
			name:       "sqlite aiosqlite memory",
			url:        "sqlite+aiosqlite:///:memory:",
			wantDriver: "sqlite",
			wantDSN:    ":memory:",
		},
		{
			name:       "sqlite absolute",
			url:        "sqlite:////tmp/my.db",
			wantDriver: "sqlite",
			wantDSN:    "/tmp/my.db",
		},
		{
			name:    "unsupported scheme",
			url:     "mysql+aiomysql://user:pass@localhost/db",
			wantErr: "unsupported sqlalchemy url scheme",
		},
		{
			name:    "empty url",
			url:     "",
			wantErr: "sqlalchemy url is required",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			driver, dsn, err := normalizeSQLAlchemyURL(tc.url)
			if tc.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.wantDriver, driver)
			assert.Equal(t, tc.wantDSN, dsn)
		})
	}
}

func TestNewSQLAlchemySession_Validation(t *testing.T) {
	ctx := t.Context()

	_, err := NewSQLAlchemySession(ctx, SQLAlchemySessionParams{})
	require.ErrorContains(t, err, "sqlalchemy url is required")

	_, err = NewSQLAlchemySession(ctx, SQLAlchemySessionParams{URL: "sqlite:///:memory:"})
	require.ErrorContains(t, err, "session id is required")

	_, err = NewSQLAlchemySession(ctx, SQLAlchemySessionParams{SessionID: "x", URL: "unknown://foo"})
	require.ErrorContains(t, err, "unsupported sqlalchemy url scheme")
}

func TestNormalizeSQLiteURLPath(t *testing.T) {
	assert.Equal(t, "file::memory:?cache=shared", normalizeSQLiteURLPath(""))
	assert.Equal(t, ":memory:", normalizeSQLiteURLPath("/:memory:"))
	assert.Equal(t, ":memory:", normalizeSQLiteURLPath(":memory:"))
	assert.Equal(t, "/tmp/test.db", normalizeSQLiteURLPath("//tmp/test.db"))
	assert.Equal(t, "file:test.db", normalizeSQLiteURLPath("/file:test.db"))
	assert.Equal(t, "relative.db", normalizeSQLiteURLPath("relative.db"))
	assert.Equal(t, "/tmp/test.db", normalizeSQLiteURLPath("///tmp/test.db"))
}

func TestSQLAlchemySession_SessionIDDelegates(t *testing.T) {
	ctx := t.Context()
	dbPath := filepath.Join(t.TempDir(), "sqlalchemy_session_id.db")

	session, err := NewSQLAlchemySession(ctx, SQLAlchemySessionParams{
		SessionID: "sqlalchemy-id",
		URL:       fmt.Sprintf("sqlite:///%s", dbPath),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close(ctx)) })

	assert.Equal(t, "sqlalchemy-id", session.SessionID(ctx))
}
