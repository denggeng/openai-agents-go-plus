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
	"cmp"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"sync"

	_ "github.com/mattn/go-sqlite3"
)

// AsyncSQLiteSession is a SQLite-based session implementation mirroring the
// async Python session semantics.
type AsyncSQLiteSession struct {
	sessionID     string
	dbDSN         string
	sessionTable  string
	messagesTable string
	sessionSettings *SessionSettings
	db            *sql.DB
	mu            sync.Mutex
}

type AsyncSQLiteSessionParams struct {
	// Unique identifier for the conversation session.
	SessionID string

	// Optional database data source name. Defaults to `file::memory:?cache=shared`.
	DBDataSourceName string

	// Optional name of the table to store session metadata.
	SessionTable string

	// Optional name of the table to store message data.
	MessagesTable string

	// Optional session settings (e.g., default history limit).
	SessionSettings *SessionSettings
}

// NewAsyncSQLiteSession initializes the async SQLite session.
func NewAsyncSQLiteSession(ctx context.Context, params AsyncSQLiteSessionParams) (_ *AsyncSQLiteSession, err error) {
	settings := params.SessionSettings
	if settings == nil {
		settings = &SessionSettings{}
	}
	s := &AsyncSQLiteSession{
		sessionID:     params.SessionID,
		dbDSN:         cmp.Or(params.DBDataSourceName, "file::memory:?cache=shared"),
		sessionTable:  cmp.Or(params.SessionTable, "agent_sessions"),
		messagesTable: cmp.Or(params.MessagesTable, "agent_messages"),
		sessionSettings: settings,
	}

	defer func() {
		if err != nil {
			if e := s.Close(); e != nil {
				err = errors.Join(err, e)
			}
		}
	}()

	s.db, err = sql.Open("sqlite3", s.dbDSN)
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite3 database: %w", err)
	}

	_, err = s.db.ExecContext(ctx, `PRAGMA journal_mode=WAL`)
	if err != nil {
		return nil, fmt.Errorf("failed to set journal mode: %w", err)
	}

	if err := s.initDB(ctx); err != nil {
		return nil, err
	}

	return s, nil
}

func (s *AsyncSQLiteSession) SessionID(context.Context) string {
	return s.sessionID
}

func (s *AsyncSQLiteSession) SessionSettings() *SessionSettings {
	return s.sessionSettings
}

func (s *AsyncSQLiteSession) GetItems(ctx context.Context, limit int) (_ []TResponseInputItem, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var rows *sql.Rows
	if limit <= 0 {
		rows, err = s.db.QueryContext(ctx, fmt.Sprintf(`
			SELECT message_data FROM "%s"
			WHERE session_id = ?
			ORDER BY id ASC
		`, s.messagesTable), s.sessionID)
	} else {
		rows, err = s.db.QueryContext(ctx, fmt.Sprintf(`
			SELECT message_data FROM "%s"
			WHERE session_id = ?
			ORDER BY id DESC
			LIMIT ?
		`, s.messagesTable), s.sessionID, limit)
	}
	if err != nil {
		return nil, fmt.Errorf("error querying session items: %w", err)
	}
	defer func() {
		if e := rows.Close(); e != nil {
			err = errors.Join(err, fmt.Errorf("error closing sql.Rows: %w", e))
		}
	}()

	var items []TResponseInputItem
	for rows.Next() {
		var messageData string
		if err = rows.Scan(&messageData); err != nil {
			return nil, fmt.Errorf("sql rows scan error: %w", err)
		}

		item, err := unmarshalMessageData(messageData)
		if err != nil {
			continue
		}
		items = append(items, item)
	}
	if err = rows.Err(); err != nil {
		return nil, fmt.Errorf("sql rows scan error: %w", err)
	}

	if limit > 0 {
		slices.Reverse(items)
	}

	return items, nil
}

func (s *AsyncSQLiteSession) AddItems(ctx context.Context, items []TResponseInputItem) error {
	if len(items) == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.ExecContext(
		ctx,
		fmt.Sprintf(`INSERT OR IGNORE INTO "%s" (session_id) VALUES (?)`, s.sessionTable),
		s.sessionID,
	)
	if err != nil {
		return err
	}

	for _, item := range items {
		jsonItem, err := item.MarshalJSON()
		if err != nil {
			return fmt.Errorf("error JSON marshaling item: %w", err)
		}
		_, err = s.db.ExecContext(
			ctx,
			fmt.Sprintf(`INSERT INTO "%s" (session_id, message_data) VALUES (?, ?)`, s.messagesTable),
			s.sessionID, string(jsonItem),
		)
		if err != nil {
			return fmt.Errorf("error inserting item in messages table: %w", err)
		}
	}

	_, err = s.db.ExecContext(
		ctx,
		fmt.Sprintf(`UPDATE "%s" SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?`, s.sessionTable),
		s.sessionID,
	)
	if err != nil {
		return fmt.Errorf("error updating session timestamp: %w", err)
	}

	return nil
}

func (s *AsyncSQLiteSession) PopItem(ctx context.Context) (*TResponseInputItem, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var messageData string
	err := s.db.QueryRowContext(
		ctx,
		fmt.Sprintf(`
			DELETE FROM "%s"
			WHERE id = (
				SELECT id FROM "%s"
				WHERE session_id = ?
				ORDER BY id DESC
				LIMIT 1
			)
			RETURNING message_data
		`, s.messagesTable, s.messagesTable),
		s.sessionID,
	).Scan(&messageData)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("sql delete error: %w", err)
	}

	item, err := unmarshalMessageData(messageData)
	if err != nil {
		return nil, nil
	}
	return &item, nil
}

func (s *AsyncSQLiteSession) ClearSession(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.ExecContext(ctx, fmt.Sprintf(
		`DELETE FROM "%s" WHERE session_id = ?`, s.messagesTable,
	), s.sessionID)
	if err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, fmt.Sprintf(
		`DELETE FROM "%s" WHERE session_id = ?`, s.sessionTable,
	), s.sessionID)
	return err
}

// Close the database connection.
func (s *AsyncSQLiteSession) Close() error {
	if s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *AsyncSQLiteSession) initDB(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, fmt.Sprintf(
		`CREATE TABLE IF NOT EXISTS "%s" (
			session_id TEXT PRIMARY KEY,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`, s.sessionTable,
	))
	if err != nil {
		return fmt.Errorf("error creating session table: %w", err)
	}

	_, err = s.db.ExecContext(ctx, fmt.Sprintf(
		`CREATE TABLE IF NOT EXISTS "%s" (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT NOT NULL,
			message_data TEXT NOT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY (session_id) REFERENCES "%s" (session_id) ON DELETE CASCADE
		)`, s.messagesTable, s.sessionTable,
	))
	if err != nil {
		return fmt.Errorf("error creating messages table: %w", err)
	}

	_, err = s.db.ExecContext(ctx, fmt.Sprintf(
		`CREATE INDEX IF NOT EXISTS "idx_%s_session_id" ON "%s" (session_id, id)`,
		s.messagesTable, s.messagesTable,
	))
	if err != nil {
		return fmt.Errorf("error creating index: %w", err)
	}

	return nil
}
