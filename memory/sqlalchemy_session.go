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
	"fmt"
	"strings"
)

// SQLAlchemySession is a compatibility session wrapper that maps SQLAlchemy-like
// URLs to Go session implementations.
//
// Supported URL schemes:
// - PostgreSQL: postgresql+asyncpg://..., postgresql://..., postgres://...
// - SQLite: sqlite+aiosqlite://..., sqlite://...
type SQLAlchemySession struct {
	session Session
	closeFn func(context.Context) error
}

type SQLAlchemySessionParams struct {
	// Optional pre-configured backend session.
	Session Session

	// SQLAlchemy-like connection URL.
	URL string

	// Session id used when URL-based construction is used.
	SessionID string

	// Optional table overrides used when URL-based construction is used.
	SessionsTable string
	MessagesTable string
}

func NewSQLAlchemySession(ctx context.Context, params SQLAlchemySessionParams) (*SQLAlchemySession, error) {
	if params.Session != nil {
		s := &SQLAlchemySession{session: params.Session}
		s.closeFn = inferSessionCloseFn(params.Session)
		return s, nil
	}

	driver, dsn, err := normalizeSQLAlchemyURL(params.URL)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(params.SessionID) == "" {
		return nil, fmt.Errorf("session id is required")
	}

	s := &SQLAlchemySession{}
	switch driver {
	case "sqlite":
		sqliteSession, err := NewSQLiteSession(ctx, SQLiteSessionParams{
			SessionID:        params.SessionID,
			DBDataSourceName: dsn,
			SessionTable:     params.SessionsTable,
			MessagesTable:    params.MessagesTable,
		})
		if err != nil {
			return nil, err
		}
		s.session = sqliteSession
		s.closeFn = func(context.Context) error { return sqliteSession.Close() }
	case "postgres":
		pgSession, err := NewPgSession(ctx, PgSessionParams{
			SessionID:        params.SessionID,
			ConnectionString: dsn,
			SessionTable:     params.SessionsTable,
			MessagesTable:    params.MessagesTable,
		})
		if err != nil {
			return nil, err
		}
		s.session = pgSession
		s.closeFn = pgSession.Close
	default:
		return nil, fmt.Errorf("unsupported SQLAlchemy driver %q", driver)
	}

	return s, nil
}

func (s *SQLAlchemySession) SessionID(ctx context.Context) string {
	return s.session.SessionID(ctx)
}

func (s *SQLAlchemySession) GetItems(ctx context.Context, limit int) ([]TResponseInputItem, error) {
	return s.session.GetItems(ctx, limit)
}

func (s *SQLAlchemySession) AddItems(ctx context.Context, items []TResponseInputItem) error {
	return s.session.AddItems(ctx, items)
}

func (s *SQLAlchemySession) PopItem(ctx context.Context) (*TResponseInputItem, error) {
	return s.session.PopItem(ctx)
}

func (s *SQLAlchemySession) ClearSession(ctx context.Context) error {
	return s.session.ClearSession(ctx)
}

func (s *SQLAlchemySession) Close(ctx context.Context) error {
	if s.closeFn != nil {
		return s.closeFn(ctx)
	}
	return nil
}

func normalizeSQLAlchemyURL(rawURL string) (driver, dsn string, err error) {
	u := strings.TrimSpace(rawURL)
	if u == "" {
		return "", "", fmt.Errorf("sqlalchemy url is required")
	}

	switch {
	case strings.HasPrefix(u, "postgresql+asyncpg://"):
		return "postgres", "postgres://" + strings.TrimPrefix(u, "postgresql+asyncpg://"), nil
	case strings.HasPrefix(u, "postgresql://"):
		return "postgres", u, nil
	case strings.HasPrefix(u, "postgres://"):
		return "postgres", u, nil
	case strings.HasPrefix(u, "sqlite+aiosqlite://"):
		return "sqlite", normalizeSQLiteURLPath(strings.TrimPrefix(u, "sqlite+aiosqlite://")), nil
	case strings.HasPrefix(u, "sqlite://"):
		return "sqlite", normalizeSQLiteURLPath(strings.TrimPrefix(u, "sqlite://")), nil
	default:
		return "", "", fmt.Errorf("unsupported sqlalchemy url scheme: %q", u)
	}
}

func normalizeSQLiteURLPath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" || path == "/" {
		return "file::memory:?cache=shared"
	}

	if path == "/:memory:" || path == ":memory:" {
		return ":memory:"
	}

	if strings.HasPrefix(path, "/file:") {
		return strings.TrimPrefix(path, "/")
	}

	// SQLAlchemy absolute SQLite URLs commonly end up with double-leading slash here.
	if strings.HasPrefix(path, "//") {
		return "/" + strings.TrimLeft(path, "/")
	}

	// Relative path from `sqlite:///relative.db`.
	if strings.HasPrefix(path, "/") {
		return strings.TrimPrefix(path, "/")
	}

	return path
}

func inferSessionCloseFn(session Session) func(context.Context) error {
	if pgLike, ok := session.(interface{ Close(context.Context) error }); ok {
		return pgLike.Close
	}
	if sqliteLike, ok := session.(interface{ Close() error }); ok {
		return func(context.Context) error { return sqliteLike.Close() }
	}
	return nil
}

var _ Session = (*SQLAlchemySession)(nil)
