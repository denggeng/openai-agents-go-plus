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

package realtime

import (
	"context"
	"os"
	"time"
)

// RealtimeAPIKeyProvider dynamically resolves API keys.
type RealtimeAPIKeyProvider func(context.Context) (string, error)

// RealtimeTransportConfig configures low-level websocket transport behavior.
type RealtimeTransportConfig struct {
	PingInterval     *time.Duration
	PingTimeout      *time.Duration
	HandshakeTimeout *time.Duration
}

// RealtimeWebSocketConn is the minimal websocket contract used by realtime transports.
type RealtimeWebSocketConn interface {
	ReadMessage() (messageType int, p []byte, err error)
	WriteJSON(v any) error
	Close() error
}

// RealtimeWebSocketDialer dials a websocket connection for realtime transport usage.
type RealtimeWebSocketDialer func(
	context.Context,
	string,
	map[string]string,
	*RealtimeTransportConfig,
) (RealtimeWebSocketConn, error)

// RealtimeModelConfig contains transport connection options.
type RealtimeModelConfig struct {
	APIKey          string
	APIKeyProvider  RealtimeAPIKeyProvider
	URL             string
	Headers         map[string]string
	EnableTransport bool
	TransportDialer RealtimeWebSocketDialer
	TransportConfig *RealtimeTransportConfig
	InitialSettings RealtimeSessionModelSettings
	PlaybackTracker *RealtimePlaybackTracker
	CallID          string
}

// ResolveAPIKey resolves API key from config string/provider/environment.
func (c RealtimeModelConfig) ResolveAPIKey(ctx context.Context) (string, error) {
	if c.APIKey != "" {
		return c.APIKey, nil
	}
	if c.APIKeyProvider != nil {
		return c.APIKeyProvider(ctx)
	}
	return os.Getenv("OPENAI_API_KEY"), nil
}

// RealtimeModelListener receives events emitted by a realtime model transport.
type RealtimeModelListener interface {
	OnEvent(context.Context, RealtimeModelEvent) error
}

// RealtimeModel defines the realtime transport contract.
type RealtimeModel interface {
	Connect(context.Context, RealtimeModelConfig) error
	AddListener(RealtimeModelListener)
	RemoveListener(RealtimeModelListener)
	SendEvent(context.Context, RealtimeModelSendEvent) error
	Close(context.Context) error
}
