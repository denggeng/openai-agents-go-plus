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
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newWebSocketTestServer(delay time.Duration) (*httptest.Server, string) {
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if delay > 0 {
			time.Sleep(delay)
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	}))
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")
	return server, wsURL
}

func TestTransportPingKeepsConnectionAlive(t *testing.T) {
	server, wsURL := newWebSocketTestServer(0)
	defer server.Close()

	pingInterval := 50 * time.Millisecond
	pingTimeout := 250 * time.Millisecond

	model := NewOpenAIRealtimeWebSocketModel()
	require.NoError(t, model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		URL:             wsURL,
		EnableTransport: true,
		TransportConfig: &RealtimeTransportConfig{
			PingInterval: &pingInterval,
			PingTimeout:  &pingTimeout,
		},
		InitialSettings: RealtimeSessionModelSettings{},
	}))

	time.Sleep(200 * time.Millisecond)

	select {
	case <-model.websocketDone:
		t.Fatal("websocket connection closed unexpectedly")
	default:
	}

	require.NoError(t, model.Close(t.Context()))
}

func TestTransportHandshakeTimeout(t *testing.T) {
	server, wsURL := newWebSocketTestServer(200 * time.Millisecond)
	defer server.Close()

	shortTimeout := 20 * time.Millisecond
	model := NewOpenAIRealtimeWebSocketModel()
	err := model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		URL:             wsURL,
		EnableTransport: true,
		TransportConfig: &RealtimeTransportConfig{
			HandshakeTimeout: &shortTimeout,
		},
		InitialSettings: RealtimeSessionModelSettings{},
	})
	require.Error(t, err)

	longTimeout := 500 * time.Millisecond
	model2 := NewOpenAIRealtimeWebSocketModel()
	require.NoError(t, model2.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		URL:             wsURL,
		EnableTransport: true,
		TransportConfig: &RealtimeTransportConfig{
			HandshakeTimeout: &longTimeout,
		},
		InitialSettings: RealtimeSessionModelSettings{},
	}))
	require.NoError(t, model2.Close(t.Context()))
}

func TestTransportPingDisabledVsEnabled(t *testing.T) {
	server, wsURL := newWebSocketTestServer(0)
	defer server.Close()

	model := NewOpenAIRealtimeWebSocketModel()
	require.NoError(t, model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		URL:             wsURL,
		EnableTransport: true,
		TransportConfig: &RealtimeTransportConfig{
			PingInterval: nil,
			PingTimeout:  nil,
		},
		InitialSettings: RealtimeSessionModelSettings{},
	}))
	assert.Nil(t, model.pingStop)
	require.NoError(t, model.Close(t.Context()))

	pingInterval := 50 * time.Millisecond
	pingTimeout := 200 * time.Millisecond
	model2 := NewOpenAIRealtimeWebSocketModel()
	require.NoError(t, model2.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		URL:             wsURL,
		EnableTransport: true,
		TransportConfig: &RealtimeTransportConfig{
			PingInterval: &pingInterval,
			PingTimeout:  &pingTimeout,
		},
		InitialSettings: RealtimeSessionModelSettings{},
	}))
	assert.NotNil(t, model2.pingStop)
	require.NoError(t, model2.Close(t.Context()))
	assert.Nil(t, model2.pingStop)
}

func TestTransportPingIntervalComparisonFastVsSlow(t *testing.T) {
	server, wsURL := newWebSocketTestServer(0)
	defer server.Close()

	connectionDurations := map[string]time.Duration{}

	run := func(interval time.Duration, label string) {
		pingTimeout := 500 * time.Millisecond
		model := NewOpenAIRealtimeWebSocketModel()
		start := time.Now()
		require.NoError(t, model.Connect(t.Context(), RealtimeModelConfig{
			APIKey:          "sk-test",
			URL:             wsURL,
			EnableTransport: true,
			TransportConfig: &RealtimeTransportConfig{
				PingInterval: &interval,
				PingTimeout:  &pingTimeout,
			},
			InitialSettings: RealtimeSessionModelSettings{},
		}))

		time.Sleep(150 * time.Millisecond)
		connectionDurations[label] = time.Since(start)

		select {
		case <-model.websocketDone:
			t.Fatalf("connection closed unexpectedly for %s interval", label)
		default:
		}

		require.NoError(t, model.Close(t.Context()))
	}

	run(20*time.Millisecond, "fast")
	run(200*time.Millisecond, "slow")

	assert.Contains(t, connectionDurations, "fast")
	assert.Contains(t, connectionDurations, "slow")
}

func TestConnectToLocalServerSendsSessionUpdate(t *testing.T) {
	done := make(chan struct{})
	messageCh := make(chan map[string]any, 1)
	upgrader := websocket.Upgrader{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()

		_, payload, err := conn.ReadMessage()
		if err != nil {
			return
		}
		var message map[string]any
		if err := json.Unmarshal(payload, &message); err != nil {
			return
		}
		messageCh <- message
		<-done
	}))
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")
	model := NewOpenAIRealtimeWebSocketModel()
	require.NoError(t, model.Connect(t.Context(), RealtimeModelConfig{
		APIKey:          "sk-test",
		URL:             wsURL,
		EnableTransport: true,
		InitialSettings: RealtimeSessionModelSettings{
			"model_name": "gpt-realtime-mini",
		},
	}))

	select {
	case message := <-messageCh:
		assert.Equal(t, "session.update", message["type"])
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timeout waiting for session.update message")
	}

	close(done)
	require.NoError(t, model.Close(t.Context()))
}
