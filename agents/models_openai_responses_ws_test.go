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

package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/gorilla/websocket"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIResponsesWSModelGetResponseReusesPersistentConnection(t *testing.T) {
	var (
		mu           sync.Mutex
		connectCount int
		requestPaths []string
		frames       []map[string]any
	)

	server := newResponsesWSTestServer(t, func(conn *websocket.Conn, r *http.Request) {
		mu.Lock()
		connectCount++
		requestPaths = append(requestPaths, r.URL.Path)
		mu.Unlock()

		requestIndex := 0
		for {
			var frame map[string]any
			if err := conn.ReadJSON(&frame); err != nil {
				return
			}
			requestIndex++
			mu.Lock()
			frames = append(frames, cloneJSONMap(frame))
			mu.Unlock()
			if err := conn.WriteMessage(
				websocket.TextMessage,
				[]byte(wsCompletedFrame(fmt.Sprintf("resp-%d", requestIndex), requestIndex)),
			); err != nil {
				return
			}
		}
	})

	client := NewOpenaiClient(param.NewOpt(server.URL+"/v1"), param.NewOpt("test-key"))
	model := NewOpenAIResponsesWSModel("gpt-4.1", client, "")

	first, err := model.GetResponse(t.Context(), ModelResponseParams{
		Input: InputString("hi"),
	})
	require.NoError(t, err)

	second, err := model.GetResponse(t.Context(), ModelResponseParams{
		Input:              InputString("next"),
		PreviousResponseID: "resp-1",
	})
	require.NoError(t, err)

	assert.Equal(t, "resp-1", first.ResponseID)
	assert.Equal(t, "resp-2", second.ResponseID)

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, 1, connectCount)
	require.Len(t, frames, 2)
	assert.Equal(t, "/v1/responses", requestPaths[0])
	assert.Equal(t, "response.create", frames[0]["type"])
	assert.Equal(t, true, frames[0]["stream"])
	assert.Equal(t, "gpt-4.1", frames[0]["model"])
	assert.Equal(t, "resp-1", frames[1]["previous_response_id"])
}

func TestOpenAIResponsesWSModelReconnectsAfterServerClose(t *testing.T) {
	var (
		mu           sync.Mutex
		connectCount int
		firstClosed  = make(chan struct{})
	)

	server := newResponsesWSTestServer(t, func(conn *websocket.Conn, _ *http.Request) {
		mu.Lock()
		connectCount++
		connectionNumber := connectCount
		mu.Unlock()

		var frame map[string]any
		if err := conn.ReadJSON(&frame); err != nil {
			return
		}
		if err := conn.WriteMessage(
			websocket.TextMessage,
			[]byte(wsCompletedFrame(fmt.Sprintf("resp-%d", connectionNumber), 1)),
		); err != nil {
			return
		}
		if connectionNumber != 1 {
			return
		}

		if tcpConn, ok := conn.UnderlyingConn().(*net.TCPConn); ok {
			_ = tcpConn.SetLinger(0)
		}
		_ = conn.Close()
		close(firstClosed)
	})

	client := NewOpenaiClient(param.NewOpt(server.URL+"/v1"), param.NewOpt("test-key"))
	model := NewOpenAIResponsesWSModel("gpt-4.1", client, "")

	first, err := model.GetResponse(t.Context(), ModelResponseParams{
		Input: InputString("hi"),
	})
	require.NoError(t, err)
	assert.Equal(t, "resp-1", first.ResponseID)

	select {
	case <-firstClosed:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for first websocket to close")
	}

	second, err := model.GetResponse(t.Context(), ModelResponseParams{
		Input: InputString("next"),
	})
	require.NoError(t, err)
	assert.Equal(t, "resp-2", second.ResponseID)

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, 2, connectCount)
}

func TestOpenAIResponsesWSModelUsesExplicitWebsocketBaseURLAndHeaders(t *testing.T) {
	var (
		handshakePath  string
		handshakeQuery string
		authHeader     string
		orgHeader      string
		userAgent      string
		customHeader   string
		overrideHeader string
	)

	server := newResponsesWSTestServer(t, func(conn *websocket.Conn, r *http.Request) {
		handshakePath = r.URL.Path
		handshakeQuery = r.URL.RawQuery
		authHeader = r.Header.Get("Authorization")
		orgHeader = r.Header.Get("OpenAI-Organization")
		userAgent = r.Header.Get("User-Agent")
		customHeader = r.Header.Get("X-Custom")
		overrideHeader = r.Header.Get("X-Override")

		var frame map[string]any
		if err := conn.ReadJSON(&frame); err != nil {
			return
		}
		_ = conn.WriteMessage(websocket.TextMessage, []byte(wsCompletedFrame("resp-explicit", 1)))
	})

	client := NewOpenaiClient(param.NewOpt("http://127.0.0.1:1/v1"), param.NewOpt("test-key"))
	client.WebsocketBaseURL = param.NewOpt(server.URL + "/proxy?token=abc")
	client.DefaultHeaders = map[string]string{
		"OpenAI-Organization": "org_123",
	}

	model := NewOpenAIResponsesWSModel("gpt-4.1", client, "")
	token := ResponsesHeadersOverride.Set(map[string]string{"X-Override": "1"})
	defer ResponsesHeadersOverride.Reset(token)

	_, err := model.GetResponse(t.Context(), ModelResponseParams{
		Input: InputString("hi"),
		ModelSettings: modelsettings.ModelSettings{
			ExtraHeaders: map[string]string{"X-Custom": "demo"},
		},
	})
	require.NoError(t, err)

	assert.Equal(t, "/proxy/responses", handshakePath)
	assert.Equal(t, "token=abc", handshakeQuery)
	assert.Equal(t, "Bearer test-key", authHeader)
	assert.Equal(t, "org_123", orgHeader)
	assert.Equal(t, DefaultUserAgent(), userAgent)
	assert.Equal(t, "demo", customHeader)
	assert.Equal(t, "1", overrideHeader)
}

func TestOpenAIResponsesWSModelStreamResponseYieldsTypedEvents(t *testing.T) {
	server := newResponsesWSTestServer(t, func(conn *websocket.Conn, _ *http.Request) {
		var frame map[string]any
		if err := conn.ReadJSON(&frame); err != nil {
			return
		}
		_ = conn.WriteMessage(websocket.TextMessage, []byte(wsCreatedFrame("resp-stream", 1)))
		_ = conn.WriteMessage(websocket.TextMessage, []byte(wsCompletedFrame("resp-stream", 2)))
	})

	client := NewOpenaiClient(param.NewOpt(server.URL+"/v1"), param.NewOpt("test-key"))
	model := NewOpenAIResponsesWSModel("gpt-4.1", client, "")

	var eventTypes []string
	err := model.StreamResponse(t.Context(), ModelResponseParams{
		Input: InputString("hi"),
	}, func(_ context.Context, event TResponseStreamEvent) error {
		eventTypes = append(eventTypes, event.Type)
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, []string{"response.created", "response.completed"}, eventTypes)
}

func TestOpenAIResponsesWSModelAcceptsTerminalResponsePayloadEvents(t *testing.T) {
	for _, eventType := range []string{"response.incomplete", "response.failed"} {
		t.Run(eventType, func(t *testing.T) {
			server := newResponsesWSTestServer(t, func(conn *websocket.Conn, _ *http.Request) {
				var frame map[string]any
				if err := conn.ReadJSON(&frame); err != nil {
					return
				}
				_ = conn.WriteMessage(
					websocket.TextMessage,
					[]byte(wsTerminalResponseFrame(eventType, "resp-terminal", 1)),
				)
			})

			client := NewOpenaiClient(param.NewOpt(server.URL+"/v1"), param.NewOpt("test-key"))
			model := NewOpenAIResponsesWSModel("gpt-4.1", client, "")

			response, err := model.GetResponse(t.Context(), ModelResponseParams{
				Input: InputString("hi"),
			})
			require.NoError(t, err)
			assert.Equal(t, "resp-terminal", response.ResponseID)
		})
	}
}

func TestOpenAIResponsesWSModelReturnsResponseError(t *testing.T) {
	server := newResponsesWSTestServer(t, func(conn *websocket.Conn, _ *http.Request) {
		var frame map[string]any
		if err := conn.ReadJSON(&frame); err != nil {
			return
		}
		_ = conn.WriteMessage(
			websocket.TextMessage,
			[]byte(`{"type":"response.error","code":"invalid_request_error","message":"bad request"}`),
		)
	})

	client := NewOpenaiClient(param.NewOpt(server.URL+"/v1"), param.NewOpt("test-key"))
	model := NewOpenAIResponsesWSModel("gpt-4.1", client, "")

	_, err := model.GetResponse(t.Context(), ModelResponseParams{
		Input: InputString("hi"),
	})
	require.Error(t, err)
	var wsErr ResponsesWebSocketError
	require.ErrorAs(t, err, &wsErr)
	assert.Equal(t, "response.error", wsErr.EventType)
	assert.Equal(t, "invalid_request_error", wsErr.Code)
	assert.Equal(t, "bad request", wsErr.ErrorMessage)
}

func TestOpenAIResponsesWSModelReceiveTimeoutDropsConnection(t *testing.T) {
	server := newResponsesWSTestServer(t, func(conn *websocket.Conn, _ *http.Request) {
		var frame map[string]any
		if err := conn.ReadJSON(&frame); err != nil {
			return
		}
		time.Sleep(80 * time.Millisecond)
		_ = conn.WriteMessage(websocket.TextMessage, []byte(wsCompletedFrame("resp-timeout", 1)))
	})

	client := NewOpenaiClient(param.NewOpt(server.URL+"/v1"), param.NewOpt("test-key"))
	model := NewOpenAIResponsesWSModel("gpt-4.1", client, "")

	_, err := model.GetResponse(t.Context(), ModelResponseParams{
		Input: InputString("hi"),
		ModelSettings: modelsettings.ModelSettings{
			ExtraArgs: map[string]any{"timeout": 0.01},
		},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "receive timed out")
}

func TestOpenAIResponsesWSModelNegativeTimeoutTimesOutImmediately(t *testing.T) {
	server := newResponsesWSTestServer(t, func(conn *websocket.Conn, _ *http.Request) {
		var frame map[string]any
		if err := conn.ReadJSON(&frame); err != nil {
			return
		}
		_ = conn.WriteMessage(websocket.TextMessage, []byte(wsCompletedFrame("resp-negative", 1)))
	})

	client := NewOpenaiClient(param.NewOpt(server.URL+"/v1"), param.NewOpt("test-key"))
	client.RequestTimeout = time.Second
	model := NewOpenAIResponsesWSModel("gpt-4.1", client, "")

	_, err := model.GetResponse(t.Context(), ModelResponseParams{
		Input: InputString("hi"),
		ModelSettings: modelsettings.ModelSettings{
			ExtraArgs: map[string]any{"timeout": -1.0},
		},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "timed out after -1 seconds")
}

func newResponsesWSTestServer(
	t *testing.T,
	handler func(*websocket.Conn, *http.Request),
) *httptest.Server {
	t.Helper()

	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer conn.Close()
		handler(conn, r)
	}))
	t.Cleanup(server.Close)
	return server
}

func wsCreatedFrame(responseID string, sequenceNumber int) string {
	return fmt.Sprintf(
		`{"type":"response.created","sequence_number":%d,"response":{"id":%q,"output":[]}}`,
		sequenceNumber,
		responseID,
	)
}

func wsCompletedFrame(responseID string, sequenceNumber int) string {
	return wsTerminalResponseFrame("response.completed", responseID, sequenceNumber)
}

func wsTerminalResponseFrame(eventType, responseID string, sequenceNumber int) string {
	return fmt.Sprintf(
		`{"type":%q,"sequence_number":%d,"response":{"id":%q,"output":[]}}`,
		eventType,
		sequenceNumber,
		responseID,
	)
}

func cloneJSONMap(input map[string]any) map[string]any {
	raw, _ := json.Marshal(input)
	var cloned map[string]any
	_ = json.Unmarshal(raw, &cloned)
	return cloned
}
