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

package agents

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type wsTestServer struct {
	server *httptest.Server
	url    string
	errCh  chan error
	doneCh chan struct{}
}

func newWSTestServer(t *testing.T, handler func(*websocket.Conn, *http.Request) error) *wsTestServer {
	t.Helper()

	errCh := make(chan error, 1)
	doneCh := make(chan struct{})
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			errCh <- err
			close(doneCh)
			return
		}
		defer conn.Close()
		if err := handler(conn, r); err != nil {
			errCh <- err
		}
		close(doneCh)
	}))

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")
	return &wsTestServer{
		server: server,
		url:    wsURL,
		errCh:  errCh,
		doneCh: doneCh,
	}
}

func (s *wsTestServer) Close() {
	s.server.Close()
}

func withSTTTimeouts(t *testing.T, creation, update, inactivity time.Duration) {
	t.Helper()

	oldCreation := VoiceModelsOpenAISessionCreationTimeout
	oldUpdate := VoiceModelsOpenAISessionUpdateTimeout
	oldInactivity := VoiceModelsOpenAIEventInactivityTimeout
	VoiceModelsOpenAISessionCreationTimeout = creation
	VoiceModelsOpenAISessionUpdateTimeout = update
	VoiceModelsOpenAIEventInactivityTimeout = inactivity
	t.Cleanup(func() {
		VoiceModelsOpenAISessionCreationTimeout = oldCreation
		VoiceModelsOpenAISessionUpdateTimeout = oldUpdate
		VoiceModelsOpenAIEventInactivityTimeout = oldInactivity
	})
}

func drainServerErrors(t *testing.T, server *wsTestServer) {
	t.Helper()
	select {
	case err := <-server.errCh:
		require.NoError(t, err)
	default:
	}
}

func TestOpenAISTTNonJSONMessagesShouldCrash(t *testing.T) {
	withSTTTimeouts(t, 50*time.Millisecond, 50*time.Millisecond, 50*time.Millisecond)

	server := newWSTestServer(t, func(conn *websocket.Conn, _ *http.Request) error {
		return conn.WriteMessage(websocket.TextMessage, []byte("not a json message"))
	})
	defer server.Close()

	input := NewStreamedAudioInput()
	input.AddAudio(AudioDataInt16{})

	session := NewOpenAISTTTranscriptionSession(OpenAISTTTranscriptionSessionParams{
		Input:                          input,
		Client:                         OpenaiClient{APIKey: param.NewOpt("FAKE_KEY")},
		Model:                          "whisper-1",
		Settings:                       STTModelSettings{},
		TraceIncludeSensitiveData:      false,
		TraceIncludeSensitiveAudioData: false,
		WebsocketURL:                   server.url,
	})

	turns := session.TranscribeTurns(context.Background())
	for range turns.Seq() {
	}

	err := turns.Error()
	require.Error(t, err)
	assert.ErrorContains(t, err, "error parsing events")
	var sttErr STTWebsocketConnectionError
	require.ErrorAs(t, err, &sttErr)

	require.NoError(t, session.Close(context.Background()))
	<-server.doneCh
	drainServerErrors(t, server)
}

func TestOpenAISTTSessionConnectsAndConfiguresSuccessfully(t *testing.T) {
	withSTTTimeouts(t, 200*time.Millisecond, 200*time.Millisecond, 200*time.Millisecond)

	var (
		gotAuth       string
		gotLogSession string
		gotBeta       string
		sentMessages  []string
		mu            sync.Mutex
	)

	server := newWSTestServer(t, func(conn *websocket.Conn, r *http.Request) error {
		gotAuth = r.Header.Get("Authorization")
		gotLogSession = r.Header.Get("OpenAI-Log-Session")
		gotBeta = r.Header.Get("OpenAI-Beta")

		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"transcription_session.created"}`)); err != nil {
			return err
		}

		_, msg, err := conn.ReadMessage()
		if err != nil {
			return err
		}
		mu.Lock()
		sentMessages = append(sentMessages, string(msg))
		mu.Unlock()

		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"transcription_session.updated"}`)); err != nil {
			return err
		}

		return conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
	})
	defer server.Close()

	input := NewStreamedAudioInput()
	input.AddAudio(AudioDataInt16{})

	session := NewOpenAISTTTranscriptionSession(OpenAISTTTranscriptionSessionParams{
		Input:                          input,
		Client:                         OpenaiClient{APIKey: param.NewOpt("FAKE_KEY")},
		Model:                          "whisper-1",
		Settings:                       STTModelSettings{},
		TraceIncludeSensitiveData:      false,
		TraceIncludeSensitiveAudioData: false,
		WebsocketURL:                   server.url,
	})

	turns := session.TranscribeTurns(context.Background())
	for range turns.Seq() {
	}
	require.NoError(t, turns.Error())

	require.NoError(t, session.Close(context.Background()))
	<-server.doneCh
	drainServerErrors(t, server)

	assert.Equal(t, "Bearer FAKE_KEY", gotAuth)
	assert.Equal(t, "1", gotLogSession)
	assert.Equal(t, "", gotBeta)

	mu.Lock()
	defer mu.Unlock()
	require.NotEmpty(t, sentMessages)
	var msg map[string]any
	require.NoError(t, json.Unmarshal([]byte(sentMessages[0]), &msg))
	assert.Equal(t, "session.update", msg["type"])
}

func TestOpenAISTTStreamAudioSendsCorrectJSON(t *testing.T) {
	withSTTTimeouts(t, 200*time.Millisecond, 200*time.Millisecond, 200*time.Millisecond)

	readyCh := make(chan struct{})
	var (
		configMessage string
		audioMessage  string
	)

	server := newWSTestServer(t, func(conn *websocket.Conn, _ *http.Request) error {
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"transcription_session.created"}`)); err != nil {
			return err
		}

		_, msg, err := conn.ReadMessage()
		if err != nil {
			return err
		}
		configMessage = string(msg)

		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"transcription_session.updated"}`)); err != nil {
			return err
		}
		close(readyCh)

		_, msg, err = conn.ReadMessage()
		if err != nil {
			return err
		}
		audioMessage = string(msg)

		return conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
	})
	defer server.Close()

	input := NewStreamedAudioInput()

	session := NewOpenAISTTTranscriptionSession(OpenAISTTTranscriptionSessionParams{
		Input:                          input,
		Client:                         OpenaiClient{APIKey: param.NewOpt("FAKE_KEY")},
		Model:                          "whisper-1",
		Settings:                       STTModelSettings{},
		TraceIncludeSensitiveData:      false,
		TraceIncludeSensitiveAudioData: false,
		WebsocketURL:                   server.url,
	})

	turns := session.TranscribeTurns(context.Background())
	done := make(chan error, 1)
	go func() {
		for range turns.Seq() {
		}
		done <- turns.Error()
	}()

	<-readyCh
	buffer := AudioDataInt16{1, 2, 3, 4}
	input.AddAudio(buffer)
	input.AddAudio(AudioDataInt16{})

	require.NoError(t, <-done)
	require.NoError(t, session.Close(context.Background()))
	<-server.doneCh
	drainServerErrors(t, server)

	var config map[string]any
	require.NoError(t, json.Unmarshal([]byte(configMessage), &config))
	assert.Equal(t, "session.update", config["type"])

	var audioPayload map[string]any
	require.NoError(t, json.Unmarshal([]byte(audioMessage), &audioPayload))
	assert.Equal(t, "input_audio_buffer.append", audioPayload["type"])
	expectedAudio := base64.StdEncoding.EncodeToString(buffer.Bytes())
	assert.Equal(t, expectedAudio, audioPayload["audio"])
}

func TestOpenAISTTTranscriptionEventPutsOutputInQueue(t *testing.T) {
	events := []string{
		"input_audio_transcription_completed",
		"conversation.item.input_audio_transcription.completed",
	}

	for _, eventType := range events {
		t.Run(eventType, func(t *testing.T) {
			withSTTTimeouts(t, 200*time.Millisecond, 200*time.Millisecond, 200*time.Millisecond)

			server := newWSTestServer(t, func(conn *websocket.Conn, _ *http.Request) error {
				if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"transcription_session.created"}`)); err != nil {
					return err
				}
				if _, _, err := conn.ReadMessage(); err != nil {
					return err
				}
				if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"transcription_session.updated"}`)); err != nil {
					return err
				}
				payload, err := json.Marshal(map[string]any{
					"type":       eventType,
					"transcript": "Hello world!",
				})
				if err != nil {
					return err
				}
				if err := conn.WriteMessage(websocket.TextMessage, payload); err != nil {
					return err
				}
				return conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
			})
			defer server.Close()

			input := NewStreamedAudioInput()
			input.AddAudio(AudioDataInt16{})

			session := NewOpenAISTTTranscriptionSession(OpenAISTTTranscriptionSessionParams{
				Input:                          input,
				Client:                         OpenaiClient{APIKey: param.NewOpt("FAKE_KEY")},
				Model:                          "whisper-1",
				Settings:                       STTModelSettings{},
				TraceIncludeSensitiveData:      false,
				TraceIncludeSensitiveAudioData: false,
				WebsocketURL:                   server.url,
			})

			turns := session.TranscribeTurns(context.Background())
			var collected []string
			for turn := range turns.Seq() {
				collected = append(collected, turn)
			}
			require.NoError(t, turns.Error())

			require.NoError(t, session.Close(context.Background()))
			<-server.doneCh
			drainServerErrors(t, server)

			assert.Contains(t, collected, "Hello world!")
		})
	}
}

func TestOpenAISTTTimeoutWaitingForCreatedEvent(t *testing.T) {
	withSTTTimeouts(t, 20*time.Millisecond, 20*time.Millisecond, 20*time.Millisecond)

	server := newWSTestServer(t, func(conn *websocket.Conn, _ *http.Request) error {
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"unknown"}`)); err != nil {
			return err
		}
		time.Sleep(2 * VoiceModelsOpenAISessionCreationTimeout)
		return conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
	})
	defer server.Close()

	input := NewStreamedAudioInput()
	input.AddAudio(AudioDataInt16{})

	session := NewOpenAISTTTranscriptionSession(OpenAISTTTranscriptionSessionParams{
		Input:                          input,
		Client:                         OpenaiClient{APIKey: param.NewOpt("FAKE_KEY")},
		Model:                          "whisper-1",
		Settings:                       STTModelSettings{},
		TraceIncludeSensitiveData:      false,
		TraceIncludeSensitiveAudioData: false,
		WebsocketURL:                   server.url,
	})

	turns := session.TranscribeTurns(context.Background())
	for range turns.Seq() {
	}
	err := turns.Error()
	require.Error(t, err)
	assert.ErrorContains(t, err, "Timeout waiting for transcription_session.created event")
	var sttErr STTWebsocketConnectionError
	require.ErrorAs(t, err, &sttErr)

	require.NoError(t, session.Close(context.Background()))
	<-server.doneCh
	drainServerErrors(t, server)
}

func TestOpenAISTTSessionErrorEvent(t *testing.T) {
	withSTTTimeouts(t, 200*time.Millisecond, 200*time.Millisecond, 200*time.Millisecond)

	server := newWSTestServer(t, func(conn *websocket.Conn, _ *http.Request) error {
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"transcription_session.created"}`)); err != nil {
			return err
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"transcription_session.updated"}`)); err != nil {
			return err
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"error","error":"Simulated server error!"}`)); err != nil {
			return err
		}
		return conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
	})
	defer server.Close()

	input := NewStreamedAudioInput()
	input.AddAudio(AudioDataInt16{})

	session := NewOpenAISTTTranscriptionSession(OpenAISTTTranscriptionSessionParams{
		Input:                          input,
		Client:                         OpenaiClient{APIKey: param.NewOpt("FAKE_KEY")},
		Model:                          "whisper-1",
		Settings:                       STTModelSettings{},
		TraceIncludeSensitiveData:      false,
		TraceIncludeSensitiveAudioData: false,
		WebsocketURL:                   server.url,
	})

	turns := session.TranscribeTurns(context.Background())
	for range turns.Seq() {
	}
	err := turns.Error()
	require.Error(t, err)
	assert.ErrorContains(t, err, "error event")
	var sttErr STTWebsocketConnectionError
	require.ErrorAs(t, err, &sttErr)

	require.NoError(t, session.Close(context.Background()))
	<-server.doneCh
	drainServerErrors(t, server)
}

func TestOpenAISTTInactivityTimeout(t *testing.T) {
	oldNow := voiceModelsOpenAITimeNow
	t.Cleanup(func() {
		voiceModelsOpenAITimeNow = oldNow
	})

	base := time.Unix(1000, 0)
	times := []time.Time{
		base,
		base.Add(VoiceModelsOpenAIEventInactivityTimeout + time.Second),
		base.Add(2*VoiceModelsOpenAIEventInactivityTimeout + time.Second),
		base.Add(3*VoiceModelsOpenAIEventInactivityTimeout + time.Second),
		base.Add(4 * VoiceModelsOpenAIEventInactivityTimeout),
	}
	var mu sync.Mutex
	index := 0
	voiceModelsOpenAITimeNow = func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		if index >= len(times) {
			return times[len(times)-1]
		}
		tm := times[index]
		index++
		return tm
	}

	server := newWSTestServer(t, func(conn *websocket.Conn, _ *http.Request) error {
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"unknown"}`)); err != nil {
			return err
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"unknown"}`)); err != nil {
			return err
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"transcription_session.created"}`)); err != nil {
			return err
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"transcription_session.updated"}`)); err != nil {
			return err
		}
		return conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
	})
	defer server.Close()

	input := NewStreamedAudioInput()
	input.AddAudio(AudioDataInt16{})

	session := NewOpenAISTTTranscriptionSession(OpenAISTTTranscriptionSessionParams{
		Input:                          input,
		Client:                         OpenaiClient{APIKey: param.NewOpt("FAKE_KEY")},
		Model:                          "whisper-1",
		Settings:                       STTModelSettings{},
		TraceIncludeSensitiveData:      false,
		TraceIncludeSensitiveAudioData: false,
		WebsocketURL:                   server.url,
	})

	turns := session.TranscribeTurns(context.Background())
	for range turns.Seq() {
	}
	err := turns.Error()
	require.Error(t, err)
	assert.ErrorContains(t, err, "Timeout waiting for transcription_session")
	var sttErr STTWebsocketConnectionError
	require.True(t, errors.As(err, &sttErr))

	require.NoError(t, session.Close(context.Background()))
	<-server.doneCh
	drainServerErrors(t, server)
}
