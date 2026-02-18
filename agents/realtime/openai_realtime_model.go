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
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/gorilla/websocket"
	oairealtime "github.com/openai/openai-go/v3/realtime"
	"github.com/openai/openai-go/v3/responses"

	"github.com/openai/openai-go/v3/packages/param"
)

const (
	defaultRealtimeModelName                = "gpt-realtime"
	defaultRealtimeVoice                    = "ash"
	defaultRealtimeInputAudioFormat         = "pcm16"
	defaultRealtimeOutputAudioFormat        = "pcm16"
	defaultRealtimeInputTranscriptionModel  = "gpt-4o-mini-transcribe"
	defaultRealtimeTurnDetectionType        = "semantic_vad"
	defaultRealtimeTurnDetectionInterruptOn = true
)

var defaultRealtimeOutputModalities = []string{"audio"}

var defaultRealtimeTurnDetection = map[string]any{
	"type":               defaultRealtimeTurnDetectionType,
	"interrupt_response": defaultRealtimeTurnDetectionInterruptOn,
}

var defaultRealtimeInputTranscription = map[string]any{
	"model": defaultRealtimeInputTranscriptionModel,
}

// OpenAIRealtimeWebSocketModel is the OpenAI realtime transport implementation.
//
// This baseline currently focuses on session-configuration parity.
type OpenAIRealtimeWebSocketModel struct {
	model                                string
	callID                               string
	listeners                            []RealtimeModelListener
	lastConnectURL                       string
	lastConnectHeads                     map[string]string
	connected                            bool
	sentClientEvents                     []map[string]any
	sendClientEvent                      func(context.Context, map[string]any) error
	websocketConn                        RealtimeWebSocketConn
	websocketDone                        chan struct{}
	dialWebSocket                        RealtimeWebSocketDialer
	audioStateTracker                    *ModelAudioTracker
	playbackTracker                      *RealtimePlaybackTracker
	createdSession                       *oairealtime.RealtimeSessionCreateRequestParam
	automaticResponseCancellationEnabled bool
	currentItemID                        string
	ongoingResponse                      bool
	tracingConfig                        any
	transportConfig                      *RealtimeTransportConfig
	pingStop                             chan struct{}
	listenersMutex                       sync.RWMutex
}

// NewOpenAIRealtimeWebSocketModel creates a realtime transport with sane defaults.
func NewOpenAIRealtimeWebSocketModel() *OpenAIRealtimeWebSocketModel {
	return &OpenAIRealtimeWebSocketModel{
		model: defaultRealtimeModelName,
	}
}

// SetTransportConfig updates the default transport configuration used for connections.
func (m *OpenAIRealtimeWebSocketModel) SetTransportConfig(config *RealtimeTransportConfig) {
	m.transportConfig = config
}

func (m *OpenAIRealtimeWebSocketModel) Connect(
	ctx context.Context,
	options RealtimeModelConfig,
) error {
	if m.connected {
		return errors.New("realtime model is already connected")
	}

	modelSettings := options.InitialSettings
	modelName, _ := stringFromSettings(modelSettings, "model_name")
	if strings.TrimSpace(modelName) != "" && strings.TrimSpace(options.CallID) != "" {
		return errors.New("cannot specify both `call_id` and `model_name`")
	}
	if strings.TrimSpace(modelName) != "" {
		m.model = strings.TrimSpace(modelName)
	}
	m.callID = strings.TrimSpace(options.CallID)

	sessionConfig, err := m.GetSessionConfig(modelSettings)
	if err != nil {
		return err
	}
	if tracingConfig, ok := modelSettings["tracing"]; ok {
		m.tracingConfig = tracingConfig
	} else {
		m.tracingConfig = "auto"
	}

	apiKey, err := options.ResolveAPIKey(ctx)
	if err != nil {
		return err
	}
	if strings.TrimSpace(apiKey) == "" {
		return errors.New("api key is required but was not provided")
	}

	headers := make(map[string]string)
	if len(options.Headers) > 0 {
		for key, value := range options.Headers {
			headers[key] = value
		}
	} else {
		headers["Authorization"] = "Bearer " + apiKey
	}

	m.lastConnectURL = strings.TrimSpace(options.URL)
	if m.lastConnectURL == "" {
		m.lastConnectURL = m.defaultRealtimeURL()
	}
	m.lastConnectHeads = headers
	m.sentClientEvents = nil
	m.currentItemID = ""
	m.ongoingResponse = false
	m.playbackTracker = options.PlaybackTracker
	m.audioStateTracker = NewModelAudioTracker()
	m.createdSession = sessionConfig
	m.automaticResponseCancellationEnabled = isAutomaticResponseCancellationEnabled(sessionConfig)
	if outputAudioFormat := ExtractSessionAudioFormat(*sessionConfig); outputAudioFormat != nil {
		m.audioStateTracker.SetAudioFormat(*outputAudioFormat)
		if m.playbackTracker != nil {
			m.playbackTracker.SetAudioFormat(*outputAudioFormat)
		}
	}

	if options.EnableTransport {
		transportConfig := options.TransportConfig
		if transportConfig == nil {
			transportConfig = m.transportConfig
		}
		dialer := options.TransportDialer
		if dialer == nil {
			dialer = m.dialWebSocket
		}
		if dialer == nil {
			dialer = defaultRealtimeWebSocketDialer
		}
		conn, err := dialer(ctx, m.lastConnectURL, m.lastConnectHeads, transportConfig)
		if err != nil {
			return fmt.Errorf("failed to connect websocket transport: %w", err)
		}
		m.websocketConn = conn
		m.websocketDone = make(chan struct{})
		m.configureTransport(conn, transportConfig)
		m.sendClientEvent = func(_ context.Context, payload map[string]any) error {
			return conn.WriteJSON(payload)
		}
		go m.listenForMessages()
	}

	if err := m.sendSessionUpdatePayload(ctx, sessionConfig); err != nil {
		if m.websocketConn != nil {
			_ = m.websocketConn.Close()
			m.websocketConn = nil
		}
		m.sendClientEvent = nil
		return err
	}
	m.connected = true
	if options.EnableTransport {
		_ = m.emitEvent(ctx, RealtimeModelConnectionStatusEvent{
			Status: RealtimeConnectionStatusConnected,
		})
	}

	return nil
}

func (m *OpenAIRealtimeWebSocketModel) AddListener(listener RealtimeModelListener) {
	if listener == nil {
		return
	}
	m.listenersMutex.Lock()
	defer m.listenersMutex.Unlock()
	for _, existing := range m.listeners {
		if existing == listener {
			return
		}
	}
	m.listeners = append(m.listeners, listener)
}

func (m *OpenAIRealtimeWebSocketModel) RemoveListener(listener RealtimeModelListener) {
	if listener == nil {
		return
	}
	m.listenersMutex.Lock()
	defer m.listenersMutex.Unlock()
	out := make([]RealtimeModelListener, 0, len(m.listeners))
	for _, existing := range m.listeners {
		if existing != listener {
			out = append(out, existing)
		}
	}
	m.listeners = out
}

func (m *OpenAIRealtimeWebSocketModel) SendEvent(
	ctx context.Context,
	event RealtimeModelSendEvent,
) error {
	if !m.connected {
		return errors.New("realtime model is not connected")
	}

	switch e := event.(type) {
	case RealtimeModelSendRawMessage:
		payload := TryConvertRawMessage(e)
		if payload == nil {
			return fmt.Errorf("failed to convert raw message of type %q", e.Message.Type)
		}
		return m.dispatchClientEvent(ctx, payload)

	case RealtimeModelSendUserInput:
		if err := m.dispatchClientEvent(ctx, ConvertUserInputToItemCreate(e)); err != nil {
			return err
		}
		return m.dispatchClientEvent(ctx, map[string]any{"type": "response.create"})

	case RealtimeModelSendAudio:
		if err := m.dispatchClientEvent(ctx, ConvertAudioToInputAudioBufferAppend(e)); err != nil {
			return err
		}
		if e.Commit {
			return m.dispatchClientEvent(ctx, map[string]any{"type": "input_audio_buffer.commit"})
		}
		return nil

	case RealtimeModelSendToolOutput:
		payload := ConvertToolOutput(e)
		if payload == nil {
			return errors.New("tool output payload conversion failed")
		}
		if err := m.dispatchClientEvent(ctx, payload); err != nil {
			return err
		}

		toolOutput := e.Output
		if err := m.emitEvent(ctx, RealtimeModelItemUpdatedEvent{
			Item: RealtimeToolCallItem{
				ItemID:         derefString(e.ToolCall.ID),
				PreviousItemID: e.ToolCall.PreviousItemID,
				CallID:         e.ToolCall.CallID,
				Type:           "function_call",
				Status:         "completed",
				Arguments:      e.ToolCall.Arguments,
				Name:           e.ToolCall.Name,
				Output:         &toolOutput,
			},
		}); err != nil {
			return err
		}

		if e.StartResponse {
			return m.dispatchClientEvent(ctx, map[string]any{"type": "response.create"})
		}
		return nil

	case RealtimeModelSendInterrupt:
		if err := m.sendInterrupt(ctx, e); err != nil {
			return err
		}

		shouldCancel := e.ForceResponseCancel || !m.automaticResponseCancellationEnabled
		if shouldCancel {
			return m.cancelResponse(ctx)
		}
		return nil

	case RealtimeModelSendSessionUpdate:
		sessionConfig, err := m.GetSessionConfig(e.SessionSettings)
		if err != nil {
			return err
		}
		m.createdSession = sessionConfig
		m.automaticResponseCancellationEnabled = isAutomaticResponseCancellationEnabled(sessionConfig)
		if outputAudioFormat := ExtractSessionAudioFormat(*sessionConfig); outputAudioFormat != nil {
			m.audioStateTracker.SetAudioFormat(*outputAudioFormat)
			if m.playbackTracker != nil {
				m.playbackTracker.SetAudioFormat(*outputAudioFormat)
			}
		}
		return m.sendSessionUpdatePayload(ctx, sessionConfig)

	default:
		return fmt.Errorf("unsupported realtime send event %T", event)
	}
}

func (m *OpenAIRealtimeWebSocketModel) Close(ctx context.Context) error {
	if m.pingStop != nil {
		close(m.pingStop)
		m.pingStop = nil
	}
	if m.websocketConn != nil {
		_ = m.websocketConn.Close()
		if m.websocketDone != nil {
			select {
			case <-m.websocketDone:
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(500 * time.Millisecond):
			}
		}
		m.websocketConn = nil
		m.websocketDone = nil
	}
	m.connected = false
	m.createdSession = nil
	m.sentClientEvents = nil
	m.currentItemID = ""
	m.ongoingResponse = false
	m.tracingConfig = nil
	if err := m.emitEvent(ctx, RealtimeModelConnectionStatusEvent{
		Status: RealtimeConnectionStatusDisconnected,
	}); err != nil {
		return err
	}
	return nil
}

// SetCallID configures the model to attach to an existing realtime call.
func (m *OpenAIRealtimeWebSocketModel) SetCallID(callID string) {
	m.callID = strings.TrimSpace(callID)
}

func (m *OpenAIRealtimeWebSocketModel) defaultRealtimeURL() string {
	query := url.Values{}
	if strings.TrimSpace(m.callID) != "" {
		query.Set("call_id", strings.TrimSpace(m.callID))
	} else {
		modelName := strings.TrimSpace(m.model)
		if modelName == "" {
			modelName = defaultRealtimeModelName
		}
		query.Set("model", modelName)
	}
	return "wss://api.openai.com/v1/realtime?" + query.Encode()
}

func (m *OpenAIRealtimeWebSocketModel) configureTransport(
	conn RealtimeWebSocketConn,
	config *RealtimeTransportConfig,
) {
	if conn == nil || config == nil {
		return
	}
	ws, ok := conn.(*websocket.Conn)
	if !ok {
		return
	}

	pingEnabled := config.PingInterval != nil && *config.PingInterval > 0
	if pingEnabled && config.PingTimeout != nil && *config.PingTimeout > 0 {
		timeout := *config.PingTimeout
		_ = ws.SetReadDeadline(time.Now().Add(timeout))
		ws.SetPongHandler(func(string) error {
			return ws.SetReadDeadline(time.Now().Add(timeout))
		})
	}

	if !pingEnabled {
		return
	}
	if m.pingStop != nil {
		close(m.pingStop)
	}
	m.pingStop = make(chan struct{})
	pingInterval := *config.PingInterval
	deadline := pingInterval
	if config.PingTimeout != nil && *config.PingTimeout > 0 {
		deadline = *config.PingTimeout
	}

	go func() {
		ticker := time.NewTicker(pingInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				_ = ws.WriteControl(websocket.PingMessage, []byte{}, time.Now().Add(deadline))
			case <-m.websocketDone:
				return
			case <-m.pingStop:
				return
			}
		}
	}()
}

func (m *OpenAIRealtimeWebSocketModel) sendSessionUpdatePayload(
	ctx context.Context,
	sessionConfig *oairealtime.RealtimeSessionCreateRequestParam,
) error {
	if sessionConfig == nil {
		return errors.New("session config cannot be nil")
	}
	sessionPayload := sessionConfigToMap(sessionConfig)
	if sessionPayload == nil {
		return errors.New("failed to serialize session config")
	}
	return m.dispatchClientEvent(ctx, map[string]any{
		"type":    "session.update",
		"session": sessionPayload,
	})
}

func (m *OpenAIRealtimeWebSocketModel) dispatchClientEvent(
	ctx context.Context,
	event map[string]any,
) error {
	if event == nil {
		return errors.New("client event payload cannot be nil")
	}
	cloned := cloneStringAnyMap(event)
	m.sentClientEvents = append(m.sentClientEvents, cloned)
	if m.sendClientEvent != nil {
		return m.sendClientEvent(ctx, cloned)
	}
	return nil
}

func (m *OpenAIRealtimeWebSocketModel) emitEvent(
	ctx context.Context,
	event RealtimeModelEvent,
) error {
	m.listenersMutex.RLock()
	listeners := slices.Clone(m.listeners)
	m.listenersMutex.RUnlock()
	for _, listener := range listeners {
		if listener == nil {
			continue
		}
		if err := listener.OnEvent(ctx, event); err != nil {
			return err
		}
	}
	return nil
}

func (m *OpenAIRealtimeWebSocketModel) sendInterrupt(
	ctx context.Context,
	event RealtimeModelSendInterrupt,
) error {
	itemID, contentIndex, elapsedMS, ok := m.currentPlaybackState()
	if ok && elapsedMS > 0 {
		if err := m.emitEvent(ctx, RealtimeModelAudioInterruptedEvent{
			ItemID:       itemID,
			ContentIndex: contentIndex,
		}); err != nil {
			return err
		}

		truncateMS := int(math.Max(0, elapsedMS))
		_, maxAudioMS, hasAudioLimit := m.audioLengthLimit(itemID, contentIndex)
		if m.ongoingResponse || !hasAudioLimit || truncateMS < maxAudioMS {
			if err := m.dispatchClientEvent(ctx, ConvertInterrupt(itemID, contentIndex, truncateMS)); err != nil {
				return err
			}
		}
	}

	if ok {
		if m.audioStateTracker != nil {
			m.audioStateTracker.OnInterrupted()
		}
		if m.playbackTracker != nil {
			m.playbackTracker.OnInterrupted()
		}
	}

	_ = event
	return nil
}

func (m *OpenAIRealtimeWebSocketModel) cancelResponse(ctx context.Context) error {
	if err := m.dispatchClientEvent(ctx, map[string]any{"type": "response.cancel"}); err != nil {
		return err
	}
	m.ongoingResponse = false
	return nil
}

func (m *OpenAIRealtimeWebSocketModel) currentPlaybackState() (string, int, float64, bool) {
	if m.playbackTracker != nil {
		state := m.playbackTracker.GetState()
		if state.CurrentItemID != nil && state.CurrentItemContentIndex != nil && state.ElapsedMS != nil {
			return *state.CurrentItemID, *state.CurrentItemContentIndex, *state.ElapsedMS, true
		}
	}

	if m.audioStateTracker == nil {
		return "", 0, 0, false
	}
	lastItem := m.audioStateTracker.GetLastAudioItem()
	if lastItem == nil {
		return "", 0, 0, false
	}
	audioState := m.audioStateTracker.GetState(lastItem.ItemID, lastItem.ItemContentIndex)
	if audioState == nil {
		return "", 0, 0, false
	}
	elapsedMS := time.Since(audioState.InitialReceivedTime).Seconds() * 1000.0
	return lastItem.ItemID, lastItem.ItemContentIndex, elapsedMS, true
}

func isAutomaticResponseCancellationEnabled(
	session *oairealtime.RealtimeSessionCreateRequestParam,
) bool {
	if session == nil {
		return false
	}

	turnDetection := session.Audio.Input.TurnDetection
	if turnDetection.OfSemanticVad != nil {
		if turnDetection.OfSemanticVad.InterruptResponse.Valid() {
			return turnDetection.OfSemanticVad.InterruptResponse.Value
		}
		return false
	}

	if turnDetection.OfServerVad != nil {
		if turnDetection.OfServerVad.InterruptResponse.Valid() {
			return turnDetection.OfServerVad.InterruptResponse.Value
		}
		return false
	}

	return false
}

func sessionConfigToMap(session *oairealtime.RealtimeSessionCreateRequestParam) map[string]any {
	if session == nil {
		return nil
	}
	raw, err := json.Marshal(session)
	if err != nil {
		return nil
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil
	}
	return payload
}

func derefString(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func defaultRealtimeWebSocketDialer(
	ctx context.Context,
	rawURL string,
	headers map[string]string,
	transportConfig *RealtimeTransportConfig,
) (RealtimeWebSocketConn, error) {
	dialer := websocket.Dialer{}
	if transportConfig != nil && transportConfig.HandshakeTimeout != nil {
		dialer.HandshakeTimeout = *transportConfig.HandshakeTimeout
	}
	httpHeaders := make(http.Header, len(headers))
	for key, value := range headers {
		httpHeaders.Set(key, value)
	}
	conn, _, err := dialer.DialContext(ctx, rawURL, httpHeaders)
	if err != nil {
		return nil, err
	}
	return conn, nil
}

func (m *OpenAIRealtimeWebSocketModel) listenForMessages() {
	defer func() {
		if m.websocketDone != nil {
			close(m.websocketDone)
		}
	}()

	for {
		if m.websocketConn == nil {
			return
		}

		_, payload, err := m.websocketConn.ReadMessage()
		if err != nil {
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				_ = m.emitEvent(context.Background(), RealtimeModelConnectionStatusEvent{
					Status: RealtimeConnectionStatusDisconnected,
				})
				return
			}
			contextMessage := "websocket error in message listener"
			_ = m.emitEvent(context.Background(), RealtimeModelExceptionEvent{
				Exception: err,
				Context:   &contextMessage,
			})
			_ = m.emitEvent(context.Background(), RealtimeModelConnectionStatusEvent{
				Status: RealtimeConnectionStatusDisconnected,
			})
			return
		}

		_ = m.handleWSMessage(context.Background(), payload)
	}
}

func (m *OpenAIRealtimeWebSocketModel) handleWSMessage(
	ctx context.Context,
	rawMessage []byte,
) error {
	var event map[string]any
	if err := json.Unmarshal(rawMessage, &event); err != nil {
		if emitErr := m.emitEvent(ctx, RealtimeModelRawServerEvent{Data: string(rawMessage)}); emitErr != nil {
			return emitErr
		}
		return m.emitEvent(ctx, RealtimeModelErrorEvent{Error: err})
	}
	return m.handleWSEvent(ctx, event)
}

func (m *OpenAIRealtimeWebSocketModel) handleWSEvent(
	ctx context.Context,
	event map[string]any,
) error {
	if event == nil {
		return nil
	}

	if err := m.emitEvent(ctx, RealtimeModelRawServerEvent{Data: event}); err != nil {
		return err
	}

	eventType, _ := event["type"].(string)
	if strings.TrimSpace(eventType) == "" {
		return m.emitEvent(ctx, RealtimeModelErrorEvent{
			Error: errors.New("missing required field type in server event"),
		})
	}
	missingField := func(field string) error {
		return m.emitEvent(ctx, RealtimeModelErrorEvent{
			Error: fmt.Errorf("missing required field %s in %s", field, eventType),
		})
	}

	switch eventType {
	case "response.output_audio.delta", "response.audio.delta":
		responseID, ok := requiredStringField(event, "response_id")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field response_id in %s", eventType),
			})
		}
		itemID, ok := requiredStringField(event, "item_id")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		contentIndex, ok := requiredIntField(event, "content_index")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field content_index in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field output_index in %s", eventType),
			})
		}
		delta, ok := stringField(event, "delta")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field delta in %s", eventType),
			})
		}
		audioBytes, err := base64.StdEncoding.DecodeString(delta)
		if err != nil {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{Error: err})
		}

		m.currentItemID = itemID
		if m.audioStateTracker != nil {
			m.audioStateTracker.OnAudioDelta(itemID, contentIndex, audioBytes)
		}
		return m.emitEvent(ctx, RealtimeModelAudioEvent{
			Data:         audioBytes,
			ResponseID:   responseID,
			ItemID:       itemID,
			ContentIndex: contentIndex,
		})

	case "response.output_audio.done", "response.audio.done":
		if _, ok := requiredStringField(event, "response_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field response_id in %s", eventType),
			})
		}
		itemID, ok := requiredStringField(event, "item_id")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		contentIndex, ok := requiredIntField(event, "content_index")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field content_index in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field output_index in %s", eventType),
			})
		}
		return m.emitEvent(ctx, RealtimeModelAudioDoneEvent{
			ItemID:       itemID,
			ContentIndex: contentIndex,
		})

	case "input_audio_buffer.speech_started":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "audio_start_ms"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field audio_start_ms in %s", eventType),
			})
		}
		if err := m.handleSpeechStarted(ctx, event); err != nil {
			return err
		}
		return nil

	case "input_audio_buffer.speech_stopped":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "audio_end_ms"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field audio_end_ms in %s", eventType),
			})
		}
		return nil

	case "input_audio_buffer.committed":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		if rawPreviousID, hasPreviousID := event["previous_item_id"]; hasPreviousID && rawPreviousID != nil {
			previousID, ok := rawPreviousID.(string)
			if !ok || strings.TrimSpace(previousID) == "" {
				return m.emitEvent(ctx, RealtimeModelErrorEvent{
					Error: fmt.Errorf("invalid field previous_item_id in %s: expected string", eventType),
				})
			}
		}
		return nil

	case "input_audio_buffer.cleared":
		return nil

	case "response.created":
		responsePayload, ok := toStringAnyMap(event["response"])
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field response in %s", eventType),
			})
		}
		if _, ok := requiredStringField(responsePayload, "id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field response.id in %s", eventType),
			})
		}
		m.ongoingResponse = true
		return m.emitEvent(ctx, RealtimeModelTurnStartedEvent{})

	case "response.done":
		responsePayload, ok := toStringAnyMap(event["response"])
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field response in %s", eventType),
			})
		}
		if _, ok := requiredStringField(responsePayload, "id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field response.id in %s", eventType),
			})
		}
		m.ongoingResponse = false
		return m.emitEvent(ctx, RealtimeModelTurnEndedEvent{})

	case "response.queued", "response.in_progress", "response.completed", "response.failed", "response.incomplete":
		responsePayload, ok := toStringAnyMap(event["response"])
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field response in %s", eventType),
			})
		}
		if _, ok := requiredStringField(responsePayload, "id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field response.id in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field sequence_number in %s", eventType),
			})
		}
		return nil

	case "session.created", "session.updated":
		sessionPayload, ok := event["session"]
		if !ok || sessionPayload == nil {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field session in %s", eventType),
			})
		}
		if _, ok := toStringAnyMap(sessionPayload); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("invalid field session in %s: expected object", eventType),
			})
		}
		m.updateCreatedSessionFromPayload(sessionPayload)
		if eventType == "session.created" {
			return m.sendTracingConfig(ctx)
		}
		return nil

	case "error":
		errorPayload, ok := event["error"]
		if !ok || errorPayload == nil {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field error in %s", eventType),
			})
		}
		if _, ok := toStringAnyMap(errorPayload); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("invalid field error in %s: expected object", eventType),
			})
		}
		return m.emitEvent(ctx, RealtimeModelErrorEvent{Error: errorPayload})

	case "conversation.item.deleted":
		itemID, ok := requiredStringField(event, "item_id")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		return m.emitEvent(ctx, RealtimeModelItemDeletedEvent{ItemID: itemID})

	case "conversation.created":
		conversationPayload, ok := event["conversation"]
		if !ok || conversationPayload == nil {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field conversation in %s", eventType),
			})
		}
		if _, ok := toStringAnyMap(conversationPayload); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("invalid field conversation in %s: expected object", eventType),
			})
		}
		return nil

	case "conversation.item.added", "conversation.item.created", "conversation.item.retrieved":
		itemMap, ok := toStringAnyMap(event["item"])
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item in %s", eventType),
			})
		}
		itemType, ok := requiredStringField(itemMap, "type")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item.type in %s", eventType),
			})
		}
		if itemType != "message" {
			return nil
		}
		if _, ok := requiredStringField(itemMap, "id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item.id in %s", eventType),
			})
		}
		var previousItemID *string
		if eventType == "conversation.item.created" {
			rawPreviousID, hasPreviousID := event["previous_item_id"]
			if hasPreviousID {
				if rawPreviousID == nil {
					previousItemID = nil
				} else {
					previousID, ok := rawPreviousID.(string)
					if !ok {
						return m.emitEvent(ctx, RealtimeModelErrorEvent{
							Error: fmt.Errorf("invalid field previous_item_id in %s: expected string", eventType),
						})
					}
					if strings.TrimSpace(previousID) != "" {
						previousItemID = &previousID
					}
				}
			}
		}
		messageItem, err := ConversationItemToRealtimeMessageItem(itemMap, previousItemID)
		if err != nil {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{Error: err})
		}
		return m.emitEvent(ctx, RealtimeModelItemUpdatedEvent{Item: *messageItem})

	case "conversation.item.input_audio_transcription.completed", "conversation.item.truncated":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		if eventType == "conversation.item.truncated" {
			if _, ok := requiredIntField(event, "content_index"); !ok {
				return m.emitEvent(ctx, RealtimeModelErrorEvent{
					Error: fmt.Errorf("missing required field content_index in %s", eventType),
				})
			}
			if _, ok := requiredIntField(event, "audio_end_ms"); !ok {
				return m.emitEvent(ctx, RealtimeModelErrorEvent{
					Error: fmt.Errorf("missing required field audio_end_ms in %s", eventType),
				})
			}
		}
		if strings.TrimSpace(m.currentItemID) != "" {
			_ = m.dispatchClientEvent(ctx, map[string]any{
				"type":    "conversation.item.retrieve",
				"item_id": m.currentItemID,
			})
		}
		if eventType == "conversation.item.input_audio_transcription.completed" {
			transcript, ok := stringField(event, "transcript")
			if !ok {
				return m.emitEvent(ctx, RealtimeModelErrorEvent{
					Error: fmt.Errorf("missing required field transcript in %s", eventType),
				})
			}
			return m.emitEvent(ctx, RealtimeModelInputAudioTranscriptionCompletedEvent{
				ItemID:     event["item_id"].(string),
				Transcript: transcript,
			})
		}
		return nil

	case "conversation.item.input_audio_transcription.failed":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		errorPayload, ok := event["error"]
		if !ok || errorPayload == nil {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field error in %s", eventType),
			})
		}
		if _, ok := toStringAnyMap(errorPayload); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("invalid field error in %s: expected object", eventType),
			})
		}
		return nil

	case "response.output_audio_transcript.delta", "response.audio.transcript.delta":
		itemID, ok := requiredStringField(event, "item_id")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "content_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field content_index in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field output_index in %s", eventType),
			})
		}
		delta, ok := stringField(event, "delta")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field delta in %s", eventType),
			})
		}
		responseID, ok := requiredStringField(event, "response_id")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field response_id in %s", eventType),
			})
		}
		return m.emitEvent(ctx, RealtimeModelTranscriptDeltaEvent{
			ItemID:     itemID,
			Delta:      delta,
			ResponseID: responseID,
		})

	case "response.output_audio_transcript.done", "response.audio.transcript.done":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "content_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field content_index in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field output_index in %s", eventType),
			})
		}
		if _, ok := requiredStringField(event, "response_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field response_id in %s", eventType),
			})
		}
		if _, ok := stringField(event, "transcript"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field transcript in %s", eventType),
			})
		}
		return nil

	case "conversation.item.input_audio_transcription.delta":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		if _, ok := stringField(event, "delta"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field delta in %s", eventType),
			})
		}
		return nil

	case "response.output_text.delta":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field output_index in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "content_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field content_index in %s", eventType),
			})
		}
		if _, ok := stringField(event, "delta"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field delta in %s", eventType),
			})
		}
		return nil

	case "response.output_text.done":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field output_index in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "content_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field content_index in %s", eventType),
			})
		}
		if _, ok := stringField(event, "text"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field text in %s", eventType),
			})
		}
		return nil

	case "response.function_call_arguments.delta":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field output_index in %s", eventType),
			})
		}
		if _, ok := stringField(event, "delta"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field delta in %s", eventType),
			})
		}
		return nil

	case "response.function_call_arguments.done":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field output_index in %s", eventType),
			})
		}
		if _, ok := requiredStringField(event, "name"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field name in %s", eventType),
			})
		}
		if _, ok := stringField(event, "arguments"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field arguments in %s", eventType),
			})
		}
		return nil

	case "response.output_item.added", "response.output_item.done":
		if _, ok := requiredIntField(event, "output_index"); !ok {
			itemMap, ok := toStringAnyMap(event["item"])
			if !ok {
				return m.emitEvent(ctx, RealtimeModelErrorEvent{
					Error: fmt.Errorf("missing required field item in %s", eventType),
				})
			}
			itemType, _ := itemMap["type"].(string)
			if itemType != "message" {
				return m.emitEvent(ctx, RealtimeModelErrorEvent{
					Error: fmt.Errorf("missing required field output_index in %s", eventType),
				})
			}
		}
		return m.handleOutputItemEvent(ctx, event, eventType, eventType == "response.output_item.done")

	case "response.content_part.added", "response.content_part.done":
		if _, ok := requiredStringField(event, "response_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field response_id in %s", eventType),
			})
		}
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field output_index in %s", eventType),
			})
		}
		if _, ok := requiredIntField(event, "content_index"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field content_index in %s", eventType),
			})
		}
		partPayload, ok := event["part"]
		if !ok || partPayload == nil {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field part in %s", eventType),
			})
		}
		partMap, ok := toStringAnyMap(partPayload)
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("invalid field part in %s: expected object", eventType),
			})
		}
		if _, ok := requiredStringField(partMap, "type"); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field part.type in %s", eventType),
			})
		}
		return nil

	case "response.output_text.annotation.added":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "content_index"); !ok {
			return missingField("content_index")
		}
		if _, ok := requiredIntField(event, "annotation_index"); !ok {
			return missingField("annotation_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		if _, ok := event["annotation"]; !ok {
			return missingField("annotation")
		}
		return nil

	case "response.refusal.delta", "response.reasoning_text.delta":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "content_index"); !ok {
			return missingField("content_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		if _, ok := stringField(event, "delta"); !ok {
			return missingField("delta")
		}
		return nil

	case "response.refusal.done":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "content_index"); !ok {
			return missingField("content_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		if _, ok := stringField(event, "refusal"); !ok {
			return missingField("refusal")
		}
		return nil

	case "response.reasoning_text.done":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "content_index"); !ok {
			return missingField("content_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		if _, ok := stringField(event, "text"); !ok {
			return missingField("text")
		}
		return nil

	case "response.reasoning_summary_text.delta":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "summary_index"); !ok {
			return missingField("summary_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		if _, ok := stringField(event, "delta"); !ok {
			return missingField("delta")
		}
		return nil

	case "response.reasoning_summary_text.done":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "summary_index"); !ok {
			return missingField("summary_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		if _, ok := stringField(event, "text"); !ok {
			return missingField("text")
		}
		return nil

	case "response.reasoning_summary_part.added", "response.reasoning_summary_part.done":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "summary_index"); !ok {
			return missingField("summary_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		partRaw, ok := event["part"]
		if !ok || partRaw == nil {
			return missingField("part")
		}
		partMap, ok := toStringAnyMap(partRaw)
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("invalid field part in %s: expected object", eventType),
			})
		}
		if _, ok := requiredStringField(partMap, "type"); !ok {
			return missingField("part.type")
		}
		if _, ok := stringField(partMap, "text"); !ok {
			return missingField("part.text")
		}
		return nil

	case "response.web_search_call.in_progress", "response.web_search_call.searching", "response.web_search_call.completed",
		"response.file_search_call.in_progress", "response.file_search_call.searching", "response.file_search_call.completed",
		"response.image_generation_call.in_progress", "response.image_generation_call.generating", "response.image_generation_call.completed",
		"response.code_interpreter_call.in_progress", "response.code_interpreter_call.interpreting", "response.code_interpreter_call.completed",
		"response.mcp_call.in_progress", "response.mcp_call.completed", "response.mcp_call.failed",
		"response.mcp_list_tools.in_progress", "response.mcp_list_tools.completed", "response.mcp_list_tools.failed":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		return nil

	case "response.image_generation_call.partial_image":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		if _, ok := stringField(event, "partial_image_b64"); !ok {
			return missingField("partial_image_b64")
		}
		if _, ok := requiredIntField(event, "partial_image_index"); !ok {
			return missingField("partial_image_index")
		}
		return nil

	case "response.code_interpreter_call_code.delta":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		if _, ok := stringField(event, "delta"); !ok {
			return missingField("delta")
		}
		return nil

	case "response.code_interpreter_call_code.done":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		if _, ok := stringField(event, "code"); !ok {
			return missingField("code")
		}
		return nil

	case "response.mcp_call_arguments.delta", "response.custom_tool_call_input.delta":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		if _, ok := stringField(event, "delta"); !ok {
			return missingField("delta")
		}
		return nil

	case "response.mcp_call_arguments.done":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		if _, ok := stringField(event, "arguments"); !ok {
			return missingField("arguments")
		}
		return nil

	case "response.custom_tool_call_input.done":
		if _, ok := requiredStringField(event, "item_id"); !ok {
			return missingField("item_id")
		}
		if _, ok := requiredIntField(event, "output_index"); !ok {
			return missingField("output_index")
		}
		if _, ok := requiredIntField(event, "sequence_number"); !ok {
			return missingField("sequence_number")
		}
		if _, ok := stringField(event, "input"); !ok {
			return missingField("input")
		}
		return nil

	case "input_audio_buffer.timeout_triggered":
		itemID, ok := requiredStringField(event, "item_id")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item_id in %s", eventType),
			})
		}
		audioStartMS, ok := requiredIntField(event, "audio_start_ms")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field audio_start_ms in %s", eventType),
			})
		}
		audioEndMS, ok := requiredIntField(event, "audio_end_ms")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field audio_end_ms in %s", eventType),
			})
		}
		return m.emitEvent(ctx, RealtimeModelInputAudioTimeoutTriggeredEvent{
			ItemID:       itemID,
			AudioStartMS: audioStartMS,
			AudioEndMS:   audioEndMS,
		})

	case "rate_limits.updated":
		rateLimitsPayload, ok := event["rate_limits"]
		if !ok || rateLimitsPayload == nil {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field rate_limits in %s", eventType),
			})
		}
		if _, ok := rateLimitsPayload.([]any); !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("invalid field rate_limits in %s: expected array", eventType),
			})
		}
		return nil
	default:
		return m.emitEvent(ctx, RealtimeModelErrorEvent{
			Error: fmt.Errorf("unsupported realtime server event type %s", eventType),
		})
	}

	return nil
}

func (m *OpenAIRealtimeWebSocketModel) handleSpeechStarted(
	ctx context.Context,
	event map[string]any,
) error {
	if m.audioStateTracker == nil {
		return nil
	}
	lastAudio := m.audioStateTracker.GetLastAudioItem()
	if lastAudio == nil {
		return nil
	}

	if err := m.emitEvent(ctx, RealtimeModelAudioInterruptedEvent{
		ItemID:       lastAudio.ItemID,
		ContentIndex: lastAudio.ItemContentIndex,
	}); err != nil {
		return err
	}

	playbackItemID, playbackContentIndex, playbackElapsedMS, hasPlaybackState := m.currentPlaybackState()
	audioEndMSOverride, hasOverride := numericToFloat64(event["audio_end_ms"])
	effectiveElapsedMS := playbackElapsedMS
	if hasOverride && audioEndMSOverride > 0 {
		effectiveElapsedMS = audioEndMSOverride
	}

	if hasPlaybackState {
		truncatedMS := maxInt(int(math.Round(maxFloat(0, effectiveElapsedMS))), 0)
		_, maxAudioMS, hasAudioLimit := m.audioLengthLimit(playbackItemID, playbackContentIndex)
		if !(hasAudioLimit && truncatedMS >= maxAudioMS && !m.ongoingResponse) {
			if hasAudioLimit {
				truncatedMS = minInt(truncatedMS, maxAudioMS)
			}
			if err := m.dispatchClientEvent(
				ctx,
				ConvertInterrupt(playbackItemID, playbackContentIndex, truncatedMS),
			); err != nil {
				return err
			}
		}
	}

	m.audioStateTracker.OnInterrupted()
	if m.playbackTracker != nil {
		m.playbackTracker.OnInterrupted()
	}

	if !m.automaticResponseCancellationEnabled {
		return m.cancelResponse(ctx)
	}
	return nil
}

func (m *OpenAIRealtimeWebSocketModel) handleOutputItemEvent(
	ctx context.Context,
	event map[string]any,
	eventType string,
	isDone bool,
) error {
	item, ok := toStringAnyMap(event["item"])
	if !ok {
		return m.emitEvent(ctx, RealtimeModelErrorEvent{
			Error: fmt.Errorf("missing required field item in %s", eventType),
		})
	}

	itemType, ok := requiredStringField(item, "type")
	if !ok {
		return m.emitEvent(ctx, RealtimeModelErrorEvent{
			Error: fmt.Errorf("missing required field item.type in %s", eventType),
		})
	}
	switch itemType {
	case "function_call":
		status, _ := item["status"].(string)
		if status != "completed" {
			return nil
		}

		callID, ok := requiredStringField(item, "call_id")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item.call_id in %s", eventType),
			})
		}
		name, ok := requiredStringField(item, "name")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item.name in %s", eventType),
			})
		}
		arguments, ok := stringField(item, "arguments")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item.arguments in %s", eventType),
			})
		}
		itemID, _ := stringField(item, "id")

		if err := m.emitEvent(ctx, RealtimeModelItemUpdatedEvent{
			Item: RealtimeToolCallItem{
				ItemID:    itemID,
				CallID:    callID,
				Type:      "function_call",
				Status:    "in_progress",
				Arguments: arguments,
				Name:      name,
			},
		}); err != nil {
			return err
		}

		var idPtr *string
		if strings.TrimSpace(itemID) != "" {
			idPtr = &itemID
		}
		return m.emitEvent(ctx, RealtimeModelToolCallEvent{
			Name:      name,
			CallID:    callID,
			Arguments: arguments,
			ID:        idPtr,
		})

	case "message":
		itemID, ok := requiredStringField(item, "id")
		if !ok {
			return m.emitEvent(ctx, RealtimeModelErrorEvent{
				Error: fmt.Errorf("missing required field item.id in %s", eventType),
			})
		}
		role, _ := item["role"].(string)
		if strings.TrimSpace(role) == "" {
			role = "assistant"
		}

		content := make([]RealtimeMessageContent, 0)
		for _, raw := range extractContentParts(item["content"]) {
			part, ok := toStringAnyMap(raw)
			if !ok {
				continue
			}
			partType, _ := part["type"].(string)
			switch partType {
			case "audio", "output_audio":
				content = append(content, RealtimeMessageContent{
					Type:       "audio",
					Audio:      stringValuePtr(part["audio"]),
					Transcript: stringValuePtr(part["transcript"]),
				})
			case "text", "output_text":
				content = append(content, RealtimeMessageContent{
					Type: "text",
					Text: stringValuePtr(part["text"]),
				})
			}
		}

		status, _ := item["status"].(string)
		if status != "in_progress" && status != "completed" && status != "incomplete" {
			if isDone {
				status = "completed"
			} else {
				status = "in_progress"
			}
		}

		return m.emitEvent(ctx, RealtimeModelItemUpdatedEvent{
			Item: RealtimeMessageItem{
				ItemID:  itemID,
				Type:    "message",
				Role:    role,
				Content: content,
				Status:  &status,
			},
		})
	default:
		return m.emitEvent(ctx, RealtimeModelErrorEvent{
			Error: fmt.Errorf("unsupported output item type %s in %s", itemType, eventType),
		})
	}

	return nil
}

func (m *OpenAIRealtimeWebSocketModel) updateCreatedSessionFromPayload(payload any) {
	session := NormalizeSessionPayload(payload)
	if session == nil {
		return
	}
	m.createdSession = session
	m.automaticResponseCancellationEnabled = isAutomaticResponseCancellationEnabled(session)

	if outputAudioFormat := ExtractSessionAudioFormat(*session); outputAudioFormat != nil {
		if m.audioStateTracker != nil {
			m.audioStateTracker.SetAudioFormat(*outputAudioFormat)
		}
		if m.playbackTracker != nil {
			m.playbackTracker.SetAudioFormat(*outputAudioFormat)
		}
	}
}

func (m *OpenAIRealtimeWebSocketModel) audioLengthLimit(
	itemID string,
	contentIndex int,
) (float64, int, bool) {
	if m.audioStateTracker == nil {
		return 0, 0, false
	}
	audioState := m.audioStateTracker.GetState(itemID, contentIndex)
	if audioState == nil {
		return 0, 0, false
	}
	maxAudioMS := int(math.Ceil(audioState.AudioLengthMS))
	return audioState.AudioLengthMS, maxAudioMS, true
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func requiredStringField(payload map[string]any, key string) (string, bool) {
	value, ok := payload[key].(string)
	if !ok || strings.TrimSpace(value) == "" {
		return "", false
	}
	return value, true
}

func stringField(payload map[string]any, key string) (string, bool) {
	value, ok := payload[key]
	if !ok {
		return "", false
	}
	stringValue, ok := value.(string)
	if !ok {
		return "", false
	}
	return stringValue, true
}

func requiredIntField(payload map[string]any, key string) (int, bool) {
	value, ok := numericToInt64(payload[key])
	if !ok {
		return 0, false
	}
	return int(value), true
}

func (m *OpenAIRealtimeWebSocketModel) sendTracingConfig(ctx context.Context) error {
	tracingParam := ConvertTracingConfig(m.tracingConfig)
	if tracingParam == nil {
		return nil
	}

	modelName := strings.TrimSpace(m.model)
	if modelName == "" {
		modelName = defaultRealtimeModelName
	}

	return m.sendSessionUpdatePayload(ctx, &oairealtime.RealtimeSessionCreateRequestParam{
		Type:    "realtime",
		Model:   oairealtime.RealtimeSessionCreateRequestModel(modelName),
		Tracing: *tracingParam,
	})
}

// GetSessionConfig builds a session configuration payload from model settings.
func (m *OpenAIRealtimeWebSocketModel) GetSessionConfig(
	modelSettings RealtimeSessionModelSettings,
) (*oairealtime.RealtimeSessionCreateRequestParam, error) {
	if modelSettings == nil {
		modelSettings = RealtimeSessionModelSettings{}
	}

	audioConfig, _ := toStringAnyMap(modelSettings["audio"])
	inputAudioConfig, _ := toStringAnyMap(audioConfig["input"])
	outputAudioConfig, _ := toStringAnyMap(audioConfig["output"])

	var audioInput oairealtime.RealtimeAudioConfigInputParam
	var audioOutput oairealtime.RealtimeAudioConfigOutputParam

	inputFormatSource, hasInputFormat := inputAudioConfig["format"]
	if !hasInputFormat {
		if m.callID != "" {
			inputFormatSource = modelSettings["input_audio_format"]
		} else if explicitInput, ok := modelSettings["input_audio_format"]; ok {
			inputFormatSource = explicitInput
		} else {
			inputFormatSource = defaultRealtimeInputAudioFormat
		}
	}
	if inputFormat := ToRealtimeAudioFormat(inputFormatSource); inputFormat != nil {
		audioInput.Format = *inputFormat
	}

	if noiseReductionSource, ok := inputAudioConfig["noise_reduction"]; ok {
		if noiseReduction, ok := toNoiseReductionParam(noiseReductionSource); ok {
			audioInput.NoiseReduction = noiseReduction
		}
	} else if noiseReductionSource, ok := modelSettings["input_audio_noise_reduction"]; ok {
		if noiseReduction, ok := toNoiseReductionParam(noiseReductionSource); ok {
			audioInput.NoiseReduction = noiseReduction
		}
	}

	if transcriptionSource, ok := inputAudioConfig["transcription"]; ok {
		if transcription, ok := toInputTranscriptionParam(transcriptionSource); ok {
			audioInput.Transcription = transcription
		}
	} else if transcriptionSource, ok := modelSettings["input_audio_transcription"]; ok {
		if transcription, ok := toInputTranscriptionParam(transcriptionSource); ok {
			audioInput.Transcription = transcription
		}
	} else if transcription, ok := toInputTranscriptionParam(defaultRealtimeInputTranscription); ok {
		audioInput.Transcription = transcription
	}

	turnDetectionSource, hasTurnDetection := inputAudioConfig["turn_detection"]
	if !hasTurnDetection {
		turnDetectionSource, hasTurnDetection = modelSettings["turn_detection"]
	}
	if hasTurnDetection {
		if turnDetectionSource == nil {
			audioInput.SetExtraFields(map[string]any{"turn_detection": nil})
		} else if turnDetection, ok := toTurnDetectionParam(turnDetectionSource); ok {
			audioInput.TurnDetection = turnDetection
		}
	} else if turnDetection, ok := toTurnDetectionParam(defaultRealtimeTurnDetection); ok {
		audioInput.TurnDetection = turnDetection
	}

	requestedVoice, _ := outputAudioConfig["voice"].(string)
	if strings.TrimSpace(requestedVoice) == "" {
		if configuredVoice, ok := stringFromSettings(modelSettings, "voice"); ok &&
			strings.TrimSpace(configuredVoice) != "" {
			requestedVoice = configuredVoice
		} else {
			requestedVoice = defaultRealtimeVoice
		}
	}
	audioOutput.Voice = oairealtime.RealtimeAudioConfigOutputVoice(requestedVoice)

	outputFormatSource, hasOutputFormat := outputAudioConfig["format"]
	if !hasOutputFormat {
		if m.callID != "" {
			outputFormatSource = modelSettings["output_audio_format"]
		} else if explicitOutput, ok := modelSettings["output_audio_format"]; ok {
			outputFormatSource = explicitOutput
		} else {
			outputFormatSource = defaultRealtimeOutputAudioFormat
		}
	}
	if outputFormat := ToRealtimeAudioFormat(outputFormatSource); outputFormat != nil {
		audioOutput.Format = *outputFormat
	}

	if speedSource, ok := outputAudioConfig["speed"]; ok {
		if speed, ok := numericToFloat64(speedSource); ok {
			audioOutput.Speed = param.NewOpt(speed)
		}
	} else if speedSource, ok := modelSettings["speed"]; ok {
		if speed, ok := numericToFloat64(speedSource); ok {
			audioOutput.Speed = param.NewOpt(speed)
		}
	}

	outputModalities := stringSliceFromAny(modelSettings["output_modalities"])
	if len(outputModalities) == 0 {
		outputModalities = stringSliceFromAny(modelSettings["modalities"])
	}
	if len(outputModalities) == 0 {
		outputModalities = slices.Clone(defaultRealtimeOutputModalities)
	}

	modelName, _ := stringFromSettings(modelSettings, "model_name")
	if strings.TrimSpace(modelName) == "" {
		modelName = m.model
	}
	if strings.TrimSpace(modelName) == "" {
		modelName = defaultRealtimeModelName
	}

	tools := toolsFromSettings(modelSettings["tools"])
	handoffs := handoffsFromSettings(modelSettings["handoffs"])
	sessionTools, err := m.toolsToSessionTools(tools, handoffs)
	if err != nil {
		return nil, err
	}

	session := &oairealtime.RealtimeSessionCreateRequestParam{
		Type:             "realtime",
		Model:            oairealtime.RealtimeSessionCreateRequestModel(modelName),
		OutputModalities: outputModalities,
		Audio: oairealtime.RealtimeAudioConfigParam{
			Input:  audioInput,
			Output: audioOutput,
		},
		Tools: sessionTools,
	}

	if instructions, ok := stringFromSettings(modelSettings, "instructions"); ok {
		session.Instructions = param.NewOpt(instructions)
	}

	if tracingParam := ConvertTracingConfig(modelSettings["tracing"]); tracingParam != nil {
		session.Tracing = *tracingParam
	}

	if maxOutputTokens, ok := numericToInt64(modelSettings["max_output_tokens"]); ok {
		session.MaxOutputTokens = oairealtime.RealtimeSessionCreateRequestMaxOutputTokensUnionParam{
			OfInt: param.NewOpt(maxOutputTokens),
		}
	}

	if toolChoice, ok := modelSettings["tool_choice"].(string); ok {
		toolChoice = strings.TrimSpace(toolChoice)
		if toolChoice != "" {
			session.ToolChoice = oairealtime.RealtimeToolChoiceConfigUnionParam{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptions(toolChoice)),
			}
		}
	}

	if promptMap, ok := toStringAnyMap(modelSettings["prompt"]); ok {
		if promptID, ok := promptMap["id"].(string); ok && strings.TrimSpace(promptID) != "" {
			prompt := responses.ResponsePromptParam{ID: promptID}
			if version, ok := promptMap["version"].(string); ok && strings.TrimSpace(version) != "" {
				prompt.Version = param.NewOpt(version)
			}
			if rawVars, ok := toStringAnyMap(promptMap["variables"]); ok {
				variables := make(map[string]responses.ResponsePromptVariableUnionParam, len(rawVars))
				for key, raw := range rawVars {
					if value, ok := raw.(string); ok {
						variables[key] = responses.ResponsePromptVariableUnionParam{
							OfString: param.NewOpt(value),
						}
					}
				}
				prompt.Variables = variables
			}
			session.Prompt = prompt
		}
	}

	return session, nil
}

func (m *OpenAIRealtimeWebSocketModel) toolsToSessionTools(
	tools []agents.Tool,
	handoffs []agents.Handoff,
) (oairealtime.RealtimeToolsConfigParam, error) {
	converted := make(oairealtime.RealtimeToolsConfigParam, 0, len(tools)+len(handoffs))
	for _, tool := range tools {
		functionTool, ok := tool.(agents.FunctionTool)
		if !ok {
			return nil, fmt.Errorf(
				"tool %T is unsupported: realtime only supports function tools",
				tool,
			)
		}
		converted = append(converted, oairealtime.RealtimeToolsConfigUnionParam{
			OfFunction: &oairealtime.RealtimeFunctionToolParam{
				Name:        param.NewOpt(functionTool.Name),
				Description: param.NewOpt(functionTool.Description),
				Parameters:  functionTool.ParamsJSONSchema,
				Type:        "function",
			},
		})
	}

	for _, handoff := range handoffs {
		converted = append(converted, oairealtime.RealtimeToolsConfigUnionParam{
			OfFunction: &oairealtime.RealtimeFunctionToolParam{
				Name:        param.NewOpt(handoff.ToolName),
				Description: param.NewOpt(handoff.ToolDescription),
				Parameters:  handoff.InputJSONSchema,
				Type:        "function",
			},
		})
	}
	return converted, nil
}

func toInputTranscriptionParam(input any) (oairealtime.AudioTranscriptionParam, bool) {
	mapping, ok := toStringAnyMap(input)
	if !ok {
		return oairealtime.AudioTranscriptionParam{}, false
	}
	model, _ := mapping["model"].(string)
	if strings.TrimSpace(model) == "" {
		return oairealtime.AudioTranscriptionParam{}, false
	}
	out := oairealtime.AudioTranscriptionParam{
		Model: oairealtime.AudioTranscriptionModel(model),
	}
	if language, ok := mapping["language"].(string); ok && strings.TrimSpace(language) != "" {
		out.Language = param.NewOpt(language)
	}
	if prompt, ok := mapping["prompt"].(string); ok && strings.TrimSpace(prompt) != "" {
		out.Prompt = param.NewOpt(prompt)
	}
	return out, true
}

func toNoiseReductionParam(
	input any,
) (oairealtime.RealtimeAudioConfigInputNoiseReductionParam, bool) {
	if input == nil {
		return oairealtime.RealtimeAudioConfigInputNoiseReductionParam{}, false
	}
	mapping, ok := toStringAnyMap(input)
	if !ok {
		return oairealtime.RealtimeAudioConfigInputNoiseReductionParam{}, false
	}
	noiseType, _ := mapping["type"].(string)
	noiseType = strings.TrimSpace(noiseType)
	if noiseType == "" {
		return oairealtime.RealtimeAudioConfigInputNoiseReductionParam{}, false
	}
	return oairealtime.RealtimeAudioConfigInputNoiseReductionParam{
		Type: oairealtime.NoiseReductionType(noiseType),
	}, true
}

func toTurnDetectionParam(
	input any,
) (oairealtime.RealtimeAudioInputTurnDetectionUnionParam, bool) {
	if input == nil {
		return oairealtime.RealtimeAudioInputTurnDetectionUnionParam{}, false
	}
	mapping, ok := toStringAnyMap(NormalizeTurnDetectionConfig(input))
	if !ok {
		return oairealtime.RealtimeAudioInputTurnDetectionUnionParam{}, false
	}
	tdType, _ := mapping["type"].(string)
	tdType = strings.TrimSpace(tdType)
	switch tdType {
	case "semantic_vad":
		semantic := oairealtime.RealtimeAudioInputTurnDetectionSemanticVadParam{
			Type: "semantic_vad",
		}
		if modelVersion, ok := mapping["model_version"].(string); ok &&
			strings.TrimSpace(modelVersion) != "" {
			semantic.SetExtraFields(map[string]any{"model_version": modelVersion})
		}
		if createResponse, ok := mapping["create_response"].(bool); ok {
			semantic.CreateResponse = param.NewOpt(createResponse)
		}
		if interruptResponse, ok := mapping["interrupt_response"].(bool); ok {
			semantic.InterruptResponse = param.NewOpt(interruptResponse)
		}
		if eagerness, ok := mapping["eagerness"].(string); ok && strings.TrimSpace(eagerness) != "" {
			semantic.Eagerness = eagerness
		}
		return oairealtime.RealtimeAudioInputTurnDetectionUnionParam{
			OfSemanticVad: &semantic,
		}, true
	case "server_vad":
		server := oairealtime.RealtimeAudioInputTurnDetectionServerVadParam{
			Type: "server_vad",
		}
		if modelVersion, ok := mapping["model_version"].(string); ok &&
			strings.TrimSpace(modelVersion) != "" {
			server.SetExtraFields(map[string]any{"model_version": modelVersion})
		}
		if idleTimeoutMS, ok := numericToInt64(mapping["idle_timeout_ms"]); ok {
			server.IdleTimeoutMs = param.NewOpt(idleTimeoutMS)
		}
		if createResponse, ok := mapping["create_response"].(bool); ok {
			server.CreateResponse = param.NewOpt(createResponse)
		}
		if interruptResponse, ok := mapping["interrupt_response"].(bool); ok {
			server.InterruptResponse = param.NewOpt(interruptResponse)
		}
		if prefixPaddingMS, ok := numericToInt64(mapping["prefix_padding_ms"]); ok {
			server.PrefixPaddingMs = param.NewOpt(prefixPaddingMS)
		}
		if silenceDurationMS, ok := numericToInt64(mapping["silence_duration_ms"]); ok {
			server.SilenceDurationMs = param.NewOpt(silenceDurationMS)
		}
		if threshold, ok := numericToFloat64(mapping["threshold"]); ok {
			server.Threshold = param.NewOpt(threshold)
		}
		return oairealtime.RealtimeAudioInputTurnDetectionUnionParam{
			OfServerVad: &server,
		}, true
	default:
		return oairealtime.RealtimeAudioInputTurnDetectionUnionParam{}, false
	}
}

func stringFromSettings(settings RealtimeSessionModelSettings, key string) (string, bool) {
	if settings == nil {
		return "", false
	}
	value, ok := settings[key].(string)
	return value, ok
}

func stringSliceFromAny(value any) []string {
	switch v := value.(type) {
	case []string:
		return slices.Clone(v)
	case []any:
		out := make([]string, 0, len(v))
		for _, raw := range v {
			if item, ok := raw.(string); ok && strings.TrimSpace(item) != "" {
				out = append(out, item)
			}
		}
		return out
	default:
		return nil
	}
}

func numericToFloat64(value any) (float64, bool) {
	switch v := value.(type) {
	case int:
		return float64(v), true
	case int8:
		return float64(v), true
	case int16:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case uint:
		return float64(v), true
	case uint8:
		return float64(v), true
	case uint16:
		return float64(v), true
	case uint32:
		return float64(v), true
	case uint64:
		return float64(v), true
	case float32:
		return float64(v), true
	case float64:
		return v, true
	default:
		return 0, false
	}
}

func toolsFromSettings(value any) []agents.Tool {
	switch v := value.(type) {
	case []agents.Tool:
		return slices.Clone(v)
	case []any:
		out := make([]agents.Tool, 0, len(v))
		for _, each := range v {
			if tool, ok := each.(agents.Tool); ok {
				out = append(out, tool)
			}
		}
		return out
	default:
		return nil
	}
}

func handoffsFromSettings(value any) []agents.Handoff {
	switch v := value.(type) {
	case []agents.Handoff:
		return slices.Clone(v)
	case []any:
		out := make([]agents.Handoff, 0, len(v))
		for _, each := range v {
			if handoff, ok := each.(agents.Handoff); ok {
				out = append(out, handoff)
			}
		}
		return out
	default:
		return nil
	}
}

// OpenAIRealtimeSIPModel is a realtime transport that requires call_id attachment.
type OpenAIRealtimeSIPModel struct {
	*OpenAIRealtimeWebSocketModel
}

// NewOpenAIRealtimeSIPModel creates a SIP-attached realtime model wrapper.
func NewOpenAIRealtimeSIPModel() *OpenAIRealtimeSIPModel {
	return &OpenAIRealtimeSIPModel{
		OpenAIRealtimeWebSocketModel: NewOpenAIRealtimeWebSocketModel(),
	}
}

func (m *OpenAIRealtimeSIPModel) Connect(ctx context.Context, options RealtimeModelConfig) error {
	if strings.TrimSpace(options.CallID) == "" {
		return errors.New("OpenAIRealtimeSIPModel requires `call_id` in model configuration")
	}
	return m.OpenAIRealtimeWebSocketModel.Connect(ctx, options)
}
