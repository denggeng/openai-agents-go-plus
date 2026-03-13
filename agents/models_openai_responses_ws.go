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
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"reflect"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/denggeng/openai-agents-go-plus/tracing"
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/gorilla/websocket"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
)

const (
	defaultOpenAIResponsesWebsocketBaseURL = "https://api.openai.com/v1"
)

type responsesWebsocketTimeout struct {
	Duration time.Duration
	Set      bool
}

type responsesWebsocketRequestTimeouts struct {
	Lock    responsesWebsocketTimeout
	Connect responsesWebsocketTimeout
	Send    responsesWebsocketTimeout
	Recv    responsesWebsocketTimeout
}

// ResponsesWebSocketError surfaces `error` / `response.error` websocket frames.
type ResponsesWebSocketError struct {
	*AgentsError
	EventType    string
	ErrorType    string
	Code         string
	ErrorMessage string
	Payload      map[string]any
}

func (err ResponsesWebSocketError) Error() string {
	if err.AgentsError == nil {
		return "ResponsesWebSocketError"
	}
	return err.AgentsError.Error()
}

func (err ResponsesWebSocketError) Unwrap() error {
	return err.AgentsError
}

// OpenAIResponsesWSModel is the websocket-transport wrapper for the Responses API.
type OpenAIResponsesWSModel struct {
	httpModel        OpenAIResponsesModel
	websocketBaseURL string
	closed           atomic.Bool

	requestSlot chan struct{}

	connMu       sync.Mutex
	conn         *websocket.Conn
	connIdentity string
}

func NewOpenAIResponsesWSModel(
	model openai.ChatModel,
	client OpenaiClient,
	websocketBaseURL string,
) *OpenAIResponsesWSModel {
	if websocketBaseURL == "" {
		websocketBaseURL = client.WebsocketBaseURL.Or("")
	}
	requestSlot := make(chan struct{}, 1)
	requestSlot <- struct{}{}
	return &OpenAIResponsesWSModel{
		httpModel: OpenAIResponsesModel{
			Model:  model,
			client: client,
		},
		websocketBaseURL: websocketBaseURL,
		requestSlot:      requestSlot,
	}
}

func (m *OpenAIResponsesWSModel) GetResponse(
	ctx context.Context,
	params ModelResponseParams,
) (*ModelResponse, error) {
	if m.closed.Load() {
		return nil, UserErrorf("responses websocket model is closed")
	}

	var (
		finalResponse *responses.Response
		u             *usage.Usage
	)

	err := tracing.ResponseSpan(
		ctx,
		tracing.ResponseSpanParams{Disabled: params.Tracing.IsDisabled()},
		func(ctx context.Context, spanResponse tracing.Span) (err error) {
			defer func() {
				if err != nil {
					var value string
					if params.Tracing.IncludeData() {
						value = err.Error()
					} else {
						value = fmt.Sprintf("%T", err)
					}
					spanResponse.SetError(tracing.SpanError{
						Message: "Error getting response",
						Data:    map[string]any{"error": value},
					})
				}
			}()

			finalResponse, err = m.executeRequest(ctx, params, nil)
			if err != nil {
				return err
			}

			u = usage.NewUsage()
			if finalResponse != nil && !isZeroResponseUsage(finalResponse.Usage) {
				*u = usage.Usage{
					Requests:            1,
					InputTokens:         uint64(finalResponse.Usage.InputTokens),
					InputTokensDetails:  finalResponse.Usage.InputTokensDetails,
					OutputTokens:        uint64(finalResponse.Usage.OutputTokens),
					OutputTokensDetails: finalResponse.Usage.OutputTokensDetails,
					TotalTokens:         uint64(finalResponse.Usage.TotalTokens),
				}
			}

			if params.Tracing.IncludeData() && finalResponse != nil {
				spanData := spanResponse.SpanData().(*tracing.ResponseSpanData)
				spanData.Response = finalResponse
				spanData.Input = params.Input
			}
			return nil
		},
	)
	if err != nil {
		return nil, err
	}

	return &ModelResponse{
		Output:     finalResponse.Output,
		Usage:      u,
		ResponseID: finalResponse.ID,
	}, nil
}

func (m *OpenAIResponsesWSModel) StreamResponse(
	ctx context.Context,
	params ModelResponseParams,
	yield ModelStreamResponseCallback,
) error {
	if m.closed.Load() {
		return UserErrorf("responses websocket model is closed")
	}
	if yield == nil {
		yield = func(context.Context, TResponseStreamEvent) error { return nil }
	}

	return tracing.ResponseSpan(
		ctx,
		tracing.ResponseSpanParams{Disabled: params.Tracing.IsDisabled()},
		func(ctx context.Context, spanResponse tracing.Span) (err error) {
			defer func() {
				if err != nil {
					var value string
					if params.Tracing.IncludeData() {
						value = err.Error()
					} else {
						value = fmt.Sprintf("%T", err)
					}
					spanResponse.SetError(tracing.SpanError{
						Message: "Error streaming response",
						Data:    map[string]any{"error": value},
					})
				}
			}()

			finalResponse, err := m.executeRequest(ctx, params, yield)
			if err != nil {
				return err
			}
			if params.Tracing.IncludeData() && finalResponse != nil {
				spanData := spanResponse.SpanData().(*tracing.ResponseSpanData)
				spanData.Response = finalResponse
				spanData.Input = params.Input
			}
			return nil
		},
	)
}

func (m *OpenAIResponsesWSModel) executeRequest(
	ctx context.Context,
	params ModelResponseParams,
	yield ModelStreamResponseCallback,
) (*responses.Response, error) {
	if m.closed.Load() {
		return nil, UserErrorf("responses websocket model is closed")
	}

	timeouts := websocketRequestTimeoutsFromModelSettings(params.ModelSettings, m.httpModel.client)
	release, err := m.acquireRequestSlot(ctx, timeouts.Lock)
	if err != nil {
		return nil, err
	}
	defer release()

	requestFrame, wsURL, headers, err := m.prepareWebsocketRequest(ctx, params)
	if err != nil {
		return nil, err
	}

	conn, reused, err := m.ensureConnection(ctx, wsURL, headers, timeouts.Connect)
	if err != nil {
		return nil, err
	}

	requestDone := make(chan struct{})
	watchConnection := func(requestConn *websocket.Conn) {
		go func() {
			select {
			case <-ctx.Done():
				m.dropSpecificConnection(requestConn)
			case <-requestDone:
			}
		}()
	}
	watchConnection(conn)
	defer close(requestDone)

	sendRequestFrame := func(requestConn *websocket.Conn, reusedConnection bool) (*websocket.Conn, error) {
		writeErr := m.writeFrame(ctx, requestConn, requestFrame, timeouts.Send)
		if writeErr == nil {
			return requestConn, nil
		}
		m.dropSpecificConnection(requestConn)
		if reusedConnection && shouldRetryResponsesWebsocketPreSend(writeErr) && ctx.Err() == nil {
			retryConn, _, retryOpenErr := m.ensureConnection(ctx, wsURL, headers, timeouts.Connect)
			if retryOpenErr != nil {
				return nil, retryOpenErr
			}
			watchConnection(retryConn)
			if retryErr := m.writeFrame(ctx, retryConn, requestFrame, timeouts.Send); retryErr == nil {
				return retryConn, nil
			} else {
				m.dropSpecificConnection(retryConn)
				if ctxErr := ctx.Err(); ctxErr != nil {
					return nil, ctxErr
				}
				return nil, retryErr
			}
		}
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, writeErr
	}

	conn, err = sendRequestFrame(conn, reused)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, err
	}

	var sawTerminal bool
	for {
		event, payload, err := m.readEvent(ctx, conn, timeouts.Recv)
		if err != nil {
			m.dropSpecificConnection(conn)
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, ctxErr
			}
			return nil, err
		}

		switch event.Type {
		case "error", "response.error":
			m.dropSpecificConnection(conn)
			return nil, newResponsesWebSocketError(payload)
		}

		if yield != nil {
			if yieldErr := yield(ctx, event); yieldErr != nil {
				if !isTerminalResponsesEventType(event.Type) {
					m.dropSpecificConnection(conn)
				}
				return nil, yieldErr
			}
		}

		if !isTerminalResponsesEventType(event.Type) {
			continue
		}
		sawTerminal = true
		if isZeroResponse(event.Response) {
			return nil, fmt.Errorf(
				"Responses websocket stream ended without a terminal response payload. Terminal event: `%s`.",
				event.Type,
			)
		}
		return &event.Response, nil
	}

	if !sawTerminal {
		return nil, errors.New("Responses websocket stream ended without a terminal response payload")
	}
	return nil, nil
}

func (m *OpenAIResponsesWSModel) prepareWebsocketRequest(
	ctx context.Context,
	params ModelResponseParams,
) (map[string]any, string, map[string]string, error) {
	body, _, err := m.httpModel.prepareRequest(
		ctx,
		params.SystemInstructions,
		params.Input,
		params.ModelSettings,
		params.Tools,
		params.OutputType,
		params.Handoffs,
		params.PreviousResponseID,
		params.ConversationID,
		true,
		params.Prompt,
	)
	if err != nil {
		return nil, "", nil, err
	}

	raw, err := json.Marshal(body)
	if err != nil {
		return nil, "", nil, fmt.Errorf("marshal websocket response.create payload: %w", err)
	}

	var frame map[string]any
	if err := json.Unmarshal(raw, &frame); err != nil {
		return nil, "", nil, fmt.Errorf("decode websocket response.create payload: %w", err)
	}

	for key, value := range mergedModelExtraJSON(params.ModelSettings) {
		frame[key] = value
	}
	delete(frame, "timeout")
	frame["type"] = "response.create"
	frame["stream"] = true

	wsURL, err := m.prepareWebsocketURL(params.ModelSettings.ExtraQuery)
	if err != nil {
		return nil, "", nil, err
	}

	headers := m.prepareWebsocketHeaders(params.ModelSettings.ExtraHeaders)
	return frame, wsURL, headers, nil
}

func (m *OpenAIResponsesWSModel) prepareWebsocketURL(extraQuery map[string]string) (string, error) {
	base := strings.TrimSpace(m.websocketBaseURL)
	if base == "" {
		base = strings.TrimSpace(m.httpModel.client.WebsocketBaseURL.Or(""))
	}
	if base == "" {
		base = strings.TrimSpace(m.httpModel.client.BaseURL.Or(""))
	}
	if base == "" {
		base = defaultOpenAIResponsesWebsocketBaseURL
	}

	parsed, err := url.Parse(base)
	if err != nil {
		return "", fmt.Errorf("parse websocket base URL: %w", err)
	}

	switch parsed.Scheme {
	case "http":
		parsed.Scheme = "ws"
	case "https":
		parsed.Scheme = "wss"
	}

	values := parsed.Query()
	for key, value := range m.httpModel.client.DefaultQuery {
		values.Set(key, value)
	}
	for key, value := range extraQuery {
		values.Set(key, value)
	}

	path := strings.TrimRight(parsed.Path, "/")
	if path == "" {
		path = "/responses"
	} else {
		path += "/responses"
	}
	parsed.Path = path
	parsed.RawQuery = values.Encode()
	return parsed.String(), nil
}

func (m *OpenAIResponsesWSModel) prepareWebsocketHeaders(extraHeaders map[string]string) map[string]string {
	headers := make(map[string]string)
	if m.httpModel.client.APIKey.Valid() {
		headers["Authorization"] = "Bearer " + m.httpModel.client.APIKey.Value
	}
	headers["User-Agent"] = DefaultUserAgent()
	for key, value := range m.httpModel.client.DefaultHeaders {
		setCaseInsensitiveStringMap(headers, key, value)
	}
	for key, value := range extraHeaders {
		setCaseInsensitiveStringMap(headers, key, value)
	}
	if override := ResponsesHeadersOverride.Get(); len(override) > 0 {
		for key, value := range override {
			setCaseInsensitiveStringMap(headers, key, value)
		}
	}
	return headers
}

func (m *OpenAIResponsesWSModel) acquireRequestSlot(
	ctx context.Context,
	timeout responsesWebsocketTimeout,
) (func(), error) {
	release := sync.OnceFunc(func() {
		m.requestSlot <- struct{}{}
	})

	switch {
	case !timeout.Set:
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-m.requestSlot:
			return release, nil
		}
	case timeout.Duration < 0:
		return nil, fmt.Errorf(
			"Responses websocket request lock wait timed out after %g seconds",
			timeout.Duration.Seconds(),
		)
	case timeout.Duration == 0:
		select {
		case <-m.requestSlot:
			return release, nil
		default:
			return nil, fmt.Errorf("Responses websocket request lock wait timed out after 0 seconds")
		}
	default:
		timer := time.NewTimer(timeout.Duration)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-m.requestSlot:
			return release, nil
		case <-timer.C:
			return nil, fmt.Errorf(
				"Responses websocket request lock wait timed out after %g seconds",
				timeout.Duration.Seconds(),
			)
		}
	}
}

func (m *OpenAIResponsesWSModel) ensureConnection(
	ctx context.Context,
	wsURL string,
	headers map[string]string,
	timeout responsesWebsocketTimeout,
) (*websocket.Conn, bool, error) {
	identity := websocketConnectionIdentity(wsURL, headers)

	m.connMu.Lock()
	if m.closed.Load() {
		m.connMu.Unlock()
		return nil, false, UserErrorf("responses websocket model is closed")
	}
	if m.conn != nil && m.connIdentity == identity {
		if isReusableResponsesWebsocketConnection(m.conn) {
			conn := m.conn
			m.connMu.Unlock()
			return conn, true, nil
		}
	}
	oldConn := m.conn
	m.conn = nil
	m.connIdentity = ""
	m.connMu.Unlock()

	if oldConn != nil {
		_ = oldConn.Close()
	}

	conn, err := openResponsesWebsocketConnection(ctx, wsURL, headers, timeout)
	if err != nil {
		return nil, false, err
	}

	m.connMu.Lock()
	defer m.connMu.Unlock()
	if m.closed.Load() {
		_ = conn.Close()
		return nil, false, UserErrorf("responses websocket model is closed")
	}
	m.conn = conn
	m.connIdentity = identity
	return conn, false, nil
}

func isReusableResponsesWebsocketConnection(conn *websocket.Conn) bool {
	if conn == nil {
		return false
	}
	if responsesWebsocketBufferedReadBytes(conn) > 0 {
		return false
	}

	rawConn := conn.UnderlyingConn()
	if rawConn == nil {
		return false
	}
	if err := rawConn.SetReadDeadline(time.Now()); err != nil {
		return false
	}
	defer func() { _ = rawConn.SetReadDeadline(time.Time{}) }()

	var probe [1]byte
	n, err := rawConn.Read(probe[:])
	if n > 0 {
		return false
	}
	return isNetTimeout(err)
}

func responsesWebsocketBufferedReadBytes(conn *websocket.Conn) int {
	if conn == nil {
		return 0
	}

	connValue := reflect.ValueOf(conn)
	if connValue.Kind() != reflect.Pointer || connValue.IsNil() {
		return 0
	}
	connValue = connValue.Elem()

	readerValue := connValue.FieldByName("br")
	if !readerValue.IsValid() || readerValue.IsNil() {
		return 0
	}
	if !readerValue.CanAddr() {
		return 0
	}

	// gorilla/websocket does not expose buffered read state; inspect it to avoid
	// reusing a socket that already has close-frame bytes waiting.
	reader := *(**bufio.Reader)(unsafe.Pointer(readerValue.UnsafeAddr()))
	if reader == nil {
		return 0
	}
	return reader.Buffered()
}

func shouldRetryResponsesWebsocketPreSend(err error) bool {
	if err == nil {
		return false
	}
	if websocket.IsCloseError(
		err,
		websocket.CloseNormalClosure,
		websocket.CloseGoingAway,
		websocket.CloseAbnormalClosure,
		websocket.CloseNoStatusReceived,
	) {
		return true
	}
	if errors.Is(err, io.EOF) ||
		errors.Is(err, io.ErrUnexpectedEOF) ||
		errors.Is(err, net.ErrClosed) ||
		errors.Is(err, websocket.ErrCloseSent) ||
		errors.Is(err, syscall.EPIPE) ||
		errors.Is(err, syscall.ECONNRESET) {
		return true
	}

	message := strings.ToLower(err.Error())
	return strings.Contains(message, "broken pipe") ||
		strings.Contains(message, "unexpected eof") ||
		strings.Contains(message, "use of closed network connection")
}

func (m *OpenAIResponsesWSModel) writeFrame(
	ctx context.Context,
	conn *websocket.Conn,
	frame map[string]any,
	timeout responsesWebsocketTimeout,
) error {
	payload, err := json.Marshal(frame)
	if err != nil {
		return fmt.Errorf("marshal websocket request frame: %w", err)
	}

	if deadline, ok := websocketOperationDeadline(ctx, timeout); ok {
		if err := conn.SetWriteDeadline(deadline); err != nil {
			return err
		}
		defer func() { _ = conn.SetWriteDeadline(time.Time{}) }()
	}
	if err := conn.WriteMessage(websocket.TextMessage, payload); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		if timeout.Set && isNetTimeout(err) {
			return fmt.Errorf("Responses websocket send timed out after %g seconds", timeout.Duration.Seconds())
		}
		return err
	}
	return nil
}

func (m *OpenAIResponsesWSModel) readEvent(
	ctx context.Context,
	conn *websocket.Conn,
	timeout responsesWebsocketTimeout,
) (responses.ResponseStreamEventUnion, map[string]any, error) {
	if deadline, ok := websocketOperationDeadline(ctx, timeout); ok {
		if err := conn.SetReadDeadline(deadline); err != nil {
			return responses.ResponseStreamEventUnion{}, nil, err
		}
		defer func() { _ = conn.SetReadDeadline(time.Time{}) }()
	}

	_, payloadBytes, err := conn.ReadMessage()
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return responses.ResponseStreamEventUnion{}, nil, ctxErr
		}
		if timeout.Set && isNetTimeout(err) {
			return responses.ResponseStreamEventUnion{}, nil, fmt.Errorf(
				"Responses websocket receive timed out after %g seconds",
				timeout.Duration.Seconds(),
			)
		}
		return responses.ResponseStreamEventUnion{}, nil, err
	}

	var payload map[string]any
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		return responses.ResponseStreamEventUnion{}, nil, fmt.Errorf("decode websocket event payload: %w", err)
	}

	var event responses.ResponseStreamEventUnion
	if err := json.Unmarshal(payloadBytes, &event); err != nil {
		return responses.ResponseStreamEventUnion{}, payload, fmt.Errorf("decode websocket event: %w", err)
	}
	return event, payload, nil
}

func (m *OpenAIResponsesWSModel) WebsocketBaseURL() string {
	return m.websocketBaseURL
}

func (m *OpenAIResponsesWSModel) Close() error {
	m.closed.Store(true)
	m.connMu.Lock()
	conn := m.conn
	m.conn = nil
	m.connIdentity = ""
	m.connMu.Unlock()
	if conn != nil {
		return conn.Close()
	}
	return nil
}

func (m *OpenAIResponsesWSModel) dropSpecificConnection(conn *websocket.Conn) {
	if conn == nil {
		return
	}

	m.connMu.Lock()
	if m.conn == conn {
		m.conn = nil
		m.connIdentity = ""
	}
	m.connMu.Unlock()
	_ = conn.Close()
}

func openResponsesWebsocketConnection(
	ctx context.Context,
	wsURL string,
	headers map[string]string,
	timeout responsesWebsocketTimeout,
) (*websocket.Conn, error) {
	dialer := websocket.Dialer{}
	if timeout.Set && timeout.Duration > 0 {
		dialer.HandshakeTimeout = timeout.Duration
	}

	header := http.Header{}
	for key, value := range headers {
		header.Set(key, value)
	}

	dialCtx := ctx
	var cancel context.CancelFunc
	if timeout.Set {
		dialCtx, cancel = context.WithTimeout(ctx, timeout.Duration)
		defer cancel()
	}

	conn, _, err := dialer.DialContext(dialCtx, wsURL, header)
	if err != nil {
		if timeout.Set && (errors.Is(err, context.DeadlineExceeded) || isNetTimeout(err)) {
			return nil, fmt.Errorf("Responses websocket connect timed out after %g seconds", timeout.Duration.Seconds())
		}
		return nil, err
	}
	return conn, nil
}

func newResponsesWebSocketError(payload map[string]any) ResponsesWebSocketError {
	eventType, _ := payload["type"].(string)
	code, _ := payload["code"].(string)
	message, _ := payload["message"].(string)
	errorType, _ := payload["error_type"].(string)

	if nestedError, ok := payload["error"].(map[string]any); ok {
		if code == "" {
			code, _ = nestedError["code"].(string)
		}
		if message == "" {
			message, _ = nestedError["message"].(string)
		}
		if errorType == "" {
			errorType, _ = nestedError["type"].(string)
		}
	}
	if errorType == "" {
		errorType = code
	}

	messageParts := []string{"Responses websocket error"}
	if eventType != "" {
		messageParts = append(messageParts, eventType)
	}
	if code != "" {
		messageParts = append(messageParts, code)
	} else if errorType != "" {
		messageParts = append(messageParts, errorType)
	}
	messageText := strings.Join(messageParts, ": ")
	if message != "" {
		messageText += ": " + message
	}

	return ResponsesWebSocketError{
		AgentsError:  NewAgentsError(messageText),
		EventType:    eventType,
		ErrorType:    errorType,
		Code:         code,
		ErrorMessage: message,
		Payload:      payload,
	}
}

func websocketRequestTimeoutsFromModelSettings(
	settings modelsettings.ModelSettings,
	client OpenaiClient,
) responsesWebsocketRequestTimeouts {
	timeout := responsesWebsocketTimeout{}
	if settings.ExtraArgs != nil {
		if value, ok := settings.ExtraArgs["timeout"]; ok {
			if parsed, ok := parseResponsesWebsocketTimeout(value); ok {
				timeout = parsed
			}
		}
	}
	if !timeout.Set && client.RequestTimeout > 0 {
		timeout = responsesWebsocketTimeout{
			Duration: client.RequestTimeout,
			Set:      true,
		}
	}

	return responsesWebsocketRequestTimeouts{
		Lock:    timeout,
		Connect: timeout,
		Send:    timeout,
		Recv:    timeout,
	}
}

func parseResponsesWebsocketTimeout(value any) (responsesWebsocketTimeout, bool) {
	switch typed := value.(type) {
	case time.Duration:
		return responsesWebsocketTimeout{Duration: typed, Set: true}, true
	case float32:
		if math.IsNaN(float64(typed)) || math.IsInf(float64(typed), 0) {
			return responsesWebsocketTimeout{}, false
		}
		return responsesWebsocketTimeout{
			Duration: time.Duration(float64(typed) * float64(time.Second)),
			Set:      true,
		}, true
	case float64:
		if math.IsNaN(typed) || math.IsInf(typed, 0) {
			return responsesWebsocketTimeout{}, false
		}
		return responsesWebsocketTimeout{
			Duration: time.Duration(typed * float64(time.Second)),
			Set:      true,
		}, true
	case int:
		return responsesWebsocketTimeout{Duration: time.Duration(typed) * time.Second, Set: true}, true
	case int32:
		return responsesWebsocketTimeout{Duration: time.Duration(typed) * time.Second, Set: true}, true
	case int64:
		return responsesWebsocketTimeout{Duration: time.Duration(typed) * time.Second, Set: true}, true
	}
	return responsesWebsocketTimeout{}, false
}

func websocketOperationDeadline(ctx context.Context, timeout responsesWebsocketTimeout) (time.Time, bool) {
	var deadline time.Time
	hasDeadline := false

	if timeout.Set {
		deadline = time.Now().Add(timeout.Duration)
		hasDeadline = true
	}
	if ctxDeadline, ok := ctx.Deadline(); ok {
		if !hasDeadline || ctxDeadline.Before(deadline) {
			deadline = ctxDeadline
			hasDeadline = true
		}
	}
	return deadline, hasDeadline
}

func websocketConnectionIdentity(wsURL string, headers map[string]string) string {
	pairs := make([]string, 0, len(headers))
	for key, value := range headers {
		pairs = append(pairs, strings.ToLower(key)+"="+value)
	}
	sort.Strings(pairs)
	return wsURL + "\n" + strings.Join(pairs, "\n")
}

func setCaseInsensitiveStringMap(values map[string]string, key, value string) {
	lowerKey := strings.ToLower(key)
	for existingKey := range values {
		if strings.ToLower(existingKey) == lowerKey {
			delete(values, existingKey)
		}
	}
	values[key] = value
}

func isTerminalResponsesEventType(eventType string) bool {
	switch eventType {
	case "response.completed", "response.failed", "response.incomplete":
		return true
	default:
		return false
	}
}

func isZeroResponse(response responses.Response) bool {
	return response.ID == "" && len(response.Output) == 0
}

func isZeroResponseUsage(u responses.ResponseUsage) bool {
	return u.InputTokens == 0 &&
		u.OutputTokens == 0 &&
		u.TotalTokens == 0 &&
		u.InputTokensDetails.CachedTokens == 0 &&
		u.OutputTokensDetails.ReasoningTokens == 0
}

func isNetTimeout(err error) bool {
	var netErr net.Error
	return errors.As(err, &netErr) && netErr.Timeout()
}
