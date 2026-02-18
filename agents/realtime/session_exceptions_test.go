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
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type testGuardrailTask struct {
	done     bool
	canceled bool
}

func (t *testGuardrailTask) Cancel() { t.canceled = true }
func (t *testGuardrailTask) Done() bool {
	return t.done
}

func drainSessionEvents(t *testing.T, ch <-chan RealtimeSessionEvent) []RealtimeSessionEvent {
	t.Helper()

	events := make([]RealtimeSessionEvent, 0)
	timer := time.NewTimer(time.Second)
	defer timer.Stop()

	for {
		select {
		case ev, ok := <-ch:
			if !ok {
				return events
			}
			events = append(events, ev)
		case <-timer.C:
			t.Fatal("timeout waiting for realtime session events to close")
		}
	}
}

func findRealtimeErrorEvent(events []RealtimeSessionEvent) (RealtimeErrorEvent, bool) {
	for _, ev := range events {
		if errEvent, ok := ev.(RealtimeErrorEvent); ok {
			return errEvent, true
		}
	}
	return RealtimeErrorEvent{}, false
}

func TestRealtimeSessionExceptionClosesAndStoresError(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.Enter(t.Context()))
	<-session.Events()

	testErr := errors.New("test error")
	contextMessage := "Test context"
	require.NoError(t, model.listener.OnEvent(t.Context(), RealtimeModelExceptionEvent{
		Exception: testErr,
		Context:   &contextMessage,
	}))

	events := drainSessionEvents(t, session.Events())

	assert.True(t, session.closed)
	assert.Same(t, testErr, session.storedException)
	assert.Nil(t, model.listener)
	assert.Equal(t, 1, model.closeCount)

	errEvent, ok := findRealtimeErrorEvent(events)
	require.True(t, ok)
	errMap, ok := errEvent.Error.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, contextMessage, errMap["message"])
	assert.Same(t, testErr, errMap["exception"])
}

func TestRealtimeSessionExceptionContextPreserved(t *testing.T) {
	testCases := []struct {
		name    string
		context string
		err     error
	}{
		{name: "audio", context: "Failed to send audio", err: errors.New("Audio encoding failed")},
		{name: "websocket", context: "WebSocket error in message listener", err: errors.New("Network error")},
		{name: "event", context: "Failed to send event: response.create", err: errors.New("Socket closed")},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			model := &mockRealtimeModel{}
			session := NewRealtimeSession(
				model,
				&RealtimeAgent[any]{Name: "agent"},
				nil,
				RealtimeModelConfig{},
				RealtimeRunConfig{},
			)

			require.NoError(t, session.Enter(t.Context()))
			<-session.Events()

			require.NoError(t, model.listener.OnEvent(t.Context(), RealtimeModelExceptionEvent{
				Exception: testCase.err,
				Context:   &testCase.context,
			}))

			events := drainSessionEvents(t, session.Events())
			assert.Same(t, testCase.err, session.storedException)
			assert.True(t, session.closed)

			errEvent, ok := findRealtimeErrorEvent(events)
			require.True(t, ok)
			errMap, ok := errEvent.Error.(map[string]any)
			require.True(t, ok)
			assert.Equal(t, testCase.context, errMap["message"])
		})
	}
}

func TestRealtimeSessionJSONParsingErrorStored(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.Enter(t.Context()))
	<-session.Events()

	jsonErr := &json.SyntaxError{Offset: 3}
	contextMessage := "Failed to parse WebSocket message as JSON"
	require.NoError(t, model.listener.OnEvent(t.Context(), RealtimeModelExceptionEvent{
		Exception: jsonErr,
		Context:   &contextMessage,
	}))

	_ = drainSessionEvents(t, session.Events())
	assert.Same(t, jsonErr, session.storedException)
	assert.True(t, session.closed)
}

func TestRealtimeSessionMultipleExceptionsStoreFirst(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.Enter(t.Context()))
	<-session.Events()

	firstErr := errors.New("first error")
	secondErr := errors.New("second error")

	require.NoError(t, model.listener.OnEvent(t.Context(), RealtimeModelExceptionEvent{
		Exception: firstErr,
		Context:   nil,
	}))
	_ = session.OnEvent(context.Background(), RealtimeModelExceptionEvent{
		Exception: secondErr,
		Context:   nil,
	})

	_ = drainSessionEvents(t, session.Events())
	assert.Same(t, firstErr, session.storedException)
	assert.True(t, session.closed)
}

func TestRealtimeSessionExceptionCancelsGuardrailTasks(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	task1 := &testGuardrailTask{done: false}
	task2 := &testGuardrailTask{done: true}
	session.guardrailTasks = map[guardrailTask]struct{}{
		task1: {},
		task2: {},
	}

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelExceptionEvent{
		Exception: errors.New("Processing error"),
		Context:   nil,
	}))

	_ = drainSessionEvents(t, session.Events())
	assert.True(t, task1.canceled)
	assert.False(t, task2.canceled)
	assert.Len(t, session.guardrailTasks, 0)
}

func TestRealtimeSessionNormalEventsBeforeException(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.Enter(t.Context()))
	<-session.Events()

	require.NoError(t, model.listener.OnEvent(t.Context(), RealtimeModelErrorEvent{
		Error: map[string]any{"message": "Normal error"},
	}))
	require.NoError(t, model.listener.OnEvent(t.Context(), RealtimeModelExceptionEvent{
		Exception: errors.New("Fatal error"),
		Context:   nil,
	}))

	events := drainSessionEvents(t, session.Events())

	var errorEvents []RealtimeErrorEvent
	for _, ev := range events {
		if errEvent, ok := ev.(RealtimeErrorEvent); ok {
			errorEvents = append(errorEvents, errEvent)
		}
	}
	require.NotEmpty(t, errorEvents)
}
