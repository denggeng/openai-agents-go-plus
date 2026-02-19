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
	"bytes"
	"context"
	"io"
	"sync/atomic"
	"testing"
	"time"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockRealtimeModel struct {
	listener     RealtimeModelListener
	connectCount int
	closeCount   int
	connectCfg   RealtimeModelConfig
	sentEvents   []RealtimeModelSendEvent
}

func (m *mockRealtimeModel) Connect(_ context.Context, cfg RealtimeModelConfig) error {
	m.connectCount++
	m.connectCfg = cfg
	return nil
}

func (m *mockRealtimeModel) AddListener(listener RealtimeModelListener) {
	m.listener = listener
}

func (m *mockRealtimeModel) RemoveListener(listener RealtimeModelListener) {
	if m.listener == listener {
		m.listener = nil
	}
}

func (m *mockRealtimeModel) SendEvent(_ context.Context, event RealtimeModelSendEvent) error {
	m.sentEvents = append(m.sentEvents, event)
	return nil
}

func (m *mockRealtimeModel) Close(context.Context) error {
	m.closeCount++
	return nil
}

func TestRealtimeSessionEnterConnectsAndEmitsInitialHistory(t *testing.T) {
	model := &mockRealtimeModel{}
	agent := &RealtimeAgent[any]{
		Name:         "agent",
		Instructions: "hello",
	}

	session := NewRealtimeSession(
		model,
		agent,
		nil,
		RealtimeModelConfig{
			InitialSettings: RealtimeSessionModelSettings{"model_name": "gpt-realtime"},
		},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.Enter(t.Context()))
	assert.Equal(t, 1, model.connectCount)
	require.NotNil(t, model.listener)

	event := <-session.Events()
	historyUpdated, ok := event.(RealtimeHistoryUpdatedEvent)
	require.True(t, ok)
	assert.Len(t, historyUpdated.History, 0)
}

func TestRealtimeSessionModelReturnsUnderlyingModel(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	assert.Equal(t, model, session.Model())
}

func TestRealtimeSessionSendHelpers(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.Enter(t.Context()))
	<-session.Events() // initial history event

	require.NoError(t, session.SendMessage(t.Context(), "hi"))
	require.NoError(t, session.SendAudio(t.Context(), []byte("abc"), true))
	require.NoError(t, session.Interrupt(t.Context()))

	hasUser := false
	hasAudio := false
	hasInterrupt := false
	for _, event := range model.sentEvents {
		switch typed := event.(type) {
		case RealtimeModelSendUserInput:
			hasUser = true
		case RealtimeModelSendAudio:
			if typed.Commit {
				hasAudio = true
			}
		case RealtimeModelSendInterrupt:
			hasInterrupt = true
		}
	}
	assert.True(t, hasUser)
	assert.True(t, hasAudio)
	assert.True(t, hasInterrupt)
}

func TestRealtimeSessionEventsStopOnClose(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	done := make(chan struct{})
	go func() {
		for range session.Events() {
		}
		close(done)
	}()

	require.NoError(t, session.Close(t.Context()))

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("event loop did not stop after close")
	}
}

func TestRealtimeSessionItemUpdatedLogsOnInvalidContentPart(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	session.history = []any{
		map[string]any{
			"type":    "message",
			"role":    "assistant",
			"item_id": "a1",
			"content": []any{
				map[string]any{"type": "audio", "transcript": "t"},
			},
		},
	}

	var buf bytes.Buffer
	sessionLogger.SetOutput(&buf)
	sessionLogger.SetFlags(0)
	t.Cleanup(func() {
		sessionLogger.SetOutput(io.Discard)
	})

	incoming := map[string]any{
		"type":    "message",
		"role":    "assistant",
		"item_id": "a1",
		"content": []any{struct{ Bad string }{Bad: "oops"}},
	}

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{Item: incoming}))
	assert.NotEmpty(t, buf.String())
}

func TestRealtimeSessionHandoffWithoutTargetEmitsError(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	handoff := agents.Handoff{
		ToolName:        "switch",
		ToolDescription: "",
		InputJSONSchema: map[string]any{},
		OnInvokeHandoff: func(context.Context, string) (*agents.Agent, error) {
			return nil, nil
		},
		AgentName: "",
		IsEnabled: agents.HandoffEnabled(),
	}

	session.handleHandoffCall(
		t.Context(),
		RealtimeModelToolCallEvent{Name: "switch", CallID: "c1", Arguments: "{}"},
		&RealtimeAgent[any]{Name: "agent"},
		handoff,
		nil,
	)

	event := <-session.Events()
	errorEvent, ok := event.(RealtimeErrorEvent)
	require.True(t, ok)
	errMap, ok := errorEvent.Error.(map[string]any)
	require.True(t, ok)
	assert.Contains(t, errMap["message"], "returned no target agent")
}

func TestRealtimeSessionGuardrailErrorEmitsErrorEvent(t *testing.T) {
	model := &mockRealtimeModel{}
	guardrail := agents.OutputGuardrail{
		Name: "boom",
		GuardrailFunction: func(context.Context, *agents.Agent, any) (agents.GuardrailFunctionOutput, error) {
			return agents.GuardrailFunctionOutput{}, assert.AnError
		},
	}

	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent", OutputGuardrails: []agents.OutputGuardrail{guardrail}},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	session.runOutputGuardrails(t.Context(), "message", "")

	event := <-session.Events()
	errorEvent, ok := event.(RealtimeErrorEvent)
	require.True(t, ok)
	errMap, ok := errorEvent.Error.(map[string]any)
	require.True(t, ok)
	assert.Contains(t, errMap["message"], "output guardrail")
}

func TestRealtimeSessionEnterConnectsWithToolsAndHandoffs(t *testing.T) {
	model := &mockRealtimeModel{}
	tool := agents.FunctionTool{
		Name:             "tool_one",
		Description:      "Tool one",
		ParamsJSONSchema: map[string]any{"type": "object"},
		OnInvokeTool: func(_ context.Context, _ string) (any, error) {
			return "ok", nil
		},
	}
	childAgent := &RealtimeAgent[any]{Name: "one"}
	parentAgent := &RealtimeAgent[any]{
		Name:         "two",
		Instructions: "instr_two",
		Tools:        []agents.Tool{tool},
		Handoffs:     []any{childAgent},
	}

	session := NewRealtimeSession(
		model,
		parentAgent,
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.Enter(t.Context()))
	<-session.Events() // initial history event

	settings := model.connectCfg.InitialSettings
	require.NotNil(t, settings)
	assert.Equal(t, "instr_two", settings["instructions"])

	tools, ok := settings["tools"].([]agents.Tool)
	require.True(t, ok)
	require.Len(t, tools, 1)
	assert.Equal(t, "tool_one", tools[0].ToolName())

	handoffs, ok := settings["handoffs"].([]agents.Handoff)
	require.True(t, ok)
	require.Len(t, handoffs, 1)
	assert.Equal(t, "transfer_to_one", handoffs[0].ToolName)
	assert.Equal(t, "one", handoffs[0].AgentName)
}

func TestRealtimeSessionSendMethodsForwardEvents(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)
	require.NoError(t, session.Enter(t.Context()))
	<-session.Events() // initial history event

	require.NoError(t, session.SendMessage(t.Context(), "hello"))
	require.NoError(t, session.SendAudio(t.Context(), []byte("abc"), true))
	require.NoError(t, session.Interrupt(t.Context()))

	require.Len(t, model.sentEvents, 3)
	assert.Equal(t, realtimeSendEventTypeUserInput, model.sentEvents[0].Type())
	assert.Equal(t, realtimeSendEventTypeAudio, model.sentEvents[1].Type())
	assert.Equal(t, realtimeSendEventTypeInterrupt, model.sentEvents[2].Type())
}

func TestRealtimeSessionErrorEventTransforms(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	errPayload := map[string]any{"message": "boom"}
	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelErrorEvent{Error: errPayload}))

	raw, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	assert.Equal(t, errPayload, raw.Data.(RealtimeModelErrorEvent).Error)

	realtimeErr, ok := (<-session.Events()).(RealtimeErrorEvent)
	require.True(t, ok)
	assert.Equal(t, errPayload, realtimeErr.Error)
}

func TestRealtimeSessionAudioEventsTransform(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	audioEvent := RealtimeModelAudioEvent{
		Data:         []byte("audio"),
		ResponseID:   "resp_1",
		ItemID:       "item_1",
		ContentIndex: 0,
	}
	require.NoError(t, session.OnEvent(t.Context(), audioEvent))
	_, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	audioSessionEvent, ok := (<-session.Events()).(RealtimeAudioEvent)
	require.True(t, ok)
	assert.Equal(t, audioEvent, audioSessionEvent.Audio)
	assert.Equal(t, "item_1", audioSessionEvent.ItemID)

	interrupted := RealtimeModelAudioInterruptedEvent{ItemID: "item_1", ContentIndex: 0}
	require.NoError(t, session.OnEvent(t.Context(), interrupted))
	_, ok = (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	_, ok = (<-session.Events()).(RealtimeAudioInterruptedEvent)
	require.True(t, ok)

	done := RealtimeModelAudioDoneEvent{ItemID: "item_1", ContentIndex: 0}
	require.NoError(t, session.OnEvent(t.Context(), done))
	_, ok = (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	_, ok = (<-session.Events()).(RealtimeAudioEndEvent)
	require.True(t, ok)
}

func TestRealtimeSessionTurnEventsTransform(t *testing.T) {
	model := &mockRealtimeModel{}
	agent := &RealtimeAgent[any]{Name: "agent"}
	session := NewRealtimeSession(
		model,
		agent,
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTurnStartedEvent{}))
	_, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	startEvent, ok := (<-session.Events()).(RealtimeAgentStartEvent)
	require.True(t, ok)
	assert.Equal(t, agent, startEvent.Agent)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTurnEndedEvent{}))
	_, ok = (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	endEvent, ok := (<-session.Events()).(RealtimeAgentEndEvent)
	require.True(t, ok)
	assert.Equal(t, agent, endEvent.Agent)
}

func TestRealtimeSessionIgnoredEventsOnlyEmitRaw(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTranscriptDeltaEvent{
		ItemID:     "item_1",
		Delta:      "hi",
		ResponseID: "resp_1",
	}))
	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelConnectionStatusEvent{
		Status: RealtimeConnectionStatusConnected,
	}))
	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelOtherEvent{
		Data: map[string]any{"custom": "data"},
	}))

	require.Len(t, session.eventQueue, 3)
	for i := 0; i < 3; i++ {
		_, ok := (<-session.Events()).(RealtimeRawModelEvent)
		require.True(t, ok)
	}
}

func TestRealtimeSessionOnEventHistoryUpdateAndDelete(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)
	require.NoError(t, session.Enter(t.Context()))
	<-session.Events() // initial history event

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{"item_id": "a", "type": "message"},
	}))
	raw := <-session.Events()
	_, ok := raw.(RealtimeRawModelEvent)
	require.True(t, ok)
	added := <-session.Events()
	_, ok = added.(RealtimeHistoryAddedEvent)
	require.True(t, ok)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{"item_id": "a", "type": "message", "status": "completed"},
	}))
	<-session.Events() // raw
	updated := <-session.Events()
	_, ok = updated.(RealtimeHistoryUpdatedEvent)
	require.True(t, ok)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemDeletedEvent{ItemID: "a"}))
	<-session.Events() // raw
	deletedUpdate := <-session.Events()
	historyUpdated, ok := deletedUpdate.(RealtimeHistoryUpdatedEvent)
	require.True(t, ok)
	assert.Len(t, historyUpdated.History, 0)
}

func TestRealtimeSessionOnEventHistoryUpdateInsertsAfterPreviousItemID(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)
	require.NoError(t, session.Enter(t.Context()))
	<-session.Events() // initial history event

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{"item_id": "item_1", "type": "message"},
	}))
	<-session.Events() // raw
	<-session.Events() // history_added

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{"item_id": "item_3", "type": "message"},
	}))
	<-session.Events() // raw
	<-session.Events() // history_added

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{
			"item_id":          "item_2",
			"type":             "message",
			"previous_item_id": "item_1",
		},
	}))
	<-session.Events() // raw
	<-session.Events() // history_added

	history := session.History()
	require.Len(t, history, 3)
	assert.Equal(t, "item_1", itemIDFromAny(history[0]))
	assert.Equal(t, "item_2", itemIDFromAny(history[1]))
	assert.Equal(t, "item_3", itemIDFromAny(history[2]))
}

func TestRealtimeSessionOnEventHistoryUpdateUnknownPreviousItemIDAppends(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)
	require.NoError(t, session.Enter(t.Context()))
	<-session.Events() // initial history event

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{"item_id": "item_1", "type": "message"},
	}))
	<-session.Events() // raw
	<-session.Events() // history_added

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{
			"item_id":          "item_2",
			"type":             "message",
			"previous_item_id": "missing",
		},
	}))
	<-session.Events() // raw
	<-session.Events() // history_added

	history := session.History()
	require.Len(t, history, 2)
	assert.Equal(t, "item_1", itemIDFromAny(history[0]))
	assert.Equal(t, "item_2", itemIDFromAny(history[1]))
}

func TestRealtimeSessionAssistantTranscriptPreservedOnItemUpdate(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{
			"item_id": "assist_1",
			"type":    "message",
			"role":    "assistant",
			"content": []map[string]any{
				{
					"type":       "audio",
					"transcript": "Hello there",
				},
			},
		},
	}))
	<-session.Events() // raw
	_, ok := (<-session.Events()).(RealtimeHistoryAddedEvent)
	require.True(t, ok)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{
			"item_id": "assist_1",
			"type":    "message",
			"role":    "assistant",
			"content": []map[string]any{
				{
					"type": "audio",
				},
			},
		},
	}))
	<-session.Events() // raw
	_, ok = (<-session.Events()).(RealtimeHistoryUpdatedEvent)
	require.True(t, ok)

	history := session.History()
	require.Len(t, history, 1)
	item, ok := history[0].(map[string]any)
	require.True(t, ok)
	content := extractContentParts(item["content"])
	require.Len(t, content, 1)
	part, ok := toStringAnyMap(content[0])
	require.True(t, ok)
	assert.Equal(t, "Hello there", part["transcript"])
}

func TestRealtimeSessionAssistantTranscriptFallbackToDeltaCacheOnItemUpdate(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	session.history = []any{
		map[string]any{
			"item_id": "assist_2",
			"type":    "message",
			"role":    "assistant",
			"content": []map[string]any{
				{
					"type": "audio",
				},
			},
		},
	}
	session.itemTranscripts["assist_2"] = "partial transcript"

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{
			"item_id": "assist_2",
			"type":    "message",
			"role":    "assistant",
			"content": []map[string]any{
				{
					"type": "audio",
				},
			},
		},
	}))
	<-session.Events() // raw
	_, ok := (<-session.Events()).(RealtimeHistoryUpdatedEvent)
	require.True(t, ok)

	history := session.History()
	require.Len(t, history, 1)
	item, ok := history[0].(map[string]any)
	require.True(t, ok)
	content := extractContentParts(item["content"])
	require.Len(t, content, 1)
	part, ok := toStringAnyMap(content[0])
	require.True(t, ok)
	assert.Equal(t, "partial transcript", part["transcript"])
}

func TestRealtimeSessionUserInputAudioTranscriptPreservedOnItemUpdate(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{
			"item_id": "user_1",
			"type":    "message",
			"role":    "user",
			"content": []map[string]any{
				{
					"type":       "input_audio",
					"transcript": "hello user",
				},
			},
		},
	}))
	<-session.Events() // raw
	_, ok := (<-session.Events()).(RealtimeHistoryAddedEvent)
	require.True(t, ok)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{
			"item_id": "user_1",
			"type":    "message",
			"role":    "user",
			"content": []map[string]any{
				{
					"type": "input_audio",
				},
			},
		},
	}))
	<-session.Events() // raw
	_, ok = (<-session.Events()).(RealtimeHistoryUpdatedEvent)
	require.True(t, ok)

	history := session.History()
	require.Len(t, history, 1)
	item, ok := history[0].(map[string]any)
	require.True(t, ok)
	content := extractContentParts(item["content"])
	require.Len(t, content, 1)
	part, ok := toStringAnyMap(content[0])
	require.True(t, ok)
	assert.Equal(t, "hello user", part["transcript"])
}

func TestRealtimeSessionAssistantTextPreservedOnItemUpdate(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{
			"item_id": "assist_text_1",
			"type":    "message",
			"role":    "assistant",
			"content": []map[string]any{
				{
					"type": "text",
					"text": "original assistant text",
				},
			},
		},
	}))
	<-session.Events() // raw
	_, ok := (<-session.Events()).(RealtimeHistoryAddedEvent)
	require.True(t, ok)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{
			"item_id": "assist_text_1",
			"type":    "message",
			"role":    "assistant",
			"content": []map[string]any{
				{
					"type": "text",
				},
			},
		},
	}))
	<-session.Events() // raw
	_, ok = (<-session.Events()).(RealtimeHistoryUpdatedEvent)
	require.True(t, ok)

	history := session.History()
	require.Len(t, history, 1)
	item, ok := history[0].(map[string]any)
	require.True(t, ok)
	content := extractContentParts(item["content"])
	require.Len(t, content, 1)
	part, ok := toStringAnyMap(content[0])
	require.True(t, ok)
	assert.Equal(t, "original assistant text", part["text"])
}

func TestRealtimeSessionUserItemUpdatePreservesMissingImageAndText(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{
			"item_id": "user_merged_1",
			"type":    "message",
			"role":    "user",
			"content": []map[string]any{
				{
					"type":      "input_image",
					"image_url": "https://example.com/a.png",
				},
				{
					"type": "input_text",
					"text": "hello user",
				},
			},
		},
	}))
	<-session.Events() // raw
	_, ok := (<-session.Events()).(RealtimeHistoryAddedEvent)
	require.True(t, ok)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{
			"item_id": "user_merged_1",
			"type":    "message",
			"role":    "user",
			"content": []map[string]any{
				{
					"type": "input_text",
				},
			},
		},
	}))
	<-session.Events() // raw
	_, ok = (<-session.Events()).(RealtimeHistoryUpdatedEvent)
	require.True(t, ok)

	history := session.History()
	require.Len(t, history, 1)
	item, ok := history[0].(map[string]any)
	require.True(t, ok)
	content := extractContentParts(item["content"])
	require.Len(t, content, 2)

	first, ok := toStringAnyMap(content[0])
	require.True(t, ok)
	assert.Equal(t, "input_image", first["type"])
	assert.Equal(t, "https://example.com/a.png", first["image_url"])

	second, ok := toStringAnyMap(content[1])
	require.True(t, ok)
	assert.Equal(t, "input_text", second["type"])
	assert.Equal(t, "hello user", second["text"])
}

func TestRealtimeSessionSystemTextPreservedOnItemUpdate(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{
			"item_id": "system_1",
			"type":    "message",
			"role":    "system",
			"content": []map[string]any{
				{
					"type": "input_text",
					"text": "system text",
				},
			},
		},
	}))
	<-session.Events() // raw
	_, ok := (<-session.Events()).(RealtimeHistoryAddedEvent)
	require.True(t, ok)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: map[string]any{
			"item_id": "system_1",
			"type":    "message",
			"role":    "system",
			"content": []map[string]any{
				{
					"type": "input_text",
				},
			},
		},
	}))
	<-session.Events() // raw
	_, ok = (<-session.Events()).(RealtimeHistoryUpdatedEvent)
	require.True(t, ok)

	history := session.History()
	require.Len(t, history, 1)
	item, ok := history[0].(map[string]any)
	require.True(t, ok)
	content := extractContentParts(item["content"])
	require.Len(t, content, 1)
	part, ok := toStringAnyMap(content[0])
	require.True(t, ok)
	assert.Equal(t, "system text", part["text"])
}

func TestRealtimeSessionTranscriptDeltaUpdatesStructuredAssistantHistoryItem(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	status := "in_progress"
	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelItemUpdatedEvent{
		Item: RealtimeMessageItem{
			ItemID: "assist_struct",
			Type:   "message",
			Role:   "assistant",
			Status: &status,
			Content: []RealtimeMessageContent{
				{Type: "audio"},
			},
		},
	}))
	<-session.Events() // raw
	_, ok := (<-session.Events()).(RealtimeHistoryAddedEvent)
	require.True(t, ok)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTranscriptDeltaEvent{
		ItemID:     "assist_struct",
		Delta:      "abc",
		ResponseID: "resp_struct",
	}))
	<-session.Events() // raw

	history := session.History()
	require.Len(t, history, 1)
	item, ok := history[0].(map[string]any)
	require.True(t, ok)
	content := extractContentParts(item["content"])
	require.Len(t, content, 1)
	part, ok := toStringAnyMap(content[0])
	require.True(t, ok)
	assert.Equal(t, "abc", part["transcript"])
}

func TestRealtimeSessionUpdateAgentSendsSessionUpdate(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent-a"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)
	require.NoError(t, session.Enter(t.Context()))
	<-session.Events() // initial history

	newAgent := &RealtimeAgent[any]{Name: "agent-b", Instructions: "sys-b"}
	require.NoError(t, session.UpdateAgent(t.Context(), newAgent))
	require.NotEmpty(t, model.sentEvents)
	assert.Equal(
		t,
		realtimeSendEventTypeSessionUpdate,
		model.sentEvents[len(model.sentEvents)-1].Type(),
	)
}

func TestRealtimeSessionClose(t *testing.T) {
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
	require.NoError(t, session.Close(t.Context()))
	assert.Equal(t, 1, model.closeCount)
	assert.Nil(t, model.listener)
}

func TestRealtimeSessionFunctionCallRunsToolAndSendsOutput(t *testing.T) {
	model := &mockRealtimeModel{}
	tool := agents.FunctionTool{
		Name:             "echo",
		Description:      "Echo tool",
		ParamsJSONSchema: map[string]any{"type": "object"},
		OnInvokeTool: func(_ context.Context, arguments string) (any, error) {
			return "echo:" + arguments, nil
		},
	}
	agent := &RealtimeAgent[any]{
		Name:  "agent",
		Tools: []agents.Tool{tool},
	}
	session := NewRealtimeSession(
		model,
		agent,
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      "echo",
		CallID:    "call_1",
		Arguments: `{"value":"x"}`,
	}))

	rawEvent := <-session.Events()
	_, ok := rawEvent.(RealtimeRawModelEvent)
	require.True(t, ok)

	toolStart := <-session.Events()
	startEvent, ok := toolStart.(RealtimeToolStartEvent)
	require.True(t, ok)
	assert.Equal(t, "echo", startEvent.Tool.ToolName())

	toolEnd := <-session.Events()
	endEvent, ok := toolEnd.(RealtimeToolEndEvent)
	require.True(t, ok)
	assert.Equal(t, "echo:{\"value\":\"x\"}", endEvent.Output)

	require.Len(t, model.sentEvents, 1)
	outputEvent, ok := model.sentEvents[0].(RealtimeModelSendToolOutput)
	require.True(t, ok)
	assert.Equal(t, "call_1", outputEvent.ToolCall.CallID)
	assert.Equal(t, "echo:{\"value\":\"x\"}", outputEvent.Output)
	assert.True(t, outputEvent.StartResponse)
}

func TestRealtimeSessionFunctionCallRunsAsyncByDefault(t *testing.T) {
	model := &mockRealtimeModel{}
	tool := agents.FunctionTool{
		Name:             "echo",
		Description:      "Echo tool",
		ParamsJSONSchema: map[string]any{"type": "object"},
		OnInvokeTool: func(_ context.Context, arguments string) (any, error) {
			return "echo:" + arguments, nil
		},
	}
	agent := &RealtimeAgent[any]{
		Name:  "agent",
		Tools: []agents.Tool{tool},
	}
	session := NewRealtimeSession(
		model,
		agent,
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{}, // async_tool_calls defaults to true
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      "echo",
		CallID:    "call_async_1",
		Arguments: `{"value":"y"}`,
	}))

	started := false
	ended := false
	timeout := time.After(time.Second)
	for !(started && ended) {
		select {
		case event := <-session.Events():
			switch event.(type) {
			case RealtimeToolStartEvent:
				started = true
			case RealtimeToolEndEvent:
				ended = true
			}
		case <-timeout:
			t.Fatalf("timed out waiting for async tool events (start=%v end=%v)", started, ended)
		}
	}

	require.Len(t, model.sentEvents, 1)
	outputEvent, ok := model.sentEvents[0].(RealtimeModelSendToolOutput)
	require.True(t, ok)
	assert.Equal(t, "call_async_1", outputEvent.ToolCall.CallID)
	assert.Equal(t, "echo:{\"value\":\"y\"}", outputEvent.Output)
}

func TestRealtimeSessionFunctionCallConvertsNonStringOutputForModel(t *testing.T) {
	type toolResult struct {
		Count int
		Label string
	}

	model := &mockRealtimeModel{}
	expected := toolResult{Count: 1, Label: "ok"}
	tool := agents.FunctionTool{
		Name:             "struct_tool",
		Description:      "Returns struct",
		ParamsJSONSchema: map[string]any{"type": "object"},
		OnInvokeTool: func(_ context.Context, arguments string) (any, error) {
			return expected, nil
		},
	}
	agent := &RealtimeAgent[any]{
		Name:  "agent",
		Tools: []agents.Tool{tool},
	}
	session := NewRealtimeSession(
		model,
		agent,
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      "struct_tool",
		CallID:    "call_struct_1",
		Arguments: `{"value":"x"}`,
	}))

	_, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	_, ok = (<-session.Events()).(RealtimeToolStartEvent)
	require.True(t, ok)
	toolEnd, ok := (<-session.Events()).(RealtimeToolEndEvent)
	require.True(t, ok)
	assert.Equal(t, expected, toolEnd.Output)

	require.Len(t, model.sentEvents, 1)
	outputEvent, ok := model.sentEvents[0].(RealtimeModelSendToolOutput)
	require.True(t, ok)
	assert.Equal(t, "{1 ok}", outputEvent.Output)
}

func TestRealtimeSessionFunctionCallUnknownToolEmitsError(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      "unknown_tool",
		CallID:    "call_2",
		Arguments: `{}`,
	}))

	_, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)

	errorEvent, ok := (<-session.Events()).(RealtimeErrorEvent)
	require.True(t, ok)
	errorMap, ok := errorEvent.Error.(map[string]any)
	require.True(t, ok)
	assert.Contains(t, errorMap["message"], "unknown_tool")
	assert.Empty(t, model.sentEvents)
}

func TestRealtimeSessionFunctionCallToolPanicEmitsError(t *testing.T) {
	model := &mockRealtimeModel{}
	tool := agents.FunctionTool{
		Name:             "panic_tool",
		Description:      "Panics",
		ParamsJSONSchema: map[string]any{"type": "object"},
		OnInvokeTool: func(_ context.Context, _ string) (any, error) {
			panic("boom")
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent", Tools: []agents.Tool{tool}},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      "panic_tool",
		CallID:    "panic_call_1",
		Arguments: `{}`,
	}))

	_, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	_, ok = (<-session.Events()).(RealtimeToolStartEvent)
	require.True(t, ok)
	errorEvent, ok := (<-session.Events()).(RealtimeErrorEvent)
	require.True(t, ok)
	errorMap, ok := errorEvent.Error.(map[string]any)
	require.True(t, ok)
	assert.Contains(t, errorMap["message"], "panic while running tool panic_tool")
	assert.Empty(t, model.sentEvents)
}

func TestRealtimeSessionFunctionCallHandoffSwitchesAgent(t *testing.T) {
	model := &mockRealtimeModel{}
	child := &RealtimeAgent[any]{
		Name:         "child",
		Instructions: "child-instructions",
	}
	parent := &RealtimeAgent[any]{
		Name:     "parent",
		Handoffs: []any{child},
	}
	session := NewRealtimeSession(
		model,
		parent,
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	handoffTool := RealtimeHandoff(child).ToolName
	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      handoffTool,
		CallID:    "call_3",
		Arguments: `{}`,
	}))

	_, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)

	handoffEvent, ok := (<-session.Events()).(RealtimeHandoffEvent)
	require.True(t, ok)
	assert.Equal(t, "parent", handoffEvent.FromAgent.(*RealtimeAgent[any]).Name)
	assert.Equal(t, "child", handoffEvent.ToAgent.(*RealtimeAgent[any]).Name)
	assert.Equal(t, "child", session.currentAgent.Name)

	require.Len(t, model.sentEvents, 2)
	sessionUpdateEvent, ok := model.sentEvents[0].(RealtimeModelSendSessionUpdate)
	require.True(t, ok)
	assert.Equal(t, "child-instructions", sessionUpdateEvent.SessionSettings["instructions"])

	toolOutputEvent, ok := model.sentEvents[1].(RealtimeModelSendToolOutput)
	require.True(t, ok)
	assert.Equal(t, "call_3", toolOutputEvent.ToolCall.CallID)
	assert.Contains(t, toolOutputEvent.Output, "\"assistant\":\"child\"")
}

func TestRealtimeSessionFunctionCallHandoffPanicEmitsError(t *testing.T) {
	model := &mockRealtimeModel{}
	panicHandoff := agents.Handoff{
		ToolName:        "handoff_panic",
		ToolDescription: "panic handoff",
		OnInvokeHandoff: func(context.Context, string) (*agents.Agent, error) {
			panic("handoff boom")
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{
			Name: "agent",
			Handoffs: []any{
				panicHandoff,
			},
		},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      "handoff_panic",
		CallID:    "handoff_panic_1",
		Arguments: `{}`,
	}))

	_, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	errorEvent, ok := (<-session.Events()).(RealtimeErrorEvent)
	require.True(t, ok)
	errorMap, ok := errorEvent.Error.(map[string]any)
	require.True(t, ok)
	assert.Contains(t, errorMap["message"], "panic while handling handoff handoff_panic")
	assert.Empty(t, model.sentEvents)
}

func TestRealtimeSessionFunctionCallRequiresApprovalAndApproveResumes(t *testing.T) {
	model := &mockRealtimeModel{}
	tool := agents.FunctionTool{
		Name:             "secure_tool",
		Description:      "Tool that needs approval",
		ParamsJSONSchema: map[string]any{"type": "object"},
		NeedsApproval:    agents.FunctionToolNeedsApprovalEnabled(),
		OnInvokeTool: func(_ context.Context, arguments string) (any, error) {
			return "approved:" + arguments, nil
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent", Tools: []agents.Tool{tool}},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      "secure_tool",
		CallID:    "approval_call_1",
		Arguments: `{"value":"x"}`,
	}))

	_, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	approvalEvent, ok := (<-session.Events()).(RealtimeToolApprovalRequiredEvent)
	require.True(t, ok)
	assert.Equal(t, "approval_call_1", approvalEvent.CallID)
	assert.Empty(t, model.sentEvents)

	require.NoError(t, session.ApproveToolCall(t.Context(), "approval_call_1", false))

	toolStart, ok := (<-session.Events()).(RealtimeToolStartEvent)
	require.True(t, ok)
	assert.Equal(t, "secure_tool", toolStart.Tool.ToolName())

	toolEnd, ok := (<-session.Events()).(RealtimeToolEndEvent)
	require.True(t, ok)
	assert.Equal(t, "approved:{\"value\":\"x\"}", toolEnd.Output)

	require.Len(t, model.sentEvents, 1)
	outputEvent, ok := model.sentEvents[0].(RealtimeModelSendToolOutput)
	require.True(t, ok)
	assert.Equal(t, "approval_call_1", outputEvent.ToolCall.CallID)
}

func TestRealtimeSessionFunctionCallRequiresApprovalAndRejectSendsRejection(t *testing.T) {
	model := &mockRealtimeModel{}
	tool := agents.FunctionTool{
		Name:             "secure_tool",
		Description:      "Tool that needs approval",
		ParamsJSONSchema: map[string]any{"type": "object"},
		NeedsApproval:    agents.FunctionToolNeedsApprovalEnabled(),
		OnInvokeTool: func(_ context.Context, arguments string) (any, error) {
			return "unexpected:" + arguments, nil
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent", Tools: []agents.Tool{tool}},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      "secure_tool",
		CallID:    "approval_call_2",
		Arguments: `{"value":"x"}`,
	}))
	<-session.Events() // raw
	_, ok := (<-session.Events()).(RealtimeToolApprovalRequiredEvent)
	require.True(t, ok)

	require.NoError(t, session.RejectToolCall(t.Context(), "approval_call_2", false))
	toolEnd, ok := (<-session.Events()).(RealtimeToolEndEvent)
	require.True(t, ok)
	assert.Equal(t, defaultRealtimeApprovalRejectionMessage, toolEnd.Output)

	require.Len(t, model.sentEvents, 1)
	outputEvent, ok := model.sentEvents[0].(RealtimeModelSendToolOutput)
	require.True(t, ok)
	assert.Equal(t, defaultRealtimeApprovalRejectionMessage, outputEvent.Output)
}

func TestRealtimeSessionFunctionCallRequiresApprovalAndRejectUsesRunLevelFormatter(t *testing.T) {
	model := &mockRealtimeModel{}
	tool := agents.FunctionTool{
		Name:             "secure_tool",
		Description:      "Tool that needs approval",
		ParamsJSONSchema: map[string]any{"type": "object"},
		NeedsApproval:    agents.FunctionToolNeedsApprovalEnabled(),
		OnInvokeTool: func(_ context.Context, arguments string) (any, error) {
			return "unexpected:" + arguments, nil
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent", Tools: []agents.Tool{tool}},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{
			"async_tool_calls": false,
			"tool_error_formatter": RealtimeToolErrorFormatter(func(
				args RealtimeToolErrorFormatterArgs,
			) any {
				return "run-level " + args.ToolName + " denied (" + args.CallID + ")"
			}),
		},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      "secure_tool",
		CallID:    "approval_call_custom",
		Arguments: `{"value":"x"}`,
	}))
	<-session.Events() // raw
	_, ok := (<-session.Events()).(RealtimeToolApprovalRequiredEvent)
	require.True(t, ok)

	require.NoError(t, session.RejectToolCall(t.Context(), "approval_call_custom", false))
	toolEnd, ok := (<-session.Events()).(RealtimeToolEndEvent)
	require.True(t, ok)
	assert.Equal(t, "run-level secure_tool denied (approval_call_custom)", toolEnd.Output)

	require.Len(t, model.sentEvents, 1)
	outputEvent, ok := model.sentEvents[0].(RealtimeModelSendToolOutput)
	require.True(t, ok)
	assert.Equal(t, "run-level secure_tool denied (approval_call_custom)", outputEvent.Output)
}

func TestRealtimeSessionFunctionCallRequiresApprovalRejectFormatterFallbackToDefault(t *testing.T) {
	model := &mockRealtimeModel{}
	tool := agents.FunctionTool{
		Name:             "secure_tool",
		Description:      "Tool that needs approval",
		ParamsJSONSchema: map[string]any{"type": "object"},
		NeedsApproval:    agents.FunctionToolNeedsApprovalEnabled(),
		OnInvokeTool: func(_ context.Context, arguments string) (any, error) {
			return "unexpected:" + arguments, nil
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent", Tools: []agents.Tool{tool}},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{
			"async_tool_calls": false,
			"tool_error_formatter": RealtimeToolErrorFormatter(func(
				RealtimeToolErrorFormatterArgs,
			) any {
				return 123
			}),
		},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      "secure_tool",
		CallID:    "approval_call_bad_formatter",
		Arguments: `{"value":"x"}`,
	}))
	<-session.Events() // raw
	_, ok := (<-session.Events()).(RealtimeToolApprovalRequiredEvent)
	require.True(t, ok)

	require.NoError(t, session.RejectToolCall(t.Context(), "approval_call_bad_formatter", false))
	toolEnd, ok := (<-session.Events()).(RealtimeToolEndEvent)
	require.True(t, ok)
	assert.Equal(t, defaultRealtimeApprovalRejectionMessage, toolEnd.Output)

	require.Len(t, model.sentEvents, 1)
	outputEvent, ok := model.sentEvents[0].(RealtimeModelSendToolOutput)
	require.True(t, ok)
	assert.Equal(t, defaultRealtimeApprovalRejectionMessage, outputEvent.Output)
}

func TestRealtimeSessionFunctionCallUsesPreApprovedDecision(t *testing.T) {
	model := &mockRealtimeModel{}
	tool := agents.FunctionTool{
		Name:             "secure_tool",
		Description:      "Tool that needs approval",
		ParamsJSONSchema: map[string]any{"type": "object"},
		NeedsApproval:    agents.FunctionToolNeedsApprovalEnabled(),
		OnInvokeTool: func(_ context.Context, arguments string) (any, error) {
			return "approved:" + arguments, nil
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent", Tools: []agents.Tool{tool}},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	session.contextWrapper.ApproveTool(agents.ToolApprovalItem{
		ToolName: "secure_tool",
		RawItem: map[string]any{
			"call_id": "approval_call_3",
		},
	}, false)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      "secure_tool",
		CallID:    "approval_call_3",
		Arguments: `{"value":"y"}`,
	}))

	<-session.Events() // raw
	_, ok := (<-session.Events()).(RealtimeToolStartEvent)
	require.True(t, ok)
	_, ok = (<-session.Events()).(RealtimeToolEndEvent)
	require.True(t, ok)
	assert.Len(t, model.sentEvents, 1)
}

func TestRealtimeSessionFunctionCallUsesPreRejectedDecision(t *testing.T) {
	model := &mockRealtimeModel{}
	tool := agents.FunctionTool{
		Name:             "secure_tool",
		Description:      "Tool that needs approval",
		ParamsJSONSchema: map[string]any{"type": "object"},
		NeedsApproval:    agents.FunctionToolNeedsApprovalEnabled(),
		OnInvokeTool: func(_ context.Context, arguments string) (any, error) {
			return "approved:" + arguments, nil
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent", Tools: []agents.Tool{tool}},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	session.contextWrapper.RejectTool(agents.ToolApprovalItem{
		ToolName: "secure_tool",
		RawItem: map[string]any{
			"call_id": "approval_call_4",
		},
	}, false)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      "secure_tool",
		CallID:    "approval_call_4",
		Arguments: `{"value":"z"}`,
	}))

	<-session.Events() // raw
	toolEnd, ok := (<-session.Events()).(RealtimeToolEndEvent)
	require.True(t, ok)
	assert.Equal(t, defaultRealtimeApprovalRejectionMessage, toolEnd.Output)
	assert.Len(t, model.sentEvents, 1)
}

func TestRealtimeSessionTranscriptionCompletedUpdatesExistingHistoryItem(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)
	session.history = []any{
		map[string]any{
			"type":    "message",
			"role":    "user",
			"item_id": "item_1",
			"content": []any{
				map[string]any{
					"type":       "input_audio",
					"transcript": nil,
				},
				map[string]any{
					"type": "input_text",
					"text": "before",
				},
			},
		},
	}

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelInputAudioTranscriptionCompletedEvent{
		ItemID:     "item_1",
		Transcript: "hello",
	}))

	_, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	_, ok = (<-session.Events()).(RealtimeHistoryUpdatedEvent)
	require.True(t, ok)

	history := session.History()
	require.Len(t, history, 1)
	item, ok := history[0].(map[string]any)
	require.True(t, ok)
	content := extractContentParts(item["content"])
	require.Len(t, content, 2)
	firstPart, ok := toStringAnyMap(content[0])
	require.True(t, ok)
	assert.Equal(t, "hello", firstPart["transcript"])
}

func TestRealtimeSessionTranscriptionCompletedAppendsNewHistoryItem(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelInputAudioTranscriptionCompletedEvent{
		ItemID:     "item_new",
		Transcript: "new text",
	}))

	_, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	addedEvent, ok := (<-session.Events()).(RealtimeHistoryAddedEvent)
	require.True(t, ok)
	item, ok := addedEvent.Item.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "item_new", item["item_id"])

	history := session.History()
	require.Len(t, history, 1)
}

func TestRealtimeSessionInputAudioTimeoutTriggeredEmitsEvent(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	require.NoError(t, session.Enter(t.Context()))
	<-session.Events() // initial history event

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelInputAudioTimeoutTriggeredEvent{
		ItemID:       "item_1",
		AudioStartMS: 0,
		AudioEndMS:   100,
	}))

	_, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	_, ok = (<-session.Events()).(RealtimeInputAudioTimeoutTriggeredEvent)
	require.True(t, ok)
}

func TestRealtimeSessionTranscriptDeltaTriggersGuardrailTripwire(t *testing.T) {
	model := &mockRealtimeModel{}
	var guardrailRuns atomic.Int32
	guardrail := agents.OutputGuardrail{
		Name: "safety_check",
		GuardrailFunction: func(context.Context, *agents.Agent, any) (agents.GuardrailFunctionOutput, error) {
			guardrailRuns.Add(1)
			return agents.GuardrailFunctionOutput{TripwireTriggered: true}, nil
		},
	}

	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{
			Name:             "agent",
			OutputGuardrails: []agents.OutputGuardrail{guardrail},
		},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{
			"guardrails_settings": map[string]any{"debounce_text_length": 5},
		},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTranscriptDeltaEvent{
		ItemID:     "assistant_item_1",
		Delta:      "hi",
		ResponseID: "resp_1",
	}))
	_, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTranscriptDeltaEvent{
		ItemID:     "assistant_item_1",
		Delta:      "abc",
		ResponseID: "resp_1",
	}))
	_, ok = (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)

	require.Eventually(t, func() bool {
		return guardrailRuns.Load() >= 1
	}, time.Second, 10*time.Millisecond)

	require.Eventually(t, func() bool {
		if len(model.sentEvents) < 2 {
			return false
		}
		_, hasInterrupt := model.sentEvents[0].(RealtimeModelSendInterrupt)
		_, hasUserInput := model.sentEvents[1].(RealtimeModelSendUserInput)
		return hasInterrupt && hasUserInput
	}, time.Second, 10*time.Millisecond)

	history := session.History()
	require.Len(t, history, 1)
	item, ok := history[0].(map[string]any)
	require.True(t, ok)
	content := extractContentParts(item["content"])
	require.Len(t, content, 1)
	part, ok := toStringAnyMap(content[0])
	require.True(t, ok)
	assert.Equal(t, "hiabc", part["transcript"])
}

func TestRealtimeSessionTranscriptDeltaAvoidsDuplicateInterruptForSameResponse(t *testing.T) {
	model := &mockRealtimeModel{}
	guardrail := agents.OutputGuardrail{
		Name: "trip",
		GuardrailFunction: func(context.Context, *agents.Agent, any) (agents.GuardrailFunctionOutput, error) {
			return agents.GuardrailFunctionOutput{TripwireTriggered: true}, nil
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{
			Name:             "agent",
			OutputGuardrails: []agents.OutputGuardrail{guardrail},
		},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"guardrails_settings": map[string]any{"debounce_text_length": 2}},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTranscriptDeltaEvent{
		ItemID:     "assistant_item",
		Delta:      "ab",
		ResponseID: "resp_same",
	}))
	<-session.Events() // raw

	require.Eventually(t, func() bool { return len(model.sentEvents) >= 2 }, time.Second, 10*time.Millisecond)
	initialCount := len(model.sentEvents)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTranscriptDeltaEvent{
		ItemID:     "assistant_item",
		Delta:      "cd",
		ResponseID: "resp_same",
	}))
	<-session.Events() // raw

	time.Sleep(80 * time.Millisecond)
	assert.Equal(t, initialCount, len(model.sentEvents))
}

func TestRealtimeSessionTranscriptDeltaGuardrailPanicDoesNotCrashSession(t *testing.T) {
	model := &mockRealtimeModel{}
	guardrail := agents.OutputGuardrail{
		Name: "panic_guardrail",
		GuardrailFunction: func(context.Context, *agents.Agent, any) (agents.GuardrailFunctionOutput, error) {
			panic("guardrail boom")
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{
			Name:             "agent",
			OutputGuardrails: []agents.OutputGuardrail{guardrail},
		},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"guardrails_settings": map[string]any{"debounce_text_length": 1}},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTranscriptDeltaEvent{
		ItemID:     "assistant_item",
		Delta:      "a",
		ResponseID: "resp_1",
	}))
	<-session.Events() // raw

	time.Sleep(80 * time.Millisecond)
	assert.Empty(t, model.sentEvents)
	for {
		select {
		case event := <-session.Events():
			if _, ok := event.(RealtimeErrorEvent); ok {
				continue
			}
		default:
			goto drained
		}
	}
drained:

	// Session should still process subsequent events after guardrail panic.
	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTurnEndedEvent{}))
	<-session.Events() // raw
	_, ok := (<-session.Events()).(RealtimeAgentEndEvent)
	require.True(t, ok)
}

func TestRealtimeSessionOutputGuardrailsDedupesAgentAndRunConfig(t *testing.T) {
	model := &mockRealtimeModel{}
	var runs atomic.Int32
	guardrail := agents.OutputGuardrail{
		Name: "dedupe_guardrail",
		GuardrailFunction: func(context.Context, *agents.Agent, any) (agents.GuardrailFunctionOutput, error) {
			runs.Add(1)
			return agents.GuardrailFunctionOutput{TripwireTriggered: false}, nil
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{
			Name:             "agent",
			OutputGuardrails: []agents.OutputGuardrail{guardrail},
		},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{
			"guardrails_settings": map[string]any{"debounce_text_length": 1},
			"output_guardrails":   []agents.OutputGuardrail{guardrail},
		},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTranscriptDeltaEvent{
		ItemID:     "assistant_item",
		Delta:      "a",
		ResponseID: "resp_1",
	}))
	<-session.Events() // raw

	require.Eventually(t, func() bool {
		return runs.Load() == 1
	}, time.Second, 10*time.Millisecond)
	time.Sleep(80 * time.Millisecond)
	assert.Equal(t, int32(1), runs.Load())
	assert.Empty(t, model.sentEvents)
}

func TestRealtimeSessionTranscriptDeltaMultipleGuardrailsCombined(t *testing.T) {
	model := &mockRealtimeModel{}
	guardrailA := agents.OutputGuardrail{
		Name: "guardrail_a",
		GuardrailFunction: func(context.Context, *agents.Agent, any) (agents.GuardrailFunctionOutput, error) {
			return agents.GuardrailFunctionOutput{TripwireTriggered: true}, nil
		},
	}
	guardrailB := agents.OutputGuardrail{
		Name: "guardrail_b",
		GuardrailFunction: func(context.Context, *agents.Agent, any) (agents.GuardrailFunctionOutput, error) {
			return agents.GuardrailFunctionOutput{TripwireTriggered: true}, nil
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{
			Name:             "agent",
			OutputGuardrails: []agents.OutputGuardrail{guardrailA, guardrailB},
		},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"guardrails_settings": map[string]any{"debounce_text_length": 1}},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTranscriptDeltaEvent{
		ItemID:     "assistant_item",
		Delta:      "a",
		ResponseID: "resp_1",
	}))
	<-session.Events() // raw

	require.Eventually(t, func() bool {
		return len(model.sentEvents) >= 2 && len(session.eventQueue) >= 1
	}, time.Second, 10*time.Millisecond)

	trippedEvent, ok := (<-session.Events()).(RealtimeGuardrailTrippedEvent)
	require.True(t, ok)
	assert.Len(t, trippedEvent.GuardrailResults, 2)

	require.IsType(t, RealtimeModelSendInterrupt{}, model.sentEvents[0])
	userInputEvent, ok := model.sentEvents[1].(RealtimeModelSendUserInput)
	require.True(t, ok)
	assert.Contains(t, userInputEvent.UserInput, "guardrail_a")
	assert.Contains(t, userInputEvent.UserInput, "guardrail_b")
}

func TestRealtimeSessionTurnEndedClearsTranscriptDebounceState(t *testing.T) {
	model := &mockRealtimeModel{}
	var runs atomic.Int32
	guardrail := agents.OutputGuardrail{
		Name: "trip",
		GuardrailFunction: func(context.Context, *agents.Agent, any) (agents.GuardrailFunctionOutput, error) {
			runs.Add(1)
			return agents.GuardrailFunctionOutput{TripwireTriggered: true}, nil
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{
			Name:             "agent",
			OutputGuardrails: []agents.OutputGuardrail{guardrail},
		},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"guardrails_settings": map[string]any{"debounce_text_length": 3}},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTranscriptDeltaEvent{
		ItemID:     "assistant_item",
		Delta:      "abc",
		ResponseID: "resp_1",
	}))
	<-session.Events() // raw
	require.Eventually(t, func() bool { return runs.Load() == 1 }, time.Second, 10*time.Millisecond)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTurnEndedEvent{}))
	<-session.Events() // raw
	<-session.Events() // agent_end

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTranscriptDeltaEvent{
		ItemID:     "assistant_item",
		Delta:      "abc",
		ResponseID: "resp_2",
	}))
	<-session.Events() // raw
	require.Eventually(t, func() bool { return runs.Load() == 2 }, time.Second, 10*time.Millisecond)
}

func TestRealtimeSessionTurnEndedKeepsPendingToolApprovals(t *testing.T) {
	model := &mockRealtimeModel{}
	tool := agents.FunctionTool{
		Name:             "secure_tool",
		Description:      "Tool that needs approval",
		ParamsJSONSchema: map[string]any{"type": "object"},
		NeedsApproval:    agents.FunctionToolNeedsApprovalEnabled(),
		OnInvokeTool: func(_ context.Context, arguments string) (any, error) {
			return "approved:" + arguments, nil
		},
	}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent", Tools: []agents.Tool{tool}},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{"async_tool_calls": false},
	)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelToolCallEvent{
		Name:      "secure_tool",
		CallID:    "approval_call_turn",
		Arguments: `{"value":"z"}`,
	}))

	<-session.Events() // raw
	_, ok := (<-session.Events()).(RealtimeToolApprovalRequiredEvent)
	require.True(t, ok)
	require.Len(t, session.pendingToolCalls, 1)

	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelTurnEndedEvent{}))
	<-session.Events() // raw
	_, ok = (<-session.Events()).(RealtimeAgentEndEvent)
	require.True(t, ok)
	require.Len(t, session.pendingToolCalls, 1)

	require.NoError(t, session.ApproveToolCall(t.Context(), "approval_call_turn", false))
	_, ok = (<-session.Events()).(RealtimeToolStartEvent)
	require.True(t, ok)
	_, ok = (<-session.Events()).(RealtimeToolEndEvent)
	require.True(t, ok)
	require.Len(t, model.sentEvents, 1)
}

func TestRealtimeSessionModelExceptionEventEmitsRealtimeError(t *testing.T) {
	model := &mockRealtimeModel{}
	session := NewRealtimeSession(
		model,
		&RealtimeAgent[any]{Name: "agent"},
		nil,
		RealtimeModelConfig{},
		RealtimeRunConfig{},
	)

	contextMessage := "test listener context"
	require.NoError(t, session.OnEvent(t.Context(), RealtimeModelExceptionEvent{
		Exception: context.Canceled,
		Context:   &contextMessage,
	}))

	_, ok := (<-session.Events()).(RealtimeRawModelEvent)
	require.True(t, ok)
	realtimeErr, ok := (<-session.Events()).(RealtimeErrorEvent)
	require.True(t, ok)
	errMap, ok := realtimeErr.Error.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, contextMessage, errMap["message"])
}
