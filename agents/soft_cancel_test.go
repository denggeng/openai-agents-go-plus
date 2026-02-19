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
	"path/filepath"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/memory"
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type softCancelFakeModel struct {
	turnOutputs    []softCancelTurnOutput
	hardcodedUsage *usage.Usage
}

type softCancelTurnOutput struct {
	Value []TResponseOutputItem
	Error error
}

func newSoftCancelFakeModel(initial *softCancelTurnOutput) *softCancelFakeModel {
	model := &softCancelFakeModel{}
	if initial != nil {
		model.turnOutputs = append(model.turnOutputs, *initial)
	}
	return model
}

func (m *softCancelFakeModel) SetNextOutput(output softCancelTurnOutput) {
	m.turnOutputs = append(m.turnOutputs, output)
}

func (m *softCancelFakeModel) AddMultipleTurnOutputs(outputs []softCancelTurnOutput) {
	m.turnOutputs = append(m.turnOutputs, outputs...)
}

func (m *softCancelFakeModel) getNextOutput() softCancelTurnOutput {
	if len(m.turnOutputs) == 0 {
		return softCancelTurnOutput{}
	}
	output := m.turnOutputs[0]
	m.turnOutputs = m.turnOutputs[1:]
	return output
}

func (m *softCancelFakeModel) GetResponse(context.Context, ModelResponseParams) (*ModelResponse, error) {
	output := m.getNextOutput()
	if output.Error != nil {
		return nil, output.Error
	}
	u := m.hardcodedUsage
	if u == nil {
		u = usage.NewUsage()
	}
	return &ModelResponse{
		Output:     output.Value,
		Usage:      u,
		ResponseID: "resp-soft-cancel",
	}, nil
}

func (m *softCancelFakeModel) StreamResponse(
	ctx context.Context,
	_ ModelResponseParams,
	yield ModelStreamResponseCallback,
) error {
	output := m.getNextOutput()
	if output.Error != nil {
		return output.Error
	}
	response := softCancelResponseObj(output.Value, "resp-soft-cancel", m.hardcodedUsage)
	return yield(ctx, TResponseStreamEvent{
		Type:           "response.completed",
		Response:       response,
		SequenceNumber: 0,
	})
}

func softCancelResponseObj(
	output []TResponseOutputItem,
	responseID string,
	u *usage.Usage,
) responses.Response {
	if responseID == "" {
		responseID = "resp-soft-cancel"
	}

	var responseUsage responses.ResponseUsage
	if u != nil {
		responseUsage = responses.ResponseUsage{
			InputTokens: int64(u.InputTokens),
			InputTokensDetails: responses.ResponseUsageInputTokensDetails{
				CachedTokens: 0,
			},
			OutputTokens: int64(u.OutputTokens),
			OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
				ReasoningTokens: 0,
			},
			TotalTokens: int64(u.TotalTokens),
		}
	}

	return responses.Response{
		ID:        responseID,
		CreatedAt: 123,
		Model:     "test_model",
		Object:    "response",
		Output:    output,
		ToolChoice: responses.ResponseToolChoiceUnion{
			OfToolChoiceMode: responses.ToolChoiceOptionsNone,
		},
		Tools:             nil,
		TopP:              0,
		ParallelToolCalls: false,
		Usage:             responseUsage,
	}
}

func softCancelTextMessage(content string) responses.ResponseOutputItemUnion {
	return responses.ResponseOutputItemUnion{
		ID:   "1",
		Type: "message",
		Role: constant.ValueOf[constant.Assistant](),
		Content: []responses.ResponseOutputMessageContentUnion{{
			Text:        content,
			Type:        "output_text",
			Annotations: nil,
		}},
		Status: string(responses.ResponseOutputMessageStatusCompleted),
	}
}

func softCancelFunctionTool(name string, returnValue string) FunctionTool {
	return FunctionTool{
		Name: name,
		ParamsJSONSchema: map[string]any{
			"title":                name + "_args",
			"type":                 "object",
			"required":             []string{},
			"additionalProperties": false,
			"properties":           map[string]any{},
		},
		OnInvokeTool: func(context.Context, string) (any, error) {
			return returnValue, nil
		},
	}
}

func softCancelFunctionToolCall(name string, arguments string) responses.ResponseOutputItemUnion {
	return responses.ResponseOutputItemUnion{
		ID:        "1",
		CallID:    "2",
		Type:      "function_call",
		Name:      name,
		Arguments: arguments,
	}
}

func TestSoftCancelCompletesTurn(t *testing.T) {
	model := newSoftCancelFakeModel(&softCancelTurnOutput{
		Value: []TResponseOutputItem{softCancelTextMessage("hello")},
	})
	agent := New("Assistant").WithModelInstance(model)

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	eventCount := 0
	err = result.StreamEvents(func(event StreamEvent) error {
		eventCount++
		if eventCount == 1 {
			result.Cancel(CancelModeAfterTurn)
		}
		return nil
	})
	require.NoError(t, err)
	assert.Greater(t, eventCount, 1)
	assert.True(t, result.IsComplete())
}

func TestSoftCancelVsImmediate(t *testing.T) {
	model1 := newSoftCancelFakeModel(&softCancelTurnOutput{
		Value: []TResponseOutputItem{softCancelTextMessage("hello")},
	})
	agent1 := New("A1").WithModelInstance(model1)
	result1, err := Runner{}.RunStreamed(t.Context(), agent1, "Hello")
	require.NoError(t, err)

	immediateEvents := 0
	_ = result1.StreamEvents(func(event StreamEvent) error {
		immediateEvents++
		if immediateEvents == 1 {
			result1.Cancel()
		}
		return nil
	})

	model2 := newSoftCancelFakeModel(&softCancelTurnOutput{
		Value: []TResponseOutputItem{softCancelTextMessage("hello")},
	})
	agent2 := New("A2").WithModelInstance(model2)
	result2, err := Runner{}.RunStreamed(t.Context(), agent2, "Hello")
	require.NoError(t, err)

	softEvents := 0
	_ = result2.StreamEvents(func(event StreamEvent) error {
		softEvents++
		if softEvents == 1 {
			result2.Cancel(CancelModeAfterTurn)
		}
		return nil
	})

	assert.Greater(t, softEvents, immediateEvents)
}

func TestSoftCancelWithToolCalls(t *testing.T) {
	model := newSoftCancelFakeModel(nil)
	agent := New("Assistant").WithModelInstance(model).WithTools(
		softCancelFunctionTool("calc", "42"),
	)

	model.AddMultipleTurnOutputs([]softCancelTurnOutput{
		{Value: []TResponseOutputItem{
			softCancelTextMessage("calc"),
			softCancelFunctionToolCall("calc", "{}"),
		}},
		{Value: []TResponseOutputItem{
			softCancelTextMessage("result"),
		}},
	})

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Calculate")
	require.NoError(t, err)

	toolCallSeen := false
	toolOutputSeen := false
	err = result.StreamEvents(func(event StreamEvent) error {
		runEvent, ok := event.(RunItemStreamEvent)
		if !ok {
			return nil
		}
		switch runEvent.Name {
		case StreamEventToolCalled:
			toolCallSeen = true
			result.Cancel(CancelModeAfterTurn)
		case StreamEventToolOutput:
			toolOutputSeen = true
		}
		return nil
	})
	require.NoError(t, err)
	assert.True(t, toolCallSeen)
	assert.True(t, toolOutputSeen)
}

func TestSoftCancelSavesSession(t *testing.T) {
	session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
		SessionID:        "test_soft_cancel_session",
		DBDataSourceName: filepath.Join(t.TempDir(), "test.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	model := newSoftCancelFakeModel(&softCancelTurnOutput{
		Value: []TResponseOutputItem{softCancelTextMessage("hello")},
	})
	agent := New("Assistant").WithModelInstance(model)

	runner := Runner{Config: RunConfig{Session: session}}
	result, err := runner.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	err = result.StreamEvents(func(event StreamEvent) error {
		if _, ok := event.(RunItemStreamEvent); ok {
			result.Cancel(CancelModeAfterTurn)
		}
		return nil
	})
	require.NoError(t, err)

	items, err := session.GetItems(t.Context(), 0)
	require.NoError(t, err)
	assert.NotEmpty(t, items)

	result2, err := runner.Run(t.Context(), agent, "Continue")
	require.NoError(t, err)
	assert.NotNil(t, result2.FinalOutput)
}

func TestSoftCancelTracksUsage(t *testing.T) {
	model := newSoftCancelFakeModel(&softCancelTurnOutput{
		Value: []TResponseOutputItem{softCancelTextMessage("hello")},
	})
	agent := New("Assistant").WithModelInstance(model)

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	err = result.StreamEvents(func(event StreamEvent) error {
		if _, ok := event.(RawResponsesStreamEvent); ok {
			result.Cancel(CancelModeAfterTurn)
		}
		return nil
	})
	require.NoError(t, err)

	u, ok := usage.FromContext(result.context)
	require.True(t, ok)
	assert.Greater(t, u.Requests, uint64(0))
}

func TestSoftCancelStopsNextTurn(t *testing.T) {
	model := newSoftCancelFakeModel(nil)
	agent := New("Assistant").WithModelInstance(model).WithTools(
		softCancelFunctionTool("tool1", "result1"),
	)

	model.AddMultipleTurnOutputs([]softCancelTurnOutput{
		{Value: []TResponseOutputItem{softCancelFunctionToolCall("tool1", "{}")}},
		{Value: []TResponseOutputItem{softCancelTextMessage("Turn 2")}},
		{Value: []TResponseOutputItem{softCancelTextMessage("Turn 3")}},
	})

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	turnsCompleted := 0
	err = result.StreamEvents(func(event StreamEvent) error {
		runEvent, ok := event.(RunItemStreamEvent)
		if !ok {
			return nil
		}
		if runEvent.Name == StreamEventToolOutput {
			turnsCompleted++
			if turnsCompleted == 1 {
				result.Cancel(CancelModeAfterTurn)
			}
		}
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 1, turnsCompleted)
}

func TestCancelModeBackwardCompatibility(t *testing.T) {
	model := newSoftCancelFakeModel(&softCancelTurnOutput{
		Value: []TResponseOutputItem{softCancelTextMessage("hello")},
	})
	agent := New("Assistant").WithModelInstance(model)

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	events := 0
	_ = result.StreamEvents(func(event StreamEvent) error {
		events++
		if events == 1 {
			result.Cancel()
		}
		return nil
	})

	assert.Equal(t, 1, events)
	assert.True(t, result.IsComplete())
	assert.Equal(t, CancelModeImmediate, result.CancelMode())
}

func TestSoftCancelIdempotent(t *testing.T) {
	model := newSoftCancelFakeModel(&softCancelTurnOutput{
		Value: []TResponseOutputItem{softCancelTextMessage("hello")},
	})
	agent := New("Assistant").WithModelInstance(model)

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	calledTwice := false
	_ = result.StreamEvents(func(event StreamEvent) error {
		if !calledTwice {
			result.Cancel(CancelModeAfterTurn)
			result.Cancel(CancelModeAfterTurn)
			calledTwice = true
		}
		return nil
	})

	assert.True(t, result.IsComplete())
}

func TestSoftCancelBeforeStreaming(t *testing.T) {
	model := newSoftCancelFakeModel(&softCancelTurnOutput{
		Value: []TResponseOutputItem{softCancelTextMessage("hello")},
	})
	agent := New("Assistant").WithModelInstance(model)

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)
	result.Cancel(CancelModeAfterTurn)

	events := make([]StreamEvent, 0)
	_ = result.StreamEvents(func(event StreamEvent) error {
		events = append(events, event)
		return nil
	})

	assert.LessOrEqual(t, len(events), 1)
	assert.True(t, result.IsComplete())
}

func TestSoftCancelMixedModes(t *testing.T) {
	model := newSoftCancelFakeModel(&softCancelTurnOutput{
		Value: []TResponseOutputItem{softCancelTextMessage("hello")},
	})
	agent := New("Assistant").WithModelInstance(model)

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	result.Cancel(CancelModeAfterTurn)
	result.Cancel(CancelModeImmediate)

	_ = result.StreamEvents(func(event StreamEvent) error { return nil })

	assert.Equal(t, CancelModeImmediate, result.CancelMode())
}

func TestSoftCancelExplicitImmediateMode(t *testing.T) {
	model := newSoftCancelFakeModel(&softCancelTurnOutput{
		Value: []TResponseOutputItem{softCancelTextMessage("hello")},
	})
	agent := New("Assistant").WithModelInstance(model)

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	events := 0
	_ = result.StreamEvents(func(event StreamEvent) error {
		events++
		if events == 1 {
			result.Cancel(CancelModeImmediate)
		}
		return nil
	})

	assert.True(t, result.IsComplete())
	assert.Equal(t, CancelModeImmediate, result.CancelMode())
	assert.Equal(t, 1, events)
}

func TestSoftCancelWithMultipleToolCalls(t *testing.T) {
	model := newSoftCancelFakeModel(nil)
	agent := New("Assistant").WithModelInstance(model).WithTools(
		softCancelFunctionTool("tool1", "result1"),
		softCancelFunctionTool("tool2", "result2"),
	)

	model.AddMultipleTurnOutputs([]softCancelTurnOutput{
		{Value: []TResponseOutputItem{
			softCancelFunctionToolCall("tool1", "{}"),
			softCancelFunctionToolCall("tool2", "{}"),
		}},
		{Value: []TResponseOutputItem{softCancelTextMessage("done")}},
	})

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Execute")
	require.NoError(t, err)

	toolOutputsSeen := 0
	err = result.StreamEvents(func(event StreamEvent) error {
		runEvent, ok := event.(RunItemStreamEvent)
		if !ok {
			return nil
		}
		switch runEvent.Name {
		case StreamEventToolCalled:
			if toolOutputsSeen == 0 {
				result.Cancel(CancelModeAfterTurn)
			}
		case StreamEventToolOutput:
			toolOutputsSeen++
		}
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 2, toolOutputsSeen)
}

func TestSoftCancelPreservesState(t *testing.T) {
	model := newSoftCancelFakeModel(nil)
	agent := New("Assistant").WithModelInstance(model).WithTools(
		softCancelFunctionTool("tool1", "result"),
	)

	model.AddMultipleTurnOutputs([]softCancelTurnOutput{
		{Value: []TResponseOutputItem{softCancelFunctionToolCall("tool1", "{}")}},
		{Value: []TResponseOutputItem{softCancelTextMessage("done")}},
	})

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	err = result.StreamEvents(func(event StreamEvent) error {
		runEvent, ok := event.(RunItemStreamEvent)
		if !ok {
			return nil
		}
		if runEvent.Name == StreamEventToolOutput {
			result.Cancel(CancelModeAfterTurn)
		}
		return nil
	})
	require.NoError(t, err)

	assert.True(t, result.IsComplete())
	assert.NotEmpty(t, result.NewItems())
	assert.NotEmpty(t, result.RawResponses())

	u, ok := usage.FromContext(result.context)
	require.True(t, ok)
	assert.Greater(t, u.Requests, uint64(0))
}

func TestImmediateCancelClearsQueues(t *testing.T) {
	model := newSoftCancelFakeModel(&softCancelTurnOutput{
		Value: []TResponseOutputItem{softCancelTextMessage("hello")},
	})
	agent := New("Assistant").WithModelInstance(model)

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	_ = result.StreamEvents(func(event StreamEvent) error {
		result.Cancel(CancelModeImmediate)
		return nil
	})

	assert.True(t, result.eventQueue.IsEmpty())
	assert.True(t, result.inputGuardrailQueue.IsEmpty())
}

func TestSoftCancelDoesNotClearQueuesImmediately(t *testing.T) {
	model := newSoftCancelFakeModel(&softCancelTurnOutput{
		Value: []TResponseOutputItem{softCancelTextMessage("hello")},
	})
	agent := New("Assistant").WithModelInstance(model)

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	result.Cancel(CancelModeAfterTurn)
	assert.Equal(t, CancelModeAfterTurn, result.CancelMode())

	_ = result.StreamEvents(func(event StreamEvent) error { return nil })
}

func TestSoftCancelWithHandoff(t *testing.T) {
	model := newSoftCancelFakeModel(nil)
	agent2 := New("Agent2").WithModelInstance(model)
	handoff := HandoffFromAgent(HandoffFromAgentParams{Agent: agent2})
	agent1 := New("Agent1").WithModelInstance(model).WithHandoffs(handoff)

	model.AddMultipleTurnOutputs([]softCancelTurnOutput{
		{Value: []TResponseOutputItem{softCancelFunctionToolCall(DefaultHandoffToolName(agent2), "{}")}},
		{Value: []TResponseOutputItem{softCancelTextMessage("Agent2 response")}},
	})

	session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
		SessionID:        "test_soft_cancel_handoff",
		DBDataSourceName: filepath.Join(t.TempDir(), "handoff.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	result, err := Runner{Config: RunConfig{Session: session}}.RunStreamed(t.Context(), agent1, "Hello")
	require.NoError(t, err)

	handoffSeen := false
	err = result.StreamEvents(func(event StreamEvent) error {
		runEvent, ok := event.(RunItemStreamEvent)
		if !ok {
			return nil
		}
		if runEvent.Name == StreamEventHandoffRequested {
			handoffSeen = true
			result.Cancel(CancelModeAfterTurn)
		}
		return nil
	})
	require.NoError(t, err)
	assert.True(t, handoffSeen)

	items, err := session.GetItems(t.Context(), 0)
	require.NoError(t, err)
	assert.NotEmpty(t, items)
}

func TestSoftCancelWithSessionAndMultipleTurns(t *testing.T) {
	model := newSoftCancelFakeModel(nil)
	agent := New("Assistant").WithModelInstance(model).WithTools(
		softCancelFunctionTool("tool1", "result1"),
	)

	session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
		SessionID:        "test_soft_cancel_multi",
		DBDataSourceName: filepath.Join(t.TempDir(), "multi.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	model.AddMultipleTurnOutputs([]softCancelTurnOutput{
		{Value: []TResponseOutputItem{softCancelFunctionToolCall("tool1", "{}")}},
		{Value: []TResponseOutputItem{softCancelFunctionToolCall("tool1", "{}")}},
		{Value: []TResponseOutputItem{softCancelTextMessage("Final")}},
	})

	result, err := Runner{Config: RunConfig{Session: session}}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	turnsSeen := 0
	err = result.StreamEvents(func(event StreamEvent) error {
		runEvent, ok := event.(RunItemStreamEvent)
		if !ok {
			return nil
		}
		if runEvent.Name == StreamEventToolOutput {
			turnsSeen++
			if turnsSeen == 2 {
				result.Cancel(CancelModeAfterTurn)
			}
		}
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 2, turnsSeen)

	items, err := session.GetItems(t.Context(), 0)
	require.NoError(t, err)
	assert.NotEmpty(t, items)
}
