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

package agents_test

import (
	"context"
	"testing"
	"time"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/denggeng/openai-agents-go-plus/tracing"
	"github.com/denggeng/openai-agents-go-plus/tracing/tracingtesting"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func inputGuardrail(delay time.Duration, trip bool, name string) agents.InputGuardrail {
	return agents.InputGuardrail{
		Name: name,
		GuardrailFunction: func(_ context.Context, _ *agents.Agent, _ agents.Input) (agents.GuardrailFunctionOutput, error) {
			if delay > 0 {
				time.Sleep(delay)
			}
			return agents.GuardrailFunctionOutput{
				OutputInfo: map[string]any{
					"delay_ms": delay.Milliseconds(),
				},
				TripwireTriggered: trip,
			}, nil
		},
	}
}

func makeInputGuardrail(delay time.Duration, trip bool) agents.InputGuardrail {
	name := "delayed_input_guardrail"
	if trip {
		name = "tripping_input_guardrail"
	}
	return inputGuardrail(delay, trip, name)
}

func streamEventType(event agents.StreamEvent) string {
	switch v := event.(type) {
	case agents.RawResponsesStreamEvent:
		return v.Type
	case agents.RunItemStreamEvent:
		return v.Type
	case agents.AgentUpdatedStreamEvent:
		return v.Type
	default:
		return ""
	}
}

func spanByType(spans []tracing.Span, spanType string) tracing.Span {
	for _, span := range spans {
		if span.SpanData().Type() == spanType {
			return span
		}
	}
	return nil
}

type slowCompleteFakeModel struct {
	*agentstesting.FakeModel
	delay time.Duration
}

func (m *slowCompleteFakeModel) StreamResponse(
	ctx context.Context,
	params agents.ModelResponseParams,
	yield agents.ModelStreamResponseCallback,
) error {
	return m.FakeModel.StreamResponse(ctx, params, func(ctx context.Context, event agents.TResponseStreamEvent) error {
		if event.Type == "response.completed" && m.delay > 0 {
			time.Sleep(m.delay)
		}
		return yield(ctx, event)
	})
}

func TestStreamInputGuardrailResultsFollowCompletionOrder(t *testing.T) {
	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Final response")},
	})

	agent := agents.New("TimingAgentOrder").
		WithModelInstance(model).
		WithInputGuardrails([]agents.InputGuardrail{
			inputGuardrail(50*time.Millisecond, false, "slow_guardrail"),
			inputGuardrail(0, false, "fast_guardrail"),
		})

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)
	require.NoError(t, result.StreamEvents(func(agents.StreamEvent) error { return nil }))

	results := result.InputGuardrailResults()
	require.Len(t, results, 2)
	delay0 := results[0].Output.OutputInfo.(map[string]any)["delay_ms"].(int64)
	delay1 := results[1].Output.OutputInfo.(map[string]any)["delay_ms"].(int64)
	assert.Equal(t, []int64{0, 50}, []int64{delay0, delay1})
}

func TestRunStreamedInputGuardrailTimingIsConsistent(t *testing.T) {
	for _, delay := range []time.Duration{0, 200 * time.Millisecond} {
		t.Run(delay.String(), func(t *testing.T) {
			model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
				Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Final response")},
			})

			agent := agents.New("TimingAgent").
				WithModelInstance(model).
				WithInputGuardrails([]agents.InputGuardrail{makeInputGuardrail(delay, false)})

			result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Hello")
			require.NoError(t, err)

			var eventTypes []string
			require.NoError(t, result.StreamEvents(func(event agents.StreamEvent) error {
				eventTypes = append(eventTypes, streamEventType(event))
				return nil
			}))

			require.Len(t, result.InputGuardrailResults(), 1)
			assert.Equal(t, "delayed_input_guardrail", result.InputGuardrailResults()[0].Guardrail.Name)
			assert.False(t, result.InputGuardrailResults()[0].Output.TripwireTriggered)
			assert.Equal(t, "Final response", result.FinalOutput())

			require.GreaterOrEqual(t, len(eventTypes), 3)
			assert.Equal(t, "agent_updated_stream_event", eventTypes[0])
			assert.Contains(t, eventTypes, "raw_response_event")
		})
	}
}

func TestRunStreamedInputGuardrailSequencesMatchBetweenFastAndSlow(t *testing.T) {
	runOnce := func(delay time.Duration) []string {
		model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Final response")},
		})
		agent := agents.New("TimingAgent").
			WithModelInstance(model).
			WithInputGuardrails([]agents.InputGuardrail{makeInputGuardrail(delay, false)})

		result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Hello")
		require.NoError(t, err)

		var events []string
		require.NoError(t, result.StreamEvents(func(event agents.StreamEvent) error {
			events = append(events, streamEventType(event))
			return nil
		}))
		return events
	}

	eventsFast := runOnce(0)
	eventsSlow := runOnce(200 * time.Millisecond)
	assert.Equal(t, eventsFast, eventsSlow)
}

func TestRunStreamedInputGuardrailTripwireRaises(t *testing.T) {
	for _, delay := range []time.Duration{0, 200 * time.Millisecond} {
		t.Run(delay.String(), func(t *testing.T) {
			model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
				Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Final response")},
			})

			agent := agents.New("TimingAgentTrip").
				WithModelInstance(model).
				WithInputGuardrails([]agents.InputGuardrail{makeInputGuardrail(delay, true)})

			result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Hello")
			require.NoError(t, err)

			err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
			var tripErr agents.InputGuardrailTripwireTriggeredError
			require.ErrorAs(t, err, &tripErr)
			assert.True(t, tripErr.GuardrailResult.Output.TripwireTriggered)
			assert.Equal(t, "tripping_input_guardrail", tripErr.GuardrailResult.Guardrail.Name)
			require.NotNil(t, tripErr.AgentsError)
			require.NotNil(t, tripErr.AgentsError.RunData)
			assert.Len(t, tripErr.AgentsError.RunData.InputGuardrailResults, 1)
			assert.Equal(t, "tripping_input_guardrail", tripErr.AgentsError.RunData.InputGuardrailResults[0].Guardrail.Name)
		})
	}
}

func TestParentSpanAndTraceFinishAfterSlowInputGuardrail(t *testing.T) {
	tracingtesting.Setup(t)

	model := agentstesting.NewFakeModel(true, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Final response")},
	})
	agent := agents.New("TimingAgentTrace").
		WithModelInstance(model).
		WithInputGuardrails([]agents.InputGuardrail{makeInputGuardrail(200*time.Millisecond, false)})

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)
	require.NoError(t, result.StreamEvents(func(agents.StreamEvent) error { return nil }))

	spans := tracingtesting.FetchOrderedSpans(false)
	agentSpan := spanByType(spans, "agent")
	guardrailSpan := spanByType(spans, "guardrail")
	generationSpan := spanByType(spans, "generation")
	require.NotNil(t, agentSpan)
	require.NotNil(t, guardrailSpan)
	require.NotNil(t, generationSpan)

	assert.True(t, !agentSpan.EndedAt().Before(guardrailSpan.EndedAt()))
	assert.True(t, !agentSpan.EndedAt().Before(generationSpan.EndedAt()))

	events := tracingtesting.FetchEvents()
	require.NotEmpty(t, events)
	assert.Equal(t, tracingtesting.TraceEnd, events[len(events)-1])
}

func TestParentSpanAndTraceFinishAfterSlowModel(t *testing.T) {
	tracingtesting.Setup(t)

	model := &slowCompleteFakeModel{
		FakeModel: agentstesting.NewFakeModel(true, &agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Final response")},
		}),
		delay: 200 * time.Millisecond,
	}
	agent := agents.New("TimingAgentTrace").
		WithModelInstance(model).
		WithInputGuardrails([]agents.InputGuardrail{makeInputGuardrail(0, false)})

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)
	require.NoError(t, result.StreamEvents(func(agents.StreamEvent) error { return nil }))

	spans := tracingtesting.FetchOrderedSpans(false)
	agentSpan := spanByType(spans, "agent")
	guardrailSpan := spanByType(spans, "guardrail")
	generationSpan := spanByType(spans, "generation")
	require.NotNil(t, agentSpan)
	require.NotNil(t, guardrailSpan)
	require.NotNil(t, generationSpan)

	assert.True(t, !agentSpan.EndedAt().Before(guardrailSpan.EndedAt()))
	assert.True(t, !agentSpan.EndedAt().Before(generationSpan.EndedAt()))

	events := tracingtesting.FetchEvents()
	require.NotEmpty(t, events)
	assert.Equal(t, tracingtesting.TraceEnd, events[len(events)-1])
}
