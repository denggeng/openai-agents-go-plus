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

	agents "github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStreamEventsMain(t *testing.T) {
	model := newStreamEventsFakeModel(
		[]agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("foo", ""),
		},
		[]agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		},
	)
	agent := &agents.Agent{
		Name:  "Joker",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("foo", "success!"),
		},
	}

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	toolCallIndex := -1
	toolOutputIndex := -1
	eventIndex := 0
	err = result.StreamEvents(func(event agents.StreamEvent) error {
		if runEvent, ok := event.(agents.RunItemStreamEvent); ok {
			switch runEvent.Item.(type) {
			case agents.ToolCallItem:
				if toolCallIndex < 0 {
					toolCallIndex = eventIndex
				}
			case agents.ToolCallOutputItem:
				if toolOutputIndex < 0 {
					toolOutputIndex = eventIndex
				}
			}
		}
		eventIndex++
		return nil
	})
	require.NoError(t, err)
	require.Greater(t, toolCallIndex, -1, "tool_call_item was not observed")
	require.Greater(t, toolOutputIndex, -1, "tool_call_output_item was not observed")
	assert.Less(t, toolCallIndex, toolOutputIndex, "tool call ended before it started")
}

func TestStreamEventsMainWithHandoff(t *testing.T) {
	englishAgent := &agents.Agent{
		Name: "EnglishAgent",
		Model: param.NewOpt(agents.NewAgentModel(agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetTextMessage("Done"),
			},
		}))),
	}

	model := newStreamEventsFakeModel(
		[]agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Hello"),
			agentstesting.GetFunctionToolCall("foo", `{"args": "arg1"}`),
			agentstesting.GetHandoffToolCall(englishAgent, "", `{"args": "arg1"}`),
		},
		[]agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Done"),
		},
	)
	triageAgent := &agents.Agent{
		Name:         "TriageAgent",
		Instructions: agents.InstructionsStr("Handoff to the appropriate agent based on the language of the request."),
		Model:        param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("foo", "foo_result_arg1"),
		},
		AgentHandoffs: []*agents.Agent{englishAgent},
	}

	result, err := agents.Runner{}.RunStreamed(t.Context(), triageAgent, "Start")
	require.NoError(t, err)

	handoffRequestedSeen := false
	agentSwitched := false
	err = result.StreamEvents(func(event agents.StreamEvent) error {
		switch e := event.(type) {
		case agents.RunItemStreamEvent:
			if _, ok := e.Item.(agents.HandoffCallItem); ok {
				handoffRequestedSeen = true
			}
		case agents.AgentUpdatedStreamEvent:
			if e.NewAgent != nil && e.NewAgent.Name == "EnglishAgent" {
				agentSwitched = true
			}
		}
		return nil
	})
	require.NoError(t, err)
	assert.True(t, handoffRequestedSeen, "handoff_requested event not observed")
	assert.True(t, agentSwitched, "Agent did not switch to EnglishAgent")
}

func TestCompleteStreamingEvents(t *testing.T) {
	model := newStreamEventsFakeModel(
		[]agents.TResponseOutputItem{
			getReasoningOutputItem(),
			agentstesting.GetFunctionToolCall("foo", `{"arg": "value"}`),
		},
		[]agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Final response"),
		},
	)
	agent := &agents.Agent{
		Name:  "TestAgent",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("foo", "success!"),
		},
	}

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)

	var events []agents.StreamEvent
	err = result.StreamEvents(func(event agents.StreamEvent) error {
		events = append(events, event)
		return nil
	})
	require.NoError(t, err)

	require.Len(t, events, 27)

	agentEvent, ok := events[0].(agents.AgentUpdatedStreamEvent)
	require.True(t, ok)
	assert.Equal(t, "TestAgent", agentEvent.NewAgent.Name)

	requireRawEventType(t, events[1], "response.created")
	requireRawEventType(t, events[2], "response.in_progress")
	requireRawEventType(t, events[3], "response.output_item.added")
	requireRawEventType(t, events[4], "response.reasoning_summary_part.added")
	requireRawEventType(t, events[5], "response.reasoning_summary_text.delta")
	requireRawEventType(t, events[6], "response.reasoning_summary_text.done")
	requireRawEventType(t, events[7], "response.reasoning_summary_part.done")
	requireRawEventType(t, events[8], "response.output_item.done")

	runEvent, ok := events[9].(agents.RunItemStreamEvent)
	require.True(t, ok)
	assert.Equal(t, agents.StreamEventReasoningItemCreated, runEvent.Name)
	assert.IsType(t, agents.ReasoningItem{}, runEvent.Item)

	requireRawEventType(t, events[10], "response.output_item.added")
	requireRawEventType(t, events[11], "response.function_call_arguments.delta")
	requireRawEventType(t, events[12], "response.function_call_arguments.done")
	requireRawEventType(t, events[13], "response.output_item.done")

	runEvent, ok = events[14].(agents.RunItemStreamEvent)
	require.True(t, ok)
	assert.Equal(t, agents.StreamEventToolCalled, runEvent.Name)
	assert.IsType(t, agents.ToolCallItem{}, runEvent.Item)

	requireRawEventType(t, events[15], "response.completed")

	runEvent, ok = events[16].(agents.RunItemStreamEvent)
	require.True(t, ok)
	assert.Equal(t, agents.StreamEventToolOutput, runEvent.Name)
	assert.IsType(t, agents.ToolCallOutputItem{}, runEvent.Item)

	requireRawEventType(t, events[17], "response.created")
	requireRawEventType(t, events[18], "response.in_progress")
	requireRawEventType(t, events[19], "response.output_item.added")
	requireRawEventType(t, events[20], "response.content_part.added")
	requireRawEventType(t, events[21], "response.output_text.delta")
	requireRawEventType(t, events[22], "response.output_text.done")
	requireRawEventType(t, events[23], "response.content_part.done")
	requireRawEventType(t, events[24], "response.output_item.done")
	requireRawEventType(t, events[25], "response.completed")

	runEvent, ok = events[26].(agents.RunItemStreamEvent)
	require.True(t, ok)
	assert.Equal(t, agents.StreamEventMessageOutputCreated, runEvent.Name)
	assert.IsType(t, agents.MessageOutputItem{}, runEvent.Item)
}

func requireRawEventType(t *testing.T, event agents.StreamEvent, expected string) {
	t.Helper()
	raw, ok := event.(agents.RawResponsesStreamEvent)
	require.True(t, ok)
	assert.Equal(t, expected, raw.Data.Type)
}

func getReasoningOutputItem() responses.ResponseOutputItemUnion {
	return responses.ResponseOutputItemUnion{
		ID:   "rid",
		Type: "reasoning",
		Summary: []responses.ResponseReasoningItemSummary{
			{
				Text: "thinking",
				Type: constant.ValueOf[constant.SummaryText](),
			},
		},
	}
}

type streamEventsFakeModel struct {
	outputs [][]agents.TResponseOutputItem
	index   int
}

func newStreamEventsFakeModel(outputs ...[]agents.TResponseOutputItem) *streamEventsFakeModel {
	return &streamEventsFakeModel{outputs: outputs}
}

func (m *streamEventsFakeModel) nextOutput() []agents.TResponseOutputItem {
	if m.index >= len(m.outputs) {
		return nil
	}
	output := m.outputs[m.index]
	m.index++
	return output
}

func (m *streamEventsFakeModel) GetResponse(context.Context, agents.ModelResponseParams) (*agents.ModelResponse, error) {
	output := m.nextOutput()
	return &agents.ModelResponse{
		Output:     output,
		Usage:      usage.NewUsage(),
		ResponseID: "resp-test",
	}, nil
}

func (m *streamEventsFakeModel) StreamResponse(
	ctx context.Context,
	_ agents.ModelResponseParams,
	yield agents.ModelStreamResponseCallback,
) error {
	output := m.nextOutput()
	response := agentstesting.GetResponseObj(output, "resp-test", nil)
	sequence := int64(0)
	emit := func(event responses.ResponseStreamEventUnion) error {
		event.SequenceNumber = sequence
		sequence++
		return yield(ctx, event)
	}

	if err := emit(responses.ResponseStreamEventUnion{
		Type:     "response.created",
		Response: response,
	}); err != nil {
		return err
	}
	if err := emit(responses.ResponseStreamEventUnion{
		Type:     "response.in_progress",
		Response: response,
	}); err != nil {
		return err
	}

	for outputIndex, outputItem := range output {
		outputIndex := int64(outputIndex)
		if err := emit(responses.ResponseStreamEventUnion{
			Type:        "response.output_item.added",
			Item:        outputItem,
			OutputIndex: outputIndex,
		}); err != nil {
			return err
		}

		switch outputItem.Type {
		case "reasoning":
			for summaryIndex, summary := range outputItem.Summary {
				part := responses.ResponseStreamEventUnionPart{
					Text: summary.Text,
					Type: string(summary.Type),
				}
				if err := emit(responses.ResponseStreamEventUnion{
					Type:         "response.reasoning_summary_part.added",
					ItemID:       outputItem.ID,
					OutputIndex:  outputIndex,
					SummaryIndex: int64(summaryIndex),
					Part:         part,
				}); err != nil {
					return err
				}
				if err := emit(responses.ResponseStreamEventUnion{
					Type:         "response.reasoning_summary_text.delta",
					ItemID:       outputItem.ID,
					OutputIndex:  outputIndex,
					SummaryIndex: int64(summaryIndex),
					Delta:        summary.Text,
				}); err != nil {
					return err
				}
				if err := emit(responses.ResponseStreamEventUnion{
					Type:         "response.reasoning_summary_text.done",
					ItemID:       outputItem.ID,
					OutputIndex:  outputIndex,
					SummaryIndex: int64(summaryIndex),
					Text:         summary.Text,
				}); err != nil {
					return err
				}
				if err := emit(responses.ResponseStreamEventUnion{
					Type:         "response.reasoning_summary_part.done",
					ItemID:       outputItem.ID,
					OutputIndex:  outputIndex,
					SummaryIndex: int64(summaryIndex),
					Part:         part,
				}); err != nil {
					return err
				}
			}
		case "function_call":
			if err := emit(responses.ResponseStreamEventUnion{
				Type:        "response.function_call_arguments.delta",
				ItemID:      outputItem.CallID,
				OutputIndex: outputIndex,
				Delta:       outputItem.Arguments,
			}); err != nil {
				return err
			}
			if err := emit(responses.ResponseStreamEventUnion{
				Type:        "response.function_call_arguments.done",
				ItemID:      outputItem.CallID,
				OutputIndex: outputIndex,
				Arguments:   outputItem.Arguments,
				Name:        outputItem.Name,
			}); err != nil {
				return err
			}
		case "message":
			for contentIndex, contentPart := range outputItem.Content {
				if contentPart.Type != "output_text" {
					continue
				}
				part := responses.ResponseStreamEventUnionPart{
					Text: contentPart.Text,
					Type: contentPart.Type,
				}
				if err := emit(responses.ResponseStreamEventUnion{
					Type:         "response.content_part.added",
					ItemID:       outputItem.ID,
					OutputIndex:  outputIndex,
					ContentIndex: int64(contentIndex),
					Part:         part,
				}); err != nil {
					return err
				}
				if err := emit(responses.ResponseStreamEventUnion{
					Type:         "response.output_text.delta",
					ItemID:       outputItem.ID,
					OutputIndex:  outputIndex,
					ContentIndex: int64(contentIndex),
					Delta:        contentPart.Text,
				}); err != nil {
					return err
				}
				if err := emit(responses.ResponseStreamEventUnion{
					Type:         "response.output_text.done",
					ItemID:       outputItem.ID,
					OutputIndex:  outputIndex,
					ContentIndex: int64(contentIndex),
					Text:         contentPart.Text,
				}); err != nil {
					return err
				}
				if err := emit(responses.ResponseStreamEventUnion{
					Type:         "response.content_part.done",
					ItemID:       outputItem.ID,
					OutputIndex:  outputIndex,
					ContentIndex: int64(contentIndex),
					Part:         part,
				}); err != nil {
					return err
				}
			}
		}

		if err := emit(responses.ResponseStreamEventUnion{
			Type:        "response.output_item.done",
			Item:        outputItem,
			OutputIndex: outputIndex,
		}); err != nil {
			return err
		}
	}

	if err := emit(responses.ResponseStreamEventUnion{
		Type:     "response.completed",
		Response: response,
	}); err != nil {
		return err
	}
	return nil
}
