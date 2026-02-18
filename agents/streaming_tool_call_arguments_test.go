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
	"encoding/json"
	"fmt"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/stretchr/testify/require"
)

type streamingFakeModel struct {
	turnOutputs [][]agents.TResponseOutputItem
	lastTurnArgs agentstesting.FakeModelLastTurnArgs
}

func (m *streamingFakeModel) SetNextOutput(output []agents.TResponseOutputItem) {
	m.turnOutputs = append(m.turnOutputs, output)
}

func (m *streamingFakeModel) getNextOutput() []agents.TResponseOutputItem {
	if len(m.turnOutputs) == 0 {
		return nil
	}
	out := m.turnOutputs[0]
	m.turnOutputs = m.turnOutputs[1:]
	return out
}

func (m *streamingFakeModel) GetResponse(context.Context, agents.ModelResponseParams) (*agents.ModelResponse, error) {
	return nil, fmt.Errorf("streaming fake model only supports StreamResponse")
}

func (m *streamingFakeModel) StreamResponse(
	ctx context.Context,
	params agents.ModelResponseParams,
	yield agents.ModelStreamResponseCallback,
) error {
	m.lastTurnArgs = agentstesting.FakeModelLastTurnArgs{
		SystemInstructions: params.SystemInstructions,
		Input:              params.Input,
		ModelSettings:      params.ModelSettings,
		Tools:              params.Tools,
		OutputType:         params.OutputType,
		PreviousResponseID: params.PreviousResponseID,
		ConversationID:     params.ConversationID,
	}

	output := m.getNextOutput()
	sequenceNumber := int64(0)

	for idx, item := range output {
		if item.Type != "function_call" {
			continue
		}

		emptyArgsItem := item
		emptyArgsItem.Arguments = ""

		if err := yield(ctx, agents.TResponseStreamEvent{
			Type:           "response.output_item.added",
			Item:           emptyArgsItem,
			OutputIndex:    int64(idx),
			SequenceNumber: sequenceNumber,
		}); err != nil {
			return err
		}
		sequenceNumber++

		if err := yield(ctx, agents.TResponseStreamEvent{
			Type:           "response.output_item.done",
			Item:           item,
			OutputIndex:    int64(idx),
			SequenceNumber: sequenceNumber,
		}); err != nil {
			return err
		}
		sequenceNumber++
	}

	response := agentstesting.GetResponseObj(output, "", nil)
	return yield(ctx, agents.TResponseStreamEvent{
		Type:           "response.completed",
		Response:       response,
		SequenceNumber: sequenceNumber,
	})
}

func collectToolCalledArgs(t *testing.T, result *agents.RunResultStreaming) []string {
	t.Helper()
	args := make([]string, 0)
	err := result.StreamEvents(func(event agents.StreamEvent) error {
		runEvent, ok := event.(agents.RunItemStreamEvent)
		if !ok || runEvent.Name != agents.StreamEventToolCalled {
			return nil
		}
		switch item := runEvent.Item.(type) {
		case agents.ToolCallItem:
			if raw, ok := item.RawItem.(agents.ResponseFunctionToolCall); ok {
				args = append(args, raw.Arguments)
			}
		case *agents.ToolCallItem:
			if item == nil {
				return nil
			}
			if raw, ok := item.RawItem.(agents.ResponseFunctionToolCall); ok {
				args = append(args, raw.Arguments)
			}
		}
		return nil
	})
	require.NoError(t, err)
	return args
}

func TestStreamingToolCallArgumentsNotEmpty(t *testing.T) {
	type Args struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	calculateSum := agents.NewFunctionTool("calculate_sum", "", func(_ context.Context, args Args) (string, error) {
		return fmt.Sprintf("%d", args.A+args.B), nil
	})

	model := &streamingFakeModel{}
	toolCall := agentstesting.GetFunctionToolCall("calculate_sum", `{"a": 5, "b": 3}`)
	toolCall.CallID = "call_123"
	model.SetNextOutput([]agents.TResponseOutputItem{toolCall})
	model.SetNextOutput([]agents.TResponseOutputItem{agentstesting.GetTextMessage("done")})

	agent := agents.New("TestAgent").WithModelInstance(model).WithTools(calculateSum)
	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Add 5 and 3")
	require.NoError(t, err)

	args := collectToolCalledArgs(t, result)
	require.Len(t, args, 1)
	require.NotEmpty(t, args[0])
	require.Equal(t, `{"a": 5, "b": 3}`, args[0])

	var parsed map[string]int
	require.NoError(t, json.Unmarshal([]byte(args[0]), &parsed))
	require.Equal(t, map[string]int{"a": 5, "b": 3}, parsed)
}

func TestStreamingToolCallArgumentsComplex(t *testing.T) {
	type Args struct {
		Name    string `json:"name"`
		Message string `json:"message"`
		Urgent  bool   `json:"urgent"`
	}
	formatMessage := agents.NewFunctionTool("format_message", "", func(_ context.Context, args Args) (string, error) {
		prefix := ""
		if args.Urgent {
			prefix = "URGENT: "
		}
		return fmt.Sprintf("%sHello %s, %s", prefix, args.Name, args.Message), nil
	})

	model := &streamingFakeModel{}
	toolCall := agentstesting.GetFunctionToolCall(
		"format_message",
		`{"name": "Alice", "message": "Your meeting is starting soon", "urgent": true}`,
	)
	toolCall.CallID = "call_456"
	model.SetNextOutput([]agents.TResponseOutputItem{toolCall})
	model.SetNextOutput([]agents.TResponseOutputItem{agentstesting.GetTextMessage("done")})

	agent := agents.New("TestAgent").WithModelInstance(model).WithTools(formatMessage)
	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Format a message for Alice")
	require.NoError(t, err)

	args := collectToolCalledArgs(t, result)
	require.Len(t, args, 1)
	require.NotEmpty(t, args[0])
	require.Equal(t, `{"name": "Alice", "message": "Your meeting is starting soon", "urgent": true}`, args[0])

	var parsed map[string]any
	require.NoError(t, json.Unmarshal([]byte(args[0]), &parsed))
	require.Equal(t, map[string]any{
		"name":    "Alice",
		"message": "Your meeting is starting soon",
		"urgent":  true,
	}, parsed)
}

func TestStreamingMultipleToolCallsArguments(t *testing.T) {
	type SumArgs struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	type MsgArgs struct {
		Name    string `json:"name"`
		Message string `json:"message"`
	}
	calculateSum := agents.NewFunctionTool("calculate_sum", "", func(_ context.Context, args SumArgs) (string, error) {
		return fmt.Sprintf("%d", args.A+args.B), nil
	})
	formatMessage := agents.NewFunctionTool("format_message", "", func(_ context.Context, args MsgArgs) (string, error) {
		return fmt.Sprintf("Hello %s, %s", args.Name, args.Message), nil
	})

	model := &streamingFakeModel{}
	call1 := agentstesting.GetFunctionToolCall("calculate_sum", `{"a": 10, "b": 20}`)
	call1.CallID = "call_1"
	call2 := agentstesting.GetFunctionToolCall("format_message", `{"name": "Bob", "message": "Test"}`)
	call2.CallID = "call_2"
	model.SetNextOutput([]agents.TResponseOutputItem{call1, call2})
	model.SetNextOutput([]agents.TResponseOutputItem{agentstesting.GetTextMessage("done")})

	agent := agents.New("TestAgent").WithModelInstance(model).WithTools(calculateSum, formatMessage)
	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Do some calculations")
	require.NoError(t, err)

	args := collectToolCalledArgs(t, result)
	require.Len(t, args, 2)
	require.Equal(t, `{"a": 10, "b": 20}`, args[0])
	require.Equal(t, `{"name": "Bob", "message": "Test"}`, args[1])
}

func TestStreamingToolCallWithEmptyArguments(t *testing.T) {
	getCurrentTime := agents.NewFunctionTool("get_current_time", "", func(context.Context, struct{}) (string, error) {
		return "2024-01-15 10:30:00", nil
	})

	model := &streamingFakeModel{}
	toolCall := agentstesting.GetFunctionToolCall("get_current_time", "{}")
	toolCall.CallID = "call_time"
	model.SetNextOutput([]agents.TResponseOutputItem{toolCall})
	model.SetNextOutput([]agents.TResponseOutputItem{agentstesting.GetTextMessage("done")})

	agent := agents.New("TestAgent").WithModelInstance(model).WithTools(getCurrentTime)
	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "What time is it?")
	require.NoError(t, err)

	args := collectToolCalledArgs(t, result)
	require.Len(t, args, 1)
	require.Equal(t, "{}", args[0])

	var parsed map[string]any
	require.NoError(t, json.Unmarshal([]byte(args[0]), &parsed))
	require.Empty(t, parsed)
}
