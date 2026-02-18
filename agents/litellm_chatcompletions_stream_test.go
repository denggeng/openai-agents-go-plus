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

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLiteLLMStreamResponseYieldsEventsForTextContent(t *testing.T) {
	type m = map[string]any
	chunk1 := m{
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"content": "He"}}},
	}
	chunk2 := m{
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"content": "llo"}}},
		"usage": m{
			"completion_tokens":         5,
			"prompt_tokens":             7,
			"total_tokens":              12,
			"prompt_tokens_details":     m{"cached_tokens": 6},
			"completion_tokens_details": m{"reasoning_tokens": 2},
		},
	}

	dummyClient := makeOpenaiClientWithStreamResponse(t, chunk1, chunk2)
	provider := agents.NewLiteLLMProvider(agents.LiteLLMProviderParams{
		OpenaiClient: &dummyClient,
	})
	model, err := provider.GetModel("gpt-4")
	require.NoError(t, err)

	var outputEvents []agents.TResponseStreamEvent
	err = model.StreamResponse(
		t.Context(),
		agents.ModelResponseParams{
			SystemInstructions: param.Opt[string]{},
			Input:              agents.InputString(""),
			ModelSettings:      modelsettings.ModelSettings{},
			Tools:              nil,
			OutputType:         nil,
			Handoffs:           nil,
			Tracing:            agents.ModelTracingDisabled,
			PreviousResponseID: "",
			Prompt:             responses.ResponsePromptParam{},
		},
		func(_ context.Context, event agents.TResponseStreamEvent) error {
			outputEvents = append(outputEvents, event)
			return nil
		},
	)
	require.NoError(t, err)
	require.Len(t, outputEvents, 8)

	assert.Equal(t, "response.created", outputEvents[0].Type)
	assert.Equal(t, "response.output_item.added", outputEvents[1].Type)
	assert.Equal(t, "response.content_part.added", outputEvents[2].Type)
	assert.Equal(t, "response.output_text.delta", outputEvents[3].Type)
	assert.Equal(t, "He", outputEvents[3].Delta)
	assert.Equal(t, "response.output_text.delta", outputEvents[4].Type)
	assert.Equal(t, "llo", outputEvents[4].Delta)
	assert.Equal(t, "response.content_part.done", outputEvents[5].Type)
	assert.Equal(t, "response.output_item.done", outputEvents[6].Type)
	assert.Equal(t, "response.completed", outputEvents[7].Type)

	completed := outputEvents[7].Response
	require.Len(t, completed.Output, 1)
	assert.Equal(t, "message", completed.Output[0].Type)
	assert.Equal(t, "output_text", completed.Output[0].Content[0].Type)
	assert.Equal(t, "Hello", completed.Output[0].Content[0].Text)

	assert.Equal(t, int64(7), completed.Usage.InputTokens)
	assert.Equal(t, int64(5), completed.Usage.OutputTokens)
	assert.Equal(t, int64(12), completed.Usage.TotalTokens)
	assert.Equal(t, int64(6), completed.Usage.InputTokensDetails.CachedTokens)
	assert.Equal(t, int64(2), completed.Usage.OutputTokensDetails.ReasoningTokens)
}

func TestLiteLLMStreamResponseYieldsEventsForRefusalContent(t *testing.T) {
	type m = map[string]any
	chunk1 := m{
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"refusal": "No"}}},
	}
	chunk2 := m{
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"refusal": "Thanks"}}},
		"usage":   m{"completion_tokens": 2, "prompt_tokens": 2, "total_tokens": 4},
	}

	dummyClient := makeOpenaiClientWithStreamResponse(t, chunk1, chunk2)
	provider := agents.NewLiteLLMProvider(agents.LiteLLMProviderParams{
		OpenaiClient: &dummyClient,
	})
	model, err := provider.GetModel("gpt-4")
	require.NoError(t, err)

	var outputEvents []agents.TResponseStreamEvent
	err = model.StreamResponse(
		t.Context(),
		agents.ModelResponseParams{
			SystemInstructions: param.Opt[string]{},
			Input:              agents.InputString(""),
			ModelSettings:      modelsettings.ModelSettings{},
			Tools:              nil,
			OutputType:         nil,
			Handoffs:           nil,
			Tracing:            agents.ModelTracingDisabled,
			PreviousResponseID: "",
			Prompt:             responses.ResponsePromptParam{},
		},
		func(_ context.Context, event agents.TResponseStreamEvent) error {
			outputEvents = append(outputEvents, event)
			return nil
		},
	)
	require.NoError(t, err)
	require.Len(t, outputEvents, 8)

	assert.Equal(t, "response.created", outputEvents[0].Type)
	assert.Equal(t, "response.output_item.added", outputEvents[1].Type)
	assert.Equal(t, "response.content_part.added", outputEvents[2].Type)
	assert.Equal(t, "response.refusal.delta", outputEvents[3].Type)
	assert.Equal(t, "No", outputEvents[3].Delta)
	assert.Equal(t, "response.refusal.delta", outputEvents[4].Type)
	assert.Equal(t, "Thanks", outputEvents[4].Delta)
	assert.Equal(t, "response.content_part.done", outputEvents[5].Type)
	assert.Equal(t, "response.output_item.done", outputEvents[6].Type)
	assert.Equal(t, "response.completed", outputEvents[7].Type)

	completed := outputEvents[7].Response
	require.Len(t, completed.Output, 1)
	assert.Equal(t, "message", completed.Output[0].Type)
	assert.Equal(t, "refusal", completed.Output[0].Content[0].Type)
	assert.Equal(t, "NoThanks", completed.Output[0].Content[0].Refusal)
}

func TestLiteLLMStreamResponseYieldsEventsForToolCall(t *testing.T) {
	type m = map[string]any
	toolCallDelta1 := m{
		"index": 0,
		"id":    "tool-id",
		"function": m{
			"name":      "my_func",
			"arguments": "arg1",
		},
		"type": "function",
	}
	toolCallDelta2 := m{
		"index": 0,
		"id":    "tool-id",
		"function": m{
			"arguments": "arg2",
		},
		"type": "function",
	}
	chunk1 := m{
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"tool_calls": []m{toolCallDelta1}}}},
	}
	chunk2 := m{
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"tool_calls": []m{toolCallDelta2}}}},
		"usage":   m{"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2},
	}

	dummyClient := makeOpenaiClientWithStreamResponse(t, chunk1, chunk2)
	provider := agents.NewLiteLLMProvider(agents.LiteLLMProviderParams{
		OpenaiClient: &dummyClient,
	})
	model, err := provider.GetModel("gpt-4")
	require.NoError(t, err)

	var outputEvents []agents.TResponseStreamEvent
	err = model.StreamResponse(
		t.Context(),
		agents.ModelResponseParams{
			SystemInstructions: param.Opt[string]{},
			Input:              agents.InputString(""),
			ModelSettings:      modelsettings.ModelSettings{},
			Tools:              nil,
			OutputType:         nil,
			Handoffs:           nil,
			Tracing:            agents.ModelTracingDisabled,
			PreviousResponseID: "",
			Prompt:             responses.ResponsePromptParam{},
		},
		func(_ context.Context, event agents.TResponseStreamEvent) error {
			outputEvents = append(outputEvents, event)
			return nil
		},
	)
	require.NoError(t, err)

	require.Len(t, outputEvents, 6)
	assert.Equal(t, "response.created", outputEvents[0].Type)
	assert.Equal(t, "response.output_item.added", outputEvents[1].Type)
	assert.Equal(t, "function_call", outputEvents[1].Item.Type)
	assert.Equal(t, "my_func", outputEvents[1].Item.Name)
	assert.Equal(t, "", outputEvents[1].Item.Arguments)
	assert.Equal(t, "response.function_call_arguments.delta", outputEvents[2].Type)
	assert.Equal(t, "arg1", outputEvents[2].Delta)
	assert.Equal(t, "response.function_call_arguments.delta", outputEvents[3].Type)
	assert.Equal(t, "arg2", outputEvents[3].Delta)
	assert.Equal(t, "response.output_item.done", outputEvents[4].Type)
	assert.Equal(t, "response.completed", outputEvents[5].Type)

	finalFn := outputEvents[4].Item
	assert.Equal(t, "my_func", finalFn.Name)
	assert.Equal(t, "arg1arg2", finalFn.Arguments)
}

func TestLiteLLMStreamResponseYieldsRealTimeFunctionCallArguments(t *testing.T) {
	type m = map[string]any
	toolCallDelta1 := m{
		"index": 0,
		"id":    "litellm-call-456",
		"function": m{
			"name":      "generate_code",
			"arguments": "",
		},
		"type": "function",
	}
	toolCallDelta2 := m{
		"index": 0,
		"function": m{
			"arguments": `{"language": "`,
		},
		"type": "function",
	}
	toolCallDelta3 := m{
		"index": 0,
		"function": m{
			"arguments": `python", "task": "`,
		},
		"type": "function",
	}
	toolCallDelta4 := m{
		"index": 0,
		"function": m{
			"arguments": `hello world"}`,
		},
		"type": "function",
	}
	chunk1 := m{
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"tool_calls": []m{toolCallDelta1}}}},
	}
	chunk2 := m{
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"tool_calls": []m{toolCallDelta2}}}},
	}
	chunk3 := m{
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"tool_calls": []m{toolCallDelta3}}}},
	}
	chunk4 := m{
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"tool_calls": []m{toolCallDelta4}}}},
		"usage":   m{"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2},
	}

	dummyClient := makeOpenaiClientWithStreamResponse(t, chunk1, chunk2, chunk3, chunk4)
	provider := agents.NewLiteLLMProvider(agents.LiteLLMProviderParams{
		OpenaiClient: &dummyClient,
	})
	model, err := provider.GetModel("gpt-4")
	require.NoError(t, err)

	var outputEvents []agents.TResponseStreamEvent
	err = model.StreamResponse(
		t.Context(),
		agents.ModelResponseParams{
			SystemInstructions: param.Opt[string]{},
			Input:              agents.InputString(""),
			ModelSettings:      modelsettings.ModelSettings{},
			Tools:              nil,
			OutputType:         nil,
			Handoffs:           nil,
			Tracing:            agents.ModelTracingDisabled,
			PreviousResponseID: "",
			Prompt:             responses.ResponsePromptParam{},
		},
		func(_ context.Context, event agents.TResponseStreamEvent) error {
			outputEvents = append(outputEvents, event)
			return nil
		},
	)
	require.NoError(t, err)

	var deltaEvents []agents.TResponseStreamEvent
	var addedEvents []agents.TResponseStreamEvent
	for _, event := range outputEvents {
		switch event.Type {
		case "response.function_call_arguments.delta":
			deltaEvents = append(deltaEvents, event)
		case "response.output_item.added":
			addedEvents = append(addedEvents, event)
		}
	}

	require.Len(t, deltaEvents, 3)
	require.Len(t, addedEvents, 1)

	expectedDeltas := []string{`{"language": "`, `python", "task": "`, `hello world"}`}
	for i, event := range deltaEvents {
		assert.Equal(t, expectedDeltas[i], event.Delta)
	}

	assert.Equal(t, "function_call", addedEvents[0].Item.Type)
	assert.Equal(t, "generate_code", addedEvents[0].Item.Name)
	assert.Equal(t, "litellm-call-456", addedEvents[0].Item.CallID)
}
