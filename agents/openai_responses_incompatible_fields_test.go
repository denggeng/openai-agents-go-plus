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
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRemoveOpenAIResponsesAPIIncompatibleFieldsReturnsUnchangedWhenNoProviderData(t *testing.T) {
	model := OpenAIResponsesModel{}
	listInput := []map[string]any{
		{"type": "message", "content": "hello"},
		{"type": "function_call", "call_id": "call_123", "name": "test"},
	}

	result := model.removeOpenAIResponsesAPIIncompatibleFields(listInput)

	require.Equal(t, listInput, result)
	if len(listInput) > 0 {
		require.Equal(t, &listInput[0], &result[0])
	}
}

func TestRemoveOpenAIResponsesAPIIncompatibleFieldsRemovesReasoningItemsWithProviderData(t *testing.T) {
	model := OpenAIResponsesModel{}
	listInput := []map[string]any{
		{"type": "message", "content": "hello"},
		{"type": "reasoning", "provider_data": map[string]any{"model": "gemini/gemini-3"}},
		{"type": "function_call", "call_id": "call_123"},
	}

	result := model.removeOpenAIResponsesAPIIncompatibleFields(listInput)

	require.Len(t, result, 2)
	require.Equal(t, map[string]any{"type": "message", "content": "hello"}, result[0])
	require.Equal(t, map[string]any{"type": "function_call", "call_id": "call_123"}, result[1])
}

func TestRemoveOpenAIResponsesAPIIncompatibleFieldsKeepsReasoningItemsWithoutProviderData(t *testing.T) {
	model := OpenAIResponsesModel{}
	listInput := []map[string]any{
		{"type": "reasoning", "summary": []any{}},
		{"type": "message", "content": "hello", "provider_data": map[string]any{"foo": "bar"}},
	}

	result := model.removeOpenAIResponsesAPIIncompatibleFields(listInput)

	require.Len(t, result, 2)
	require.Equal(t, map[string]any{"type": "reasoning", "summary": []any{}}, result[0])
	require.Equal(t, map[string]any{"type": "message", "content": "hello"}, result[1])
}

func TestRemoveOpenAIResponsesAPIIncompatibleFieldsRemovesProviderDataFromAllItems(t *testing.T) {
	model := OpenAIResponsesModel{}
	listInput := []map[string]any{
		{"type": "message", "content": "hello", "provider_data": map[string]any{"model": "gemini/gemini-3"}},
		{"type": "function_call", "call_id": "call_123", "provider_data": map[string]any{"model": "gemini/gemini-3"}},
	}

	result := model.removeOpenAIResponsesAPIIncompatibleFields(listInput)

	require.Len(t, result, 2)
	_, ok := result[0]["provider_data"]
	require.False(t, ok)
	_, ok = result[1]["provider_data"]
	require.False(t, ok)
}

func TestRemoveOpenAIResponsesAPIIncompatibleFieldsRemovesFakeResponsesID(t *testing.T) {
	model := OpenAIResponsesModel{}
	listInput := []map[string]any{
		{"type": "message", "id": FakeResponsesID, "content": "hello", "provider_data": map[string]any{"model": "gemini/gemini-3"}},
	}

	result := model.removeOpenAIResponsesAPIIncompatibleFields(listInput)

	require.Len(t, result, 1)
	_, ok := result[0]["id"]
	require.False(t, ok)
	require.Equal(t, "hello", result[0]["content"])
}

func TestRemoveOpenAIResponsesAPIIncompatibleFieldsPreservesRealIDs(t *testing.T) {
	model := OpenAIResponsesModel{}
	listInput := []map[string]any{
		{"type": "message", "id": "msg_real123", "content": "hello", "provider_data": map[string]any{}},
	}

	result := model.removeOpenAIResponsesAPIIncompatibleFields(listInput)

	require.Equal(t, "msg_real123", result[0]["id"])
}

func TestRemoveOpenAIResponsesAPIIncompatibleFieldsHandlesEmptyList(t *testing.T) {
	model := OpenAIResponsesModel{}
	listInput := []map[string]any{}

	result := model.removeOpenAIResponsesAPIIncompatibleFields(listInput)

	require.Equal(t, []map[string]any{}, result)
}

func TestRemoveOpenAIResponsesAPIIncompatibleFieldsCombinedScenario(t *testing.T) {
	model := OpenAIResponsesModel{}
	listInput := []map[string]any{
		{"type": "message", "content": "user input"},
		{"type": "reasoning", "summary": []any{}, "provider_data": map[string]any{"model": "gemini/gemini-3"}},
		{"type": "function_call", "call_id": "call_abc_123", "name": "get_weather", "provider_data": map[string]any{"model": "gemini/gemini-3"}},
		{"type": "function_call_output", "call_id": "call_abc_123", "output": `{"temp": 72}`},
		{"type": "message", "id": FakeResponsesID, "content": "The weather is 72F", "provider_data": map[string]any{"model": "gemini/gemini-3"}},
	}

	result := model.removeOpenAIResponsesAPIIncompatibleFields(listInput)

	require.Len(t, result, 4)
	require.Equal(t, map[string]any{"type": "message", "content": "user input"}, result[0])
	require.Equal(t, "function_call", result[1]["type"])
	require.Equal(t, "call_abc_123", result[1]["call_id"])
	_, ok := result[1]["provider_data"]
	require.False(t, ok)

	require.Equal(t, "function_call_output", result[2]["type"])
	require.Equal(t, "call_abc_123", result[2]["call_id"])

	require.Equal(t, "message", result[3]["type"])
	require.Equal(t, "The weather is 72F", result[3]["content"])
	_, ok = result[3]["id"]
	require.False(t, ok)
	_, ok = result[3]["provider_data"]
	require.False(t, ok)
}
