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

package tool_output_trimmer

import (
	"encoding/json"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaults(t *testing.T) {
	trimmer := NewToolOutputTrimmer()
	assert.Equal(t, 2, trimmer.RecentTurns)
	assert.Equal(t, 500, trimmer.MaxOutputChars)
	assert.Equal(t, 200, trimmer.PreviewChars)
	assert.Nil(t, trimmer.TrimmableTools)
}

func TestValidation(t *testing.T) {
	trimmer := NewToolOutputTrimmer()
	trimmer.RecentTurns = 0
	assert.ErrorContains(t, trimmer.Validate(), "recent_turns must be >= 1")

	trimmer = NewToolOutputTrimmer()
	trimmer.MaxOutputChars = 0
	assert.ErrorContains(t, trimmer.Validate(), "max_output_chars must be >= 1")

	trimmer = NewToolOutputTrimmer()
	trimmer.PreviewChars = -1
	assert.ErrorContains(t, trimmer.Validate(), "preview_chars must be >= 0")

	trimmer = NewToolOutputTrimmer()
	trimmer.PreviewChars = 0
	assert.NoError(t, trimmer.Validate())
}

func TestFindRecentBoundary(t *testing.T) {
	trimmer := NewToolOutputTrimmer()
	assert.Equal(t, 0, trimmer.findRecentBoundary(nil))

	items := []agents.TResponseInputItem{userItem(t, "q1")}
	assert.Equal(t, 0, trimmer.findRecentBoundary(items))

	items = []agents.TResponseInputItem{
		userItem(t, "q1"),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		assistantItem(t, "a2"),
	}
	assert.Equal(t, 0, trimmer.findRecentBoundary(items))

	items = []agents.TResponseInputItem{
		userItem(t, "q1"),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		assistantItem(t, "a2"),
		userItem(t, "q3"),
		assistantItem(t, "a3"),
	}
	assert.Equal(t, 2, trimmer.findRecentBoundary(items))

	trimmer.RecentTurns = 3
	items = []agents.TResponseInputItem{
		userItem(t, "q1"),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		assistantItem(t, "a2"),
		userItem(t, "q3"),
		assistantItem(t, "a3"),
		userItem(t, "q4"),
		assistantItem(t, "a4"),
	}
	assert.Equal(t, 2, trimmer.findRecentBoundary(items))
}

func TestTrimmingBehavior(t *testing.T) {
	large := repeat("x", 1000)
	items := []agents.TResponseInputItem{
		userItem(t, "q"),
		functionCallItem(t, "c1", "search"),
		functionOutputItem(t, "c1", large),
		assistantItem(t, "a"),
	}

	trimmer := NewToolOutputTrimmer()
	result, err := trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	assert.Equal(t, large, outputAt(t, result, 2))

	items = []agents.TResponseInputItem{
		userItem(t, "q1"),
		functionCallItem(t, "c1", "search"),
		functionOutputItem(t, "c1", large),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		assistantItem(t, "a2"),
		userItem(t, "q3"),
		assistantItem(t, "a3"),
	}
	result, err = trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	trimmed := outputAt(t, result, 2)
	assert.Contains(t, trimmed, "[Trimmed:")
	assert.Contains(t, trimmed, "search")
	assert.Contains(t, trimmed, "1000 chars")
	assert.Less(t, len(trimmed), len(large))

	small := repeat("x", 100)
	items = []agents.TResponseInputItem{
		userItem(t, "q1"),
		functionCallItem(t, "c1", "search"),
		functionOutputItem(t, "c1", small),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		assistantItem(t, "a2"),
		userItem(t, "q3"),
		assistantItem(t, "a3"),
	}
	trimmer = NewToolOutputTrimmer()
	result, err = trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	assert.Equal(t, small, outputAt(t, result, 2))
}

func TestTrimmingAllowlist(t *testing.T) {
	large := repeat("x", 1000)
	items := []agents.TResponseInputItem{
		userItem(t, "q1"),
		functionCallItem(t, "c1", "search"),
		functionOutputItem(t, "c1", large),
		functionCallItem(t, "c2", "resolve_entity"),
		functionOutputItem(t, "c2", large),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		assistantItem(t, "a2"),
		userItem(t, "q3"),
		assistantItem(t, "a3"),
	}
	trimmer := NewToolOutputTrimmer()
	trimmer.TrimmableTools = map[string]struct{}{"search": {}}
	result, err := trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	assert.Contains(t, outputAt(t, result, 2), "[Trimmed:")
	assert.Equal(t, large, outputAt(t, result, 4))
}

func TestPreservesRecentLargeOutput(t *testing.T) {
	large := repeat("x", 1000)
	items := []agents.TResponseInputItem{
		userItem(t, "q1"),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		functionCallItem(t, "c1", "search"),
		functionOutputItem(t, "c1", large),
		assistantItem(t, "a2"),
		userItem(t, "q3"),
		assistantItem(t, "a3"),
	}
	trimmer := NewToolOutputTrimmer()
	result, err := trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	assert.Equal(t, large, outputAt(t, result, 4))
}

func TestDoesNotMutateOriginalItems(t *testing.T) {
	large := repeat("x", 1000)
	items := []agents.TResponseInputItem{
		userItem(t, "q1"),
		functionCallItem(t, "c1", "search"),
		functionOutputItem(t, "c1", large),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		assistantItem(t, "a2"),
		userItem(t, "q3"),
		assistantItem(t, "a3"),
	}
	before := marshalItems(t, items)
	trimmer := NewToolOutputTrimmer()
	_, err := trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	after := marshalItems(t, items)
	assert.Equal(t, before, after)
}

func TestPreservesInstructions(t *testing.T) {
	items := []agents.TResponseInputItem{userItem(t, "hi")}
	trimmer := NewToolOutputTrimmer()
	data := agents.CallModelData{
		ModelData: agents.ModelInputData{
			Input:        items,
			Instructions: param.NewOpt("Custom prompt"),
		},
	}
	result, err := trimmer.Filter(t.Context(), data)
	require.NoError(t, err)
	assert.Equal(t, "Custom prompt", result.Instructions.Value)
}

func TestMultipleOldOutputsTrimmed(t *testing.T) {
	large1 := repeat("a", 1000)
	large2 := repeat("b", 2000)
	items := []agents.TResponseInputItem{
		userItem(t, "q1"),
		functionCallItem(t, "c1", "search"),
		functionOutputItem(t, "c1", large1),
		functionCallItem(t, "c2", "execute"),
		functionOutputItem(t, "c2", large2),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		assistantItem(t, "a2"),
		userItem(t, "q3"),
		assistantItem(t, "a3"),
	}
	trimmer := NewToolOutputTrimmer()
	result, err := trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	assert.Contains(t, outputAt(t, result, 2), "[Trimmed:")
	assert.Contains(t, outputAt(t, result, 4), "[Trimmed:")
	assert.Contains(t, outputAt(t, result, 2), "search")
	assert.Contains(t, outputAt(t, result, 4), "execute")
}

func TestCustomPreviewChars(t *testing.T) {
	large := repeat("abcdefghij", 100) // 1000 chars
	items := []agents.TResponseInputItem{
		userItem(t, "q1"),
		functionCallItem(t, "c1", "search"),
		functionOutputItem(t, "c1", large),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		assistantItem(t, "a2"),
		userItem(t, "q3"),
		assistantItem(t, "a3"),
	}
	trimmer := NewToolOutputTrimmer()
	trimmer.PreviewChars = 50
	result, err := trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	trimmed := outputAt(t, result, 2)
	assert.Contains(t, trimmed, repeat("abcdefghij", 5))
}

func TestPreservesUserAndAssistantMessages(t *testing.T) {
	items := []agents.TResponseInputItem{
		userItem(t, "important"),
		assistantItem(t, repeat("detailed ", 100)),
		userItem(t, "follow up"),
		assistantItem(t, "another"),
		userItem(t, "final"),
		assistantItem(t, "done"),
	}
	trimmer := NewToolOutputTrimmer()
	result, err := trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	assert.Equal(t, marshalItems(t, items), marshalItems(t, result.Input))
}

func TestSlidingWindow(t *testing.T) {
	large := repeat("x", 1000)
	items := []agents.TResponseInputItem{
		userItem(t, "q1"),
		functionCallItem(t, "c1", "search"),
		functionOutputItem(t, "c1", large),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		functionCallItem(t, "c2", "search"),
		functionOutputItem(t, "c2", large),
		assistantItem(t, "a2"),
		userItem(t, "q3"),
		assistantItem(t, "a3"),
	}
	trimmer := NewToolOutputTrimmer()
	result, err := trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	assert.Contains(t, outputAt(t, result, 2), "[Trimmed:")
	assert.Equal(t, large, outputAt(t, result, 6))

	items = []agents.TResponseInputItem{
		userItem(t, "q1"),
		functionCallItem(t, "c1", "s"),
		functionOutputItem(t, "c1", large),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		functionCallItem(t, "c2", "s"),
		functionOutputItem(t, "c2", large),
		assistantItem(t, "a2"),
		userItem(t, "q3"),
		functionCallItem(t, "c3", "s"),
		functionOutputItem(t, "c3", large),
		assistantItem(t, "a3"),
		userItem(t, "q4"),
		assistantItem(t, "a4"),
	}
	result, err = trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	assert.Contains(t, outputAt(t, result, 2), "[Trimmed:")
	assert.Contains(t, outputAt(t, result, 6), "[Trimmed:")
	assert.Equal(t, large, outputAt(t, result, 10))
}

func TestEdgeCases(t *testing.T) {
	borderline := repeat("x", 501)
	items := []agents.TResponseInputItem{
		userItem(t, "q1"),
		functionCallItem(t, "c1", "search"),
		functionOutputItem(t, "c1", borderline),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		assistantItem(t, "a2"),
		userItem(t, "q3"),
		assistantItem(t, "a3"),
	}
	trimmer := NewToolOutputTrimmer()
	trimmer.MaxOutputChars = 500
	trimmer.PreviewChars = 490
	result, err := trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	assert.Equal(t, borderline, outputAt(t, result, 2))

	large := repeat("x", 1000)
	items = []agents.TResponseInputItem{
		userItem(t, "q1"),
		functionOutputItem(t, "orphan_id", large),
		assistantItem(t, "a1"),
		userItem(t, "q2"),
		assistantItem(t, "a2"),
		userItem(t, "q3"),
		assistantItem(t, "a3"),
	}
	trimmer = NewToolOutputTrimmer()
	result, err = trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	trimmed := outputAt(t, result, 1)
	assert.Contains(t, trimmed, "unknown_tool")
	assert.Contains(t, trimmed, "[Trimmed:")

	trimmer = NewToolOutputTrimmer()
	trimmer.TrimmableTools = map[string]struct{}{"search": {}}
	result, err = trimmer.Filter(t.Context(), makeData(items))
	require.NoError(t, err)
	assert.Equal(t, large, outputAt(t, result, 1))
}

func makeData(items []agents.TResponseInputItem) agents.CallModelData {
	return agents.CallModelData{
		ModelData: agents.ModelInputData{
			Input:        items,
			Instructions: param.NewOpt("You are helpful."),
		},
	}
}

func outputAt(t *testing.T, result *agents.ModelInputData, idx int) string {
	t.Helper()
	require.NotNil(t, result)
	require.True(t, idx < len(result.Input))
	payload, ok := inputItemPayload(result.Input[idx])
	require.True(t, ok)
	out, _ := payload["output"].(string)
	return out
}

func inputItemFromMap(t *testing.T, payload map[string]any) agents.TResponseInputItem {
	t.Helper()
	raw, err := json.Marshal(payload)
	require.NoError(t, err)
	var item agents.TResponseInputItem
	param.SetJSON(raw, &item)
	return item
}

func userItem(t *testing.T, text string) agents.TResponseInputItem {
	return inputItemFromMap(t, map[string]any{"role": "user", "content": text, "type": "message"})
}

func assistantItem(t *testing.T, text string) agents.TResponseInputItem {
	return inputItemFromMap(t, map[string]any{"role": "assistant", "content": text, "type": "message"})
}

func functionCallItem(t *testing.T, callID, name string) agents.TResponseInputItem {
	return inputItemFromMap(t, map[string]any{
		"type":      "function_call",
		"call_id":   callID,
		"name":      name,
		"arguments": "{}",
	})
}

func functionOutputItem(t *testing.T, callID, output string) agents.TResponseInputItem {
	return inputItemFromMap(t, map[string]any{
		"type":    "function_call_output",
		"call_id": callID,
		"output":  output,
	})
}

func marshalItems(t *testing.T, items []agents.TResponseInputItem) []string {
	t.Helper()
	out := make([]string, len(items))
	for i, item := range items {
		raw, err := item.MarshalJSON()
		require.NoError(t, err)
		out[i] = string(raw)
	}
	return out
}

func repeat(s string, n int) string {
	if n <= 0 {
		return ""
	}
	buf := make([]byte, 0, len(s)*n)
	for i := 0; i < n; i++ {
		buf = append(buf, s...)
	}
	return string(buf)
}
