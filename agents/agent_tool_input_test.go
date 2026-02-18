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
	"encoding/json"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/stretchr/testify/require"
)

func TestAgentAsToolInputSchemaAcceptsString(t *testing.T) {
	_, err := agents.ParseAgentAsToolInput(map[string]any{"input": "hi"})
	require.NoError(t, err)

	_, err = agents.ParseAgentAsToolInput(map[string]any{"input": []any{}})
	require.Error(t, err)
}

func TestResolveAgentToolInputReturnsStringInput(t *testing.T) {
	result, err := agents.ResolveAgentToolInput(map[string]any{"input": "hello"}, nil, nil)
	require.NoError(t, err)
	require.Equal(t, "hello", result)
}

func TestResolveAgentToolInputFallsBackToJSON(t *testing.T) {
	params := map[string]any{"foo": "bar"}
	result, err := agents.ResolveAgentToolInput(params, nil, nil)
	require.NoError(t, err)

	expected, err := json.Marshal(params)
	require.NoError(t, err)
	require.Equal(t, string(expected), result)
}

func TestResolveAgentToolInputPreservesInputWithExtraFields(t *testing.T) {
	params := map[string]any{"input": "hello", "target": "world"}
	result, err := agents.ResolveAgentToolInput(params, nil, nil)
	require.NoError(t, err)

	expected, err := json.Marshal(params)
	require.NoError(t, err)
	require.Equal(t, string(expected), result)
}

func TestResolveAgentToolInputUsesDefaultBuilderWhenSchemaInfoExists(t *testing.T) {
	schemaInfo := &agents.StructuredInputSchemaInfo{Summary: "Summary"}
	result, err := agents.ResolveAgentToolInput(map[string]any{"foo": "bar"}, schemaInfo, nil)
	require.NoError(t, err)

	text, ok := result.(string)
	require.True(t, ok)
	require.Contains(t, text, "Input Schema Summary:")
	require.Contains(t, text, "Summary")
}

func TestResolveAgentToolInputReturnsBuilderItems(t *testing.T) {
	items := []agents.TResponseInputItem{agentstesting.GetTextInputItem("custom input")}

	builder := func(_ agents.StructuredToolInputBuilderOptions) (agents.StructuredToolInputResult, error) {
		return items, nil
	}

	result, err := agents.ResolveAgentToolInput(map[string]any{"input": "ignored"}, nil, builder)
	require.NoError(t, err)
	require.Equal(t, items, result)
}
