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

package agents

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestChatCompletionsToolSchemaMaterializesIterables(t *testing.T) {
	called := 0
	requiredSeq := func(yield func(string) bool) {
		called++
		_ = yield("foo")
	}

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"foo": map[string]any{"type": "string"},
		},
		"required": requiredSeq,
	}

	tool := FunctionTool{
		Name:             "tool",
		ParamsJSONSchema: schema,
	}

	converted, err := ChatCmplConverter().ToolToOpenai(tool)
	require.NoError(t, err)
	require.NotNil(t, converted)

	funcDef := converted.GetFunction()
	require.NotNil(t, funcDef)
	require.Equal(t, 1, called)

	required, ok := funcDef.Parameters["required"].([]string)
	require.True(t, ok)
	require.Equal(t, []string{"foo"}, required)
}

func TestResponsesToolSchemaMaterializesIterables(t *testing.T) {
	called := 0
	requiredSeq := func(yield func(string) bool) {
		called++
		_ = yield("bar")
	}

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"bar": map[string]any{"type": "string"},
		},
		"required": requiredSeq,
	}

	tool := FunctionTool{
		Name:             "tool",
		ParamsJSONSchema: schema,
	}

	converted, _, err := responsesConverter{}.convertTool(context.Background(), tool)
	require.NoError(t, err)
	require.NotNil(t, converted)
	require.NotNil(t, converted.OfFunction)
	require.Equal(t, 1, called)

	required, ok := converted.OfFunction.Parameters["required"].([]string)
	require.True(t, ok)
	require.Equal(t, []string{"bar"}, required)
}
