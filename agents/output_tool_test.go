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

package agents_test

import (
	"context"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type outputToolFoo struct {
	Bar string `json:"bar"`
}

func TestOutputToolPlainText(t *testing.T) {
	agent := agents.Agent{Name: "test"}
	assert.Nil(t, agent.OutputType)

	outputType := agents.OutputType[string]()
	_, err := outputType.JSONSchema()
	require.ErrorAs(t, err, &agents.UserError{})
}

func TestOutputToolStructuredOutput(t *testing.T) {
	outputType := agents.OutputType[outputToolFoo]()

	schema, err := outputType.JSONSchema()
	require.NoError(t, err)
	assert.Equal(t, "object", schema["type"])

	validated, err := outputType.ValidateJSON(t.Context(), `{"bar": "baz"}`)
	require.NoError(t, err)
	assert.Equal(t, outputToolFoo{Bar: "baz"}, validated)
}

func TestOutputToolListOutput(t *testing.T) {
	outputType := agents.OutputType[[]string]()

	schema, err := outputType.JSONSchema()
	require.NoError(t, err)
	props, ok := schema["properties"].(map[string]any)
	require.True(t, ok)
	_, ok = props["response"]
	assert.True(t, ok)

	validated, err := outputType.ValidateJSON(t.Context(), `{"response": ["foo", "bar"]}`)
	require.NoError(t, err)
	assert.Equal(t, []string{"foo", "bar"}, validated)
}

func TestOutputToolBadJSONRaisesError(t *testing.T) {
	outputType := agents.OutputType[outputToolFoo]()

	_, err := outputType.ValidateJSON(t.Context(), "not valid json")
	require.ErrorAs(t, err, &agents.ModelBehaviorError{})
}

type outputToolCustom struct{}

func (outputToolCustom) IsPlainText() bool { return false }
func (outputToolCustom) Name() string      { return "FooBarBaz" }
func (outputToolCustom) JSONSchema() (map[string]any, error) {
	return map[string]any{"type": "object"}, nil
}
func (outputToolCustom) IsStrictJSONSchema() bool { return false }
func (outputToolCustom) ValidateJSON(context.Context, string) (any, error) {
	return []string{"some", "output"}, nil
}

func TestOutputToolCustomOutputSchema(t *testing.T) {
	outputType := outputToolCustom{}

	assert.False(t, outputType.IsPlainText())
	assert.Equal(t, "FooBarBaz", outputType.Name())
	assert.False(t, outputType.IsStrictJSONSchema())

	schema, err := outputType.JSONSchema()
	require.NoError(t, err)
	assert.Equal(t, "object", schema["type"])

	validated, err := outputType.ValidateJSON(context.Background(), `{"foo":"bar"}`)
	require.NoError(t, err)
	assert.Equal(t, []string{"some", "output"}, validated)
}
