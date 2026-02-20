package agents

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type customSchema struct{}

func (customSchema) IsPlainText() bool { return false }
func (customSchema) Name() string      { return "CustomSchema" }
func (customSchema) JSONSchema() (map[string]any, error) {
	return map[string]any{"type": "object"}, nil
}
func (customSchema) IsStrictJSONSchema() bool { return true }
func (customSchema) ValidateJSON(_ context.Context, jsonStr string) (any, error) {
	var out any
	if err := json.Unmarshal([]byte(jsonStr), &out); err != nil {
		return nil, err
	}
	return out, nil
}

func TestFormatFinalOutputTextHandlesWrappedPayload(t *testing.T) {
	agent := &Agent{Name: "wrapped-output", OutputType: OutputType[[]string]()}
	output := map[string]any{"response": []string{"a", "b"}}

	rendered := formatFinalOutputText(agent, output)
	var decoded map[string]any
	require.NoError(t, json.Unmarshal([]byte(rendered), &decoded))

	expected := map[string]any{}
	expectedBytes, _ := json.Marshal(output)
	require.NoError(t, json.Unmarshal(expectedBytes, &expected))
	assert.Equal(t, expected, decoded)
}

func TestValidateHandlerFinalOutputAcceptsWrappedPayload(t *testing.T) {
	agent := &Agent{Name: "wrapped-validate", OutputType: OutputType[[]string]()}
	output := map[string]any{"response": []string{"ok"}}

	validated, err := validateHandlerFinalOutput(context.Background(), agent, output)
	require.NoError(t, err)
	assert.Equal(t, []string{"ok"}, validated)
}

func TestFormatFinalOutputTextUsesCustomSchemaAndFallback(t *testing.T) {
	agent := &Agent{Name: "custom-format", OutputType: customSchema{}}

	rendered := formatFinalOutputText(agent, map[string]any{"ok": true})
	var decoded map[string]any
	require.NoError(t, json.Unmarshal([]byte(rendered), &decoded))
	assert.Equal(t, map[string]any{"ok": true}, decoded)

	value := make(chan int)
	fallback := formatFinalOutputText(agent, value)
	assert.Equal(t, fmt.Sprint(value), fallback)
}

func TestValidateHandlerFinalOutputRaisesForUnserializableData(t *testing.T) {
	agent := &Agent{Name: "custom-validate", OutputType: customSchema{}}
	bad := map[string]any{
		"bad": make(chan int),
	}

	_, err := validateHandlerFinalOutput(context.Background(), agent, bad)
	var userErr UserError
	assert.ErrorAs(t, err, &userErr)
	assert.Contains(t, err.Error(), "Invalid run error handler final_output")
}

func TestResolveRunErrorHandlerResultCoversValidationPaths(t *testing.T) {
	agent := &Agent{Name: "max-turns"}
	contextWrapper := NewRunContextWrapper[any](nil)
	runData := RunErrorData{
		Input:     InputString("hello"),
		NewItems:  nil,
		History:   nil,
		Output:    nil,
		LastAgent: agent,
	}
	maxTurnsErr := MaxTurnsExceededErrorf("too many turns")

	noHandler, err := resolveRunErrorHandlerResult(
		context.Background(),
		RunErrorHandlers{},
		maxTurnsErr,
		contextWrapper,
		runData,
	)
	require.NoError(t, err)
	assert.Nil(t, noHandler)

	handlerResult, err := resolveRunErrorHandlerResult(
		context.Background(),
		RunErrorHandlers{
			MaxTurns: func(context.Context, RunErrorHandlerInput) (any, error) {
				return nil, nil
			},
		},
		maxTurnsErr,
		contextWrapper,
		runData,
	)
	require.NoError(t, err)
	assert.Nil(t, handlerResult)

	_, err = resolveRunErrorHandlerResult(
		context.Background(),
		RunErrorHandlers{
			MaxTurns: func(context.Context, RunErrorHandlerInput) (any, error) {
				return map[string]any{"final_output": "x", "extra": "y"}, nil
			},
		},
		maxTurnsErr,
		contextWrapper,
		runData,
	)
	var userErr UserError
	assert.ErrorAs(t, err, &userErr)
	assert.Contains(t, err.Error(), "Invalid run error handler result")
}

func TestResolveRunErrorHandlerResultAcceptsRunErrorHandlerResult(t *testing.T) {
	agent := &Agent{Name: "max-turns"}
	contextWrapper := NewRunContextWrapper[any](nil)
	runData := RunErrorData{Input: InputString("hello"), LastAgent: agent}
	err := MaxTurnsExceededErrorf("too many turns")

	include := false
	result, handlerErr := resolveRunErrorHandlerResult(
		context.Background(),
		RunErrorHandlers{
			MaxTurns: func(context.Context, RunErrorHandlerInput) (any, error) {
				return RunErrorHandlerResult{
					FinalOutput:      "done",
					IncludeInHistory: &include,
				}, nil
			},
		},
		err,
		contextWrapper,
		runData,
	)
	require.NoError(t, handlerErr)
	require.NotNil(t, result)
	assert.Equal(t, "done", result.FinalOutput)
	assert.False(t, includeInHistoryDefault(result.IncludeInHistory))
}

func TestValidateHandlerFinalOutputPropagatesModelErrors(t *testing.T) {
	agent := &Agent{Name: "bad-schema", OutputType: badSchema{}}
	_, err := validateHandlerFinalOutput(context.Background(), agent, map[string]any{"x": "y"})
	var userErr UserError
	assert.ErrorAs(t, err, &userErr)
}

type badSchema struct{}

func (badSchema) IsPlainText() bool { return false }
func (badSchema) Name() string      { return "BadSchema" }
func (badSchema) JSONSchema() (map[string]any, error) {
	return map[string]any{"type": "object"}, nil
}
func (badSchema) IsStrictJSONSchema() bool { return true }
func (badSchema) ValidateJSON(context.Context, string) (any, error) {
	return nil, errors.New("bad json")
}
