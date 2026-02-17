package agents_test

import (
	"context"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunErrorIncludesData(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("test").
		WithModelInstance(model).
		WithTools(agentstesting.GetFunctionTool("foo", "res"))

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("1"),
			agentstesting.GetFunctionToolCall("foo", `{"a": "b"}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	runner := agents.Runner{Config: agents.RunConfig{MaxTurns: 1}}
	_, err := runner.Run(t.Context(), agent, "hello")

	var target agents.MaxTurnsExceededError
	require.ErrorAs(t, err, &target)
	data := target.AgentsError.RunData
	require.NotNil(t, data)
	assert.Same(t, agent, data.LastAgent)
	assert.Len(t, data.RawResponses, 1)
	assert.NotEmpty(t, data.NewItems)
	assert.Empty(t, data.ToolInputGuardrailResults)
	assert.Empty(t, data.ToolOutputGuardrailResults)
}

func TestStreamedRunErrorIncludesData(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("test").
		WithModelInstance(model).
		WithTools(agentstesting.GetFunctionTool("foo", "res"))

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("1"),
			agentstesting.GetFunctionToolCall("foo", `{"a": "b"}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	runner := agents.Runner{Config: agents.RunConfig{MaxTurns: 1}}
	result, err := runner.RunStreamed(t.Context(), agent, "hello")
	require.NoError(t, err)

	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })

	var target agents.MaxTurnsExceededError
	require.ErrorAs(t, err, &target)
	data := target.AgentsError.RunData
	require.NotNil(t, data)
	assert.Same(t, agent, data.LastAgent)
	assert.Len(t, data.RawResponses, 1)
	assert.NotEmpty(t, data.NewItems)
	assert.Empty(t, data.ToolInputGuardrailResults)
	assert.Empty(t, data.ToolOutputGuardrailResults)
}

func TestRunErrorIncludesToolGuardrailData(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)

	tool := agentstesting.GetFunctionTool("foo", "res")
	tool.ToolInputGuardrails = []agents.ToolInputGuardrail{
		{
			Name: "tool_input_guard",
			GuardrailFunction: func(context.Context, agents.ToolInputGuardrailData) (agents.ToolGuardrailFunctionOutput, error) {
				return agents.ToolGuardrailAllow("input_checked"), nil
			},
		},
	}
	tool.ToolOutputGuardrails = []agents.ToolOutputGuardrail{
		{
			Name: "tool_output_guard",
			GuardrailFunction: func(context.Context, agents.ToolOutputGuardrailData) (agents.ToolGuardrailFunctionOutput, error) {
				return agents.ToolGuardrailAllow("output_checked"), nil
			},
		},
	}

	agent := agents.New("test").
		WithModelInstance(model).
		WithTools(tool)

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("foo", `{"a":"b"}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	runner := agents.Runner{Config: agents.RunConfig{MaxTurns: 1}}
	_, err := runner.Run(t.Context(), agent, "hello")

	var target agents.MaxTurnsExceededError
	require.ErrorAs(t, err, &target)
	data := target.AgentsError.RunData
	require.NotNil(t, data)
	require.Len(t, data.ToolInputGuardrailResults, 1)
	require.Len(t, data.ToolOutputGuardrailResults, 1)
}
