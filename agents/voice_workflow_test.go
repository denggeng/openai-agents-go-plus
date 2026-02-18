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
	"context"
	"fmt"
	"iter"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/openaitypes"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeStreamingModel struct {
	turnOutputs [][]TResponseOutputItem
}

func (m *fakeStreamingModel) setNextOutput(output []TResponseOutputItem) {
	m.turnOutputs = append(m.turnOutputs, output)
}

func (m *fakeStreamingModel) addMultipleTurnOutputs(outputs [][]TResponseOutputItem) {
	m.turnOutputs = append(m.turnOutputs, outputs...)
}

func (m *fakeStreamingModel) getNextOutput() []TResponseOutputItem {
	if len(m.turnOutputs) == 0 {
		return nil
	}
	out := m.turnOutputs[0]
	m.turnOutputs = m.turnOutputs[1:]
	return out
}

func (m *fakeStreamingModel) GetResponse(context.Context, ModelResponseParams) (*ModelResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *fakeStreamingModel) StreamResponse(
	ctx context.Context,
	_ ModelResponseParams,
	yield ModelStreamResponseCallback,
) error {
	output := m.getNextOutput()
	seq := int64(0)
	for _, item := range output {
		if item.Type != "message" || len(item.Content) != 1 {
			continue
		}
		part := item.Content[0]
		if part.Type != "output_text" {
			continue
		}
		seq++
		event := responses.ResponseStreamEventUnion{
			Type:           "response.output_text.delta",
			Delta:          part.Text,
			ContentIndex:   0,
			OutputIndex:    0,
			ItemID:         item.ID,
			SequenceNumber: seq,
		}
		if err := yield(ctx, event); err != nil {
			return err
		}
	}

	response := responses.Response{
		ID:     "resp-test",
		Output: output,
	}
	seq++
	return yield(ctx, responses.ResponseStreamEventUnion{
		Type:           "response.completed",
		Response:       response,
		SequenceNumber: seq,
	})
}

func voiceWorkflowFunctionTool(name, output string) FunctionTool {
	return FunctionTool{
		Name: name,
		ParamsJSONSchema: map[string]any{
			"title":                name + "_args",
			"type":                 "object",
			"required":             []string{},
			"additionalProperties": false,
			"properties":           map[string]any{},
		},
		OnInvokeTool: func(context.Context, string) (any, error) {
			return output, nil
		},
	}
}

type voiceWorkflowRunResult struct {
	ctx    context.Context
	result VoiceWorkflowBaseRunResult
}

func (r *voiceWorkflowRunResult) Seq() iter.Seq[string] {
	return r.result.Seq()
}

func (r *voiceWorkflowRunResult) Error() error { return r.result.Error() }

func TestSingleAgentVoiceWorkflow(t *testing.T) {
	model := &fakeStreamingModel{}
	model.addMultipleTurnOutputs([][]TResponseOutputItem{
		{
			getFunctionToolCall("some_function", `{"a": "b"}`, ""),
			getTextMessage("a_message"),
		},
		{
			getTextMessage("done"),
		},
	})

	agent := &Agent{
		Name:  "initial_agent",
		Model: param.NewOpt(NewAgentModel(model)),
		Tools: []Tool{voiceWorkflowFunctionTool("some_function", "tool_result")},
	}

	workflow := NewSingleAgentVoiceWorkflow(agent, nil)
	var output []string
	run := workflow.Run(context.Background(), "transcription_1")
	for chunk := range run.Seq() {
		output = append(output, chunk)
	}
	require.NoError(t, run.Error())

	assert.Equal(t, []string{"a_message", "done"}, output)

	expected := []TResponseInputItem{
		{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt("transcription_1"),
				},
				Role: responses.EasyInputMessageRoleUser,
				Type: responses.EasyInputMessageTypeMessage,
			},
		},
		openaitypes.ResponseInputItemUnionParamFromResponseOutputItemUnion(
			getFunctionToolCall("some_function", `{"a": "b"}`, ""),
		),
		openaitypes.ResponseInputItemUnionParamFromResponseOutputItemUnion(getTextMessage("a_message")),
		openaitypes.ResponseInputItemUnionParamFromResponseInputItemFunctionCallOutputParam(
			responses.ResponseInputItemFunctionCallOutputParam{
				CallID: "2",
				Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
					OfString: param.NewOpt("tool_result"),
				},
				Type: constant.ValueOf[constant.FunctionCallOutput](),
			},
		),
		openaitypes.ResponseInputItemUnionParamFromResponseOutputItemUnion(getTextMessage("done")),
	}

	assert.Equal(t, expected, workflow.inputHistory)
	assert.Equal(t, agent, workflow.currentAgent)

	model.setNextOutput([]TResponseOutputItem{getTextMessage("done_2")})

	output = nil
	run = workflow.Run(context.Background(), "transcription_2")
	for chunk := range run.Seq() {
		output = append(output, chunk)
	}
	require.NoError(t, run.Error())

	expected = append(expected, TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{
				OfString: param.NewOpt("transcription_2"),
			},
			Role: responses.EasyInputMessageRoleUser,
			Type: responses.EasyInputMessageTypeMessage,
		},
	})
	expected = append(expected, openaitypes.ResponseInputItemUnionParamFromResponseOutputItemUnion(getTextMessage("done_2")))

	assert.Equal(t, expected, workflow.inputHistory)
	assert.Equal(t, agent, workflow.currentAgent)
}
