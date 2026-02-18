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
	"runtime"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/require"
)

type resultCastFoo struct {
	Bar int
}

func createRunResult(finalOutput any, newItems []agents.RunItem, lastAgent *agents.Agent) *agents.RunResult {
	if lastAgent == nil {
		lastAgent = &agents.Agent{Name: "test-agent"}
	}
	return &agents.RunResult{
		Input:                      agents.InputString("test"),
		NewItems:                   newItems,
		RawResponses:               nil,
		FinalOutput:                finalOutput,
		InputGuardrailResults:      nil,
		OutputGuardrailResults:     nil,
		ToolInputGuardrailResults:  nil,
		ToolOutputGuardrailResults: nil,
		Interruptions:              nil,
		LastAgent:                  lastAgent,
	}
}

func messageOutput(text string) responses.ResponseOutputMessage {
	return responses.ResponseOutputMessage{
		ID: "msg",
		Content: []responses.ResponseOutputMessageContentUnion{{
			Text:        text,
			Type:        "output_text",
			Annotations: nil,
		}},
		Role:   constant.ValueOf[constant.Assistant](),
		Status: responses.ResponseOutputMessageStatusCompleted,
		Type:   constant.ValueOf[constant.Message](),
	}
}

func messageOutputItem(agent *agents.Agent, text string) agents.MessageOutputItem {
	return agents.MessageOutputItem{
		Agent:   agent,
		RawItem: messageOutput(text),
		Type:    "message_output_item",
	}
}

func TestRunResultFinalOutputAsTypechecks(t *testing.T) {
	result := createRunResult(1, nil, nil)
	var value int
	err := result.FinalOutputAs(&value, false)
	require.NoError(t, err)
	require.Equal(t, 1, value)

	result = createRunResult("test", nil, nil)
	var text string
	err = result.FinalOutputAs(&text, false)
	require.NoError(t, err)
	require.Equal(t, "test", text)

	result = createRunResult(resultCastFoo{Bar: 1}, nil, nil)
	var foo resultCastFoo
	err = result.FinalOutputAs(&foo, false)
	require.NoError(t, err)
	require.Equal(t, resultCastFoo{Bar: 1}, foo)
}

func TestRunResultFinalOutputAsBadCastDoesNotError(t *testing.T) {
	result := createRunResult(1, nil, nil)
	var text string
	err := result.FinalOutputAs(&text, false)
	require.NoError(t, err)
	require.Equal(t, "", text)

	result = createRunResult("test", nil, nil)
	var foo resultCastFoo
	err = result.FinalOutputAs(&foo, false)
	require.NoError(t, err)
	require.Equal(t, resultCastFoo{}, foo)
}

func TestRunResultFinalOutputAsBadCastRaises(t *testing.T) {
	result := createRunResult(1, nil, nil)
	var text string
	err := result.FinalOutputAs(&text, true)
	require.Error(t, err)

	result = createRunResult("test", nil, nil)
	var foo resultCastFoo
	err = result.FinalOutputAs(&foo, true)
	require.Error(t, err)

	result = createRunResult(resultCastFoo{Bar: 1}, nil, nil)
	var value int
	err = result.FinalOutputAs(&value, true)
	require.Error(t, err)
}

func TestRunResultReleaseAgentsBreaksStrongRefs(t *testing.T) {
	agent := &agents.Agent{Name: "leak-test-agent"}
	finalized := make(chan struct{})
	runtime.SetFinalizer(agent, func(*agents.Agent) {
		close(finalized)
	})

	item := messageOutputItem(agent, "hello")
	result := createRunResult(nil, []agents.RunItem{item}, agent)
	result.ReleaseAgents()

	require.Len(t, result.NewItems, 1)
	releasedItem, ok := result.NewItems[0].(agents.MessageOutputItem)
	require.True(t, ok)
	require.Nil(t, releasedItem.Agent)
	require.Nil(t, result.LastAgent)

	item = agents.MessageOutputItem{}
	agent = nil
	waitForFinalizer(t, finalized)
}

func TestRunItemRetainsAgentAfterResultCollected(t *testing.T) {
	agent := &agents.Agent{Name: "persisted-agent"}
	item := &agents.MessageOutputItem{
		Agent:   agent,
		RawItem: messageOutput("persist"),
		Type:    "message_output_item",
	}
	result := createRunResult(nil, []agents.RunItem{item}, agent)
	_ = result
	result = nil
	runtime.GC()

	require.NotNil(t, item.Agent)
	require.Equal(t, "persisted-agent", item.Agent.Name)
}

func TestRunResultReleaseAgentsKeepsNewItemsWhenDisabled(t *testing.T) {
	itemAgent := &agents.Agent{Name: "item-agent"}
	lastAgent := &agents.Agent{Name: "last-agent"}
	item := messageOutputItem(itemAgent, "keep")
	result := createRunResult(nil, []agents.RunItem{item}, lastAgent)

	result.ReleaseAgents(false)

	require.Nil(t, result.LastAgent)
	keptItem, ok := result.NewItems[0].(agents.MessageOutputItem)
	require.True(t, ok)
	require.NotNil(t, keptItem.Agent)
	require.Equal(t, "item-agent", keptItem.Agent.Name)
}

func TestRunResultReleaseAgentsIsIdempotent(t *testing.T) {
	agent := &agents.Agent{Name: "idempotent-agent"}
	item := messageOutputItem(agent, "idempotent")
	result := createRunResult(nil, []agents.RunItem{item}, agent)

	result.ReleaseAgents()
	result.ReleaseAgents()

	require.Len(t, result.NewItems, 1)
	releasedItem, ok := result.NewItems[0].(agents.MessageOutputItem)
	require.True(t, ok)
	require.Nil(t, releasedItem.Agent)
	require.Nil(t, result.LastAgent)
}

func TestRunResultStreamingReleaseAgentsReleasesCurrentAgent(t *testing.T) {
	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Hi"),
		},
	})
	agent := &agents.Agent{
		Name:  "streaming-agent",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	finalized := make(chan struct{})
	runtime.SetFinalizer(agent, func(*agents.Agent) {
		close(finalized)
	})

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	result.ReleaseAgents()
	require.Nil(t, result.LastAgent())

	agent = nil
	waitForFinalizer(t, finalized)
}
