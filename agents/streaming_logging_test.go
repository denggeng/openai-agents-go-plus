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
	"bufio"
	"bytes"
	"encoding/json"
	"log/slog"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/require"
)

func TestRunStreamedResumeOmitsToolOutputWhenDontLog(t *testing.T) {
	prevLogger := agents.Logger()
	prevDontLog := agents.DontLogToolData
	defer func() {
		agents.SetLogger(prevLogger)
		agents.DontLogToolData = prevDontLog
	}()

	agents.DontLogToolData = true

	var buf bytes.Buffer
	agents.SetLogger(slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))

	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("ok")},
	})
	agent := agents.New("log-agent").WithModelInstance(model)

	rawItem := agents.ResponseInputItemFunctionCallOutputParam{
		CallID: "call-1",
		Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
			OfString: param.NewOpt("secret"),
		},
		Type:   constant.ValueOf[constant.FunctionCallOutput](),
	}
	runItem := agents.ToolCallOutputItem{
		Agent:  agent,
		RawItem: rawItem,
		Output: "secret",
		Type:   "tool_call_output_item",
	}

	state := agents.RunState{
		SchemaVersion:    agents.CurrentRunStateSchemaVersion,
		CurrentTurn:      0,
		MaxTurns:         1,
		CurrentAgentName: agent.Name,
		OriginalInput: []agents.TResponseInputItem{
			agentstesting.GetTextInputItem("hi"),
		},
		GeneratedRunItems: []agents.RunItem{runItem},
		GeneratedItems:    []agents.TResponseInputItem{runItem.ToInputItem()},
	}

	result, err := agents.Runner{}.RunFromStateStreamed(t.Context(), agent, state)
	require.NoError(t, err)
	require.NoError(t, result.StreamEvents(func(agents.StreamEvent) error { return nil }))

	var record map[string]any
	scanner := bufio.NewScanner(bytes.NewReader(buf.Bytes()))
	for scanner.Scan() {
		var payload map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &payload); err != nil {
			continue
		}
		if payload["msg"] == "Resuming from RunState in run_streaming()" {
			record = payload
			break
		}
	}
	require.NotNil(t, record)

	detailsAny, ok := record["generated_items_details"]
	require.True(t, ok)
	details, ok := detailsAny.([]any)
	require.True(t, ok)
	require.NotEmpty(t, details)

	first, ok := details[0].(map[string]any)
	require.True(t, ok)
	_, hasOutput := first["output"]
	require.False(t, hasOutput)
}
