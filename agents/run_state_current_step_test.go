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
	"encoding/json"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunStateSerializesCurrentStepInterruptions(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
		Interruptions: []agents.ToolApprovalItem{{
			ToolName: "sensitive_tool",
			RawItem: map[string]any{
				"name":    "sensitive_tool",
				"call_id": "call-1",
				"type":    "function_call",
			},
		}},
	}

	raw, err := state.ToJSONWithOptions(agents.RunStateSerializeOptions{})
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(raw, &payload))

	step, ok := payload["current_step"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "next_step_interruption", step["type"])

	data, ok := step["data"].(map[string]any)
	require.True(t, ok)
	interruptions, ok := data["interruptions"].([]any)
	require.True(t, ok)
	require.Len(t, interruptions, 1)

	item, ok := interruptions[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "tool_approval_item", item["type"])
	assert.Equal(t, "sensitive_tool", item["tool_name"])
}

func TestRunStateDeserializesCurrentStepInterruptions(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
		Interruptions: []agents.ToolApprovalItem{{
			ToolName: "sensitive_tool",
			RawItem: map[string]any{
				"name":    "sensitive_tool",
				"call_id": "call-1",
				"type":    "function_call",
			},
		}},
	}

	raw, err := state.ToJSONWithOptions(agents.RunStateSerializeOptions{})
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(raw, &payload))
	delete(payload, "interruptions")

	raw, err = json.Marshal(payload)
	require.NoError(t, err)

	decoded, err := agents.RunStateFromJSONWithOptions(raw, agents.RunStateDeserializeOptions{})
	require.NoError(t, err)
	require.Len(t, decoded.Interruptions, 1)
	assert.Equal(t, "sensitive_tool", decoded.Interruptions[0].ToolName)
}
