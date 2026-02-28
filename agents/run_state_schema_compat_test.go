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
	"fmt"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunStateAcceptsSupportedSchemaVersions(t *testing.T) {
	for _, version := range []string{"1.0", "1.1", "1.2", "1.3", "1.4"} {
		t.Run(version, func(t *testing.T) {
			_, err := agents.RunStateFromJSONString(
				fmt.Sprintf(`{"$schemaVersion":"%s","current_turn":1,"max_turns":1}`, version),
			)
			require.NoError(t, err)
		})
	}
}

func TestRunStateDefaultsToLatestSchemaVersion(t *testing.T) {
	state := agents.RunState{
		CurrentTurn: 1,
		MaxTurns:    1,
	}
	raw, err := state.ToJSON()
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(raw, &payload))
	assert.Equal(t, agents.CurrentRunStateSchemaVersion, payload["$schemaVersion"])
}

func TestRunStateModelResponseRequestIDRoundTrip(t *testing.T) {
	state := agents.RunState{
		SchemaVersion: agents.CurrentRunStateSchemaVersion,
		CurrentTurn:   1,
		MaxTurns:      1,
		ModelResponses: []agents.ModelResponse{
			{
				ResponseID: "resp_1",
				RequestID:  "req_1",
			},
		},
	}

	raw, err := state.ToJSON()
	require.NoError(t, err)

	decoded, err := agents.RunStateFromJSON(raw)
	require.NoError(t, err)
	require.Len(t, decoded.ModelResponses, 1)
	assert.Equal(t, "resp_1", decoded.ModelResponses[0].ResponseID)
	assert.Equal(t, "req_1", decoded.ModelResponses[0].RequestID)
}

func TestRunStateModelResponseLegacyFieldsStillDecode(t *testing.T) {
	decoded, err := agents.RunStateFromJSONString(`{
		"$schemaVersion":"1.4",
		"current_turn":1,
		"max_turns":1,
		"model_responses":[{"ResponseID":"resp_old","RequestID":"req_old"}]
	}`)
	require.NoError(t, err)
	require.Len(t, decoded.ModelResponses, 1)
	assert.Equal(t, "resp_old", decoded.ModelResponses[0].ResponseID)
	assert.Equal(t, "req_old", decoded.ModelResponses[0].RequestID)
}
