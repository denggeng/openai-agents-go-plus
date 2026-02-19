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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMaterializeJSONValueConvertsIterators(t *testing.T) {
	contentIter := func(yield func(map[string]any) bool) {
		yield(map[string]any{
			"type":        "output_text",
			"text":        "Hey, what's up?",
			"annotations": []any{},
			"logprobs":    []any{},
		})
	}

	input := map[string]any{
		"id":      "a75654dc-7492-4d1c-bce0-89e8312fbdd7",
		"content": contentIter,
		"role":    "assistant",
		"status":  "completed",
		"type":    "message",
	}

	materialized := materializeJSONValue(input)
	mapping, ok := materialized.(map[string]any)
	require.True(t, ok)

	raw, err := json.Marshal(mapping)
	require.NoError(t, err)

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(raw, &decoded))

	expected := map[string]any{
		"id": "a75654dc-7492-4d1c-bce0-89e8312fbdd7",
		"content": []any{
			map[string]any{
				"type":        "output_text",
				"text":        "Hey, what's up?",
				"annotations": []any{},
				"logprobs":    []any{},
			},
		},
		"role":   "assistant",
		"status": "completed",
		"type":   "message",
	}
	assert.Equal(t, expected, decoded)
}
