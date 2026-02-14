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
	"testing"

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

func TestMergedModelExtraJSON(t *testing.T) {
	t.Run("nil when no extras", func(t *testing.T) {
		assert.Nil(t, mergedModelExtraJSON(modelsettings.ModelSettings{}))
	})

	t.Run("extra args override except reasoning_effort", func(t *testing.T) {
		extras := mergedModelExtraJSON(modelsettings.ModelSettings{
			ExtraBody: map[string]any{
				"foo":              "body",
				"reasoning_effort": "none",
			},
			ExtraArgs: map[string]any{
				"foo":              "args",
				"bar":              123,
				"reasoning_effort": "low",
			},
		})

		assert.Equal(t, map[string]any{
			"foo":              "args",
			"bar":              123,
			"reasoning_effort": "none",
		}, extras)
	})

	t.Run("explicit reasoning drops extra reasoning_effort", func(t *testing.T) {
		extras := mergedModelExtraJSON(modelsettings.ModelSettings{
			Reasoning: openai.ReasoningParam{Effort: openai.ReasoningEffortLow},
			ExtraBody: map[string]any{
				"reasoning_effort": "none",
				"foo":              "ok",
			},
			ExtraArgs: map[string]any{
				"reasoning_effort": "high",
				"bar":              true,
			},
		})

		assert.Equal(t, map[string]any{
			"foo": "ok",
			"bar": true,
		}, extras)
	})
}
