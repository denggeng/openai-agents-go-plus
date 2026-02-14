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

package realtime

import (
	"fmt"
	"slices"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

// RealtimeInstructionsFunc computes dynamic instructions for realtime agents.
type RealtimeInstructionsFunc[T any] func(
	*agents.RunContextWrapper[T],
	*RealtimeAgent[T],
) (string, error)

// RealtimeInstructionsSyncFunc computes dynamic instructions without an error return.
type RealtimeInstructionsSyncFunc[T any] func(
	*agents.RunContextWrapper[T],
	*RealtimeAgent[T],
) string

// RealtimeAgent is a simplified agent shape used by realtime sessions.
//
// Instructions accepts:
// 1. string
// 2. RealtimeInstructionsFunc[T]
// 3. RealtimeInstructionsSyncFunc[T]
type RealtimeAgent[T any] struct {
	Name             string
	Instructions     any
	Prompt           map[string]any
	Handoffs         []any
	Tools            []agents.Tool
	OutputGuardrails []agents.OutputGuardrail
}

// Clone returns a shallow copy of the realtime agent.
func (a *RealtimeAgent[T]) Clone() *RealtimeAgent[T] {
	if a == nil {
		return nil
	}
	cloned := *a
	cloned.Prompt = cloneStringAnyMap(a.Prompt)
	cloned.Handoffs = slices.Clone(a.Handoffs)
	cloned.Tools = slices.Clone(a.Tools)
	cloned.OutputGuardrails = slices.Clone(a.OutputGuardrails)
	return &cloned
}

// GetSystemPrompt resolves static or dynamic instructions for this realtime agent.
func (a *RealtimeAgent[T]) GetSystemPrompt(
	runContext *agents.RunContextWrapper[T],
) (string, error) {
	switch v := a.Instructions.(type) {
	case nil:
		return "", nil
	case string:
		return v, nil
	case RealtimeInstructionsFunc[T]:
		return v(runContext, a)
	case RealtimeInstructionsSyncFunc[T]:
		return v(runContext, a), nil
	default:
		return "", fmt.Errorf("instructions must be a string or function, got %T", a.Instructions)
	}
}

// GetAllTools returns currently enabled static tools for the realtime agent.
func (a *RealtimeAgent[T]) GetAllTools(
	_ *agents.RunContextWrapper[T],
) ([]agents.Tool, error) {
	return slices.Clone(a.Tools), nil
}

func cloneStringAnyMap(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}
