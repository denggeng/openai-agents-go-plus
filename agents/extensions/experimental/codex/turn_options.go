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

package codex

import (
	"slices"

	"github.com/denggeng/openai-agents-go-plus/agents"
)

// TurnOptions stores run-specific options for a Codex thread turn.
type TurnOptions struct {
	OutputSchema       map[string]any
	Signal             <-chan struct{}
	IdleTimeoutSeconds *float64
}

// CoerceTurnOptions accepts either nil, TurnOptions, or map-based options.
func CoerceTurnOptions(options any) (*TurnOptions, error) {
	switch typed := options.(type) {
	case nil:
		return nil, nil
	case TurnOptions:
		return cloneTurnOptions(&typed), nil
	case *TurnOptions:
		if typed == nil {
			return nil, nil
		}
		return cloneTurnOptions(typed), nil
	case map[string]any:
		return coerceTurnOptionsMap(typed)
	default:
		return nil, agents.NewUserError("TurnOptions must be a TurnOptions or a mapping.")
	}
}

func coerceTurnOptionsMap(values map[string]any) (*TurnOptions, error) {
	allowed := map[string]struct{}{
		"output_schema":        {},
		"signal":               {},
		"idle_timeout_seconds": {},
	}
	unknown := make([]string, 0)
	for key := range values {
		if _, ok := allowed[key]; !ok {
			unknown = append(unknown, key)
		}
	}
	if len(unknown) > 0 {
		slices.Sort(unknown)
		return nil, agents.UserErrorf("Unknown TurnOptions field(s): %v", unknown)
	}

	out := &TurnOptions{}
	for key, raw := range values {
		switch key {
		case "output_schema":
			value, err := optionalStringAnyMap(raw, "output_schema")
			if err != nil {
				return nil, err
			}
			out.OutputSchema = value
		case "signal":
			value, err := optionalSignal(raw, "signal")
			if err != nil {
				return nil, err
			}
			out.Signal = value
		case "idle_timeout_seconds":
			value, err := optionalFloat64(raw, "idle_timeout_seconds")
			if err != nil {
				return nil, err
			}
			out.IdleTimeoutSeconds = value
		}
	}
	return out, nil
}

func cloneTurnOptions(options *TurnOptions) *TurnOptions {
	if options == nil {
		return nil
	}
	clone := *options
	clone.OutputSchema = cloneStringAnyMap(options.OutputSchema)
	if options.IdleTimeoutSeconds != nil {
		value := *options.IdleTimeoutSeconds
		clone.IdleTimeoutSeconds = &value
	}
	return &clone
}

func optionalSignal(value any, fieldName string) (<-chan struct{}, error) {
	if value == nil {
		return nil, nil
	}
	switch typed := value.(type) {
	case <-chan struct{}:
		return typed, nil
	case chan struct{}:
		return typed, nil
	default:
		return nil, agents.UserErrorf("%s must be a channel or nil", fieldName)
	}
}

func optionalFloat64(value any, fieldName string) (*float64, error) {
	if value == nil {
		return nil, nil
	}
	switch typed := value.(type) {
	case float64:
		return &typed, nil
	case float32:
		converted := float64(typed)
		return &converted, nil
	case int:
		converted := float64(typed)
		return &converted, nil
	case int8:
		converted := float64(typed)
		return &converted, nil
	case int16:
		converted := float64(typed)
		return &converted, nil
	case int32:
		converted := float64(typed)
		return &converted, nil
	case int64:
		converted := float64(typed)
		return &converted, nil
	case uint:
		converted := float64(typed)
		return &converted, nil
	case uint8:
		converted := float64(typed)
		return &converted, nil
	case uint16:
		converted := float64(typed)
		return &converted, nil
	case uint32:
		converted := float64(typed)
		return &converted, nil
	case uint64:
		converted := float64(typed)
		return &converted, nil
	default:
		return nil, agents.UserErrorf("%s must be numeric or nil", fieldName)
	}
}

func optionalStringAnyMap(value any, fieldName string) (map[string]any, error) {
	if value == nil {
		return nil, nil
	}
	typed, ok := value.(map[string]any)
	if !ok {
		return nil, agents.UserErrorf("%s must be a plain JSON object", fieldName)
	}
	return cloneStringAnyMap(typed), nil
}

func cloneStringAnyMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	out := make(map[string]any, len(value))
	for key, each := range value {
		out[key] = each
	}
	return out
}
