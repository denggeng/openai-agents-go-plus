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
	"fmt"
	"slices"

	"github.com/denggeng/openai-agents-go-plus/agents"
)

// CodexOptions stores top-level Codex extension settings.
type CodexOptions struct {
	CodexPathOverride               *string
	BaseURL                         *string
	APIKey                          *string
	Env                             map[any]any
	CodexSubprocessStreamLimitBytes *int
}

// CoerceCodexOptions accepts either nil, CodexOptions, or map-based options.
func CoerceCodexOptions(options any) (*CodexOptions, error) {
	switch typed := options.(type) {
	case nil:
		return nil, nil
	case CodexOptions:
		return cloneCodexOptions(&typed), nil
	case *CodexOptions:
		if typed == nil {
			return nil, nil
		}
		return cloneCodexOptions(typed), nil
	case map[string]any:
		return coerceCodexOptionsMap(typed)
	default:
		return nil, agents.NewUserError("CodexOptions must be a CodexOptions or a mapping.")
	}
}

func coerceCodexOptionsMap(values map[string]any) (*CodexOptions, error) {
	allowed := map[string]struct{}{
		"codex_path_override":                 {},
		"base_url":                            {},
		"api_key":                             {},
		"env":                                 {},
		"codex_subprocess_stream_limit_bytes": {},
	}
	unknown := make([]string, 0)
	for key := range values {
		if _, ok := allowed[key]; !ok {
			unknown = append(unknown, key)
		}
	}
	if len(unknown) > 0 {
		slices.Sort(unknown)
		return nil, agents.UserErrorf("Unknown CodexOptions field(s): %v", unknown)
	}

	out := &CodexOptions{}
	for key, raw := range values {
		switch key {
		case "codex_path_override":
			value, err := optionalString(raw, "codex_path_override")
			if err != nil {
				return nil, err
			}
			out.CodexPathOverride = value
		case "base_url":
			value, err := optionalString(raw, "base_url")
			if err != nil {
				return nil, err
			}
			out.BaseURL = value
		case "api_key":
			value, err := optionalString(raw, "api_key")
			if err != nil {
				return nil, err
			}
			out.APIKey = value
		case "env":
			value, err := optionalAnyAnyMap(raw, "env")
			if err != nil {
				return nil, err
			}
			out.Env = value
		case "codex_subprocess_stream_limit_bytes":
			value, err := optionalInt(raw, "codex_subprocess_stream_limit_bytes")
			if err != nil {
				return nil, err
			}
			out.CodexSubprocessStreamLimitBytes = value
		}
	}
	return out, nil
}

func cloneCodexOptions(options *CodexOptions) *CodexOptions {
	if options == nil {
		return nil
	}
	clone := *options
	clone.Env = cloneAnyAnyMap(options.Env)
	if options.CodexPathOverride != nil {
		value := *options.CodexPathOverride
		clone.CodexPathOverride = &value
	}
	if options.BaseURL != nil {
		value := *options.BaseURL
		clone.BaseURL = &value
	}
	if options.APIKey != nil {
		value := *options.APIKey
		clone.APIKey = &value
	}
	if options.CodexSubprocessStreamLimitBytes != nil {
		value := *options.CodexSubprocessStreamLimitBytes
		clone.CodexSubprocessStreamLimitBytes = &value
	}
	return &clone
}

func optionalString(value any, fieldName string) (*string, error) {
	if value == nil {
		return nil, nil
	}
	typed, ok := value.(string)
	if !ok {
		return nil, agents.UserErrorf("%s must be a string or nil", fieldName)
	}
	return &typed, nil
}

func optionalAnyAnyMap(value any, fieldName string) (map[any]any, error) {
	if value == nil {
		return nil, nil
	}
	switch typed := value.(type) {
	case map[any]any:
		return cloneAnyAnyMap(typed), nil
	case map[string]any:
		out := make(map[any]any, len(typed))
		for key, each := range typed {
			out[key] = each
		}
		return out, nil
	case map[string]string:
		out := make(map[any]any, len(typed))
		for key, each := range typed {
			out[key] = each
		}
		return out, nil
	default:
		return nil, agents.UserErrorf("%s must be a mapping or nil", fieldName)
	}
}

func optionalInt(value any, fieldName string) (*int, error) {
	if value == nil {
		return nil, nil
	}
	switch typed := value.(type) {
	case int:
		return &typed, nil
	case int8:
		v := int(typed)
		return &v, nil
	case int16:
		v := int(typed)
		return &v, nil
	case int32:
		v := int(typed)
		return &v, nil
	case int64:
		v := int(typed)
		return &v, nil
	case uint:
		v := int(typed)
		return &v, nil
	case uint8:
		v := int(typed)
		return &v, nil
	case uint16:
		v := int(typed)
		return &v, nil
	case uint32:
		v := int(typed)
		return &v, nil
	case uint64:
		v := int(typed)
		return &v, nil
	case float64:
		integer := int(typed)
		if float64(integer) != typed {
			return nil, agents.UserErrorf("%s must be an integer or nil", fieldName)
		}
		return &integer, nil
	case float32:
		integer := int(typed)
		if float32(integer) != typed {
			return nil, agents.UserErrorf("%s must be an integer or nil", fieldName)
		}
		return &integer, nil
	default:
		return nil, agents.UserErrorf("%s must be an integer or nil", fieldName)
	}
}

func cloneAnyAnyMap(value map[any]any) map[any]any {
	if value == nil {
		return nil
	}
	out := make(map[any]any, len(value))
	for key, each := range value {
		out[key] = each
	}
	return out
}

func anyToString(value any) string {
	return fmt.Sprint(value)
}
