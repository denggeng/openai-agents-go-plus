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

// ThreadOptions stores execution options used for a Codex thread.
type ThreadOptions struct {
	Model                 *string
	SandboxMode           *string
	WorkingDirectory      *string
	SkipGitRepoCheck      *bool
	ModelReasoningEffort  *string
	NetworkAccessEnabled  *bool
	WebSearchMode         *string
	WebSearchEnabled      *bool
	ApprovalPolicy        *string
	AdditionalDirectories []string
}

// CoerceThreadOptions accepts either nil, ThreadOptions, or map-based options.
func CoerceThreadOptions(options any) (*ThreadOptions, error) {
	switch typed := options.(type) {
	case nil:
		return nil, nil
	case ThreadOptions:
		return cloneThreadOptions(&typed), nil
	case *ThreadOptions:
		if typed == nil {
			return nil, nil
		}
		return cloneThreadOptions(typed), nil
	case map[string]any:
		return coerceThreadOptionsMap(typed)
	default:
		return nil, agents.NewUserError("ThreadOptions must be a ThreadOptions or a mapping.")
	}
}

func coerceThreadOptionsMap(values map[string]any) (*ThreadOptions, error) {
	allowed := map[string]struct{}{
		"model":                  {},
		"sandbox_mode":           {},
		"working_directory":      {},
		"skip_git_repo_check":    {},
		"model_reasoning_effort": {},
		"network_access_enabled": {},
		"web_search_mode":        {},
		"web_search_enabled":     {},
		"approval_policy":        {},
		"additional_directories": {},
	}
	unknown := make([]string, 0)
	for key := range values {
		if _, ok := allowed[key]; !ok {
			unknown = append(unknown, key)
		}
	}
	if len(unknown) > 0 {
		slices.Sort(unknown)
		return nil, agents.UserErrorf("Unknown ThreadOptions field(s): %v", unknown)
	}

	out := &ThreadOptions{}
	for key, raw := range values {
		switch key {
		case "model":
			value, err := optionalString(raw, "model")
			if err != nil {
				return nil, err
			}
			out.Model = value
		case "sandbox_mode":
			value, err := optionalString(raw, "sandbox_mode")
			if err != nil {
				return nil, err
			}
			out.SandboxMode = value
		case "working_directory":
			value, err := optionalString(raw, "working_directory")
			if err != nil {
				return nil, err
			}
			out.WorkingDirectory = value
		case "skip_git_repo_check":
			value, err := optionalBool(raw, "skip_git_repo_check")
			if err != nil {
				return nil, err
			}
			out.SkipGitRepoCheck = value
		case "model_reasoning_effort":
			value, err := optionalString(raw, "model_reasoning_effort")
			if err != nil {
				return nil, err
			}
			out.ModelReasoningEffort = value
		case "network_access_enabled":
			value, err := optionalBool(raw, "network_access_enabled")
			if err != nil {
				return nil, err
			}
			out.NetworkAccessEnabled = value
		case "web_search_mode":
			value, err := optionalString(raw, "web_search_mode")
			if err != nil {
				return nil, err
			}
			out.WebSearchMode = value
		case "web_search_enabled":
			value, err := optionalBool(raw, "web_search_enabled")
			if err != nil {
				return nil, err
			}
			out.WebSearchEnabled = value
		case "approval_policy":
			value, err := optionalString(raw, "approval_policy")
			if err != nil {
				return nil, err
			}
			out.ApprovalPolicy = value
		case "additional_directories":
			value, err := optionalStringSlice(raw, "additional_directories")
			if err != nil {
				return nil, err
			}
			out.AdditionalDirectories = value
		}
	}
	return out, nil
}

func cloneThreadOptions(options *ThreadOptions) *ThreadOptions {
	if options == nil {
		return nil
	}
	clone := *options
	clone.AdditionalDirectories = slices.Clone(options.AdditionalDirectories)
	if options.Model != nil {
		value := *options.Model
		clone.Model = &value
	}
	if options.SandboxMode != nil {
		value := *options.SandboxMode
		clone.SandboxMode = &value
	}
	if options.WorkingDirectory != nil {
		value := *options.WorkingDirectory
		clone.WorkingDirectory = &value
	}
	if options.SkipGitRepoCheck != nil {
		value := *options.SkipGitRepoCheck
		clone.SkipGitRepoCheck = &value
	}
	if options.ModelReasoningEffort != nil {
		value := *options.ModelReasoningEffort
		clone.ModelReasoningEffort = &value
	}
	if options.NetworkAccessEnabled != nil {
		value := *options.NetworkAccessEnabled
		clone.NetworkAccessEnabled = &value
	}
	if options.WebSearchMode != nil {
		value := *options.WebSearchMode
		clone.WebSearchMode = &value
	}
	if options.WebSearchEnabled != nil {
		value := *options.WebSearchEnabled
		clone.WebSearchEnabled = &value
	}
	if options.ApprovalPolicy != nil {
		value := *options.ApprovalPolicy
		clone.ApprovalPolicy = &value
	}
	return &clone
}

func optionalBool(value any, fieldName string) (*bool, error) {
	if value == nil {
		return nil, nil
	}
	typed, ok := value.(bool)
	if !ok {
		return nil, agents.UserErrorf("%s must be a bool or nil", fieldName)
	}
	return &typed, nil
}

func optionalStringSlice(value any, fieldName string) ([]string, error) {
	if value == nil {
		return nil, nil
	}
	switch typed := value.(type) {
	case []string:
		return slices.Clone(typed), nil
	case []any:
		out := make([]string, 0, len(typed))
		for _, each := range typed {
			text, ok := each.(string)
			if !ok {
				return nil, agents.UserErrorf("%s must contain only strings", fieldName)
			}
			out = append(out, text)
		}
		return out, nil
	default:
		return nil, agents.UserErrorf("%s must be an array of strings or nil", fieldName)
	}
}
