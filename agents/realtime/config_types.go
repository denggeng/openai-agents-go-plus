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

import "github.com/denggeng/openai-agents-go-plus/agents"

// RealtimeSessionModelSettings stores transport model settings for realtime sessions.
type RealtimeSessionModelSettings map[string]any

// RealtimeRunConfig stores runtime configuration for realtime sessions.
type RealtimeRunConfig map[string]any

// RealtimeModelTracingConfig stores trace metadata options for realtime sessions.
type RealtimeModelTracingConfig struct {
	GroupID      *string
	Metadata     map[string]any
	WorkflowName *string
}

const (
	// RealtimeToolErrorKindApprovalRejected indicates a tool call was rejected by approval policy.
	RealtimeToolErrorKindApprovalRejected = "approval_rejected"
)

// RealtimeToolErrorFormatterArgs contains metadata passed to run-level tool error formatters.
type RealtimeToolErrorFormatterArgs struct {
	Kind           string
	ToolType       string
	ToolName       string
	CallID         string
	DefaultMessage string
	RunContext     *agents.RunContextWrapper[any]
}

// RealtimeToolErrorFormatter resolves model-visible error text for realtime tool failures.
type RealtimeToolErrorFormatter func(args RealtimeToolErrorFormatterArgs) any
