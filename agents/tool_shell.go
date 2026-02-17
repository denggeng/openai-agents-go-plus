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
	"context"
	"strings"
)

// ShellTool allows the model to execute shell commands.
type ShellTool struct {
	Executor ShellExecutor
	Name     string

	// Optional approval policy for shell tool calls.
	NeedsApproval ShellNeedsApproval

	// Optional handler to auto-approve or reject when approval is required.
	OnApproval ShellOnApprovalFunc

	// Execution environment for shell commands. Defaults to {"type":"local"}.
	Environment map[string]any
}

func (t ShellTool) ToolName() string {
	if t.Name != "" {
		return t.Name
	}
	return "shell"
}

func (ShellTool) isTool() {}

// Normalize validates and normalizes shell tool configuration.
func (t *ShellTool) Normalize() error {
	if t == nil {
		return nil
	}
	env, err := normalizeShellToolEnvironment(t.Environment)
	if err != nil {
		return err
	}
	t.Environment = env

	envType := "local"
	if value, ok := env["type"]; ok {
		if str, ok := coerceStringValue(value); ok && str != "" {
			envType = strings.ToLower(str)
			env["type"] = str
		}
	}

	if envType == "local" {
		if t.Executor == nil {
			return UserErrorf("ShellTool with local environment requires an executor.")
		}
		return nil
	}

	if t.Executor != nil {
		return UserErrorf("ShellTool with hosted environment does not accept an executor.")
	}

	if t.NeedsApproval != nil {
		if flag, ok := t.NeedsApproval.(ShellNeedsApprovalFlag); ok {
			if flag.Enabled() {
				return UserErrorf("ShellTool with hosted environment does not support needs_approval or on_approval.")
			}
		} else if flagPtr, ok := t.NeedsApproval.(*ShellNeedsApprovalFlag); ok {
			if flagPtr.Enabled() {
				return UserErrorf("ShellTool with hosted environment does not support needs_approval or on_approval.")
			}
		} else {
			return UserErrorf("ShellTool with hosted environment does not support needs_approval or on_approval.")
		}
	}

	if t.OnApproval != nil {
		return UserErrorf("ShellTool with hosted environment does not support needs_approval or on_approval.")
	}

	return nil
}

type ShellExecutor func(context.Context, ShellCommandRequest) (any, error)

// ShellNeedsApproval determines whether a shell call requires approval.
type ShellNeedsApproval interface {
	NeedsApproval(
		ctx context.Context,
		runContext *RunContextWrapper[any],
		action ShellActionRequest,
		callID string,
	) (bool, error)
}

// ShellNeedsApprovalFlag is a static approval policy.
type ShellNeedsApprovalFlag struct {
	needsApproval bool
}

func (f ShellNeedsApprovalFlag) NeedsApproval(
	context.Context,
	*RunContextWrapper[any],
	ShellActionRequest,
	string,
) (bool, error) {
	return f.needsApproval, nil
}

func (f ShellNeedsApprovalFlag) Enabled() bool {
	return f.needsApproval
}

// NewShellNeedsApprovalFlag creates a static shell approval policy.
func NewShellNeedsApprovalFlag(needsApproval bool) ShellNeedsApprovalFlag {
	return ShellNeedsApprovalFlag{needsApproval: needsApproval}
}

// ShellNeedsApprovalEnabled always requires approval.
func ShellNeedsApprovalEnabled() ShellNeedsApproval {
	return NewShellNeedsApprovalFlag(true)
}

// ShellNeedsApprovalDisabled never requires approval.
func ShellNeedsApprovalDisabled() ShellNeedsApproval {
	return NewShellNeedsApprovalFlag(false)
}

// ShellNeedsApprovalFunc wraps a callback as an approval policy.
type ShellNeedsApprovalFunc func(
	ctx context.Context,
	runContext *RunContextWrapper[any],
	action ShellActionRequest,
	callID string,
) (bool, error)

func (f ShellNeedsApprovalFunc) NeedsApproval(
	ctx context.Context,
	runContext *RunContextWrapper[any],
	action ShellActionRequest,
	callID string,
) (bool, error) {
	return f(ctx, runContext, action, callID)
}

// ShellOnApprovalFunc allows auto-approving or rejecting shell calls.
type ShellOnApprovalFunc func(
	ctx *RunContextWrapper[any],
	approvalItem ToolApprovalItem,
) (any, error)

// ShellCallOutcome describes the terminal condition of a shell command.
type ShellCallOutcome struct {
	Type     string
	ExitCode *int
}

const (
	ShellCallOutcomeExit    = "exit"
	ShellCallOutcomeTimeout = "timeout"
)

// ShellCommandOutput is the structured output of a shell command.
type ShellCommandOutput struct {
	Stdout       string
	Stderr       string
	Outcome      ShellCallOutcome
	Command      *string
	ProviderData map[string]any
}

func (o ShellCommandOutput) ExitCode() *int {
	return o.Outcome.ExitCode
}

func (o ShellCommandOutput) Status() string {
	if strings.ToLower(o.Outcome.Type) == ShellCallOutcomeTimeout {
		return "timeout"
	}
	return "completed"
}

// ShellResult is the result returned by a shell executor.
type ShellResult struct {
	Output       []ShellCommandOutput
	MaxOutputLen *int
	ProviderData map[string]any
}

// ShellActionRequest is the action payload for a shell call.
type ShellActionRequest struct {
	Commands        []string
	TimeoutMs       *int
	MaxOutputLength *int
}

// ShellCallData is the normalized shell call data.
type ShellCallData struct {
	CallID string
	Action ShellActionRequest
	Status string
	Raw    any
}

// ShellCommandRequest is the request payload for shell executors.
type ShellCommandRequest struct {
	CtxWrapper *RunContextWrapper[any]
	Data       ShellCallData
}

func normalizeShellToolEnvironment(environment map[string]any) (map[string]any, error) {
	if environment == nil {
		return map[string]any{"type": "local"}, nil
	}
	normalized := make(map[string]any, len(environment))
	for key, value := range environment {
		normalized[key] = value
	}
	if _, ok := normalized["type"]; !ok {
		normalized["type"] = "local"
	}
	return normalized, nil
}
