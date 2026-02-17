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

import "context"

// ApplyPatchTool lets the model request file mutations via unified diffs.
type ApplyPatchTool struct {
	Editor ApplyPatchEditor
	Name   string

	// Optional approval policy for apply_patch tool calls.
	NeedsApproval ApplyPatchNeedsApproval

	// Optional handler to auto-approve or reject when approval is required.
	OnApproval ApplyPatchOnApprovalFunc
}

func (t ApplyPatchTool) ToolName() string {
	if t.Name != "" {
		return t.Name
	}
	return "apply_patch"
}

func (ApplyPatchTool) isTool() {}

// ApplyPatchNeedsApproval determines whether an apply_patch call requires approval.
type ApplyPatchNeedsApproval interface {
	NeedsApproval(
		ctx context.Context,
		runContext *RunContextWrapper[any],
		operation ApplyPatchOperation,
		callID string,
	) (bool, error)
}

// ApplyPatchNeedsApprovalFlag is a static approval policy.
type ApplyPatchNeedsApprovalFlag struct {
	needsApproval bool
}

func (f ApplyPatchNeedsApprovalFlag) NeedsApproval(
	context.Context,
	*RunContextWrapper[any],
	ApplyPatchOperation,
	string,
) (bool, error) {
	return f.needsApproval, nil
}

// NewApplyPatchNeedsApprovalFlag creates a static apply_patch approval policy.
func NewApplyPatchNeedsApprovalFlag(needsApproval bool) ApplyPatchNeedsApprovalFlag {
	return ApplyPatchNeedsApprovalFlag{needsApproval: needsApproval}
}

// ApplyPatchNeedsApprovalEnabled always requires approval.
func ApplyPatchNeedsApprovalEnabled() ApplyPatchNeedsApproval {
	return NewApplyPatchNeedsApprovalFlag(true)
}

// ApplyPatchNeedsApprovalDisabled never requires approval.
func ApplyPatchNeedsApprovalDisabled() ApplyPatchNeedsApproval {
	return NewApplyPatchNeedsApprovalFlag(false)
}

// ApplyPatchNeedsApprovalFunc wraps a callback as an approval policy.
type ApplyPatchNeedsApprovalFunc func(
	ctx context.Context,
	runContext *RunContextWrapper[any],
	operation ApplyPatchOperation,
	callID string,
) (bool, error)

func (f ApplyPatchNeedsApprovalFunc) NeedsApproval(
	ctx context.Context,
	runContext *RunContextWrapper[any],
	operation ApplyPatchOperation,
	callID string,
) (bool, error) {
	return f(ctx, runContext, operation, callID)
}

// ApplyPatchOnApprovalFunc allows auto-approving or rejecting apply_patch calls.
type ApplyPatchOnApprovalFunc func(
	ctx *RunContextWrapper[any],
	approvalItem ToolApprovalItem,
) (any, error)
