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
	"fmt"
	"testing"

	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const applyPatchRejectionMessage = "Tool execution was not approved."

type dummyApplyPatchCall struct {
	Type      string
	CallID    string
	Operation map[string]any
}

func buildApplyPatchCall(
	tool ApplyPatchTool,
	callID string,
	operation map[string]any,
	contextWrapper *RunContextWrapper[any],
) (*Agent, *RunContextWrapper[any], ToolRunApplyPatchCall) {
	ctx := contextWrapper
	if ctx == nil {
		ctx = NewRunContextWrapper[any](nil)
	}
	agent := New("patcher").WithTools(tool)
	toolRun := ToolRunApplyPatchCall{
		ToolCall: dummyApplyPatchCall{
			Type:      "apply_patch_call",
			CallID:    callID,
			Operation: operation,
		},
		ApplyPatchTool: tool,
	}
	return agent, ctx, toolRun
}

type recordingEditor struct {
	operations []ApplyPatchOperation
}

func (r *recordingEditor) CreateFile(operation ApplyPatchOperation) (any, error) {
	r.operations = append(r.operations, operation)
	return ApplyPatchResult{Output: fmt.Sprintf("Created %s", operation.Path)}, nil
}

func (r *recordingEditor) UpdateFile(operation ApplyPatchOperation) (any, error) {
	r.operations = append(r.operations, operation)
	return ApplyPatchResult{Status: ApplyPatchResultStatusCompleted, Output: fmt.Sprintf("Updated %s", operation.Path)}, nil
}

func (r *recordingEditor) DeleteFile(operation ApplyPatchOperation) (any, error) {
	r.operations = append(r.operations, operation)
	return ApplyPatchResult{Output: fmt.Sprintf("Deleted %s", operation.Path)}, nil
}

type explodingEditor struct {
	recordingEditor
}

func (e *explodingEditor) UpdateFile(operation ApplyPatchOperation) (any, error) {
	e.operations = append(e.operations, operation)
	return nil, fmt.Errorf("boom")
}

func requireApplyPatchApproval(
	context.Context,
	*RunContextWrapper[any],
	ApplyPatchOperation,
	string,
) (bool, error) {
	return true, nil
}

func rejectToolCall(
	contextWrapper *RunContextWrapper[any],
	agent *Agent,
	rawItem any,
	toolName string,
) ToolApprovalItem {
	approvalItem := ToolApprovalItem{
		ToolName: toolName,
		RawItem:  rawItem,
	}
	contextWrapper.RejectTool(approvalItem, false)
	return approvalItem
}

func makeOnApprovalCallback(approve bool, reason string) ApplyPatchOnApprovalFunc {
	return func(_ *RunContextWrapper[any], _ ToolApprovalItem) (any, error) {
		payload := map[string]any{"approve": approve}
		if reason != "" {
			payload["reason"] = reason
		}
		return payload, nil
	}
}

func TestApplyPatchToolSuccess(t *testing.T) {
	editor := &recordingEditor{}
	tool := ApplyPatchTool{Editor: editor}

	agent, contextWrapper, toolRun := buildApplyPatchCall(
		tool,
		"call_apply",
		map[string]any{"type": "update_file", "path": "tasks.md", "diff": "-a\n+b\n"},
		nil,
	)

	result, err := ApplyPatchAction().Execute(
		context.Background(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Contains(t, outputItem.Output.(string), "Updated tasks.md")

	rawItem, ok := outputItem.RawItem.(ResponseInputItemApplyPatchCallOutputParam)
	require.True(t, ok)
	assert.Equal(t, constant.ValueOf[constant.ApplyPatchCallOutput](), rawItem.Type)
	assert.Equal(t, "completed", rawItem.Status)
	assert.Equal(t, "call_apply", rawItem.CallID)
	assert.True(t, rawItem.Output.Valid())
	assert.Contains(t, rawItem.Output.Value, "Updated tasks.md")

	require.Len(t, editor.operations, 1)
	assert.Equal(t, ApplyPatchOperationUpdateFile, editor.operations[0].Type)
	assert.Equal(t, contextWrapper, editor.operations[0].CtxWrapper)

	inputPayload := outputItem.ToInputItem()
	require.NotNil(t, inputPayload.OfApplyPatchCallOutput)
	assert.Equal(t, "completed", inputPayload.OfApplyPatchCallOutput.Status)
}

func TestApplyPatchToolFailure(t *testing.T) {
	tool := ApplyPatchTool{Editor: &explodingEditor{}}
	agent, contextWrapper, toolRun := buildApplyPatchCall(
		tool,
		"call_apply_fail",
		map[string]any{"type": "update_file", "path": "tasks.md", "diff": "-a\n+b\n"},
		nil,
	)

	result, err := ApplyPatchAction().Execute(
		context.Background(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Contains(t, outputItem.Output.(string), "boom")

	rawItem, ok := outputItem.RawItem.(ResponseInputItemApplyPatchCallOutputParam)
	require.True(t, ok)
	assert.Equal(t, "failed", rawItem.Status)
	assert.True(t, rawItem.Output.Valid())

	inputPayload := outputItem.ToInputItem()
	require.NotNil(t, inputPayload.OfApplyPatchCallOutput)
	assert.Equal(t, "failed", inputPayload.OfApplyPatchCallOutput.Status)
}

func TestApplyPatchToolAcceptsMappingOperation(t *testing.T) {
	editor := &recordingEditor{}
	tool := ApplyPatchTool{Editor: editor}
	operation := map[string]any{
		"type": "create_file",
		"path": "notes.md",
		"diff": "+hello\n",
	}
	agent, contextWrapper, toolRun := buildApplyPatchCall(
		tool,
		"call_mapping",
		operation,
		NewRunContextWrapper[any](nil),
	)

	result, err := ApplyPatchAction().Execute(
		context.Background(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	rawItem, ok := outputItem.RawItem.(ResponseInputItemApplyPatchCallOutputParam)
	require.True(t, ok)
	assert.Equal(t, "call_mapping", rawItem.CallID)
	require.Len(t, editor.operations, 1)
	assert.Equal(t, "notes.md", editor.operations[0].Path)
	assert.Equal(t, contextWrapper, editor.operations[0].CtxWrapper)
}

func TestApplyPatchToolNeedsApprovalReturnsApprovalItem(t *testing.T) {
	editor := &recordingEditor{}
	tool := ApplyPatchTool{Editor: editor, NeedsApproval: ApplyPatchNeedsApprovalFunc(requireApplyPatchApproval)}

	agent, contextWrapper, toolRun := buildApplyPatchCall(
		tool,
		"call_apply",
		map[string]any{"type": "update_file", "path": "tasks.md", "diff": "-a\n+b\n"},
		nil,
	)

	result, err := ApplyPatchAction().Execute(
		context.Background(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	approvalItem, ok := result.(ToolApprovalItem)
	require.True(t, ok)
	assert.Equal(t, "apply_patch", approvalItem.ToolName)
}

func TestApplyPatchToolNeedsApprovalRejectedReturnsRejection(t *testing.T) {
	editor := &recordingEditor{}
	tool := ApplyPatchTool{Editor: editor, NeedsApproval: ApplyPatchNeedsApprovalFunc(requireApplyPatchApproval)}
	agent, contextWrapper, toolRun := buildApplyPatchCall(
		tool,
		"call_apply",
		map[string]any{"type": "update_file", "path": "tasks.md", "diff": "-a\n+b\n"},
		NewRunContextWrapper[any](map[string]any{}),
	)

	rejectToolCall(contextWrapper, agent, toolRun.ToolCall, "apply_patch")

	result, err := ApplyPatchAction().Execute(
		context.Background(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Equal(t, applyPatchRejectionMessage, outputItem.Output)

	rawItem, ok := outputItem.RawItem.(ResponseInputItemApplyPatchCallOutputParam)
	require.True(t, ok)
	assert.Equal(t, "failed", rawItem.Status)
	assert.Equal(t, applyPatchRejectionMessage, rawItem.Output.Value)
}

func TestApplyPatchRejectionUsesRunLevelFormatter(t *testing.T) {
	editor := &recordingEditor{}
	tool := ApplyPatchTool{Editor: editor, NeedsApproval: ApplyPatchNeedsApprovalFunc(requireApplyPatchApproval)}

	agent, contextWrapper, toolRun := buildApplyPatchCall(
		tool,
		"call_apply",
		map[string]any{"type": "update_file", "path": "tasks.md", "diff": "-a\n+b\n"},
		NewRunContextWrapper[any](map[string]any{}),
	)

	rejectToolCall(contextWrapper, agent, toolRun.ToolCall, "apply_patch")

	result, err := ApplyPatchAction().Execute(
		context.Background(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{ToolErrorFormatter: ToolErrorFormatter(func(args ToolErrorFormatterArgs) any {
			return fmt.Sprintf("%s denied (%s)", args.ToolName, args.CallID)
		})},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Equal(t, "apply_patch denied (call_apply)", outputItem.Output)

	rawItem, ok := outputItem.RawItem.(ResponseInputItemApplyPatchCallOutputParam)
	require.True(t, ok)
	assert.Equal(t, "apply_patch denied (call_apply)", rawItem.Output.Value)
}

func TestApplyPatchToolOnApprovalCallbackAutoApproves(t *testing.T) {
	editor := &recordingEditor{}
	tool := ApplyPatchTool{
		Editor:        editor,
		NeedsApproval: ApplyPatchNeedsApprovalFunc(requireApplyPatchApproval),
		OnApproval:    makeOnApprovalCallback(true, ""),
	}

	agent, contextWrapper, toolRun := buildApplyPatchCall(
		tool,
		"call_apply",
		map[string]any{"type": "update_file", "path": "tasks.md", "diff": "-a\n+b\n"},
		nil,
	)

	result, err := ApplyPatchAction().Execute(
		context.Background(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Contains(t, outputItem.Output.(string), "Updated tasks.md")
	assert.Len(t, editor.operations, 1)
}

func TestApplyPatchToolOnApprovalCallbackAutoRejects(t *testing.T) {
	editor := &recordingEditor{}
	tool := ApplyPatchTool{
		Editor:        editor,
		NeedsApproval: ApplyPatchNeedsApprovalFunc(requireApplyPatchApproval),
		OnApproval:    makeOnApprovalCallback(false, "Not allowed"),
	}

	agent, contextWrapper, toolRun := buildApplyPatchCall(
		tool,
		"call_apply",
		map[string]any{"type": "update_file", "path": "tasks.md", "diff": "-a\n+b\n"},
		nil,
	)

	result, err := ApplyPatchAction().Execute(
		context.Background(),
		agent,
		toolRun,
		NoOpRunHooks{},
		contextWrapper,
		RunConfig{},
	)
	require.NoError(t, err)

	outputItem, ok := result.(ToolCallOutputItem)
	require.True(t, ok)
	assert.Equal(t, applyPatchRejectionMessage, outputItem.Output)
	assert.Len(t, editor.operations, 0)
}
