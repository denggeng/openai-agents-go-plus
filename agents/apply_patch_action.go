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
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

type applyPatchAction struct{}

func ApplyPatchAction() applyPatchAction { return applyPatchAction{} }

// Execute runs an apply_patch call, returning either a ToolCallOutputItem or a ToolApprovalItem.
func (applyPatchAction) Execute(
	ctx context.Context,
	agent *Agent,
	call ToolRunApplyPatchCall,
	hooks RunHooks,
	contextWrapper *RunContextWrapper[any],
	config RunConfig,
) (any, error) {
	applyPatchTool := call.ApplyPatchTool
	if contextWrapper == nil {
		contextWrapper = NewRunContextWrapper[any](nil)
	}

	operation, err := coerceApplyPatchOperation(call.ToolCall, contextWrapper)
	if err != nil {
		return nil, err
	}

	callID, err := extractApplyPatchCallID(call.ToolCall)
	if err != nil {
		return nil, err
	}

	needsApproval, err := evaluateApplyPatchNeedsApproval(ctx, applyPatchTool, contextWrapper, operation, callID)
	if err != nil {
		return nil, err
	}

	if needsApproval {
		approvalStatus, approvalItem, err := resolveApplyPatchApprovalStatus(
			applyPatchTool,
			callID,
			call.ToolCall,
			contextWrapper,
		)
		if err != nil {
			return nil, err
		}

		if approvalStatus.known && !approvalStatus.approved {
			rejectionMessage := resolveApprovalRejectionMessage(
				contextWrapper,
				config,
				"apply_patch",
				applyPatchTool.ToolName(),
				callID,
			)
			return buildApplyPatchRejectionItem(agent, callID, rejectionMessage), nil
		}

		if !approvalStatus.known {
			return approvalItem, nil
		}
	}

	var hooksErrors [2]error

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := hooks.OnToolStart(childCtx, agent, applyPatchTool); err != nil {
			cancel()
			hooksErrors[0] = fmt.Errorf("RunHooks.OnToolStart failed: %w", err)
		}
	}()

	if agent != nil && agent.Hooks != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := agent.Hooks.OnToolStart(childCtx, agent, applyPatchTool, nil); err != nil {
				cancel()
				hooksErrors[1] = fmt.Errorf("AgentHooks.OnToolStart failed: %w", err)
			}
		}()
	}

	wg.Wait()
	if err := errors.Join(hooksErrors[:]...); err != nil {
		return nil, err
	}

	status := ApplyPatchResultStatusCompleted
	outputText := ""

	editor := applyPatchTool.Editor
	if editor == nil {
		status = ApplyPatchResultStatusFailed
		outputText = "apply_patch editor is not configured"
	} else {
		result, err := invokeApplyPatchEditor(editor, operation)
		if err != nil {
			status = ApplyPatchResultStatusFailed
			outputText = formatApplyPatchError(err)
			Logger().Error("Apply patch editor failed", "error", err)
		} else if normalized := normalizeApplyPatchResult(result); normalized != nil {
			if normalized.Status != "" {
				status = normalized.Status
			}
			if normalized.Output != "" {
				outputText = normalized.Output
			}
		}
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := hooks.OnToolEnd(childCtx, agent, applyPatchTool, outputText); err != nil {
			cancel()
			hooksErrors[0] = fmt.Errorf("RunHooks.OnToolEnd failed: %w", err)
		}
	}()

	if agent != nil && agent.Hooks != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := agent.Hooks.OnToolEnd(childCtx, agent, applyPatchTool, outputText); err != nil {
				cancel()
				hooksErrors[1] = fmt.Errorf("AgentHooks.OnToolEnd failed: %w", err)
			}
		}()
	}

	wg.Wait()
	if err := errors.Join(hooksErrors[:]...); err != nil {
		return nil, err
	}

	rawItem := responses.ResponseInputItemApplyPatchCallOutputParam{
		CallID: callID,
		Status: string(status),
		Type:   constant.ValueOf[constant.ApplyPatchCallOutput](),
	}
	if outputText != "" {
		rawItem.Output = param.NewOpt(outputText)
	}

	return ToolCallOutputItem{
		Agent:   agent,
		RawItem: ResponseInputItemApplyPatchCallOutputParam(rawItem),
		Output:  outputText,
		Type:    "tool_call_output_item",
	}, nil
}

type applyPatchApprovalStatus struct {
	approved bool
	known    bool
}

func resolveApplyPatchApprovalStatus(
	tool ApplyPatchTool,
	callID string,
	rawItem any,
	contextWrapper *RunContextWrapper[any],
) (applyPatchApprovalStatus, ToolApprovalItem, error) {
	approvalItem := ToolApprovalItem{
		ToolName: tool.ToolName(),
		RawItem:  rawItem,
	}
	approved, known := contextWrapper.GetApprovalStatus(
		tool.ToolName(),
		callID,
		&approvalItem,
	)
	if !known && tool.OnApproval != nil {
		decision, err := tool.OnApproval(contextWrapper, approvalItem)
		if err != nil {
			return applyPatchApprovalStatus{}, approvalItem, err
		}
		applyApplyPatchApprovalDecision(contextWrapper, approvalItem, decision)
		approved, known = contextWrapper.GetApprovalStatus(
			tool.ToolName(),
			callID,
			&approvalItem,
		)
	}
	return applyPatchApprovalStatus{approved: approved, known: known}, approvalItem, nil
}

func applyApplyPatchApprovalDecision(
	contextWrapper *RunContextWrapper[any],
	approvalItem ToolApprovalItem,
	decision any,
) {
	approved, ok := parseApplyPatchApprovalDecision(decision)
	if !ok {
		return
	}
	if approved {
		contextWrapper.ApproveTool(approvalItem, false)
	} else {
		contextWrapper.RejectTool(approvalItem, false)
	}
}

func parseApplyPatchApprovalDecision(decision any) (bool, bool) {
	if decision == nil {
		return false, false
	}

	switch v := decision.(type) {
	case map[string]any:
		return boolFromMap(v, "approve")
	case map[string]bool:
		value, ok := v["approve"]
		return value, ok
	default:
		return boolFromStruct(decision, "Approve")
	}
}

func boolFromMap(values map[string]any, key string) (bool, bool) {
	value, ok := values[key]
	if !ok {
		return false, false
	}
	b, ok := value.(bool)
	return b, ok
}

func boolFromStruct(value any, fieldName string) (bool, bool) {
	if value == nil {
		return false, false
	}
	v := reflect.ValueOf(value)
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return false, false
		}
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		return false, false
	}
	field := v.FieldByName(fieldName)
	if !field.IsValid() || !field.CanInterface() {
		return false, false
	}
	b, ok := field.Interface().(bool)
	return b, ok
}

func buildApplyPatchRejectionItem(
	agent *Agent,
	callID string,
	rejectionMessage string,
) ToolCallOutputItem {
	rawItem := responses.ResponseInputItemApplyPatchCallOutputParam{
		CallID: callID,
		Status: string(ApplyPatchResultStatusFailed),
		Output: param.NewOpt(rejectionMessage),
		Type:   constant.ValueOf[constant.ApplyPatchCallOutput](),
	}
	return ToolCallOutputItem{
		Agent:   agent,
		RawItem: ResponseInputItemApplyPatchCallOutputParam(rawItem),
		Output:  rejectionMessage,
		Type:    "tool_call_output_item",
	}
}

func evaluateApplyPatchNeedsApproval(
	ctx context.Context,
	tool ApplyPatchTool,
	contextWrapper *RunContextWrapper[any],
	operation ApplyPatchOperation,
	callID string,
) (bool, error) {
	if tool.NeedsApproval == nil {
		return false, nil
	}
	return tool.NeedsApproval.NeedsApproval(ctx, contextWrapper, operation, callID)
}

func invokeApplyPatchEditor(editor ApplyPatchEditor, operation ApplyPatchOperation) (any, error) {
	switch operation.Type {
	case ApplyPatchOperationCreateFile:
		return editor.CreateFile(operation)
	case ApplyPatchOperationUpdateFile:
		return editor.UpdateFile(operation)
	case ApplyPatchOperationDeleteFile:
		return editor.DeleteFile(operation)
	default:
		return nil, ModelBehaviorErrorf("Unsupported apply_patch operation: %s", operation.Type)
	}
}

func formatApplyPatchError(err error) string {
	if err == nil {
		return ""
	}
	message := err.Error()
	if message != "" {
		return message
	}
	return fmt.Sprintf("%T", err)
}

func coerceApplyPatchOperation(toolCall any, contextWrapper *RunContextWrapper[any]) (ApplyPatchOperation, error) {
	rawOperation, ok := anyFromMap(toolCall, "operation")
	if !ok {
		rawOperation, ok = anyFromField(toolCall, "Operation")
	}
	if !ok || rawOperation == nil {
		return ApplyPatchOperation{}, ModelBehaviorErrorf("Apply patch call is missing an operation payload.")
	}

	opType, ok := stringFromMap(rawOperation, "type")
	if !ok {
		opType, ok = stringFromField(rawOperation, "Type")
	}
	if !ok {
		return ApplyPatchOperation{}, ModelBehaviorErrorf("Unknown apply_patch operation: %v", opType)
	}
	opType = strings.ToLower(opType)
	if opType != string(ApplyPatchOperationCreateFile) &&
		opType != string(ApplyPatchOperationUpdateFile) &&
		opType != string(ApplyPatchOperationDeleteFile) {
		return ApplyPatchOperation{}, ModelBehaviorErrorf("Unknown apply_patch operation: %s", opType)
	}

	path, ok := stringFromMap(rawOperation, "path")
	if !ok {
		path, ok = stringFromField(rawOperation, "Path")
	}
	if !ok || path == "" {
		return ApplyPatchOperation{}, ModelBehaviorErrorf("Apply patch operation is missing a valid path.")
	}

	diff, diffOK := stringFromMap(rawOperation, "diff")
	if !diffOK {
		diff, diffOK = stringFromField(rawOperation, "Diff")
	}

	if opType == string(ApplyPatchOperationCreateFile) || opType == string(ApplyPatchOperationUpdateFile) {
		if !diffOK || diff == "" {
			return ApplyPatchOperation{}, ModelBehaviorErrorf(
				"Apply patch operation %s is missing the required diff payload.",
				opType,
			)
		}
	} else {
		diff = ""
	}

	return ApplyPatchOperation{
		Type:       ApplyPatchOperationType(opType),
		Path:       path,
		Diff:       diff,
		CtxWrapper: contextWrapper,
	}, nil
}

func normalizeApplyPatchResult(result any) *ApplyPatchResult {
	if result == nil {
		return nil
	}

	switch v := result.(type) {
	case ApplyPatchResult:
		return &v
	case *ApplyPatchResult:
		return v
	case map[string]any:
		return applyPatchResultFromMap(v)
	case map[string]string:
		converted := make(map[string]any, len(v))
		for k, value := range v {
			converted[k] = value
		}
		return applyPatchResultFromMap(converted)
	case string:
		return &ApplyPatchResult{Output: v}
	case []byte:
		return &ApplyPatchResult{Output: string(v)}
	default:
		return &ApplyPatchResult{Output: fmt.Sprintf("%v", v)}
	}
}

func applyPatchResultFromMap(values map[string]any) *ApplyPatchResult {
	statusValue, _ := values["status"]
	outputValue, _ := values["output"]

	var status ApplyPatchResultStatus
	if statusString, ok := statusValue.(string); ok {
		normalized := strings.ToLower(statusString)
		switch normalized {
		case string(ApplyPatchResultStatusCompleted):
			status = ApplyPatchResultStatusCompleted
		case string(ApplyPatchResultStatusFailed):
			status = ApplyPatchResultStatusFailed
		}
	}

	var output string
	if outputValue != nil {
		output = fmt.Sprintf("%v", outputValue)
	}

	return &ApplyPatchResult{Status: status, Output: output}
}

func extractApplyPatchCallID(toolCall any) (string, error) {
	if v, ok := stringFromMap(toolCall, "call_id"); ok {
		return v, nil
	}
	if v, ok := stringFromField(toolCall, "CallID"); ok {
		return v, nil
	}
	if v, ok := stringFromMap(toolCall, "id"); ok {
		return v, nil
	}
	if v, ok := stringFromField(toolCall, "ID"); ok {
		return v, nil
	}
	return "", ModelBehaviorErrorf("Apply patch call is missing call_id.")
}

func parseApplyPatchCustomInput(inputJSON string) (map[string]any, error) {
	return parseApplyPatchJSON(inputJSON, "input")
}

func parseApplyPatchFunctionArgs(arguments string) (map[string]any, error) {
	return parseApplyPatchJSON(arguments, "arguments")
}

func parseApplyPatchJSON(payload string, label string) (map[string]any, error) {
	if payload == "" {
		payload = "{}"
	}
	var parsed any
	if err := json.Unmarshal([]byte(payload), &parsed); err != nil {
		return nil, ModelBehaviorErrorf("Invalid apply_patch %s JSON: %w", label, err)
	}
	mapped, ok := parsed.(map[string]any)
	if !ok {
		return nil, ModelBehaviorErrorf("Apply patch %s must be a JSON object.", label)
	}
	return mapped, nil
}

func isApplyPatchName(name string, tool *ApplyPatchTool) bool {
	if name == "" {
		return false
	}
	candidate := strings.ToLower(strings.TrimSpace(name))
	if strings.HasPrefix(candidate, "apply_patch") {
		return true
	}
	if tool != nil && strings.ToLower(strings.TrimSpace(tool.ToolName())) == candidate {
		return true
	}
	return false
}

func applyPatchOperationUnionFromMap(values map[string]any) (responses.ResponseApplyPatchToolCallOperationUnion, error) {
	typeValue, _ := values["type"]
	opType, _ := typeValue.(string)
	opType = strings.ToLower(opType)

	pathValue, _ := values["path"]
	path, _ := pathValue.(string)

	diffValue, _ := values["diff"]
	diff, _ := diffValue.(string)

	if opType == "" || path == "" {
		return responses.ResponseApplyPatchToolCallOperationUnion{}, ModelBehaviorErrorf(
			"Apply patch operation is missing a valid path.",
		)
	}
	if opType != string(ApplyPatchOperationCreateFile) &&
		opType != string(ApplyPatchOperationUpdateFile) &&
		opType != string(ApplyPatchOperationDeleteFile) {
		return responses.ResponseApplyPatchToolCallOperationUnion{}, ModelBehaviorErrorf(
			"Unknown apply_patch operation: %s",
			opType,
		)
	}

	if opType == string(ApplyPatchOperationCreateFile) || opType == string(ApplyPatchOperationUpdateFile) {
		if diff == "" {
			return responses.ResponseApplyPatchToolCallOperationUnion{}, ModelBehaviorErrorf(
				"Apply patch operation %s is missing the required diff payload.",
				opType,
			)
		}
	}

	return responses.ResponseApplyPatchToolCallOperationUnion{
		Type: opType,
		Path: path,
		Diff: diff,
	}, nil
}
