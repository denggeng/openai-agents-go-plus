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
	"strconv"
	"strings"
	"sync"
)

type shellAction struct{}

func ShellAction() shellAction { return shellAction{} }

// Execute runs a shell tool call, returning either a ToolCallOutputItem or ToolApprovalItem.
func (shellAction) Execute(
	ctx context.Context,
	agent *Agent,
	call ToolRunShellCall,
	hooks RunHooks,
	contextWrapper *RunContextWrapper[any],
	config RunConfig,
) (any, error) {
	shellTool := call.ShellTool
	if contextWrapper == nil {
		contextWrapper = NewRunContextWrapper[any](nil)
	}

	shellCall, err := coerceShellCall(call.ToolCall)
	if err != nil {
		return nil, err
	}

	needsApproval, err := evaluateShellNeedsApproval(ctx, shellTool, contextWrapper, shellCall.Action, shellCall.CallID)
	if err != nil {
		return nil, err
	}

	if needsApproval {
		approvalStatus, approvalItem, err := resolveShellApprovalStatus(
			shellTool,
			shellCall.CallID,
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
				"shell",
				shellTool.ToolName(),
				shellCall.CallID,
			)
			return buildShellRejectionItem(agent, shellCall.CallID, rejectionMessage), nil
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
		if err := hooks.OnToolStart(childCtx, agent, shellTool); err != nil {
			cancel()
			hooksErrors[0] = fmt.Errorf("RunHooks.OnToolStart failed: %w", err)
		}
	}()

	if agent != nil && agent.Hooks != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := agent.Hooks.OnToolStart(childCtx, agent, shellTool, nil); err != nil {
				cancel()
				hooksErrors[1] = fmt.Errorf("AgentHooks.OnToolStart failed: %w", err)
			}
		}()
	}

	wg.Wait()
	if err := errors.Join(hooksErrors[:]...); err != nil {
		return nil, err
	}

	status := "completed"
	outputText := ""
	var shellOutputPayload []map[string]any
	var providerMeta map[string]any
	var maxOutputLength *int

	requestedMaxOutputLength := normalizeMaxOutputLength(shellCall.Action.MaxOutputLength)

	request := ShellCommandRequest{
		CtxWrapper: contextWrapper,
		Data:       shellCall,
	}

	result, err := executeShellExecutor(ctx, shellTool, request)
	if err != nil {
		status = "failed"
		outputText = formatShellError(err)
		Logger().Error("Shell executor failed", "error", err)
		if requestedMaxOutputLength != nil {
			outputText = truncateString(outputText, *requestedMaxOutputLength)
			maxOutputLength = requestedMaxOutputLength
		}
	} else {
		switch v := result.(type) {
		case ShellResult:
			outputText, shellOutputPayload, providerMeta, maxOutputLength = normalizeShellResult(v, requestedMaxOutputLength)
		case *ShellResult:
			if v != nil {
				outputText, shellOutputPayload, providerMeta, maxOutputLength = normalizeShellResult(*v, requestedMaxOutputLength)
			}
		default:
			outputText = fmt.Sprintf("%v", v)
			if requestedMaxOutputLength != nil {
				outputText = truncateString(outputText, *requestedMaxOutputLength)
				maxOutputLength = requestedMaxOutputLength
			}
		}
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := hooks.OnToolEnd(childCtx, agent, shellTool, outputText); err != nil {
			cancel()
			hooksErrors[0] = fmt.Errorf("RunHooks.OnToolEnd failed: %w", err)
		}
	}()

	if agent != nil && agent.Hooks != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := agent.Hooks.OnToolEnd(childCtx, agent, shellTool, outputText); err != nil {
				cancel()
				hooksErrors[1] = fmt.Errorf("AgentHooks.OnToolEnd failed: %w", err)
			}
		}()
	}

	wg.Wait()
	if err := errors.Join(hooksErrors[:]...); err != nil {
		return nil, err
	}

	var rawEntries []map[string]any
	if len(shellOutputPayload) > 0 {
		rawEntries = shellOutputPayload
	} else if outputText != "" {
		outcome := "success"
		if status != "completed" {
			outcome = "failure"
		}
		rawEntries = []map[string]any{
			{
				"stdout":  outputText,
				"stderr":  "",
				"status":  status,
				"outcome": outcome,
			},
		}
	}

	structuredOutput := []map[string]any{}
	if len(rawEntries) > 0 {
		structuredOutput = normalizeShellOutputEntries(rawEntries)
	}

	rawItem := ShellCallOutputRawItem{
		"type":    "shell_call_output",
		"call_id": shellCall.CallID,
		"output":  sliceToAny(structuredOutput),
		"status":  status,
	}
	if maxOutputLength != nil {
		rawItem["max_output_length"] = *maxOutputLength
	}
	if len(rawEntries) > 0 {
		rawItem["shell_output"] = sliceToAny(rawEntries)
	}
	if providerMeta != nil {
		rawItem["provider_data"] = providerMeta
	}

	return ToolCallOutputItem{
		Agent:   agent,
		RawItem: rawItem,
		Output:  outputText,
		Type:    "tool_call_output_item",
	}, nil
}

type shellApprovalStatus struct {
	approved bool
	known    bool
}

func evaluateShellNeedsApproval(
	ctx context.Context,
	tool ShellTool,
	contextWrapper *RunContextWrapper[any],
	action ShellActionRequest,
	callID string,
) (bool, error) {
	if tool.NeedsApproval == nil {
		return false, nil
	}
	return tool.NeedsApproval.NeedsApproval(ctx, contextWrapper, action, callID)
}

func resolveShellApprovalStatus(
	tool ShellTool,
	callID string,
	rawItem any,
	contextWrapper *RunContextWrapper[any],
) (shellApprovalStatus, ToolApprovalItem, error) {
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
			return shellApprovalStatus{}, approvalItem, err
		}
		applyShellApprovalDecision(contextWrapper, approvalItem, decision)
		approved, known = contextWrapper.GetApprovalStatus(
			tool.ToolName(),
			callID,
			&approvalItem,
		)
	}
	return shellApprovalStatus{approved: approved, known: known}, approvalItem, nil
}

func applyShellApprovalDecision(
	contextWrapper *RunContextWrapper[any],
	approvalItem ToolApprovalItem,
	decision any,
) {
	approved, ok := parseShellApprovalDecision(decision)
	if !ok {
		return
	}
	if approved {
		contextWrapper.ApproveTool(approvalItem, false)
	} else {
		contextWrapper.RejectTool(approvalItem, false)
	}
}

func parseShellApprovalDecision(decision any) (bool, bool) {
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
		if value, ok := boolFromStruct(decision, "Approve"); ok {
			return value, true
		}
	}
	return false, false
}

func buildShellRejectionItem(
	agent *Agent,
	callID string,
	rejectionMessage string,
) ToolCallOutputItem {
	rawItem := ShellCallOutputRawItem{
		"type":    "shell_call_output",
		"call_id": callID,
		"output": []any{
			map[string]any{
				"stdout": "",
				"stderr": rejectionMessage,
				"outcome": map[string]any{
					"type":      ShellCallOutcomeExit,
					"exit_code": 1,
				},
			},
		},
	}
	return ToolCallOutputItem{
		Agent:   agent,
		RawItem: rawItem,
		Output:  rejectionMessage,
		Type:    "tool_call_output_item",
	}
}

func executeShellExecutor(
	ctx context.Context,
	tool ShellTool,
	request ShellCommandRequest,
) (any, error) {
	if tool.Executor == nil {
		return nil, ModelBehaviorErrorf("Shell tool has no local executor configured.")
	}
	return tool.Executor(ctx, request)
}

func normalizeShellResult(
	result ShellResult,
	requestedMaxOutputLength *int,
) (string, []map[string]any, map[string]any, *int) {
	normalized := make([]ShellCommandOutput, len(result.Output))
	for i, entry := range result.Output {
		normalized[i] = normalizeShellOutput(entry)
	}

	resultMaxOutputLength := normalizeMaxOutputLength(result.MaxOutputLen)
	maxOutputLength := mergeMaxOutputLengths(resultMaxOutputLength, requestedMaxOutputLength)

	if maxOutputLength != nil {
		normalized = truncateShellOutputs(normalized, *maxOutputLength)
	}

	outputText := renderShellOutputs(normalized)
	if maxOutputLength != nil {
		outputText = truncateString(outputText, *maxOutputLength)
	}

	shellOutputPayload := make([]map[string]any, len(normalized))
	for i, entry := range normalized {
		shellOutputPayload[i] = serializeShellOutput(entry)
	}

	var providerMeta map[string]any
	if result.ProviderData != nil {
		providerMeta = result.ProviderData
	}

	return outputText, shellOutputPayload, providerMeta, maxOutputLength
}

func mergeMaxOutputLengths(resultMax, requestedMax *int) *int {
	if resultMax == nil {
		return requestedMax
	}
	if requestedMax == nil {
		return resultMax
	}
	minValue := *resultMax
	if *requestedMax < minValue {
		minValue = *requestedMax
	}
	return &minValue
}

func extractShellCallID(toolCall any) (string, error) {
	if v, ok := stringFromMap(toolCall, "call_id"); ok && v != "" {
		return v, nil
	}
	if v, ok := stringFromMap(toolCall, "id"); ok && v != "" {
		return v, nil
	}
	if v, ok := stringFromField(toolCall, "CallID"); ok && v != "" {
		return v, nil
	}
	if v, ok := stringFromField(toolCall, "ID"); ok && v != "" {
		return v, nil
	}
	return "", ModelBehaviorErrorf("Shell call is missing call_id.")
}

func coerceShellCall(toolCall any) (ShellCallData, error) {
	callID, err := extractShellCallID(toolCall)
	if err != nil {
		return ShellCallData{}, err
	}

	actionPayload, ok := anyFromMap(toolCall, "action")
	if !ok {
		actionPayload, ok = anyFromField(toolCall, "Action")
	}
	if !ok || actionPayload == nil {
		return ShellCallData{}, ModelBehaviorErrorf("Shell call is missing an action payload.")
	}

	commandsValue, ok := anyFromMap(actionPayload, "commands")
	if !ok {
		commandsValue, ok = anyFromMap(actionPayload, "command")
	}
	if !ok {
		commandsValue, ok = anyFromField(actionPayload, "Commands")
	}
	if !ok {
		commandsValue, ok = anyFromField(actionPayload, "Command")
	}
	commands, err := coerceShellCommands(commandsValue)
	if err != nil {
		return ShellCallData{}, err
	}

	timeoutValue := firstNonNil(
		valueFromMap(actionPayload, "timeout_ms"),
		valueFromMap(actionPayload, "timeoutMs"),
		valueFromMap(actionPayload, "timeout"),
		valueFromField(actionPayload, "TimeoutMs"),
		valueFromField(actionPayload, "Timeout"),
	)
	timeoutMs := intFromValue(timeoutValue)

	maxLengthValue := firstNonNil(
		valueFromMap(actionPayload, "max_output_length"),
		valueFromMap(actionPayload, "maxOutputLength"),
		valueFromField(actionPayload, "MaxOutputLength"),
	)
	maxOutputLength := intFromValue(maxLengthValue)

	action := ShellActionRequest{
		Commands:        commands,
		TimeoutMs:       timeoutMs,
		MaxOutputLength: maxOutputLength,
	}

	statusValue := ""
	if v, ok := stringFromMap(toolCall, "status"); ok {
		statusValue = v
	} else if v, ok := stringFromField(toolCall, "Status"); ok {
		statusValue = v
	}
	status := ""
	if statusValue != "" {
		normalized := strings.ToLower(statusValue)
		if normalized == "in_progress" || normalized == "completed" {
			status = normalized
		}
	}

	return ShellCallData{
		CallID: callID,
		Action: action,
		Status: status,
		Raw:    toolCall,
	}, nil
}

func coerceShellCommands(value any) ([]string, error) {
	if value == nil {
		return nil, ModelBehaviorErrorf("Shell call action is missing commands.")
	}

	switch v := value.(type) {
	case []string:
		if len(v) == 0 {
			return nil, ModelBehaviorErrorf("Shell call action must include at least one command.")
		}
		return append([]string(nil), v...), nil
	case []any:
		return normalizeCommandList(v)
	default:
		rv := reflect.ValueOf(value)
		if rv.Kind() == reflect.Slice {
			entries := make([]any, 0, rv.Len())
			for i := 0; i < rv.Len(); i++ {
				entries = append(entries, rv.Index(i).Interface())
			}
			return normalizeCommandList(entries)
		}
	}
	return nil, ModelBehaviorErrorf("Shell call action is missing commands.")
}

func normalizeCommandList(entries []any) ([]string, error) {
	commands := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry == nil {
			continue
		}
		commands = append(commands, fmt.Sprintf("%v", entry))
	}
	if len(commands) == 0 {
		return nil, ModelBehaviorErrorf("Shell call action must include at least one command.")
	}
	return commands, nil
}

func firstNonNil(values ...any) any {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func valueFromMap(raw any, key string) any {
	if value, ok := anyFromMap(raw, key); ok {
		return value
	}
	return nil
}

func valueFromField(raw any, fieldName string) any {
	if value, ok := anyFromField(raw, fieldName); ok {
		return value
	}
	return nil
}

func intFromValue(value any) *int {
	switch v := value.(type) {
	case nil:
		return nil
	case int:
		return &v
	case int64:
		out := int(v)
		return &out
	case int32:
		out := int(v)
		return &out
	case float64:
		out := int(v)
		return &out
	case float32:
		out := int(v)
		return &out
	case json.Number:
		if parsed, err := v.Int64(); err == nil {
			out := int(parsed)
			return &out
		}
	case string:
		if parsed, err := strconv.Atoi(v); err == nil {
			return &parsed
		}
	}
	return nil
}

func normalizeShellOutput(entry any) ShellCommandOutput {
	switch v := entry.(type) {
	case ShellCommandOutput:
		return v
	case *ShellCommandOutput:
		if v != nil {
			return *v
		}
	case map[string]any:
		return shellCommandOutputFromMap(v)
	case map[string]string:
		converted := make(map[string]any, len(v))
		for k, value := range v {
			converted[k] = value
		}
		return shellCommandOutputFromMap(converted)
	}
	return ShellCommandOutput{
		Stdout: fmt.Sprintf("%v", entry),
		Outcome: ShellCallOutcome{
			Type: ShellCallOutcomeExit,
		},
	}
}

func shellCommandOutputFromMap(values map[string]any) ShellCommandOutput {
	stdout := stringValue(values["stdout"])
	stderr := stringValue(values["stderr"])
	commandValue := values["command"]
	providerDataValue := values["provider_data"]
	outcomeValue := values["outcome"]

	outcomeType := ShellCallOutcomeExit
	var exitCode *int

	if outcomeMap, ok := outcomeValue.(map[string]any); ok {
		if typeValue, ok := outcomeMap["type"].(string); ok && typeValue == ShellCallOutcomeTimeout {
			outcomeType = ShellCallOutcomeTimeout
		}
		exitCode = normalizeShellExitCode(outcomeMap["exit_code"])
	} else {
		statusStr := strings.ToLower(stringValue(values["status"]))
		if statusStr == "timeout" {
			outcomeType = ShellCallOutcomeTimeout
		}
		if outcomeStr, ok := outcomeValue.(string); ok {
			switch outcomeStr {
			case "failure":
				exitCode = ptrInt(1)
			case "success":
				exitCode = ptrInt(0)
			}
		}
		if exitCode == nil {
			exitCode = normalizeShellExitCode(values["exit_code"])
		}
	}

	outcome := ShellCallOutcome{
		Type:     outcomeType,
		ExitCode: exitCode,
	}

	var command *string
	if commandValue != nil {
		value := fmt.Sprintf("%v", commandValue)
		command = &value
	}

	var providerData map[string]any
	if v, ok := providerDataValue.(map[string]any); ok {
		providerData = v
	}

	return ShellCommandOutput{
		Stdout:       stdout,
		Stderr:       stderr,
		Outcome:      outcome,
		Command:      command,
		ProviderData: providerData,
	}
}

func serializeShellOutput(output ShellCommandOutput) map[string]any {
	payload := map[string]any{
		"stdout":  output.Stdout,
		"stderr":  output.Stderr,
		"status":  output.Status(),
		"outcome": map[string]any{"type": normalizeShellOutcomeType(output.Outcome.Type)},
	}

	if output.Outcome.Type == ShellCallOutcomeExit {
		payload["outcome"].(map[string]any)["exit_code"] = output.Outcome.ExitCode
		if output.Outcome.ExitCode != nil {
			payload["exit_code"] = *output.Outcome.ExitCode
		}
	}
	if output.Command != nil {
		payload["command"] = *output.Command
	}
	if output.ProviderData != nil {
		payload["provider_data"] = output.ProviderData
	}
	return payload
}

func normalizeShellExitCode(value any) *int {
	switch v := value.(type) {
	case nil:
		return nil
	case int:
		return &v
	case *int:
		if v == nil {
			return nil
		}
		return v
	case *int64:
		if v == nil {
			return nil
		}
		out := int(*v)
		return &out
	case *int32:
		if v == nil {
			return nil
		}
		out := int(*v)
		return &out
	case *float64:
		if v == nil {
			return nil
		}
		out := int(*v)
		return &out
	case *float32:
		if v == nil {
			return nil
		}
		out := int(*v)
		return &out
	case int64:
		out := int(v)
		return &out
	case int32:
		out := int(v)
		return &out
	case float64:
		out := int(v)
		return &out
	case float32:
		out := int(v)
		return &out
	case json.Number:
		if parsed, err := v.Int64(); err == nil {
			out := int(parsed)
			return &out
		}
	case string:
		if parsed, err := strconv.Atoi(v); err == nil {
			return &parsed
		}
	}
	return nil
}

func resolveShellExitCode(rawExitCode any, outcomeStatus string) int {
	if normalized := normalizeShellExitCode(rawExitCode); normalized != nil {
		return *normalized
	}
	switch strings.ToLower(outcomeStatus) {
	case "success":
		return 0
	case "failure":
		return 1
	default:
		return 0
	}
}

func renderShellOutputs(outputs []ShellCommandOutput) string {
	if len(outputs) == 0 {
		return "(no output)"
	}

	rendered := make([]string, 0, len(outputs))
	for _, result := range outputs {
		lines := make([]string, 0, 6)
		if result.Command != nil && *result.Command != "" {
			lines = append(lines, "$ "+*result.Command)
		}

		stdout := strings.TrimRight(result.Stdout, "\n")
		stderr := strings.TrimRight(result.Stderr, "\n")

		if stdout != "" {
			lines = append(lines, stdout)
		}
		if stderr != "" {
			if stdout != "" {
				lines = append(lines, "")
			}
			lines = append(lines, "stderr:")
			lines = append(lines, stderr)
		}

		if exitCode := result.ExitCode(); exitCode != nil && *exitCode != 0 {
			lines = append(lines, fmt.Sprintf("exit code: %d", *exitCode))
		}
		if result.Status() == "timeout" {
			lines = append(lines, "status: timeout")
		}

		chunk := strings.TrimSpace(strings.Join(lines, "\n"))
		if chunk == "" {
			chunk = "(no output)"
		}
		rendered = append(rendered, chunk)
	}
	return strings.Join(rendered, "\n\n")
}

func truncateShellOutputs(outputs []ShellCommandOutput, maxLength int) []ShellCommandOutput {
	if maxLength <= 0 {
		truncated := make([]ShellCommandOutput, len(outputs))
		for i, output := range outputs {
			truncated[i] = ShellCommandOutput{
				Stdout:       "",
				Stderr:       "",
				Outcome:      output.Outcome,
				Command:      output.Command,
				ProviderData: output.ProviderData,
			}
		}
		return truncated
	}

	remaining := maxLength
	truncated := make([]ShellCommandOutput, 0, len(outputs))
	for _, output := range outputs {
		stdout := ""
		stderr := ""
		if remaining > 0 && output.Stdout != "" {
			stdout = output.Stdout
			if len(stdout) > remaining {
				stdout = stdout[:remaining]
			}
			remaining -= len(stdout)
		}
		if remaining > 0 && output.Stderr != "" {
			stderr = output.Stderr
			if len(stderr) > remaining {
				stderr = stderr[:remaining]
			}
			remaining -= len(stderr)
		}
		truncated = append(truncated, ShellCommandOutput{
			Stdout:       stdout,
			Stderr:       stderr,
			Outcome:      output.Outcome,
			Command:      output.Command,
			ProviderData: output.ProviderData,
		})
	}
	return truncated
}

func normalizeShellOutputEntries(entries []map[string]any) []map[string]any {
	structured := make([]map[string]any, 0, len(entries))
	for _, entry := range entries {
		sanitized := make(map[string]any, len(entry))
		for key, value := range entry {
			sanitized[key] = value
		}
		statusValue := sanitized["status"]
		delete(sanitized, "status")
		delete(sanitized, "provider_data")
		rawExitCode := sanitized["exit_code"]
		delete(sanitized, "exit_code")
		delete(sanitized, "command")

		switch outcomeValue := sanitized["outcome"].(type) {
		case string:
			resolvedType := ShellCallOutcomeExit
			if statusValue == "timeout" {
				resolvedType = ShellCallOutcomeTimeout
			}
			outcomePayload := map[string]any{"type": resolvedType}
			if resolvedType == ShellCallOutcomeExit {
				outcomePayload["exit_code"] = resolveShellExitCode(rawExitCode, outcomeValue)
			}
			sanitized["outcome"] = outcomePayload
		case map[string]any:
			outcomePayload := make(map[string]any, len(outcomeValue))
			for key, value := range outcomeValue {
				outcomePayload[key] = value
			}
			outcomeStatus := outcomePayload["status"]
			delete(outcomePayload, "status")
			outcomeType, _ := outcomePayload["type"].(string)
			if outcomeType != ShellCallOutcomeTimeout {
				statusStr, _ := outcomeStatus.(string)
				if _, ok := outcomePayload["exit_code"]; !ok {
					outcomePayload["exit_code"] = resolveShellExitCode(rawExitCode, statusStr)
				}
			}
			if rawExitCode, ok := outcomePayload["exit_code"]; ok {
				if normalized := normalizeShellExitCode(rawExitCode); normalized != nil {
					outcomePayload["exit_code"] = *normalized
				}
			}
			sanitized["outcome"] = outcomePayload
		}

		structured = append(structured, sanitized)
	}
	return structured
}

func normalizeMaxOutputLength(value *int) *int {
	if value == nil {
		return nil
	}
	if *value < 0 {
		return ptrInt(0)
	}
	return value
}

func formatShellError(err error) string {
	if err == nil {
		return ""
	}
	message := err.Error()
	if message != "" {
		return message
	}
	return fmt.Sprintf("%T", err)
}

func truncateString(value string, maxLength int) string {
	if maxLength <= 0 {
		return ""
	}
	if len(value) <= maxLength {
		return value
	}
	return value[:maxLength]
}

func sliceToAny[T any](values []T) []any {
	out := make([]any, len(values))
	for i, value := range values {
		out[i] = value
	}
	return out
}

func stringValue(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		return fmt.Sprintf("%v", v)
	}
}

func normalizeShellOutcomeType(value string) string {
	if value == "" {
		return ShellCallOutcomeExit
	}
	return value
}

func ptrInt(v int) *int {
	return &v
}
