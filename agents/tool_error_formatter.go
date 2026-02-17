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

import "fmt"

const DefaultApprovalRejectionMessage = "Tool execution was not approved."

// ToolErrorKind describes the category of tool error.
type ToolErrorKind string

const ToolErrorKindApprovalRejected ToolErrorKind = "approval_rejected"

// ToolErrorFormatterArgs contains metadata passed to tool error formatters.
type ToolErrorFormatterArgs struct {
	Kind           ToolErrorKind
	ToolType       string
	ToolName       string
	CallID         string
	DefaultMessage string
	RunContext     *RunContextWrapper[any]
}

// ToolErrorFormatter resolves model-visible error text for tool failures.
type ToolErrorFormatter func(args ToolErrorFormatterArgs) any

func resolveApprovalRejectionMessage(
	contextWrapper *RunContextWrapper[any],
	runConfig RunConfig,
	toolType string,
	toolName string,
	callID string,
) string {
	formatter := runConfig.ToolErrorFormatter
	if formatter == nil {
		return DefaultApprovalRejectionMessage
	}

	args := ToolErrorFormatterArgs{
		Kind:           ToolErrorKindApprovalRejected,
		ToolType:       toolType,
		ToolName:       toolName,
		CallID:         callID,
		DefaultMessage: DefaultApprovalRejectionMessage,
		RunContext:     contextWrapper,
	}

	message, ok := invokeToolErrorFormatter(formatter, args)
	if ok {
		return message
	}
	return DefaultApprovalRejectionMessage
}

func invokeToolErrorFormatter(formatter ToolErrorFormatter, args ToolErrorFormatterArgs) (string, bool) {
	defer func() {
		if r := recover(); r != nil {
			Logger().Error("Tool error formatter panicked", "error", r)
		}
	}()

	value := formatter(args)
	message, ok := toolErrorFormatterMessageFromAny(value)
	if !ok {
		Logger().Error(
			"Tool error formatter returned invalid value",
			"type",
			fmt.Sprintf("%T", value),
		)
		return "", false
	}
	return message, true
}

func toolErrorFormatterMessageFromAny(value any) (string, bool) {
	if value == nil {
		return "", false
	}
	switch v := value.(type) {
	case string:
		return v, true
	case []byte:
		return string(v), true
	case fmt.Stringer:
		return v.String(), true
	default:
		return "", false
	}
}
