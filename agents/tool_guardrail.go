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

package agents

import (
	"context"
	"reflect"
	"runtime"
	"strings"
)

// ToolGuardrailBehaviorType is the behavior returned by a tool guardrail.
type ToolGuardrailBehaviorType string

const (
	ToolGuardrailBehaviorTypeAllow          ToolGuardrailBehaviorType = "allow"
	ToolGuardrailBehaviorTypeRejectContent  ToolGuardrailBehaviorType = "reject_content"
	ToolGuardrailBehaviorTypeRaiseException ToolGuardrailBehaviorType = "raise_exception"
)

// ToolGuardrailBehavior defines how the system should respond to a guardrail result.
type ToolGuardrailBehavior struct {
	Type ToolGuardrailBehaviorType
	// Message sent back to the model when type is reject_content.
	Message string
}

// ToolGuardrailFunctionOutput is the output of a tool guardrail function.
type ToolGuardrailFunctionOutput struct {
	// Optional information about the checks performed by the guardrail.
	OutputInfo any

	// Behavior to apply. Zero value defaults to allow.
	Behavior ToolGuardrailBehavior
}

func (o ToolGuardrailFunctionOutput) resolvedBehavior() ToolGuardrailBehavior {
	if o.Behavior.Type == "" {
		return ToolGuardrailBehavior{Type: ToolGuardrailBehaviorTypeAllow}
	}
	return o.Behavior
}

// BehaviorType returns the normalized behavior type.
func (o ToolGuardrailFunctionOutput) BehaviorType() ToolGuardrailBehaviorType {
	return o.resolvedBehavior().Type
}

// BehaviorMessage returns the normalized behavior message.
func (o ToolGuardrailFunctionOutput) BehaviorMessage() string {
	return o.resolvedBehavior().Message
}

// ToolGuardrailAllow creates a guardrail output that allows execution to continue.
func ToolGuardrailAllow(outputInfo any) ToolGuardrailFunctionOutput {
	return ToolGuardrailFunctionOutput{
		OutputInfo: outputInfo,
		Behavior: ToolGuardrailBehavior{
			Type: ToolGuardrailBehaviorTypeAllow,
		},
	}
}

// ToolGuardrailRejectContent creates a guardrail output that rejects content but keeps execution running.
func ToolGuardrailRejectContent(message string, outputInfo any) ToolGuardrailFunctionOutput {
	return ToolGuardrailFunctionOutput{
		OutputInfo: outputInfo,
		Behavior: ToolGuardrailBehavior{
			Type:    ToolGuardrailBehaviorTypeRejectContent,
			Message: message,
		},
	}
}

// ToolGuardrailRaiseException creates a guardrail output that raises a tripwire error.
func ToolGuardrailRaiseException(outputInfo any) ToolGuardrailFunctionOutput {
	return ToolGuardrailFunctionOutput{
		OutputInfo: outputInfo,
		Behavior: ToolGuardrailBehavior{
			Type: ToolGuardrailBehaviorTypeRaiseException,
		},
	}
}

// ToolInputGuardrailData is passed to a tool input guardrail.
type ToolInputGuardrailData struct {
	Context ToolContextData
	Agent   *Agent
}

// ToolOutputGuardrailData is passed to a tool output guardrail.
type ToolOutputGuardrailData struct {
	ToolInputGuardrailData
	Output any
}

// ToolInputGuardrailFunction runs before invoking a function tool.
type ToolInputGuardrailFunction func(context.Context, ToolInputGuardrailData) (ToolGuardrailFunctionOutput, error)

// ToolOutputGuardrailFunction runs after invoking a function tool.
type ToolOutputGuardrailFunction func(context.Context, ToolOutputGuardrailData) (ToolGuardrailFunctionOutput, error)

// ToolInputGuardrail runs before invoking a function tool.
type ToolInputGuardrail struct {
	GuardrailFunction ToolInputGuardrailFunction
	Name              string
}

// ToolOutputGuardrail runs after invoking a function tool.
type ToolOutputGuardrail struct {
	GuardrailFunction ToolOutputGuardrailFunction
	Name              string
}

// ToolInputGuardrailResult is the result of a tool input guardrail run.
type ToolInputGuardrailResult struct {
	Guardrail ToolInputGuardrail
	Output    ToolGuardrailFunctionOutput
}

// ToolOutputGuardrailResult is the result of a tool output guardrail run.
type ToolOutputGuardrailResult struct {
	Guardrail ToolOutputGuardrail
	Output    ToolGuardrailFunctionOutput
}

func (ig ToolInputGuardrail) GetName() string {
	if ig.Name != "" {
		return ig.Name
	}
	if fnName := functionName(ig.GuardrailFunction); fnName != "" {
		return fnName
	}
	return "tool_input_guardrail"
}

func (og ToolOutputGuardrail) GetName() string {
	if og.Name != "" {
		return og.Name
	}
	if fnName := functionName(og.GuardrailFunction); fnName != "" {
		return fnName
	}
	return "tool_output_guardrail"
}

func (ig ToolInputGuardrail) Run(
	ctx context.Context,
	data ToolInputGuardrailData,
) (ToolInputGuardrailResult, error) {
	if ig.GuardrailFunction == nil {
		return ToolInputGuardrailResult{}, UserErrorf("guardrail function must be callable, got nil")
	}
	output, err := ig.GuardrailFunction(ctx, data)
	return ToolInputGuardrailResult{
		Guardrail: ig,
		Output:    output,
	}, err
}

func (og ToolOutputGuardrail) Run(
	ctx context.Context,
	data ToolOutputGuardrailData,
) (ToolOutputGuardrailResult, error) {
	if og.GuardrailFunction == nil {
		return ToolOutputGuardrailResult{}, UserErrorf("guardrail function must be callable, got nil")
	}
	output, err := og.GuardrailFunction(ctx, data)
	return ToolOutputGuardrailResult{
		Guardrail: og,
		Output:    output,
	}, err
}

func functionName(fn any) string {
	if fn == nil {
		return ""
	}
	v := reflect.ValueOf(fn)
	if v.Kind() != reflect.Func {
		return ""
	}
	rf := runtime.FuncForPC(v.Pointer())
	if rf == nil {
		return ""
	}
	name := rf.Name()
	if idx := strings.LastIndex(name, "/"); idx >= 0 {
		name = name[idx+1:]
	}
	if idx := strings.LastIndex(name, "."); idx >= 0 && idx < len(name)-1 {
		name = name[idx+1:]
	}
	return name
}
