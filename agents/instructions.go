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
	"fmt"
	"reflect"
)

// InstructionsGetter interface is implemented by objects that can provide instructions to an Agent.
type InstructionsGetter interface {
	GetInstructions(context.Context, *Agent) (string, error)
}

// InstructionsStr satisfies InstructionsGetter providing a simple constant string value.
type InstructionsStr string

// GetInstructions returns the string value and always nil error.
func (s InstructionsStr) GetInstructions(context.Context, *Agent) (string, error) {
	return s.String(), nil
}

func (s InstructionsStr) String() string {
	return string(s)
}

// InstructionsFunc lets you implement a function that dynamically generates instructions for an Agent.
type InstructionsFunc func(context.Context, *Agent) (string, error)

// GetInstructions returns the string value and always nil error.
func (fn InstructionsFunc) GetInstructions(ctx context.Context, a *Agent) (string, error) {
	return fn(ctx, a)
}

// InstructionsFromAny converts a supported instructions value into an InstructionsGetter.
// Supported inputs: string, InstructionsGetter, nil, or a function with signature
// func(context.Context, *Agent) string or func(context.Context, *Agent) (string, error).
func InstructionsFromAny(value any) (InstructionsGetter, error) {
	if value == nil {
		return nil, nil
	}
	switch v := value.(type) {
	case InstructionsGetter:
		return v, nil
	case string:
		return InstructionsStr(v), nil
	case func(context.Context, *Agent) (string, error):
		return InstructionsFunc(v), nil
	case func(context.Context, *Agent) string:
		return InstructionsFunc(func(ctx context.Context, a *Agent) (string, error) {
			return v(ctx, a), nil
		}), nil
	}

	typ := reflect.TypeOf(value)
	if typ == nil || typ.Kind() != reflect.Func {
		return nil, fmt.Errorf("Agent instructions must be a string, callable, or nil; got %T", value)
	}
	if typ.IsVariadic() || typ.NumIn() != 2 {
		return nil, fmt.Errorf("instructions must accept exactly 2 arguments, but got %d", typ.NumIn())
	}
	if typ.NumOut() != 1 && typ.NumOut() != 2 {
		return nil, fmt.Errorf("instructions must return string or (string, error)")
	}
	if typ.Out(0).Kind() != reflect.String {
		return nil, fmt.Errorf("instructions must return string or (string, error)")
	}
	if typ.NumOut() == 2 {
		if !typ.Out(1).Implements(reflect.TypeOf((*error)(nil)).Elem()) {
			return nil, fmt.Errorf("instructions must return string or (string, error)")
		}
	}
	ctxType := reflect.TypeOf((*context.Context)(nil)).Elem()
	agentType := reflect.TypeOf(&Agent{})
	if !reflect.TypeOf(context.Background()).AssignableTo(typ.In(0)) && !ctxType.AssignableTo(typ.In(0)) {
		return nil, fmt.Errorf("instructions must accept context.Context and *Agent")
	}
	if !agentType.AssignableTo(typ.In(1)) {
		return nil, fmt.Errorf("instructions must accept context.Context and *Agent")
	}

	fn := reflect.ValueOf(value)
	return instructionsFuncAny{fn: fn, hasError: typ.NumOut() == 2}, nil
}

type instructionsFuncAny struct {
	fn       reflect.Value
	hasError bool
}

func (i instructionsFuncAny) GetInstructions(ctx context.Context, a *Agent) (string, error) {
	if !i.fn.IsValid() {
		return "", fmt.Errorf("instructions must be callable")
	}
	outputs := i.fn.Call([]reflect.Value{reflect.ValueOf(ctx), reflect.ValueOf(a)})
	text := outputs[0].String()
	if i.hasError {
		if !outputs[1].IsNil() {
			if err, ok := outputs[1].Interface().(error); ok {
				return text, err
			}
			return text, fmt.Errorf("instructions returned non-error")
		}
	}
	return text, nil
}
