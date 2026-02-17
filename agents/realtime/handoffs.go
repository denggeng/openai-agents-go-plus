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

package realtime

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/tracing"
	"github.com/denggeng/openai-agents-go-plus/util"
	"github.com/denggeng/openai-agents-go-plus/util/transforms"
	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/xeipuuv/gojsonschema"
)

var (
	runContextWrapperAnyType = reflect.TypeOf(&agents.RunContextWrapper[any]{})
	realtimeAgentAnyType     = reflect.TypeOf(&RealtimeAgent[any]{})
)

// RealtimeHandoffParams configures a realtime handoff helper.
type RealtimeHandoffParams struct {
	ToolNameOverride        string
	ToolDescriptionOverride string
	OnHandoff               any
	InputType               any
	IsEnabled               any
}

// RealtimeHandoff creates a handoff tool for realtime agents and panics on invalid options.
func RealtimeHandoff[T any](agent *RealtimeAgent[T], params ...RealtimeHandoffParams) agents.Handoff {
	handoff, err := SafeRealtimeHandoff(agent, params...)
	if err != nil {
		panic(err)
	}
	return handoff
}

// SafeRealtimeHandoff creates a handoff tool for realtime agents and returns an error on invalid options.
func SafeRealtimeHandoff[T any](agent *RealtimeAgent[T], params ...RealtimeHandoffParams) (agents.Handoff, error) {
	if len(params) > 1 {
		return agents.Handoff{}, agents.NewUserError("RealtimeHandoff accepts at most one params struct")
	}
	var cfg RealtimeHandoffParams
	if len(params) == 1 {
		cfg = params[0]
	}

	agentName := ""
	if agent != nil {
		agentName = agent.Name
	}

	toolName := strings.TrimSpace(cfg.ToolNameOverride)
	if toolName == "" {
		toolName = transforms.TransformStringFunctionStyle("transfer_to_" + agentName)
	}
	toolDescription := cfg.ToolDescriptionOverride
	if toolDescription == "" {
		toolDescription = fmt.Sprintf("Handoff to the %s agent to handle the request.", agentName)
	}

	invoker, err := newRealtimeHandoffInvoker(cfg.OnHandoff, cfg.InputType)
	if err != nil {
		return agents.Handoff{}, err
	}

	inputSchema, err := realtimeHandoffSchema(invoker)
	if err != nil {
		return agents.Handoff{}, err
	}

	agentForEnable := cloneRealtimeAgentForEnable(agent)
	isEnabled, err := coerceRealtimeHandoffIsEnabled(agentForEnable, cfg.IsEnabled)
	if err != nil {
		return agents.Handoff{}, err
	}
	if isEnabled == nil {
		isEnabled = agents.HandoffEnabled()
	}

	return agents.Handoff{
		ToolName:         toolName,
		ToolDescription:  toolDescription,
		InputJSONSchema:  inputSchema,
		AgentName:        agentName,
		StrictJSONSchema: param.NewOpt(true),
		IsEnabled:        isEnabled,
		InputFilter:      nil,
		OnInvokeHandoff: func(ctx context.Context, jsonInput string) (*agents.Agent, error) {
			runContext := runContextWrapperFromContext(ctx)
			if invoker != nil {
				if err := invoker.invoke(ctx, runContext, jsonInput); err != nil {
					return nil, err
				}
			}
			if strings.TrimSpace(agentName) == "" {
				return nil, fmt.Errorf("realtime handoff target agent is missing")
			}
			return &agents.Agent{Name: agentName}, nil
		},
	}, nil
}

type realtimeHandoffInvoker struct {
	fn           reflect.Value
	withInput    bool
	inputType    reflect.Type
	schemaMap    map[string]any
	schema       *gojsonschema.Schema
	returnsError bool
}

func newRealtimeHandoffInvoker(onHandoff any, inputType any) (*realtimeHandoffInvoker, error) {
	if onHandoff == nil {
		if inputType != nil {
			return nil, agents.NewUserError("on_handoff must be provided when input_type is set")
		}
		return nil, nil
	}

	fnValue := reflect.ValueOf(onHandoff)
	fnType := fnValue.Type()
	if fnType.Kind() != reflect.Func {
		return nil, agents.NewUserError("on_handoff must be callable")
	}

	if inputType == nil {
		if fnType.NumIn() != 1 {
			return nil, agents.NewUserError("on_handoff must take one argument: context")
		}
		if !runContextWrapperAnyType.AssignableTo(fnType.In(0)) {
			return nil, agents.NewUserError("on_handoff must accept *agents.RunContextWrapper[any]")
		}
		returnsError, err := validateReturnSignature(fnType)
		if err != nil {
			return nil, err
		}
		return &realtimeHandoffInvoker{
			fn:           fnValue,
			withInput:    false,
			returnsError: returnsError,
		}, nil
	}

	if fnType.NumIn() != 2 {
		return nil, agents.NewUserError("on_handoff must take two arguments: context and input")
	}
	if !runContextWrapperAnyType.AssignableTo(fnType.In(0)) {
		return nil, agents.NewUserError("on_handoff must accept *agents.RunContextWrapper[any]")
	}

	providedType := resolveInputType(inputType)
	if providedType == nil {
		return nil, agents.NewUserError("input_type must be a non-nil type or value")
	}
	if !providedType.AssignableTo(fnType.In(1)) && !fnType.In(1).AssignableTo(providedType) {
		return nil, agents.NewUserError("input_type does not match on_handoff input parameter")
	}

	returnsError, err := validateReturnSignature(fnType)
	if err != nil {
		return nil, err
	}

	schemaMap, compiled, err := schemaForInputType(fnType.In(1))
	if err != nil {
		return nil, err
	}

	return &realtimeHandoffInvoker{
		fn:           fnValue,
		withInput:    true,
		inputType:    fnType.In(1),
		schemaMap:    schemaMap,
		schema:       compiled,
		returnsError: returnsError,
	}, nil
}

func validateReturnSignature(fnType reflect.Type) (bool, error) {
	if fnType.NumOut() == 0 {
		return false, nil
	}
	if fnType.NumOut() != 1 {
		return false, agents.NewUserError("on_handoff must return either no values or a single error")
	}
	if !fnType.Out(0).Implements(reflect.TypeOf((*error)(nil)).Elem()) {
		return false, agents.NewUserError("on_handoff must return an error when returning a value")
	}
	return true, nil
}

func realtimeHandoffSchema(invoker *realtimeHandoffInvoker) (map[string]any, error) {
	if invoker != nil && invoker.withInput {
		return invoker.schemaMap, nil
	}
	return agents.EnsureStrictJSONSchema(map[string]any{})
}

func (invoker *realtimeHandoffInvoker) invoke(
	ctx context.Context,
	runContext *agents.RunContextWrapper[any],
	jsonInput string,
) error {
	if !invoker.withInput {
		return invoker.call(runContext)
	}

	if strings.TrimSpace(jsonInput) == "" {
		agents.AttachErrorToCurrentSpan(ctx, tracing.SpanError{
			Message: "Handoff function expected non-null input",
		})
		return agents.NewModelBehaviorError("handoff function expected non-null input, but got empty value")
	}

	if invoker.schema != nil {
		if err := agents.ValidateJSON(ctx, invoker.schema, jsonInput); err != nil {
			return err
		}
	}

	inputValue, err := decodeJSONValue(jsonInput, invoker.inputType)
	if err != nil {
		return agents.ModelBehaviorErrorf("failed to parse handoff input: %w", err)
	}

	return invoker.call(runContext, inputValue.Interface())
}

func (invoker *realtimeHandoffInvoker) call(
	runContext *agents.RunContextWrapper[any],
	args ...any,
) error {
	callArgs := make([]reflect.Value, 0, 1+len(args))
	callArgs = append(callArgs, reflect.ValueOf(runContext))
	for _, arg := range args {
		callArgs = append(callArgs, reflect.ValueOf(arg))
	}

	results := invoker.fn.Call(callArgs)
	if !invoker.returnsError {
		return nil
	}
	if len(results) != 1 {
		return nil
	}
	if results[0].IsNil() {
		return nil
	}
	if err, ok := results[0].Interface().(error); ok {
		return err
	}
	return nil
}

func resolveInputType(inputType any) reflect.Type {
	if inputType == nil {
		return nil
	}
	if t, ok := inputType.(reflect.Type); ok {
		return t
	}
	return reflect.TypeOf(inputType)
}

func schemaForInputType(t reflect.Type) (map[string]any, *gojsonschema.Schema, error) {
	if t == nil {
		return nil, nil, agents.NewUserError("input_type must be a non-nil type")
	}

	valueType := t
	if valueType.Kind() == reflect.Pointer {
		valueType = valueType.Elem()
	}

	reflector := &jsonschema.Reflector{
		ExpandedStruct:             true,
		RequiredFromJSONSchemaTags: false,
		AllowAdditionalProperties:  false,
	}

	switch valueType.Kind() {
	case reflect.Bool:
		return compileSchemaMap(map[string]any{"type": "boolean"})
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return compileSchemaMap(map[string]any{"type": "integer"})
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return compileSchemaMap(map[string]any{"type": "integer"})
	case reflect.Float32, reflect.Float64:
		return compileSchemaMap(map[string]any{"type": "number"})
	case reflect.String:
		return compileSchemaMap(map[string]any{"type": "string"})
	}

	var schema *jsonschema.Schema
	if valueType.Kind() == reflect.Struct && valueType.Name() == "" && valueType.NumField() == 0 {
		schema = &jsonschema.Schema{
			Version:    jsonschema.Version,
			Type:       "object",
			Properties: jsonschema.NewProperties(),
		}
		if !reflector.AllowAdditionalProperties {
			schema.AdditionalProperties = jsonschema.FalseSchema
		}
	} else {
		schema = reflector.ReflectFromType(valueType)
	}

	schemaMap, err := util.JSONMap(schema)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to transform handoff input schema: %w", err)
	}

	return compileSchemaMap(schemaMap)
}

func compileSchemaMap(schemaMap map[string]any) (map[string]any, *gojsonschema.Schema, error) {
	schemaMap, err := agents.EnsureStrictJSONSchema(schemaMap)
	if err != nil {
		return nil, nil, err
	}

	compiled, err := gojsonschema.NewSchema(gojsonschema.NewGoLoader(schemaMap))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to compile handoff input schema: %w", err)
	}

	return schemaMap, compiled, nil
}

func decodeJSONValue(jsonStr string, t reflect.Type) (reflect.Value, error) {
	holder := reflect.New(t)
	if err := json.Unmarshal([]byte(jsonStr), holder.Interface()); err != nil {
		return reflect.Value{}, err
	}
	return holder.Elem(), nil
}

func runContextWrapperFromContext(ctx context.Context) *agents.RunContextWrapper[any] {
	if ctx == nil {
		return agents.NewRunContextWrapper[any](nil)
	}
	if value, ok := agents.RunContextValueFromContext(ctx); ok {
		if wrapper, ok := value.(*agents.RunContextWrapper[any]); ok {
			return wrapper
		}
	}
	return agents.NewRunContextWrapper[any](nil)
}

func coerceRealtimeHandoffIsEnabled(
	agent *RealtimeAgent[any],
	value any,
) (agents.HandoffEnabler, error) {
	if value == nil {
		return nil, nil
	}
	switch v := value.(type) {
	case agents.HandoffEnabler:
		return v, nil
	case bool:
		return agents.NewHandoffEnabledFlag(v), nil
	}

	fnValue := reflect.ValueOf(value)
	fnType := fnValue.Type()
	if fnType.Kind() != reflect.Func {
		return nil, agents.NewUserError("is_enabled must be a bool or callable")
	}
	if fnType.NumIn() != 2 {
		return nil, agents.NewUserError("is_enabled must take two arguments: context and agent")
	}
	if !runContextWrapperAnyType.AssignableTo(fnType.In(0)) {
		return nil, agents.NewUserError("is_enabled must accept *agents.RunContextWrapper[any]")
	}
	if !realtimeAgentAnyType.AssignableTo(fnType.In(1)) {
		return nil, agents.NewUserError("is_enabled must accept *realtime.RealtimeAgent[any]")
	}

	returnsError := false
	switch fnType.NumOut() {
	case 1:
		if fnType.Out(0).Kind() != reflect.Bool {
			return nil, agents.NewUserError("is_enabled must return bool or (bool, error)")
		}
	case 2:
		if fnType.Out(0).Kind() != reflect.Bool || !fnType.Out(1).Implements(reflect.TypeOf((*error)(nil)).Elem()) {
			return nil, agents.NewUserError("is_enabled must return bool or (bool, error)")
		}
		returnsError = true
	default:
		return nil, agents.NewUserError("is_enabled must return bool or (bool, error)")
	}

	realtimeAgent := agent
	if realtimeAgent == nil {
		realtimeAgent = &RealtimeAgent[any]{}
	}

	return agents.HandoffEnablerFunc(func(ctx context.Context, _ *agents.Agent) (bool, error) {
		runContext := runContextWrapperFromContext(ctx)
		results := fnValue.Call([]reflect.Value{
			reflect.ValueOf(runContext),
			reflect.ValueOf(realtimeAgent),
		})

		enabled := false
		if len(results) > 0 {
			enabled = results[0].Bool()
		}
		if !returnsError {
			return enabled, nil
		}
		if len(results) < 2 || results[1].IsNil() {
			return enabled, nil
		}
		if err, ok := results[1].Interface().(error); ok {
			return enabled, err
		}
		return enabled, nil
	}), nil
}

func cloneRealtimeAgentForEnable[T any](agent *RealtimeAgent[T]) *RealtimeAgent[any] {
	if agent == nil {
		return &RealtimeAgent[any]{}
	}
	return &RealtimeAgent[any]{
		Name:             agent.Name,
		Instructions:     agent.Instructions,
		Prompt:           cloneStringAnyMap(agent.Prompt),
		Handoffs:         slices.Clone(agent.Handoffs),
		Tools:            slices.Clone(agent.Tools),
		OutputGuardrails: slices.Clone(agent.OutputGuardrails),
	}
}
