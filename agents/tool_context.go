// Copyright 2025 The NLP Odyssey Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package agents

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/openai/openai-go/v3/responses"
)

// ToolContextData provides context data of a tool call.
type ToolContextData struct {
	// The name of the tool being invoked.
	ToolName string

	// The ID of the tool call.
	ToolCallID string

	// The raw JSON arguments passed by the model for this tool call.
	ToolArguments string
}

type toolContextDataKey struct{}

func ContextWithToolData(
	ctx context.Context,
	toolCallID string,
	toolCall responses.ResponseFunctionToolCall,
) context.Context {
	return context.WithValue(ctx, toolContextDataKey{}, &ToolContextData{
		ToolName:      toolCall.Name,
		ToolCallID:    toolCallID,
		ToolArguments: toolCall.Arguments,
	})
}

func ToolDataFromContext(ctx context.Context) *ToolContextData {
	v, _ := ctx.Value(toolContextDataKey{}).(*ToolContextData)
	return v
}

var (
	errToolCallIDRequired    = errors.New("tool_call_id must be passed to ToolContext")
	errToolNameRequired      = errors.New("tool_name must be passed to ToolContext")
	errToolArgumentsRequired = errors.New("tool_arguments must be passed to ToolContext")
)

// ToolContext captures the runtime context for a tool call.
type ToolContext[T any] struct {
	*RunContextWrapper[T]

	ToolName      string
	ToolCallID    string
	ToolArguments string

	ToolCall  *responses.ResponseFunctionToolCall
	Agent     *Agent
	RunConfig *RunConfig
}

// ToolContextOption configures a ToolContext at construction time.
type ToolContextOption[T any] func(*ToolContext[T])

func ToolContextWithToolCall[T any](call *responses.ResponseFunctionToolCall) ToolContextOption[T] {
	return func(tc *ToolContext[T]) {
		tc.ToolCall = call
	}
}

func ToolContextWithAgent[T any](agent *Agent) ToolContextOption[T] {
	return func(tc *ToolContext[T]) {
		tc.Agent = agent
	}
}

func ToolContextWithRunConfig[T any](runConfig *RunConfig) ToolContextOption[T] {
	return func(tc *ToolContext[T]) {
		tc.RunConfig = runConfig
	}
}

// NewToolContext constructs a ToolContext with required tool call fields.
func NewToolContext[T any](
	context T,
	toolName string,
	toolCallID string,
	toolArguments string,
	options ...ToolContextOption[T],
) (*ToolContext[T], error) {
	wrapper := NewRunContextWrapper(context)
	toolContext, err := newToolContextFromWrapper(wrapper, toolName, toolCallID, toolArguments)
	if err != nil {
		return nil, err
	}
	for _, opt := range options {
		opt(toolContext)
	}
	return toolContext, nil
}

// ToolContextFromAgentContext derives a ToolContext from an existing run context.
func ToolContextFromAgentContext[T any](
	ctx any,
	toolCallID string,
	toolCall *responses.ResponseFunctionToolCall,
	agent *Agent,
	runConfig *RunConfig,
) (*ToolContext[T], error) {
	if toolCallID == "" {
		return nil, errToolCallIDRequired
	}

	base, inheritedAgent, inheritedRunConfig, err := resolveToolContextBase[T](ctx)
	if err != nil {
		return nil, err
	}

	if toolCall == nil {
		return nil, errToolNameRequired
	}

	toolName := toolCall.Name
	if toolName == "" {
		return nil, errToolNameRequired
	}
	toolArguments := toolCall.Arguments
	if toolArguments == "" {
		return nil, errToolArgumentsRequired
	}

	toolAgent := agent
	if toolAgent == nil {
		toolAgent = inheritedAgent
	}
	toolRunConfig := runConfig
	if toolRunConfig == nil {
		toolRunConfig = inheritedRunConfig
	}

	wrapper := cloneRunContextWrapper(base)
	if wrapper == nil {
		return nil, fmt.Errorf("context wrapper is nil")
	}
	toolContext, err := newToolContextFromWrapper(wrapper, toolName, toolCallID, toolArguments)
	if err != nil {
		return nil, err
	}
	toolContext.ToolCall = toolCall
	toolContext.Agent = toolAgent
	toolContext.RunConfig = toolRunConfig
	return toolContext, nil
}

func newToolContextFromWrapper[T any](
	wrapper *RunContextWrapper[T],
	toolName string,
	toolCallID string,
	toolArguments string,
) (*ToolContext[T], error) {
	if toolName == "" {
		return nil, errToolNameRequired
	}
	if toolCallID == "" {
		return nil, errToolCallIDRequired
	}
	if toolArguments == "" {
		return nil, errToolArgumentsRequired
	}
	return &ToolContext[T]{
		RunContextWrapper: wrapper,
		ToolName:          toolName,
		ToolCallID:        toolCallID,
		ToolArguments:     toolArguments,
	}, nil
}

func resolveToolContextBase[T any](ctx any) (*RunContextWrapper[T], *Agent, *RunConfig, error) {
	switch typed := ctx.(type) {
	case *ToolContext[T]:
		if typed == nil || typed.RunContextWrapper == nil {
			return nil, nil, nil, fmt.Errorf("context wrapper is nil")
		}
		return typed.RunContextWrapper, typed.Agent, typed.RunConfig, nil
	case *RunContextWrapper[T]:
		if typed == nil {
			return nil, nil, nil, fmt.Errorf("context wrapper is nil")
		}
		return typed, nil, nil, nil
	default:
		return nil, nil, nil, fmt.Errorf("unsupported context type %T", ctx)
	}
}

func cloneRunContextWrapper[T any](wrapper *RunContextWrapper[T]) *RunContextWrapper[T] {
	if wrapper == nil {
		return nil
	}
	return &RunContextWrapper[T]{
		Context:   wrapper.Context,
		Usage:     wrapper.Usage,
		TurnInput: slices.Clone(wrapper.TurnInput),
		ToolInput: wrapper.ToolInput,
		approvals: wrapper.approvals,
	}
}
