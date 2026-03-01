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
	"errors"
	"strconv"
	"sync"

	"github.com/denggeng/openai-agents-go-plus/computer"
	"github.com/openai/openai-go/v3/responses"
)

// ComputerCreateFunc creates a computer instance for the current run.
type ComputerCreateFunc func(context.Context, *RunContextWrapper[any]) (computer.Computer, error)

// ComputerDisposeFunc disposes a computer instance after a run ends.
type ComputerDisposeFunc func(context.Context, *RunContextWrapper[any], computer.Computer) error

// ComputerProvider defines create/dispose lifecycle callbacks for computer instances.
type ComputerProvider struct {
	Create  ComputerCreateFunc
	Dispose ComputerDisposeFunc
}

// ComputerTool is a hosted tool that lets the LLM control a computer.
type ComputerTool struct {
	// The Computer implementation, which describes the environment and
	// dimensions of the computer, as well as implements the computer actions
	// like click, screenshot, etc.
	Computer computer.Computer

	// Optional factory that creates a computer instance for each run context.
	ComputerFactory ComputerCreateFunc

	// Optional lifecycle provider that creates and disposes computer instances
	// per run context. If set, it takes precedence over ComputerFactory.
	ComputerProvider *ComputerProvider

	// Optional callback to acknowledge computer tool safety checks.
	OnSafetyCheck func(context.Context, ComputerToolSafetyCheckData) (bool, error)

	// Internal stable identity used for per-run computer cache lookups.
	cacheID string
}

func (t ComputerTool) ToolName() string {
	return "computer_use_preview"
}

func (t ComputerTool) isTool() {}

// ComputerToolSafetyCheckData provides information about a computer tool safety check.
type ComputerToolSafetyCheckData struct {
	// The agent performing the computer action.
	Agent *Agent

	// The computer tool call.
	ToolCall responses.ResponseComputerToolCall

	// The pending safety check to acknowledge.
	SafetyCheck responses.ResponseComputerToolCallPendingSafetyCheck
}

type resolvedComputer struct {
	Computer computer.Computer
	Dispose  ComputerDisposeFunc
}

var (
	computerResolutionMu  sync.Mutex
	computersByRunContext = make(map[*RunContextWrapper[any]]map[string]resolvedComputer)
	computerToolIDMu      sync.Mutex
	nextComputerToolID    uint64
)

// ResolveComputer resolves a computer instance for the given run context.
// Instances created from a factory/provider are cached for the run and reused.
func ResolveComputer(
	ctx context.Context,
	tool *ComputerTool,
	runContext *RunContextWrapper[any],
) (computer.Computer, error) {
	if tool == nil {
		return nil, NewUserError("computer tool is required")
	}
	if runContext == nil {
		return nil, NewUserError("run context is required to resolve computer tools")
	}

	cacheKey := tool.cacheKey()
	if cacheKey == "" {
		return nil, NewUserError("unable to resolve computer tool cache key")
	}

	computerResolutionMu.Lock()
	if byTool, ok := computersByRunContext[runContext]; ok {
		if cached, ok := byTool[cacheKey]; ok {
			computerResolutionMu.Unlock()
			return cached.Computer, nil
		}
	}
	computerResolutionMu.Unlock()

	resolved, disposer, err := tool.createComputer(ctx, runContext)
	if err != nil {
		return nil, err
	}
	if resolved == nil {
		return nil, NewUserError("computer tool did not provide a computer instance")
	}

	computerResolutionMu.Lock()
	byTool, ok := computersByRunContext[runContext]
	if !ok {
		byTool = make(map[string]resolvedComputer)
		computersByRunContext[runContext] = byTool
	}
	if existing, ok := byTool[cacheKey]; ok {
		computerResolutionMu.Unlock()
		return existing.Computer, nil
	}
	byTool[cacheKey] = resolvedComputer{
		Computer: resolved,
		Dispose:  disposer,
	}
	computerResolutionMu.Unlock()

	return resolved, nil
}

// DisposeResolvedComputers disposes all computers associated with the run context.
func DisposeResolvedComputers(ctx context.Context, runContext *RunContextWrapper[any]) error {
	if runContext == nil {
		return nil
	}

	computerResolutionMu.Lock()
	resolvedByTool := computersByRunContext[runContext]
	delete(computersByRunContext, runContext)
	computerResolutionMu.Unlock()

	if len(resolvedByTool) == 0 {
		return nil
	}

	var errs []error
	for _, resolved := range resolvedByTool {
		if resolved.Dispose == nil {
			continue
		}
		if err := resolved.Dispose(ctx, runContext, resolved.Computer); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// InitializeComputerTools resolves computer tools ahead of model invocation.
func InitializeComputerTools(
	ctx context.Context,
	tools []Tool,
	runContext *RunContextWrapper[any],
) error {
	if runContext == nil {
		return nil
	}

	for i, tool := range tools {
		switch current := tool.(type) {
		case ComputerTool:
			toolCopy := current
			resolved, err := ResolveComputer(ctx, &toolCopy, runContext)
			if err != nil {
				return err
			}
			toolCopy.Computer = resolved
			tools[i] = &toolCopy
		case *ComputerTool:
			resolved, err := ResolveComputer(ctx, current, runContext)
			if err != nil {
				return err
			}
			current.Computer = resolved
			tools[i] = current
		}
	}
	return nil
}

func (t *ComputerTool) createComputer(
	ctx context.Context,
	runContext *RunContextWrapper[any],
) (computer.Computer, ComputerDisposeFunc, error) {
	if t.ComputerProvider != nil && t.ComputerProvider.Create != nil {
		comp, err := t.ComputerProvider.Create(ctx, runContext)
		return comp, t.ComputerProvider.Dispose, err
	}
	if t.ComputerFactory != nil {
		comp, err := t.ComputerFactory(ctx, runContext)
		return comp, nil, err
	}
	if t.Computer != nil {
		return t.Computer, nil, nil
	}
	return nil, nil, NewUserError("computer tool has no computer, factory, or provider")
}

func (t *ComputerTool) cacheKey() string {
	if t == nil {
		return ""
	}
	id := t.ensureCacheID()
	if id == "" {
		return ""
	}
	return "tool:" + id
}

func (t *ComputerTool) ensureCacheID() string {
	computerToolIDMu.Lock()
	defer computerToolIDMu.Unlock()

	if t.cacheID == "" {
		nextComputerToolID++
		t.cacheID = strconv.FormatUint(nextComputerToolID, 10)
	}
	return t.cacheID
}
