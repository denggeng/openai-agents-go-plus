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
	"math"
	"time"
)

type functionToolInvokeResult struct {
	result any
	err    error
}

// Invoke executes the function tool handler and enforces timeout semantics when configured.
func (t FunctionTool) Invoke(ctx context.Context, arguments string) (any, error) {
	if err := validateFunctionToolTimeoutConfig(t); err != nil {
		return nil, err
	}
	if t.OnInvokeTool == nil {
		return nil, UserErrorf("function tool %q has no OnInvokeTool handler", t.Name)
	}
	if t.TimeoutSeconds == nil {
		return t.OnInvokeTool(ctx, arguments)
	}

	timeoutSeconds := *t.TimeoutSeconds
	invokeCtx, cancel := context.WithTimeout(
		ctx,
		time.Duration(timeoutSeconds*float64(time.Second)),
	)
	defer cancel()

	resultCh := make(chan functionToolInvokeResult, 1)
	go func() {
		result, err := t.OnInvokeTool(invokeCtx, arguments)
		resultCh <- functionToolInvokeResult{result: result, err: err}
	}()

	select {
	case outcome := <-resultCh:
		return outcome.result, outcome.err
	case <-invokeCtx.Done():
		select {
		case outcome := <-resultCh:
			return outcome.result, outcome.err
		default:
		}

		if !errors.Is(invokeCtx.Err(), context.DeadlineExceeded) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			return nil, invokeCtx.Err()
		}

		timeoutErr := NewToolTimeoutError(t.Name, timeoutSeconds)
		if normalizeToolTimeoutBehavior(t.TimeoutBehavior) == ToolTimeoutBehaviorRaiseException {
			return nil, timeoutErr
		}

		if t.TimeoutErrorFunction == nil {
			return DefaultToolTimeoutErrorMessage(t.Name, timeoutSeconds), nil
		}
		return (*t.TimeoutErrorFunction)(ctx, timeoutErr)
	}
}

func validateFunctionToolTimeoutConfig(tool FunctionTool) error {
	if tool.TimeoutSeconds != nil {
		timeoutSeconds := *tool.TimeoutSeconds
		switch {
		case math.IsNaN(timeoutSeconds) || math.IsInf(timeoutSeconds, 0):
			return UserErrorf("FunctionTool timeout_seconds must be a finite number.")
		case timeoutSeconds <= 0:
			return UserErrorf("FunctionTool timeout_seconds must be greater than 0.")
		}
	}

	switch normalizeToolTimeoutBehavior(tool.TimeoutBehavior) {
	case ToolTimeoutBehaviorErrorAsResult, ToolTimeoutBehaviorRaiseException:
		return nil
	default:
		return UserErrorf(
			"FunctionTool timeout_behavior must be one of: 'error_as_result', 'raise_exception'.",
		)
	}
}
