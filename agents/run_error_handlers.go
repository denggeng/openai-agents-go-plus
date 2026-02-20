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

import "context"

// RunErrorData is a snapshot of run data passed to error handlers.
type RunErrorData struct {
	Input        Input
	NewItems     []RunItem
	History      []TResponseInputItem
	Output       []TResponseInputItem
	RawResponses []ModelResponse
	LastAgent    *Agent
}

// RunErrorHandlerInput bundles error data for run error handlers.
type RunErrorHandlerInput struct {
	Error   MaxTurnsExceededError
	Context *RunContextWrapper[any]
	RunData RunErrorData
}

// RunErrorHandlerResult is returned by an error handler.
type RunErrorHandlerResult struct {
	FinalOutput      any
	IncludeInHistory *bool
}

// RunErrorHandler handles run errors and may return a result or nil to fallback to default behavior.
type RunErrorHandler func(context.Context, RunErrorHandlerInput) (any, error)

// RunErrorHandlers configures error handlers keyed by error kind.
type RunErrorHandlers struct {
	MaxTurns RunErrorHandler
}

func includeInHistoryDefault(value *bool) bool {
	if value == nil {
		return true
	}
	return *value
}
