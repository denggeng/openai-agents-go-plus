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

import "context"

type runContextValueKey struct{}

// ContextWithRunContextValue stores a mutable run-context object on ctx.
// Tools can read this value via RunContextValueFromContext.
func ContextWithRunContextValue(ctx context.Context, value any) context.Context {
	return context.WithValue(ctx, runContextValueKey{}, value)
}

// RunContextValueFromContext returns a run-context object previously set on ctx.
func RunContextValueFromContext(ctx context.Context) (any, bool) {
	value := ctx.Value(runContextValueKey{})
	if value == nil {
		return nil, false
	}
	return value, true
}
