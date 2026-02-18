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
	"maps"
	"sync/atomic"
)

type HeadersOverrideToken struct {
	prev *map[string]string
}

type HeadersOverrideVar struct {
	store *atomic.Value
}

var (
	chatHeadersOverride      atomic.Value
	responsesHeadersOverride atomic.Value
)

// HeadersOverride stores per-call header overrides for chat completions (including LiteLLM).
var HeadersOverride = HeadersOverrideVar{store: &chatHeadersOverride}

// ResponsesHeadersOverride stores per-call header overrides for responses API calls.
var ResponsesHeadersOverride = HeadersOverrideVar{store: &responsesHeadersOverride}

func init() {
	chatHeadersOverride.Store((*map[string]string)(nil))
	responsesHeadersOverride.Store((*map[string]string)(nil))
}

func (v HeadersOverrideVar) Set(headers map[string]string) HeadersOverrideToken {
	prev := v.store.Load().(*map[string]string)
	if headers == nil {
		v.store.Store((*map[string]string)(nil))
		return HeadersOverrideToken{prev: prev}
	}
	clone := maps.Clone(headers)
	v.store.Store(&clone)
	return HeadersOverrideToken{prev: prev}
}

func (v HeadersOverrideVar) Reset(token HeadersOverrideToken) {
	v.store.Store(token.prev)
}

func (v HeadersOverrideVar) Get() map[string]string {
	value := v.store.Load().(*map[string]string)
	if value == nil {
		return nil
	}
	return *value
}
