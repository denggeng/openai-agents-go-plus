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

package tracing

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEnvReadOnFirstUse(t *testing.T) {
	t.Setenv("OPENAI_AGENTS_DISABLE_TRACING", "1")
	provider := NewDefaultTraceProvider()

	trace := provider.CreateTrace("demo", "", "", nil, false)
	_, ok := trace.(*NoOpTrace)
	assert.True(t, ok)
}

func TestEnvCachedAfterFirstUse(t *testing.T) {
	t.Setenv("OPENAI_AGENTS_DISABLE_TRACING", "0")
	provider := NewDefaultTraceProvider()

	first := provider.CreateTrace("first", "", "", nil, false)
	_, ok := first.(*TraceImpl)
	require.True(t, ok)

	t.Setenv("OPENAI_AGENTS_DISABLE_TRACING", "1")
	second := provider.CreateTrace("second", "", "", nil, false)
	_, ok = second.(*TraceImpl)
	assert.True(t, ok)
}

func TestManualOverrideAfterCache(t *testing.T) {
	t.Setenv("OPENAI_AGENTS_DISABLE_TRACING", "0")
	provider := NewDefaultTraceProvider()

	_ = provider.CreateTrace("warmup", "", "", nil, false)
	provider.SetDisabled(true)
	disabled := provider.CreateTrace("disabled", "", "", nil, false)
	_, ok := disabled.(*NoOpTrace)
	require.True(t, ok)

	provider.SetDisabled(false)
	enabled := provider.CreateTrace("enabled", "", "", nil, false)
	_, ok = enabled.(*TraceImpl)
	assert.True(t, ok)
}

func TestManualOverrideEnvDisable(t *testing.T) {
	t.Setenv("OPENAI_AGENTS_DISABLE_TRACING", "1")
	provider := NewDefaultTraceProvider()

	envDisabled := provider.CreateTrace("env_disabled", "", "", nil, false)
	_, ok := envDisabled.(*NoOpTrace)
	require.True(t, ok)

	provider.SetDisabled(false)
	reenabled := provider.CreateTrace("reenabled", "", "", nil, false)
	_, ok = reenabled.(*TraceImpl)
	assert.True(t, ok)
}
