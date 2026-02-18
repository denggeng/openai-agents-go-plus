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
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func withUnsetEnv(t *testing.T, key string, fn func()) {
	t.Helper()
	original, had := os.LookupEnv(key)
	if err := os.Unsetenv(key); err != nil {
		require.NoError(t, err)
	}
	t.Cleanup(func() {
		if had {
			_ = os.Setenv(key, original)
		} else {
			_ = os.Unsetenv(key)
		}
	})
	fn()
}

func TestDontLogModelDataDefault(t *testing.T) {
	withUnsetEnv(t, "OPENAI_AGENTS_DONT_LOG_MODEL_DATA", func() {
		require.True(t, loadDontLogModelData())
	})
}

func TestDontLogModelDataZero(t *testing.T) {
	t.Setenv("OPENAI_AGENTS_DONT_LOG_MODEL_DATA", "0")
	require.False(t, loadDontLogModelData())
}

func TestDontLogModelDataOne(t *testing.T) {
	t.Setenv("OPENAI_AGENTS_DONT_LOG_MODEL_DATA", "1")
	require.True(t, loadDontLogModelData())
}

func TestDontLogModelDataTrue(t *testing.T) {
	t.Setenv("OPENAI_AGENTS_DONT_LOG_MODEL_DATA", "true")
	require.True(t, loadDontLogModelData())
}

func TestDontLogModelDataFalse(t *testing.T) {
	t.Setenv("OPENAI_AGENTS_DONT_LOG_MODEL_DATA", "false")
	require.False(t, loadDontLogModelData())
}

func TestDontLogToolDataDefault(t *testing.T) {
	withUnsetEnv(t, "OPENAI_AGENTS_DONT_LOG_TOOL_DATA", func() {
		require.True(t, loadDontLogToolData())
	})
}

func TestDontLogToolDataZero(t *testing.T) {
	t.Setenv("OPENAI_AGENTS_DONT_LOG_TOOL_DATA", "0")
	require.False(t, loadDontLogToolData())
}

func TestDontLogToolDataOne(t *testing.T) {
	t.Setenv("OPENAI_AGENTS_DONT_LOG_TOOL_DATA", "1")
	require.True(t, loadDontLogToolData())
}

func TestDontLogToolDataTrue(t *testing.T) {
	t.Setenv("OPENAI_AGENTS_DONT_LOG_TOOL_DATA", "true")
	require.True(t, loadDontLogToolData())
}

func TestDontLogToolDataFalse(t *testing.T) {
	t.Setenv("OPENAI_AGENTS_DONT_LOG_TOOL_DATA", "false")
	require.False(t, loadDontLogToolData())
}
