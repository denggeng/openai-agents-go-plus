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

package codex

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCreateOutputSchemaFileNoneSchema(t *testing.T) {
	result, err := CreateOutputSchemaFile(nil)
	require.NoError(t, err)
	assert.Nil(t, result.SchemaPath)
	result.Cleanup()
}

func TestCreateOutputSchemaFileRejectsNonObject(t *testing.T) {
	_, err := CreateOutputSchemaFile([]any{"not", "an", "object"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "output_schema must be a plain JSON object")
}

func TestCreateOutputSchemaFileCreatesAndCleans(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"foo": map[string]any{"type": "string"},
		},
	}

	result, err := CreateOutputSchemaFile(schema)
	require.NoError(t, err)
	require.NotNil(t, result.SchemaPath)

	raw, err := os.ReadFile(*result.SchemaPath)
	require.NoError(t, err)

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(raw, &decoded))
	assert.Equal(t, schema["type"], decoded["type"])
	result.Cleanup()
	assert.NoFileExists(t, filepath.Dir(*result.SchemaPath))
}

func TestCreateOutputSchemaFileCleanupSwallowsRemoveAllErrors(t *testing.T) {
	schema := map[string]any{"type": "object"}

	oldRemoveAll := removeAll
	removeAll = func(_ string) error {
		return errors.New("boom")
	}
	t.Cleanup(func() { removeAll = oldRemoveAll })

	result, err := CreateOutputSchemaFile(schema)
	require.NoError(t, err)
	require.NotNil(t, result.SchemaPath)
	assert.NotPanics(t, result.Cleanup)
}

func TestCreateOutputSchemaFileCleanupOnEncodeError(t *testing.T) {
	cleanupCalled := false
	oldRemoveAll := removeAll
	oldEncodeSchema := encodeSchema
	removeAll = func(_ string) error {
		cleanupCalled = true
		return nil
	}
	encodeSchema = func(_ *os.File, _ any) error {
		return errors.New("boom")
	}
	t.Cleanup(func() {
		removeAll = oldRemoveAll
		encodeSchema = oldEncodeSchema
	})

	_, err := CreateOutputSchemaFile(map[string]any{"type": "object"})
	require.Error(t, err)
	assert.True(t, cleanupCalled)
}

func TestNormalizeEnvStringifiesValues(t *testing.T) {
	options := CodexOptions{
		Env: map[any]any{
			"FOO": 1,
			2:     "bar",
		},
	}
	assert.Equal(t, map[string]string{
		"FOO": "1",
		"2":   "bar",
	}, normalizeEnv(options))
}

func TestCoerceCodexOptionsRejectsUnknownFields(t *testing.T) {
	_, err := CoerceCodexOptions(map[string]any{
		"unknown": "value",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Unknown CodexOptions field")
}

func TestCoerceThreadOptionsRejectsUnknownFields(t *testing.T) {
	_, err := CoerceThreadOptions(map[string]any{
		"unknown": "value",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Unknown ThreadOptions field")
}

func TestCoerceTurnOptionsRejectsUnknownFields(t *testing.T) {
	_, err := CoerceTurnOptions(map[string]any{
		"unknown": "value",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Unknown TurnOptions field")
}

func TestNewCodexStartAndResumeThread(t *testing.T) {
	client, err := NewCodex(map[string]any{
		"codex_path_override": "/bin/codex",
	})
	require.NoError(t, err)

	started, err := client.StartThread(map[string]any{
		"model": "gpt-test",
	})
	require.NoError(t, err)
	assert.Nil(t, started.ID())

	resumed, err := client.ResumeThread("thread-1", map[string]any{
		"model": "gpt-test",
	})
	require.NoError(t, err)
	require.NotNil(t, resumed.ID())
	assert.Equal(t, "thread-1", *resumed.ID())
}

func TestNewCodexAcceptsSubprocessStreamLimit(t *testing.T) {
	limit := 123456
	client, err := NewCodex(map[string]any{
		"codex_path_override":                 "/bin/codex",
		"codex_subprocess_stream_limit_bytes": limit,
	})
	require.NoError(t, err)

	execClient, ok := client.execClient.(*CodexExec)
	require.True(t, ok)
	assert.Equal(t, limit, execClient.subprocessStreamLimitBytes)
}

func TestResolveSubprocessStreamLimitUsesEnv(t *testing.T) {
	t.Setenv(subprocessStreamLimitEnvVar, "131072")
	client, err := NewCodexExec(nil, nil, nil)
	require.NoError(t, err)
	assert.Equal(t, 131072, client.subprocessStreamLimitBytes)
}

func TestResolveSubprocessStreamLimitExplicitOverridesEnv(t *testing.T) {
	t.Setenv(subprocessStreamLimitEnvVar, "131072")
	explicit := 262144
	client, err := NewCodexExec(nil, nil, &explicit)
	require.NoError(t, err)
	assert.Equal(t, 262144, client.subprocessStreamLimitBytes)
}

func TestResolveSubprocessStreamLimitRejectsInvalidEnv(t *testing.T) {
	t.Setenv(subprocessStreamLimitEnvVar, "not-an-int")
	_, err := NewCodexExec(nil, nil, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), subprocessStreamLimitEnvVar)
}

func TestResolveSubprocessStreamLimitRejectsOutOfRangeValue(t *testing.T) {
	invalid := 1024
	_, err := NewCodexExec(nil, nil, &invalid)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must be between")
}

func TestNormalizeInputMergesTextAndImages(t *testing.T) {
	prompt, images, err := normalizeInput([]map[string]any{
		{"type": "text", "text": "first"},
		{"type": "local_image", "path": "/tmp/a.png"},
		{"type": "text", "text": "second"},
		{"type": "local_image", "path": ""},
	})
	require.NoError(t, err)
	assert.Equal(t, "first\n\nsecond", prompt)
	assert.Equal(t, []string{"/tmp/a.png"}, images)
}
