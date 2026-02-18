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
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCodexExecBuildCommandArgsIncludesOptions(t *testing.T) {
	client := &CodexExec{executablePath: "/bin/codex"}
	commandArgs := client.buildCommandArgs(CodexExecArgs{
		Model:                 ptr("gpt-4.1-mini"),
		SandboxMode:           ptr("read-only"),
		WorkingDirectory:      ptr("/work"),
		AdditionalDirectories: []string{"/extra-a", "/extra-b"},
		SkipGitRepoCheck:      ptr(true),
		OutputSchemaFile:      ptr("/tmp/schema.json"),
		ModelReasoningEffort:  ptr("high"),
		NetworkAccessEnabled:  ptr(true),
		WebSearchMode:         ptr("live"),
		ApprovalPolicy:        ptr("on-request"),
		ThreadID:              ptr("thread-123"),
		Images:                []string{"/tmp/img.png"},
	})

	assert.Equal(t, []string{
		"exec",
		"--experimental-json",
		"--model",
		"gpt-4.1-mini",
		"--sandbox",
		"read-only",
		"--cd",
		"/work",
		"--add-dir",
		"/extra-a",
		"--add-dir",
		"/extra-b",
		"--skip-git-repo-check",
		"--output-schema",
		"/tmp/schema.json",
		"--config",
		`model_reasoning_effort="high"`,
		"--config",
		"sandbox_workspace_write.network_access=true",
		"--config",
		`web_search="live"`,
		"--config",
		`approval_policy="on-request"`,
		"resume",
		"thread-123",
		"--image",
		"/tmp/img.png",
		"-",
	}, commandArgs)
}

func TestCodexExecBuildCommandArgsWebSearchEnabledFlags(t *testing.T) {
	client := &CodexExec{executablePath: "/bin/codex"}
	trueArgs := client.buildCommandArgs(CodexExecArgs{WebSearchEnabled: ptr(true)})
	assert.True(t, slices.Contains(trueArgs, `web_search="live"`))

	falseArgs := client.buildCommandArgs(CodexExecArgs{WebSearchEnabled: ptr(false)})
	assert.True(t, slices.Contains(falseArgs, `web_search="disabled"`))
}

func TestCodexExecBuildEnvHonorsOverrideAndInjectsBaseValues(t *testing.T) {
	client := &CodexExec{
		executablePath: "/bin/codex",
		envOverride:    map[string]string{"FOO": "bar"},
	}
	env := client.buildEnv(CodexExecArgs{
		BaseURL: ptr("https://example.com"),
		APIKey:  ptr("api-key"),
	})

	assert.Equal(t, "bar", env["FOO"])
	assert.Equal(t, typescriptSDKOriginator, env[internalOriginatorEnv])
	assert.Equal(t, "https://example.com", env["OPENAI_BASE_URL"])
	assert.Equal(t, "api-key", env["CODEX_API_KEY"])
}

func TestCodexExecRunJSONLStreamsLines(t *testing.T) {
	script := writeExecutableScript(t, `#!/bin/sh
while IFS= read -r _line; do
  break
done
echo '{"type":"turn.started"}'
echo '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}'
`)

	client, err := NewCodexExec(&script, nil, nil)
	require.NoError(t, err)

	linesCh, errsCh := client.RunJSONL(t.Context(), CodexExecArgs{Input: "hello"})
	lines, err := collectJSONLResult(linesCh, errsCh)
	require.NoError(t, err)
	assert.Equal(t, []string{
		`{"type":"turn.started"}`,
		`{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}`,
	}, lines)
}

func TestCodexExecRunJSONLHandlesLargeSingleLine(t *testing.T) {
	payload := strings.Repeat("x", (1<<16)+1)
	dir := t.TempDir()
	payloadPath := filepath.Join(dir, "payload.txt")
	require.NoError(t, os.WriteFile(payloadPath, []byte(payload+"\n"), 0o644))

	script := writeExecutableScript(t, fmt.Sprintf("#!/bin/sh\ncat %q\n", payloadPath))

	client, err := NewCodexExec(&script, nil, nil)
	require.NoError(t, err)

	linesCh, errsCh := client.RunJSONL(t.Context(), CodexExecArgs{Input: "hello"})
	lines, err := collectJSONLResult(linesCh, errsCh)
	require.NoError(t, err)
	require.Len(t, lines, 1)
	assert.Equal(t, payload, lines[0])
}

func TestResolveCodexPathUsesEnvOverride(t *testing.T) {
	t.Setenv("CODEX_PATH", "/custom/codex")
	assert.Equal(t, "/custom/codex", resolveCodexPath(nil))
}

func TestResolveCodexPathUsesLookPath(t *testing.T) {
	dir := t.TempDir()
	codexPath := filepath.Join(dir, "codex")
	require.NoError(t, os.WriteFile(codexPath, []byte("#!/bin/sh\nexit 0\n"), 0o755))

	t.Setenv("CODEX_PATH", "")
	t.Setenv("PATH", dir)

	resolved := resolveCodexPath(nil)
	assert.Equal(t, codexPath, resolved)
}

func TestResolveCodexPathFallbackWhenNotFound(t *testing.T) {
	t.Setenv("CODEX_PATH", "")
	t.Setenv("PATH", "")
	assert.Equal(t, "codex", resolveCodexPath(nil))
}

func TestCodexExecRunJSONLReturnsExitCodeError(t *testing.T) {
	script := writeExecutableScript(t, `#!/bin/sh
echo "bad stderr" 1>&2
exit 2
`)

	client, err := NewCodexExec(&script, nil, nil)
	require.NoError(t, err)

	linesCh, errsCh := client.RunJSONL(t.Context(), CodexExecArgs{Input: "hello"})
	lines, err := collectJSONLResult(linesCh, errsCh)
	assert.Empty(t, lines)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "exited with code 2")
	assert.Contains(t, err.Error(), "bad stderr")
}

func TestCodexExecRunJSONLTimeout(t *testing.T) {
	script := writeExecutableScript(t, `#!/bin/sh
sleep 1
echo '{"type":"turn.started"}'
`)

	client, err := NewCodexExec(&script, nil, nil)
	require.NoError(t, err)

	timeout := 0.01
	linesCh, errsCh := client.RunJSONL(t.Context(), CodexExecArgs{
		Input:              "hello",
		IdleTimeoutSeconds: &timeout,
	})
	lines, err := collectJSONLResult(linesCh, errsCh)
	assert.Empty(t, lines)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Codex stream idle for")
}

func collectJSONLResult(linesCh <-chan string, errsCh <-chan error) ([]string, error) {
	lines := make([]string, 0)
	var firstErr error
	for linesCh != nil || errsCh != nil {
		select {
		case line, ok := <-linesCh:
			if !ok {
				linesCh = nil
				continue
			}
			lines = append(lines, line)
		case err, ok := <-errsCh:
			if !ok {
				errsCh = nil
				continue
			}
			if err != nil && firstErr == nil && err != context.Canceled {
				firstErr = err
			}
		}
	}
	return lines, firstErr
}

func writeExecutableScript(t *testing.T, content string) string {
	t.Helper()

	dir := t.TempDir()
	scriptPath := filepath.Join(dir, "codex-script.sh")
	require.NoError(t, os.WriteFile(scriptPath, []byte(strings.TrimSpace(content)+"\n"), 0o755))
	return scriptPath
}
