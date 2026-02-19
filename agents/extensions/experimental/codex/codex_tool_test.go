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
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/tracing"
	"github.com/denggeng/openai-agents-go-plus/tracing/tracingtesting"
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type recordingExec struct {
	mu     sync.Mutex
	events []any
	args   []CodexExecArgs
}

func (r *recordingExec) RunJSONL(ctx context.Context, args CodexExecArgs) (<-chan string, <-chan error) {
	r.mu.Lock()
	r.args = append(r.args, args)
	events := append([]any(nil), r.events...)
	r.mu.Unlock()

	lines := make(chan string)
	errs := make(chan error, 1)

	go func() {
		defer close(lines)
		defer close(errs)

		for _, event := range events {
			raw, err := json.Marshal(event)
			if err != nil {
				errs <- err
				return
			}
			select {
			case <-ctx.Done():
				errs <- ctx.Err()
				return
			case lines <- string(raw):
			}
		}
	}()

	return lines, errs
}

func (r *recordingExec) Args() []CodexExecArgs {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]CodexExecArgs, len(r.args))
	copy(out, r.args)
	return out
}

type schemaCaptureExec struct {
	mu     sync.Mutex
	events []any
	args   []CodexExecArgs
	schema map[string]any
}

func (s *schemaCaptureExec) RunJSONL(ctx context.Context, args CodexExecArgs) (<-chan string, <-chan error) {
	s.mu.Lock()
	s.args = append(s.args, args)
	events := append([]any(nil), s.events...)
	if args.OutputSchemaFile != nil {
		raw, err := os.ReadFile(*args.OutputSchemaFile)
		if err == nil {
			var decoded map[string]any
			if jsonErr := json.Unmarshal(raw, &decoded); jsonErr == nil {
				s.schema = decoded
			}
		}
	}
	s.mu.Unlock()

	lines := make(chan string)
	errs := make(chan error, 1)

	go func() {
		defer close(lines)
		defer close(errs)

		for _, event := range events {
			raw, err := json.Marshal(event)
			if err != nil {
				errs <- err
				return
			}
			select {
			case <-ctx.Done():
				errs <- ctx.Err()
				return
			case lines <- string(raw):
			}
		}
	}()

	return lines, errs
}

func (s *schemaCaptureExec) Schema() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return cloneStringAnyMap(s.schema)
}

func TestCoerceCodexToolOptionsRejectsUnknownFields(t *testing.T) {
	_, err := CoerceCodexToolOptions(map[string]any{"unknown": "value"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Unknown Codex tool option")
}

func TestResolveCodexToolNameValidation(t *testing.T) {
	name := "codex_engineer"
	resolved, err := resolveCodexToolName(&name)
	require.NoError(t, err)
	assert.Equal(t, "codex_engineer", resolved)

	invalid := "engineer"
	_, err = resolveCodexToolName(&invalid)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `must be "codex" or start with "codex_"`)
}

func TestCodexToolAllowsNonAlnumSuffixWhenRunContextThreadIDDisabled(t *testing.T) {
	name := "codex_a-b"
	tool, err := NewCodexTool(CodexToolOptions{Name: &name})
	require.NoError(t, err)
	assert.Equal(t, "codex_a-b", tool.Name)
}

func TestCodexToolResultStringifies(t *testing.T) {
	threadID := "thread-1"
	result := CodexToolResult{
		ThreadID: &threadID,
		Response: "ok",
		Usage: &Usage{
			InputTokens:       1,
			CachedInputTokens: 0,
			OutputTokens:      1,
		},
	}
	raw := result.String()
	var decoded map[string]any
	require.NoError(t, json.Unmarshal([]byte(raw), &decoded))
	assert.Equal(t, "ok", decoded["response"])
	assert.Equal(t, "thread-1", decoded["thread_id"])
}

func TestParseCodexToolInputRejectsInvalidJSON(t *testing.T) {
	_, err := parseCodexToolInput("{bad")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Invalid JSON input for codex tool")
}

func TestParseCodexToolInputValidationErrors(t *testing.T) {
	_, err := parseCodexToolInput(`{"inputs":[{"type":"text","text":"","path":""}]}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `non-empty "text"`)

	_, err = parseCodexToolInput(`{"inputs":[{"type":"local_image","path":"","text":""}]}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `non-empty "path"`)

	_, err = parseCodexToolInput(`{"inputs":[{"type":"text","text":"hello","path":"x"}]}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `"path" is not allowed`)
}

func TestParseCodexToolInputThreadIDValidationErrors(t *testing.T) {
	_, err := parseCodexToolInput(`{"inputs":[{"type":"text","text":"hello"}],"thread_id":"   "}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "thread_id")
}

func TestParseCodexToolInputAllowsEmptyInputs(t *testing.T) {
	parsed, err := parseCodexToolInput(`{"inputs":[]}`)
	require.NoError(t, err)
	assert.Empty(t, parsed.Inputs)
	assert.Nil(t, parsed.ThreadID)
}

func TestParseCodexToolInputWithCustomParametersRequiresInputs(t *testing.T) {
	_, err := parseCodexToolInputWithCustomParameters(`{}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "inputs field")
}

func TestParseCodexToolInputWithCustomParametersAllowsEmptyInputs(t *testing.T) {
	parsed, err := parseCodexToolInputWithCustomParameters(`{"inputs":[]}`)
	require.NoError(t, err)
	assert.Empty(t, parsed.Inputs)
	assert.Nil(t, parsed.ThreadID)
}

func TestNormalizeCodexToolInputItemAcceptsLocalImage(t *testing.T) {
	path := " /tmp/img.png "
	item := CodexToolInputItem{Type: "local_image", Path: &path}
	normalized, err := normalizeCodexToolInputItem(item)
	require.NoError(t, err)
	assert.Equal(t, map[string]any{"type": "local_image", "path": "/tmp/img.png"}, normalized)
}

func TestCodexToolOnInvokeToolHandlesFailureErrorFunction(t *testing.T) {
	handler := agents.ToolErrorFunction(func(context.Context, error) (any, error) {
		return "handled", nil
	})
	tool, err := NewCodexTool(CodexToolOptions{FailureErrorFunction: &handler})
	require.NoError(t, err)

	result, err := tool.OnInvokeTool(t.Context(), "{bad")
	require.NoError(t, err)
	assert.Equal(t, "handled", result)
}

func TestCodexToolOnInvokeToolRaisesWithoutFailureHandler(t *testing.T) {
	tool, err := NewCodexTool(CodexToolOptions{})
	require.NoError(t, err)

	result, err := tool.OnInvokeTool(t.Context(), "{bad")
	require.Error(t, err)
	var modelErr agents.ModelBehaviorError
	assert.ErrorAs(t, err, &modelErr)
	assert.Nil(t, result)
}

func TestResolveOutputSchemaDescriptor(t *testing.T) {
	schema, err := resolveOutputSchema(map[string]any{
		"title": "Summary",
		"properties": []any{
			map[string]any{
				"name": "summary",
				"schema": map[string]any{
					"type": "string",
				},
			},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, schema)
	assert.Equal(t, "object", schema["type"])
	assert.Equal(t, false, schema["additionalProperties"])

	properties, ok := schema["properties"].(map[string]any)
	require.True(t, ok)
	summary, ok := properties["summary"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "string", summary["type"])
}

func TestResolveOutputSchemaRejectsInvalid(t *testing.T) {
	_, err := resolveOutputSchema(map[string]any{"type": "string"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), `type "object"`)
}

func TestNewCodexToolAcceptsOutputSchemaDescriptor(t *testing.T) {
	execClient := &schemaCaptureExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-1"},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "Codex done."},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	descriptor := map[string]any{
		"title": "Summary",
		"properties": []any{
			map[string]any{
				"name":        "summary",
				"description": "Short summary",
				"schema": map[string]any{
					"type":        "string",
					"description": "Summary field",
				},
			},
		},
	}

	tool, err := NewCodexTool(CodexToolOptions{
		Codex:        codexClient,
		OutputSchema: descriptor,
	})
	require.NoError(t, err)

	_, err = tool.OnInvokeTool(t.Context(), `{"inputs":[{"type":"text","text":"Check schema"}]}`)
	require.NoError(t, err)

	schema := execClient.Schema()
	require.NotNil(t, schema)
	assert.Equal(t, "object", schema["type"])
	assert.Equal(t, false, schema["additionalProperties"])
	properties, ok := schema["properties"].(map[string]any)
	require.True(t, ok)
	summary, ok := properties["summary"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "string", summary["type"])
	assert.Equal(t, "Short summary", summary["description"])
}

func TestBuildDefaultCodexToolResponse(t *testing.T) {
	assert.Equal(t, "Codex task completed with no inputs.", buildDefaultCodexToolResponse(codexToolCallArguments{}))
	assert.Equal(t, "Codex task completed with inputs.", buildDefaultCodexToolResponse(codexToolCallArguments{
		Inputs: []map[string]any{{"type": "text", "text": "hello"}},
	}))
}

func TestConsumeCodexToolEventsDefaultResponse(t *testing.T) {
	events := make(chan ThreadEvent, 1)
	errs := make(chan error, 1)
	events <- TurnCompletedEvent{
		Usage: &Usage{InputTokens: 1, CachedInputTokens: 0, OutputTokens: 1},
	}
	close(events)
	close(errs)

	threadID := "thread-1"
	thread := &Thread{id: &threadID}
	streamed := &StreamedTurn{Events: events, Errors: errs}

	response, usage, resolvedThreadID, err := consumeCodexToolEvents(
		t.Context(),
		streamed,
		codexToolCallArguments{Inputs: []map[string]any{{"type": "text", "text": "hello"}}},
		thread,
		nil,
		nil,
	)
	require.NoError(t, err)
	assert.Equal(t, "Codex task completed with inputs.", response)
	require.NotNil(t, usage)
	assert.Equal(t, 1, usage.InputTokens)
	require.NotNil(t, resolvedThreadID)
	assert.Equal(t, "thread-1", *resolvedThreadID)
}

func TestConsumeCodexToolEventsDefaultResponseNoInputs(t *testing.T) {
	events := make(chan ThreadEvent, 1)
	errs := make(chan error, 1)
	events <- TurnCompletedEvent{
		Usage: &Usage{InputTokens: 1, CachedInputTokens: 0, OutputTokens: 1},
	}
	close(events)
	close(errs)

	threadID := "thread-1"
	thread := &Thread{id: &threadID}
	streamed := &StreamedTurn{Events: events, Errors: errs}

	response, usage, resolvedThreadID, err := consumeCodexToolEvents(
		t.Context(),
		streamed,
		codexToolCallArguments{},
		thread,
		nil,
		nil,
	)
	require.NoError(t, err)
	assert.Equal(t, "Codex task completed with no inputs.", response)
	require.NotNil(t, usage)
	assert.Equal(t, 1, usage.InputTokens)
	require.NotNil(t, resolvedThreadID)
	assert.Equal(t, "thread-1", *resolvedThreadID)
}

func TestConsumeCodexToolEventsWithOnStreamError(t *testing.T) {
	events := make(chan ThreadEvent, 6)
	errs := make(chan error, 1)
	exitCode := 0
	events <- ItemStartedEvent{Item: CommandExecutionItem{ID: "cmd-1", Command: "ls", Status: "in_progress"}}
	events <- ItemCompletedEvent{Item: CommandExecutionItem{ID: "cmd-1", Command: "ls", Status: "completed", ExitCode: &exitCode}}
	events <- ItemStartedEvent{Item: McpToolCallItem{
		ID:        "mcp-1",
		Server:    "server",
		Tool:      "tool",
		Arguments: map[string]any{"q": "x"},
		Status:    "in_progress",
	}}
	events <- ItemCompletedEvent{Item: McpToolCallItem{
		ID:        "mcp-1",
		Server:    "server",
		Tool:      "tool",
		Arguments: map[string]any{"q": "x"},
		Status:    "failed",
		Error:     &McpToolCallError{Message: "boom"},
	}}
	events <- ItemCompletedEvent{Item: AgentMessageItem{ID: "agent-1", Text: "done"}}
	events <- TurnCompletedEvent{Usage: &Usage{InputTokens: 1, CachedInputTokens: 0, OutputTokens: 1}}
	close(events)
	close(errs)

	callbacks := make([]string, 0, 2)
	onStream := CodexToolStreamHandler(func(ctx context.Context, payload CodexToolStreamEvent) error {
		callbacks = append(callbacks, payload.Event.EventType())
		if payload.Event.EventType() == "item.started" {
			panic("boom")
		}
		return nil
	})

	threadID := "thread-1"
	thread := &Thread{id: &threadID}
	streamed := &StreamedTurn{Events: events, Errors: errs}

	response, usage, resolvedThreadID, err := consumeCodexToolEvents(
		t.Context(),
		streamed,
		codexToolCallArguments{Inputs: []map[string]any{{"type": "text", "text": "hello"}}},
		thread,
		onStream,
		nil,
	)
	require.NoError(t, err)
	assert.Equal(t, "done", response)
	require.NotNil(t, usage)
	assert.Equal(t, 1, usage.InputTokens)
	assert.Equal(t, 1, usage.OutputTokens)
	require.NotNil(t, resolvedThreadID)
	assert.Equal(t, "thread-1", *resolvedThreadID)
	assert.Contains(t, callbacks, "item.started")
}

func TestConsumeCodexToolEventsThreadStartedUpdatesThreadID(t *testing.T) {
	events := make(chan ThreadEvent, 2)
	errs := make(chan error, 1)
	events <- ThreadStartedEvent{ThreadID: "thread-2"}
	events <- TurnCompletedEvent{}
	close(events)
	close(errs)

	threadID := "thread-1"
	thread := &Thread{id: &threadID}
	streamed := &StreamedTurn{Events: events, Errors: errs}

	response, _, resolvedThreadID, err := consumeCodexToolEvents(
		t.Context(),
		streamed,
		codexToolCallArguments{Inputs: []map[string]any{{"type": "text", "text": "hello"}}},
		thread,
		nil,
		nil,
	)
	require.NoError(t, err)
	assert.Equal(t, "Codex task completed with inputs.", response)
	require.NotNil(t, resolvedThreadID)
	assert.Equal(t, "thread-2", *resolvedThreadID)
}

func TestConsumeCodexToolEventsThreadErrorFails(t *testing.T) {
	events := make(chan ThreadEvent, 1)
	errs := make(chan error, 1)
	events <- ThreadErrorEvent{Message: "boom"}
	close(events)
	close(errs)

	threadID := "thread-1"
	thread := &Thread{id: &threadID}
	streamed := &StreamedTurn{Events: events, Errors: errs}

	_, _, _, err := consumeCodexToolEvents(
		t.Context(),
		streamed,
		codexToolCallArguments{},
		thread,
		nil,
		nil,
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Codex stream error: boom")
}

func TestCodexToolDefaultsToOpenAIAPIKey(t *testing.T) {
	script := filepath.Join(t.TempDir(), "codex-capture.sh")
	scriptBody := `#!/bin/sh
if [ -n "$CAPTURE_PATH" ]; then
  echo "$CODEX_API_KEY" > "$CAPTURE_PATH"
fi
cat >/dev/null
echo '{"type":"thread.started","thread_id":"thread-1"}'
echo '{"type":"item.completed","item":{"id":"agent-1","type":"agent_message","text":"Codex done."}}'
echo '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}'
`
	require.NoError(t, os.WriteFile(script, []byte(scriptBody), 0o755))

	capturePath := filepath.Join(t.TempDir(), "api_key.txt")
	t.Setenv("CAPTURE_PATH", capturePath)
	t.Setenv("CODEX_PATH", script)
	t.Setenv("OPENAI_API_KEY", "openai-key")
	t.Setenv("CODEX_API_KEY", "")

	tool, err := NewCodexTool(nil)
	require.NoError(t, err)

	_, err = tool.OnInvokeTool(t.Context(), `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.NoError(t, err)

	raw, err := os.ReadFile(capturePath)
	require.NoError(t, err)
	assert.Equal(t, "openai-key", strings.TrimSpace(string(raw)))
}

func TestCodexToolDefaultsToOpenAIAPIKeyFromDefaultKey(t *testing.T) {
	script := filepath.Join(t.TempDir(), "codex-capture.sh")
	scriptBody := `#!/bin/sh
if [ -n "$CAPTURE_PATH" ]; then
  echo "$CODEX_API_KEY" > "$CAPTURE_PATH"
fi
cat >/dev/null
echo '{"type":"thread.started","thread_id":"thread-1"}'
echo '{"type":"item.completed","item":{"id":"agent-1","type":"agent_message","text":"Codex done."}}'
echo '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}'
`
	require.NoError(t, os.WriteFile(script, []byte(scriptBody), 0o755))

	capturePath := filepath.Join(t.TempDir(), "api_key.txt")
	t.Setenv("CAPTURE_PATH", capturePath)
	t.Setenv("CODEX_PATH", script)
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("CODEX_API_KEY", "")

	agents.ClearOpenaiSettings()
	agents.SetDefaultOpenaiKey("default-openai-key", false)
	defer agents.ClearOpenaiSettings()

	tool, err := NewCodexTool(nil)
	require.NoError(t, err)

	_, err = tool.OnInvokeTool(t.Context(), `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.NoError(t, err)

	raw, err := os.ReadFile(capturePath)
	require.NoError(t, err)
	assert.Equal(t, "default-openai-key", strings.TrimSpace(string(raw)))
}

func TestNewCodexToolInvokesAndAggregates(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-1"},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "done"},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        10,
					"cached_input_tokens": 1,
					"output_tokens":       5,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{Codex: codexClient})
	require.NoError(t, err)

	resultRaw, err := tool.OnInvokeTool(t.Context(), `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.NoError(t, err)

	result, ok := resultRaw.(CodexToolResult)
	require.True(t, ok)
	require.NotNil(t, result.ThreadID)
	assert.Equal(t, "thread-1", *result.ThreadID)
	assert.Equal(t, "done", result.Response)
	require.NotNil(t, result.Usage)
	assert.Equal(t, 10, result.Usage.InputTokens)
	assert.Equal(t, 1, result.Usage.CachedInputTokens)
	assert.Equal(t, 5, result.Usage.OutputTokens)
}

func TestNewCodexToolPersistedThreadIDForRecoverableTurnFailure(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-next"},
			map[string]any{"type": "turn.failed", "error": map[string]any{"message": "boom"}},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	handler := agents.ToolErrorFunction(func(context.Context, error) (any, error) {
		return "handled", nil
	})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
		FailureErrorFunction:  &handler,
	})
	require.NoError(t, err)

	runContext := map[string]any{}
	ctx := agents.ContextWithRunContextValue(t.Context(), runContext)

	first, err := tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.NoError(t, err)
	assert.Equal(t, "handled", first)

	second, err := tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"hello again"}]}`)
	require.NoError(t, err)
	assert.Equal(t, "handled", second)

	assert.Equal(t, "thread-next", runContext["codex_thread_id"])

	args := execClient.Args()
	require.Len(t, args, 2)
	assert.Nil(t, args[0].ThreadID)
	require.NotNil(t, args[1].ThreadID)
	assert.Equal(t, "thread-next", *args[1].ThreadID)
}

func TestNewCodexToolPersistedThreadIDForRaisedTurnFailure(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-next"},
			map[string]any{"type": "turn.failed", "error": map[string]any{"message": "boom"}},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
	})
	require.NoError(t, err)

	runContext := map[string]any{}
	ctx := agents.ContextWithRunContextValue(t.Context(), runContext)

	_, err = tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Codex turn failed: boom")
	assert.Equal(t, "thread-next", runContext["codex_thread_id"])

	_, err = tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"hello again"}]}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Codex turn failed: boom")
	assert.Equal(t, "thread-next", runContext["codex_thread_id"])

	args := execClient.Args()
	require.Len(t, args, 2)
	assert.Nil(t, args[0].ThreadID)
	require.NotNil(t, args[1].ThreadID)
	assert.Equal(t, "thread-next", *args[1].ThreadID)
}

func TestNewCodexToolFallsBackToCallThreadIDWhenThreadObjectIDIsNone(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "done"},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
		Parameters:            buildCodexToolSchema(true),
	})
	require.NoError(t, err)

	runContext := map[string]any{}
	ctx := agents.ContextWithRunContextValue(t.Context(), runContext)

	firstInput := `{"inputs":[{"type":"text","text":"hello"}],"thread_id":"thread-explicit"}`
	secondInput := `{"inputs":[{"type":"text","text":"hello again"}]}`

	first, err := tool.OnInvokeTool(ctx, firstInput)
	require.NoError(t, err)
	firstResult, ok := first.(CodexToolResult)
	require.True(t, ok)
	require.NotNil(t, firstResult.ThreadID)
	assert.Equal(t, "thread-explicit", *firstResult.ThreadID)

	second, err := tool.OnInvokeTool(ctx, secondInput)
	require.NoError(t, err)
	secondResult, ok := second.(CodexToolResult)
	require.True(t, ok)
	require.NotNil(t, secondResult.ThreadID)
	assert.Equal(t, "thread-explicit", *secondResult.ThreadID)

	assert.Equal(t, "thread-explicit", runContext["codex_thread_id"])

	args := execClient.Args()
	require.Len(t, args, 2)
	require.NotNil(t, args[0].ThreadID)
	assert.Equal(t, "thread-explicit", *args[0].ThreadID)
	require.NotNil(t, args[1].ThreadID)
	assert.Equal(t, "thread-explicit", *args[1].ThreadID)
}

func TestCodexToolPassesIdleTimeoutSeconds(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-1"},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "done"},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:              codexClient,
		DefaultTurnOptions: map[string]any{"idle_timeout_seconds": 3.5},
	})
	require.NoError(t, err)

	_, err = tool.OnInvokeTool(t.Context(), `{"inputs":[{"type":"text","text":"Check timeout option"}]}`)
	require.NoError(t, err)

	args := execClient.Args()
	require.Len(t, args, 1)
	require.NotNil(t, args[0].IdleTimeoutSeconds)
	assert.InDelta(t, 3.5, *args[0].IdleTimeoutSeconds, 1e-9)
}

func TestNewCodexToolResumesFromInputThreadID(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "done"},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{Codex: codexClient})
	require.NoError(t, err)

	_, err = tool.OnInvokeTool(
		t.Context(),
		`{"inputs":[{"type":"text","text":"hello"}],"thread_id":"thread-xyz"}`,
	)
	require.NoError(t, err)

	args := execClient.Args()
	require.Len(t, args, 1)
	require.NotNil(t, args[0].ThreadID)
	assert.Equal(t, "thread-xyz", *args[0].ThreadID)
}

func TestNewCodexToolPersistSessionReusesThread(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-1"},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "done"},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:          codexClient,
		PersistSession: true,
	})
	require.NoError(t, err)

	_, err = tool.OnInvokeTool(t.Context(), `{"inputs":[{"type":"text","text":"first"}]}`)
	require.NoError(t, err)
	_, err = tool.OnInvokeTool(t.Context(), `{"inputs":[{"type":"text","text":"second"}]}`)
	require.NoError(t, err)

	args := execClient.Args()
	require.Len(t, args, 2)
	assert.Nil(t, args[0].ThreadID)
	require.NotNil(t, args[1].ThreadID)
	assert.Equal(t, "thread-1", *args[1].ThreadID)
}

func TestNewCodexToolPersistSessionMismatchRaises(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "done"},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:          codexClient,
		PersistSession: true,
	})
	require.NoError(t, err)

	_, err = tool.OnInvokeTool(
		t.Context(),
		`{"inputs":[{"type":"text","text":"first"}],"thread_id":"thread-1"}`,
	)
	require.NoError(t, err)

	_, err = tool.OnInvokeTool(
		t.Context(),
		`{"inputs":[{"type":"text","text":"second"}],"thread_id":"thread-2"}`,
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already has an active thread")
}

func TestNewCodexToolDefaultResponseWithoutAgentMessage(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{Codex: codexClient})
	require.NoError(t, err)

	resultRaw, err := tool.OnInvokeTool(t.Context(), `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.NoError(t, err)
	result, ok := resultRaw.(CodexToolResult)
	require.True(t, ok)
	assert.Equal(t, "Codex task completed with inputs.", result.Response)
}

func TestNewCodexToolOnStreamCallbackPanicDoesNotFail(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-1"},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "done"},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})

	callbackCount := 0
	tool, err := NewCodexTool(CodexToolOptions{
		Codex: codexClient,
		OnStream: func(context.Context, CodexToolStreamEvent) error {
			callbackCount++
			panic("boom")
		},
	})
	require.NoError(t, err)

	_, err = tool.OnInvokeTool(t.Context(), `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.NoError(t, err)
	assert.Greater(t, callbackCount, 0)
}

func TestNewCodexToolIsEnabledBoolOption(t *testing.T) {
	execClient := &recordingExec{}
	codexClient := newCodexWithExec(execClient, CodexOptions{})

	tool, err := NewCodexTool(map[string]any{
		"codex":      codexClient,
		"is_enabled": false,
	})
	require.NoError(t, err)
	require.NotNil(t, tool.IsEnabled)
	enabled, err := tool.IsEnabled.IsEnabled(t.Context(), &agents.Agent{Name: "a"})
	require.NoError(t, err)
	assert.False(t, enabled)
}

func TestNewCodexToolStreamsEventsAndUpdatesUsageAndSpans(t *testing.T) {
	tracingtesting.Setup(t)

	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-1"},
			map[string]any{
				"type": "item.started",
				"item": map[string]any{"id": "reason-1", "type": "reasoning", "text": "Initial reasoning"},
			},
			map[string]any{
				"type": "item.updated",
				"item": map[string]any{"id": "reason-1", "type": "reasoning", "text": "Refined reasoning"},
			},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "reason-1", "type": "reasoning", "text": "Final reasoning"},
			},
			map[string]any{
				"type": "item.started",
				"item": map[string]any{
					"id":                "cmd-1",
					"type":              "command_execution",
					"command":           "pytest",
					"aggregated_output": "",
					"status":            "in_progress",
				},
			},
			map[string]any{
				"type": "item.updated",
				"item": map[string]any{
					"id":                "cmd-1",
					"type":              "command_execution",
					"command":           "pytest",
					"aggregated_output": "Running tests",
					"status":            "in_progress",
				},
			},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{
					"id":                "cmd-1",
					"type":              "command_execution",
					"command":           "pytest",
					"aggregated_output": "All good",
					"exit_code":         0,
					"status":            "completed",
				},
			},
			map[string]any{
				"type": "item.started",
				"item": map[string]any{
					"id":        "mcp-1",
					"type":      "mcp_tool_call",
					"server":    "gitmcp",
					"tool":      "search_codex_code",
					"arguments": map[string]any{"query": "foo"},
					"status":    "in_progress",
				},
			},
			map[string]any{
				"type": "item.updated",
				"item": map[string]any{
					"id":        "mcp-1",
					"type":      "mcp_tool_call",
					"server":    "gitmcp",
					"tool":      "search_codex_code",
					"arguments": map[string]any{"query": "foo"},
					"status":    "in_progress",
				},
			},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{
					"id":        "mcp-1",
					"type":      "mcp_tool_call",
					"server":    "gitmcp",
					"tool":      "search_codex_code",
					"arguments": map[string]any{"query": "foo"},
					"status":    "completed",
					"result": map[string]any{
						"content":            []any{},
						"structured_content": nil,
					},
				},
			},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "Codex finished."},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        10,
					"cached_input_tokens": 1,
					"output_tokens":       5,
				},
			},
		},
	}

	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{Codex: codexClient})
	require.NoError(t, err)

	runUsage := usage.NewUsage()
	ctx := usage.NewContext(t.Context(), runUsage)

	var resultRaw any
	err = tracing.RunTrace(ctx, tracing.TraceParams{WorkflowName: "codex-test"}, func(ctx context.Context, _ tracing.Trace) error {
		return tracing.FunctionSpan(ctx, tracing.FunctionSpanParams{Name: tool.Name}, func(ctx context.Context, _ tracing.Span) error {
			var invokeErr error
			resultRaw, invokeErr = tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"Diagnose failure"}]}`)
			return invokeErr
		})
	})
	require.NoError(t, err)

	result, ok := resultRaw.(CodexToolResult)
	require.True(t, ok)
	require.NotNil(t, result.ThreadID)
	assert.Equal(t, "thread-1", *result.ThreadID)
	assert.Equal(t, "Codex finished.", result.Response)
	require.NotNil(t, result.Usage)
	assert.Equal(t, 10, result.Usage.InputTokens)
	assert.Equal(t, 1, result.Usage.CachedInputTokens)
	assert.Equal(t, 5, result.Usage.OutputTokens)

	assert.EqualValues(t, 1, runUsage.Requests)
	assert.EqualValues(t, 15, runUsage.TotalTokens)
	assert.EqualValues(t, 10, runUsage.InputTokens)
	assert.EqualValues(t, 5, runUsage.OutputTokens)
	assert.EqualValues(t, 1, runUsage.InputTokensDetails.CachedTokens)

	spans := tracingtesting.FetchOrderedSpans(false)
	require.NotEmpty(t, spans)

	var functionSpan tracing.Span
	customSpans := make([]tracing.Span, 0)
	for _, span := range spans {
		switch data := span.SpanData().(type) {
		case *tracing.FunctionSpanData:
			if data.Name == tool.Name {
				functionSpan = span
			}
		case *tracing.CustomSpanData:
			customSpans = append(customSpans, span)
		}
	}

	require.NotNil(t, functionSpan)
	require.Len(t, customSpans, 3)
	for _, span := range customSpans {
		assert.Equal(t, functionSpan.SpanID(), span.ParentID())
	}

	spanByName := make(map[string]*tracing.CustomSpanData, len(customSpans))
	for _, span := range customSpans {
		data, _ := span.SpanData().(*tracing.CustomSpanData)
		if data != nil {
			spanByName[data.Name] = data
		}
	}

	reasoningSpan := spanByName["Codex reasoning"]
	require.NotNil(t, reasoningSpan)
	assert.Equal(t, "Final reasoning", reasoningSpan.Data["text"])

	commandSpan := spanByName["Codex command execution"]
	require.NotNil(t, commandSpan)
	assert.Equal(t, "pytest", commandSpan.Data["command"])
	assert.Equal(t, "completed", commandSpan.Data["status"])
	assert.Equal(t, "All good", commandSpan.Data["output"])
	assert.Equal(t, 0, commandSpan.Data["exit_code"])

	mcpSpan := spanByName["Codex MCP tool call"]
	require.NotNil(t, mcpSpan)
	assert.Equal(t, "gitmcp", mcpSpan.Data["server"])
	assert.Equal(t, "search_codex_code", mcpSpan.Data["tool"])
	assert.Equal(t, "completed", mcpSpan.Data["status"])
}

func TestCodexToolKeepsCommandOutputWhenCompletedMissingOutput(t *testing.T) {
	tracingtesting.Setup(t)

	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-1"},
			map[string]any{
				"type": "item.started",
				"item": map[string]any{
					"id":                "cmd-1",
					"type":              "command_execution",
					"command":           "ls",
					"aggregated_output": "",
					"status":            "in_progress",
				},
			},
			map[string]any{
				"type": "item.updated",
				"item": map[string]any{
					"id":                "cmd-1",
					"type":              "command_execution",
					"command":           "ls",
					"aggregated_output": "first output",
					"status":            "in_progress",
				},
			},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{
					"id":        "cmd-1",
					"type":      "command_execution",
					"command":   "ls",
					"exit_code": 0,
					"status":    "completed",
				},
			},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "Codex finished."},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}

	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{Codex: codexClient})
	require.NoError(t, err)

	err = tracing.RunTrace(t.Context(), tracing.TraceParams{WorkflowName: "codex-test"}, func(ctx context.Context, _ tracing.Trace) error {
		return tracing.FunctionSpan(ctx, tracing.FunctionSpanParams{Name: tool.Name}, func(ctx context.Context, _ tracing.Span) error {
			_, invokeErr := tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"List files"}]}`)
			return invokeErr
		})
	})
	require.NoError(t, err)

	spans := tracingtesting.FetchOrderedSpans(false)
	require.NotEmpty(t, spans)

	var commandSpan *tracing.CustomSpanData
	for _, span := range spans {
		data, ok := span.SpanData().(*tracing.CustomSpanData)
		if !ok || data == nil {
			continue
		}
		if data.Name == "Codex command execution" {
			commandSpan = data
			break
		}
	}
	require.NotNil(t, commandSpan)
	assert.Equal(t, "first output", commandSpan.Data["output"])
}

func TestCodexToolTruncatesSpanValues(t *testing.T) {
	value := map[string]any{"payload": strings.Repeat("x", 200)}
	truncated := truncateSpanValue(value, ptrInt(40))

	data, ok := truncated.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, true, data["truncated"])
	originalLength, ok := data["original_length"].(int)
	require.True(t, ok)
	assert.Greater(t, originalLength, 40)

	preview, ok := data["preview"].(string)
	require.True(t, ok)
	assert.LessOrEqual(t, len(preview), 40)
}

func TestCodexToolEnforcesSpanDataBudget(t *testing.T) {
	data := map[string]any{
		"command":   "run",
		"output":    strings.Repeat("x", 5000),
		"arguments": map[string]any{"payload": strings.Repeat("y", 5000)},
	}
	trimmed := enforceSpanDataBudget(data, ptrInt(512))

	assert.Contains(t, trimmed, "command")
	assert.Contains(t, trimmed, "output")
	assert.Contains(t, trimmed, "arguments")
	assert.NotEmpty(t, trimmed["command"])
	assert.LessOrEqual(t, jsonCharSize(trimmed), 512)
}

func TestCodexToolKeepsOutputPreviewWithBudget(t *testing.T) {
	data := map[string]any{"output": strings.Repeat("x", 1000)}
	trimmed := enforceSpanDataBudget(data, ptrInt(120))

	assert.Contains(t, trimmed, "output")
	output, ok := trimmed["output"].(string)
	require.True(t, ok)
	assert.NotEmpty(t, output)
	assert.LessOrEqual(t, jsonCharSize(trimmed), 120)
}

func TestCodexToolPrioritizesArgumentsOverLargeResults(t *testing.T) {
	data := map[string]any{
		"arguments": map[string]any{"foo": "bar"},
		"result":    strings.Repeat("x", 2000),
	}
	trimmed := enforceSpanDataBudget(data, ptrInt(200))

	assert.Equal(t, stringifySpanValue(map[string]any{"foo": "bar"}), trimmed["arguments"])
	assert.Contains(t, trimmed, "result")
	assert.LessOrEqual(t, jsonCharSize(trimmed), 200)
}

func TestCodexToolDuplicateNamesFailFast(t *testing.T) {
	agent := &agents.Agent{
		Name: "test",
		Tools: []agents.Tool{
			MustNewCodexTool(nil),
			MustNewCodexTool(nil),
		},
	}

	_, err := agent.GetAllTools(t.Context())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Duplicate Codex tool names found")
}

func TestCodexToolNameCollisionWithOtherToolFailsFast(t *testing.T) {
	otherTool := agents.FunctionTool{
		Name: "codex",
		OnInvokeTool: func(context.Context, string) (any, error) {
			return "ok", nil
		},
	}
	agent := &agents.Agent{
		Name: "test",
		Tools: []agents.Tool{
			MustNewCodexTool(nil),
			otherTool,
		},
	}

	_, err := agent.GetAllTools(t.Context())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Duplicate Codex tool names found")
}

func TestResolveCodexOptionsReadsEnvOverride(t *testing.T) {
	options := CodexOptions{
		CodexPathOverride: ptr(" /bin/codex "),
		Env:               map[any]any{"CODEX_API_KEY": "env-key"},
	}

	resolved, err := resolveCodexOptions(options)
	require.NoError(t, err)
	require.NotNil(t, resolved.APIKey)
	assert.Equal(t, "env-key", *resolved.APIKey)
	require.NotNil(t, resolved.CodexPathOverride)
	assert.Equal(t, " /bin/codex ", *resolved.CodexPathOverride)
}

func TestCoerceCodexToolOptionsAcceptsCodexOptionsMap(t *testing.T) {
	options, err := CoerceCodexToolOptions(map[string]any{
		"codex_options": map[string]any{"api_key": "from-options"},
	})
	require.NoError(t, err)
	require.NotNil(t, options)

	resolved, err := resolveCodexOptions(options.CodexOptions)
	require.NoError(t, err)
	require.NotNil(t, resolved.APIKey)
	assert.Equal(t, "from-options", *resolved.APIKey)
}

func TestNewCodexToolAcceptsMapOptions(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-1"},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "Codex done."},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(map[string]any{
		"codex":        codexClient,
		"sandbox_mode": "read-only",
		"name":         "codex_dict",
	})
	require.NoError(t, err)
	assert.Equal(t, "codex_dict", tool.Name)

	result, err := tool.OnInvokeTool(t.Context(), `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.NoError(t, err)
	response, ok := result.(CodexToolResult)
	require.True(t, ok)
	assert.Equal(t, "Codex done.", response.Response)
}

func TestNewCodexToolAcceptsKeywordOverrides(t *testing.T) {
	name := "codex_overrides"
	description := "desc"
	parameters := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"inputs": map[string]any{"type": "array"},
		},
	}
	tool, err := NewCodexTool(CodexToolOptions{
		Name:                  &name,
		Description:           &description,
		Parameters:            parameters,
		OutputSchema:          map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false},
		SpanDataMaxChars:      ptrInt(10),
		PersistSession:        true,
		IsEnabled:             agents.NewFunctionToolEnabledFlag(false),
		UseRunContextThreadID: true,
		RunContextThreadIDKey: ptr("thread_key"),
	})
	require.NoError(t, err)
	assert.Equal(t, "codex_overrides", tool.Name)
	assert.Equal(t, "desc", tool.Description)
	properties, ok := tool.ParamsJSONSchema["properties"].(map[string]any)
	require.True(t, ok)
	_, hasInputs := properties["inputs"]
	assert.True(t, hasInputs)
	require.NotNil(t, tool.IsEnabled)
	enabled, err := tool.IsEnabled.IsEnabled(t.Context(), &agents.Agent{Name: "a"})
	require.NoError(t, err)
	assert.False(t, enabled)
}

func TestNewCodexToolAcceptsAllMapOverrides(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-1"},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "Codex done."},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	onStream := CodexToolStreamHandler(func(context.Context, CodexToolStreamEvent) error { return nil })
	failureHandler := agents.ToolErrorFunction(func(context.Context, error) (any, error) { return "handled", nil })
	tool, err := NewCodexTool(map[string]any{
		"name":                      "codex_overrides",
		"description":               "desc",
		"parameters":                map[string]any{"type": "object", "properties": map[string]any{"inputs": map[string]any{"type": "array"}}},
		"span_data_max_chars":       10,
		"output_schema":             map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false},
		"codex":                     codexClient,
		"codex_options":             map[string]any{"api_key": "from-kwargs"},
		"default_thread_options":    map[string]any{"model": "gpt"},
		"thread_id":                 "thread-1",
		"sandbox_mode":              "read-only",
		"working_directory":         "/work",
		"skip_git_repo_check":       true,
		"default_turn_options":      map[string]any{"idle_timeout_seconds": 1.0},
		"persist_session":           true,
		"on_stream":                 onStream,
		"is_enabled":                false,
		"failure_error_function":    failureHandler,
		"use_run_context_thread_id": true,
		"run_context_thread_id_key": "thread_key",
	})
	require.NoError(t, err)
	assert.Equal(t, "codex_overrides", tool.Name)
}

func TestResolveCodexOptionsEnvOverrideWins(t *testing.T) {
	t.Setenv("CODEX_API_KEY", "env-codex")
	t.Setenv("OPENAI_API_KEY", "env-openai")
	options := CodexOptions{
		Env: map[any]any{"CODEX_API_KEY": "override-key"},
	}

	resolved, err := resolveCodexOptions(options)
	require.NoError(t, err)
	require.NotNil(t, resolved.APIKey)
	assert.Equal(t, "override-key", *resolved.APIKey)
}

func TestResolveCodexOptionsReadsEnv(t *testing.T) {
	t.Setenv("CODEX_API_KEY", "env-key")
	t.Setenv("OPENAI_API_KEY", "")

	resolved, err := resolveCodexOptions(nil)
	require.NoError(t, err)
	require.NotNil(t, resolved.APIKey)
	assert.Equal(t, "env-key", *resolved.APIKey)
}

func TestResolveThreadOptionsMergesValues(t *testing.T) {
	defaults := map[string]any{
		"model":                  "gpt",
		"sandbox_mode":           "read-only",
		"additional_directories": []any{"/extra"},
	}
	sandbox := " sandbox "
	working := " /work "
	skip := true

	resolved, err := resolveThreadOptions(defaults, &sandbox, &working, &skip)
	require.NoError(t, err)
	require.NotNil(t, resolved)
	require.NotNil(t, resolved.Model)
	assert.Equal(t, "gpt", *resolved.Model)
	require.NotNil(t, resolved.SandboxMode)
	assert.Equal(t, "sandbox", *resolved.SandboxMode)
	require.NotNil(t, resolved.WorkingDirectory)
	assert.Equal(t, "/work", *resolved.WorkingDirectory)
	require.NotNil(t, resolved.SkipGitRepoCheck)
	assert.True(t, *resolved.SkipGitRepoCheck)
	assert.Equal(t, []string{"/extra"}, resolved.AdditionalDirectories)
}

func TestResolveThreadOptionsEmptyIsNone(t *testing.T) {
	resolved, err := resolveThreadOptions(nil, nil, nil, nil)
	require.NoError(t, err)
	assert.Nil(t, resolved)

	resolved, err = resolveThreadOptions(map[string]any{}, nil, nil, nil)
	require.NoError(t, err)
	assert.Nil(t, resolved)
}

func TestBuildTurnOptionsOverridesSchema(t *testing.T) {
	outputSchema := map[string]any{"type": "object", "properties": map[string]any{}}
	defaults := &TurnOptions{
		OutputSchema:       map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "string"}}},
		IdleTimeoutSeconds: ptr(1.0),
	}
	turn := buildTurnOptions(defaults, outputSchema)
	assert.Equal(t, outputSchema, turn.OutputSchema)
	require.NotNil(t, turn.IdleTimeoutSeconds)
	assert.Equal(t, 1.0, *turn.IdleTimeoutSeconds)
}

func TestBuildTurnOptionsMergesOutputSchema(t *testing.T) {
	outputSchema := map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false}
	turn := buildTurnOptions(nil, outputSchema)
	assert.Equal(t, outputSchema, turn.OutputSchema)

	defaults := &TurnOptions{
		OutputSchema:       map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "string"}}},
		IdleTimeoutSeconds: ptr(1.0),
	}
	turn = buildTurnOptions(defaults, nil)
	assert.Equal(t, defaults.OutputSchema, turn.OutputSchema)
	require.NotNil(t, turn.IdleTimeoutSeconds)
	assert.Equal(t, 1.0, *turn.IdleTimeoutSeconds)
}

func TestResolveCodexOptionsReadsOpenAIAPIKey(t *testing.T) {
	t.Setenv("CODEX_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "openai-env-key")

	resolved, err := resolveCodexOptions(nil)
	require.NoError(t, err)
	require.NotNil(t, resolved.APIKey)
	assert.Equal(t, "openai-env-key", *resolved.APIKey)
}

func TestResolveCodexOptionsUsesDefaultOpenaiKey(t *testing.T) {
	t.Setenv("CODEX_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	agents.ClearOpenaiSettings()
	agents.SetDefaultOpenaiKey("default-openai-key", false)
	defer agents.ClearOpenaiSettings()

	resolved, err := resolveCodexOptions(nil)
	require.NoError(t, err)
	require.NotNil(t, resolved.APIKey)
	assert.Equal(t, "default-openai-key", *resolved.APIKey)
}

func TestNewCodexToolRunContextModeHidesThreadIDInDefaultSchema(t *testing.T) {
	tool, err := NewCodexTool(CodexToolOptions{
		UseRunContextThreadID: true,
	})
	require.NoError(t, err)

	properties, ok := tool.ParamsJSONSchema["properties"].(map[string]any)
	require.True(t, ok)
	_, hasThreadID := properties["thread_id"]
	assert.False(t, hasThreadID)
}

func TestNewCodexToolUseRunContextThreadIDRequiresRunContext(t *testing.T) {
	execClient := &recordingExec{}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
	})
	require.NoError(t, err)

	_, err = tool.OnInvokeTool(t.Context(), `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "use_run_context_thread_id=true")
}

func TestNewCodexToolUsesRunContextThreadIDAndPersistsLatest(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-next"},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "done"},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
	})
	require.NoError(t, err)

	runContext := map[string]any{
		"codex_thread_id": "thread-prev",
	}
	ctx := agents.ContextWithRunContextValue(t.Context(), runContext)

	resultRaw, err := tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.NoError(t, err)
	result, ok := resultRaw.(CodexToolResult)
	require.True(t, ok)
	require.NotNil(t, result.ThreadID)
	assert.Equal(t, "thread-next", *result.ThreadID)
	assert.Equal(t, "thread-next", runContext["codex_thread_id"])

	args := execClient.Args()
	require.Len(t, args, 1)
	require.NotNil(t, args[0].ThreadID)
	assert.Equal(t, "thread-prev", *args[0].ThreadID)

	_, err = tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"hello again"}]}`)
	require.NoError(t, err)
	args = execClient.Args()
	require.Len(t, args, 2)
	require.NotNil(t, args[1].ThreadID)
	assert.Equal(t, "thread-next", *args[1].ThreadID)
}

func TestNewCodexToolUsesRunContextThreadIDWithStructContext(t *testing.T) {
	type runContextStruct struct {
		UserID        string
		CodexThreadID string `json:"codex_thread_id"`
	}

	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-next"},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "done"},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
	})
	require.NoError(t, err)

	runContext := &runContextStruct{
		UserID:        "u1",
		CodexThreadID: "thread-prev",
	}
	ctx := agents.ContextWithRunContextValue(t.Context(), runContext)
	_, err = tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.NoError(t, err)

	args := execClient.Args()
	require.Len(t, args, 1)
	require.NotNil(t, args[0].ThreadID)
	assert.Equal(t, "thread-prev", *args[0].ThreadID)
	assert.Equal(t, "thread-next", runContext.CodexThreadID)
}

func TestNewCodexToolUsesRunContextThreadIDWithFieldNameMatch(t *testing.T) {
	type runContextStruct struct {
		UserID        string
		CodexThreadID string
	}

	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-next"},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "done"},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
	})
	require.NoError(t, err)

	runContext := &runContextStruct{
		UserID:        "u1",
		CodexThreadID: "thread-prev",
	}
	ctx := agents.ContextWithRunContextValue(t.Context(), runContext)
	_, err = tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.NoError(t, err)

	args := execClient.Args()
	require.Len(t, args, 1)
	require.NotNil(t, args[0].ThreadID)
	assert.Equal(t, "thread-prev", *args[0].ThreadID)
	assert.Equal(t, "thread-next", runContext.CodexThreadID)
}

func TestNewCodexToolRunContextStructMissingFieldRejected(t *testing.T) {
	type runContextStruct struct {
		UserID string
	}

	execClient := &recordingExec{}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
	})
	require.NoError(t, err)

	runContext := &runContextStruct{UserID: "u1"}
	ctx := agents.ContextWithRunContextValue(t.Context(), runContext)
	_, err = tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `support field "codex_thread_id"`)
}

func TestNewCodexToolRunContextStructUnexportedFieldRejected(t *testing.T) {
	type runContextStruct struct {
		codexThreadID string `json:"codex_thread_id"`
	}

	execClient := &recordingExec{}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
	})
	require.NoError(t, err)

	runContext := &runContextStruct{codexThreadID: "thread-prev"}
	ctx := agents.ContextWithRunContextValue(t.Context(), runContext)
	_, err = tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "writable run context field")
}

func TestNewCodexToolRunContextStructByValueRejected(t *testing.T) {
	type runContextStruct struct {
		CodexThreadID string `json:"codex_thread_id"`
	}

	execClient := &recordingExec{}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
	})
	require.NoError(t, err)

	runContext := runContextStruct{CodexThreadID: "thread-prev"}
	ctx := agents.ContextWithRunContextValue(t.Context(), runContext)
	_, err = tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "struct contexts must be passed by pointer")
}

func TestNewCodexToolRunContextRejectsNonStringKeyMap(t *testing.T) {
	execClient := &recordingExec{}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
	})
	require.NoError(t, err)

	runContext := map[any]any{"codex_thread_id": "thread-prev"}
	ctx := agents.ContextWithRunContextValue(t.Context(), runContext)
	_, err = tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "string keys")
}

func TestNewCodexToolRunContextModeRejectsThreadIDWithoutCustomParameters(t *testing.T) {
	execClient := &recordingExec{}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
	})
	require.NoError(t, err)

	runContext := map[string]any{
		"codex_thread_id": "thread-prev",
	}
	ctx := agents.ContextWithRunContextValue(t.Context(), runContext)
	_, err = tool.OnInvokeTool(
		ctx,
		`{"inputs":[{"type":"text","text":"hello"}],"thread_id":"thread-xyz"}`,
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Invalid JSON input for codex tool")
}

func TestNewCodexToolToolInputThreadIDOverridesRunContextThreadIDWithCustomParameters(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-from-tool-input"},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "done"},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
		Parameters:            buildCodexToolSchema(true),
	})
	require.NoError(t, err)

	runContext := map[string]any{
		"codex_thread_id": "thread-from-context",
	}
	ctx := agents.ContextWithRunContextValue(t.Context(), runContext)
	_, err = tool.OnInvokeTool(
		ctx,
		`{"inputs":[{"type":"text","text":"hello"}],"thread_id":"thread-from-args"}`,
	)
	require.NoError(t, err)

	args := execClient.Args()
	require.Len(t, args, 1)
	require.NotNil(t, args[0].ThreadID)
	assert.Equal(t, "thread-from-args", *args[0].ThreadID)
}

func TestNewCodexToolPersistsThreadIDForTurnFailureWithRunContext(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-next"},
			map[string]any{"type": "turn.failed", "error": map[string]any{"message": "boom"}},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:                 codexClient,
		UseRunContextThreadID: true,
	})
	require.NoError(t, err)

	runContext := map[string]any{}
	ctx := agents.ContextWithRunContextValue(t.Context(), runContext)
	_, err = tool.OnInvokeTool(ctx, `{"inputs":[{"type":"text","text":"hello"}]}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Codex turn failed: boom")
	assert.Equal(t, "thread-next", runContext["codex_thread_id"])
}

func TestResolveRunContextThreadIDKeyRejectsLossyDefaultSuffix(t *testing.T) {
	name := "codex_a-b"
	_, err := NewCodexTool(CodexToolOptions{
		Name:                  &name,
		UseRunContextThreadID: true,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "run_context_thread_id_key")
}

func TestResolveRunContextThreadIDKeyDefaultsToToolNameSuffix(t *testing.T) {
	key, err := resolveRunContextThreadIDKey("codex_engineer", nil, true)
	require.NoError(t, err)
	assert.Equal(t, "codex_thread_id_engineer", key)
}

func TestCoerceCodexToolOptionsRejectsEmptyRunContextThreadIDKey(t *testing.T) {
	_, err := CoerceCodexToolOptions(map[string]any{
		"use_run_context_thread_id": true,
		"run_context_thread_id_key": " ",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "run_context_thread_id_key")
}

func TestResolveRunContextThreadIDKeyNormalizesNameWhenNotStrict(t *testing.T) {
	key, err := resolveRunContextThreadIDKey("codex_a-b", nil, false)
	require.NoError(t, err)
	assert.Equal(t, "codex_thread_id_a_b", key)
}

func TestNewCodexToolAcceptsCustomParametersSchema(t *testing.T) {
	execClient := &recordingExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-1"},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "done"},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        1,
					"cached_input_tokens": 0,
					"output_tokens":       1,
				},
			},
		},
	}
	codexClient := newCodexWithExec(execClient, CodexOptions{})
	customParameters := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"inputs"},
		"properties": map[string]any{
			"inputs": map[string]any{
				"type":     "array",
				"minItems": 1,
				"items": map[string]any{
					"type":                 "object",
					"additionalProperties": false,
					"required":             []any{"type"},
					"properties": map[string]any{
						"type": map[string]any{
							"type": "string",
							"enum": []any{"text", "local_image"},
						},
						"text": map[string]any{"type": "string"},
						"path": map[string]any{"type": "string"},
					},
				},
			},
		},
	}
	tool, err := NewCodexTool(CodexToolOptions{
		Codex:      codexClient,
		Parameters: customParameters,
	})
	require.NoError(t, err)

	_, err = tool.OnInvokeTool(
		t.Context(),
		`{"inputs":[{"type":"text","text":"hello"}],"extra":"ignored-by-custom-parser"}`,
	)
	require.NoError(t, err)
}

func TestCodexToolTruncateSpanStringLimits(t *testing.T) {
	zero := 0
	assert.Equal(t, "", truncateSpanString("hello", &zero))

	limit := 3
	longValue := strings.Repeat("x", 100)
	assert.Equal(t, "xxx", truncateSpanString(longValue, &limit))
}

func TestCodexToolTruncateSpanValueHandlesCircularReference(t *testing.T) {
	value := []any{}
	value = append(value, value)
	limit := 1
	truncated := truncateSpanValue(value, &limit)

	truncatedMap, ok := truncated.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, true, truncatedMap["truncated"])
}

func TestCodexToolEnforceSpanDataBudgetZeroMax(t *testing.T) {
	maxChars := 0
	trimmed := enforceSpanDataBudget(map[string]any{"output": "x"}, &maxChars)
	assert.Empty(t, trimmed)
}

func TestCodexToolEnforceSpanDataBudgetTrimsValuesWhenBudgetTight(t *testing.T) {
	data := map[string]any{
		"command":   "run",
		"output":    strings.Repeat("x", 50),
		"arguments": strings.Repeat("y", 50),
	}
	base := map[string]any{
		"command":   "run",
		"output":    "",
		"arguments": "",
	}
	maxChars := jsonCharSize(base) + 1
	trimmed := enforceSpanDataBudget(data, &maxChars)

	_, hasCommand := trimmed["command"]
	_, hasOutput := trimmed["output"]
	_, hasArguments := trimmed["arguments"]
	assert.True(t, hasCommand)
	assert.True(t, hasOutput)
	assert.True(t, hasArguments)
	assert.LessOrEqual(t, jsonCharSize(trimmed), maxChars)
}

func TestCodexToolEnforceSpanDataBudgetDropsUntilBaseFits(t *testing.T) {
	data := map[string]any{
		"command": "run",
		"output":  strings.Repeat("x", 50),
	}
	base := map[string]any{
		"command": "",
		"output":  "",
	}
	maxChars := jsonCharSize(base) - 1
	trimmed := enforceSpanDataBudget(data, &maxChars)

	_, hasCommand := trimmed["command"]
	_, hasOutput := trimmed["output"]
	assert.False(t, hasCommand && hasOutput)
}

func TestCodexToolHandleItemStartedIgnoresMissingID(t *testing.T) {
	spans := map[string]tracing.Span{}
	handleCodexItemStarted(t.Context(), ReasoningItem{ID: "", Text: "hi"}, spans, nil)
	assert.Empty(t, spans)
}

func TestCodexToolHandleItemUpdatedIgnoresMissingSpan(t *testing.T) {
	spans := map[string]tracing.Span{}
	handleCodexItemUpdated(ReasoningItem{ID: "missing", Text: "hi"}, spans, nil)
	assert.Empty(t, spans)
}
