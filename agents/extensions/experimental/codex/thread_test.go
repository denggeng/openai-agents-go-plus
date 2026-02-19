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
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeExec struct {
	events   []any
	delay    time.Duration
	lastArgs CodexExecArgs
}

func (f *fakeExec) RunJSONL(ctx context.Context, args CodexExecArgs) (<-chan string, <-chan error) {
	f.lastArgs = args
	lines := make(chan string)
	errs := make(chan error, 1)

	go func() {
		defer close(lines)
		defer close(errs)

		for _, event := range f.events {
			if f.delay > 0 {
				select {
				case <-ctx.Done():
					errs <- ctx.Err()
					return
				case <-time.After(f.delay):
				}
			}

			var line string
			switch typed := event.(type) {
			case string:
				line = typed
			default:
				raw, err := json.Marshal(typed)
				if err != nil {
					errs <- err
					return
				}
				line = string(raw)
			}

			select {
			case <-ctx.Done():
				errs <- ctx.Err()
				return
			case lines <- line:
			}
		}
	}()

	return lines, errs
}

type blockingExec struct {
	lastArgs CodexExecArgs
}

func (b *blockingExec) RunJSONL(ctx context.Context, args CodexExecArgs) (<-chan string, <-chan error) {
	b.lastArgs = args
	lines := make(chan string)
	errs := make(chan error, 1)

	go func() {
		defer close(lines)
		defer close(errs)
		if args.Signal != nil {
			select {
			case <-args.Signal:
				return
			case <-ctx.Done():
				errs <- ctx.Err()
				return
			}
		}
		<-ctx.Done()
		errs <- ctx.Err()
	}()

	return lines, errs
}

func TestThreadRunStreamedPassesOptionsAndUpdatesID(t *testing.T) {
	exec := &fakeExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-42"},
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
	thread := newThread(exec, CodexOptions{
		BaseURL: ptr("https://example.com"),
		APIKey:  ptr("api-key"),
	}, ThreadOptions{
		Model:                 ptr("gpt-4.1-mini"),
		SandboxMode:           ptr("read-only"),
		WorkingDirectory:      ptr("/work"),
		AdditionalDirectories: []string{"/extra"},
		SkipGitRepoCheck:      ptr(true),
		ModelReasoningEffort:  ptr("low"),
		NetworkAccessEnabled:  ptr(false),
		WebSearchMode:         ptr("cached"),
		ApprovalPolicy:        ptr("on-request"),
	}, nil)

	streamed, err := thread.RunStreamed(
		t.Context(),
		[]map[string]any{
			{"type": "text", "text": "hello"},
			{"type": "local_image", "path": "/tmp/a.png"},
		},
		TurnOptions{OutputSchema: map[string]any{"type": "object"}},
	)
	require.NoError(t, err)

	events, err := collectStreamedEvents(streamed)
	require.NoError(t, err)
	require.NotEmpty(t, events)
	require.IsType(t, ThreadStartedEvent{}, events[0])

	require.NotNil(t, thread.ID())
	assert.Equal(t, "thread-42", *thread.ID())

	require.NotNil(t, exec.lastArgs.OutputSchemaFile)
	assert.Equal(t, "schema.json", filepath.Base(*exec.lastArgs.OutputSchemaFile))
	assert.Equal(t, "gpt-4.1-mini", optionalValue(exec.lastArgs.Model))
	assert.Equal(t, "read-only", optionalValue(exec.lastArgs.SandboxMode))
	assert.Equal(t, "/work", optionalValue(exec.lastArgs.WorkingDirectory))
	assert.Equal(t, "low", optionalValue(exec.lastArgs.ModelReasoningEffort))
	assert.Equal(t, "cached", optionalValue(exec.lastArgs.WebSearchMode))
	assert.Equal(t, "on-request", optionalValue(exec.lastArgs.ApprovalPolicy))
	require.NotNil(t, exec.lastArgs.SkipGitRepoCheck)
	assert.True(t, *exec.lastArgs.SkipGitRepoCheck)
	require.NotNil(t, exec.lastArgs.NetworkAccessEnabled)
	assert.False(t, *exec.lastArgs.NetworkAccessEnabled)
	assert.Equal(t, []string{"/extra"}, exec.lastArgs.AdditionalDirectories)
	assert.Equal(t, []string{"/tmp/a.png"}, exec.lastArgs.Images)
}

func TestThreadRunAggregatesItemsAndUsage(t *testing.T) {
	thread := newThread(&fakeExec{
		events: []any{
			map[string]any{"type": "thread.started", "thread_id": "thread-1"},
			map[string]any{
				"type": "item.completed",
				"item": map[string]any{"id": "agent-1", "type": "agent_message", "text": "done"},
			},
			map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        2,
					"cached_input_tokens": 1,
					"output_tokens":       3,
				},
			},
		},
	}, CodexOptions{}, ThreadOptions{}, nil)

	result, err := thread.Run(t.Context(), "hello", nil)
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "done", result.FinalResponse)
	require.NotNil(t, result.Usage)
	assert.Equal(t, 2, result.Usage.InputTokens)
	assert.Equal(t, 1, result.Usage.CachedInputTokens)
	assert.Equal(t, 3, result.Usage.OutputTokens)
	assert.Len(t, result.Items, 1)
}

func TestThreadRunRaisesOnFailure(t *testing.T) {
	thread := newThread(&fakeExec{
		events: []any{
			map[string]any{"type": "turn.failed", "error": map[string]any{"message": "boom"}},
		},
	}, CodexOptions{}, ThreadOptions{}, nil)

	_, err := thread.Run(t.Context(), "hello", nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "boom")
}

func TestThreadRunRaisesOnStreamError(t *testing.T) {
	thread := newThread(&fakeExec{
		events: []any{
			map[string]any{"type": "error", "message": "boom"},
		},
	}, CodexOptions{}, ThreadOptions{}, nil)

	_, err := thread.Run(t.Context(), "hello", nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Codex stream error: boom")
}

func TestThreadRunStreamedRaisesOnParseError(t *testing.T) {
	thread := newThread(&fakeExec{
		events: []any{"not-json"},
	}, CodexOptions{}, ThreadOptions{}, nil)

	streamed, err := thread.RunStreamed(t.Context(), "hello", nil)
	require.NoError(t, err)

	_, err = collectStreamedEvents(streamed)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Failed to parse event")
}

func TestThreadRunStreamedIdleTimeoutForNonNativeExec(t *testing.T) {
	thread := newThread(&fakeExec{
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
		delay: 200 * time.Millisecond,
	}, CodexOptions{}, ThreadOptions{}, nil)

	streamed, err := thread.RunStreamed(t.Context(), "hello", TurnOptions{IdleTimeoutSeconds: ptr(0.01)})
	require.NoError(t, err)

	_, err = collectStreamedEvents(streamed)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Codex stream idle for")
}

func TestThreadRunStreamedIdleTimeoutSetsSignal(t *testing.T) {
	exec := &blockingExec{}
	thread := newThread(exec, CodexOptions{}, ThreadOptions{}, nil)
	signal := make(chan struct{})

	streamed, err := thread.RunStreamed(t.Context(), "hello", TurnOptions{
		Signal:             signal,
		IdleTimeoutSeconds: ptr(0.01),
	})
	require.NoError(t, err)

	_, err = collectStreamedEvents(streamed)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Codex stream idle for")

	select {
	case <-signal:
	default:
		t.Fatal("expected signal to be closed")
	}
}

func collectStreamedEvents(streamed *StreamedTurn) ([]ThreadEvent, error) {
	out := make([]ThreadEvent, 0)
	events := streamed.Events
	errs := streamed.Errors
	for events != nil || errs != nil {
		select {
		case event, ok := <-events:
			if !ok {
				events = nil
				continue
			}
			out = append(out, event)
		case err, ok := <-errs:
			if !ok {
				errs = nil
				continue
			}
			if err != nil && !errors.Is(err, context.Canceled) {
				return out, err
			}
		}
	}
	return out, nil
}

func ptr[T any](value T) *T {
	return &value
}
