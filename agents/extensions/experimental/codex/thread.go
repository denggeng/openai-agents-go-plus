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
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/denggeng/openai-agents-go-plus/agents"
)

// TextInput is a Codex text input item.
type TextInput struct {
	Type string
	Text string
}

// LocalImageInput is a Codex local image input item.
type LocalImageInput struct {
	Type string
	Path string
}

// Turn aggregates one completed turn result.
type Turn struct {
	Items         []ThreadItem
	FinalResponse string
	Usage         *Usage
}

// RunResult is the non-streaming turn result.
type RunResult = Turn

// StreamedTurn wraps event and error streams for one run.
type StreamedTurn struct {
	Events <-chan ThreadEvent
	Errors <-chan error
}

// RunStreamedResult is the streaming turn result.
type RunStreamedResult = StreamedTurn

// Thread is a Codex conversation thread handle.
type Thread struct {
	execClient    CodexExecClient
	options       CodexOptions
	threadOptions ThreadOptions

	mu sync.RWMutex
	id *string
}

func newThread(
	execClient CodexExecClient,
	options CodexOptions,
	threadOptions ThreadOptions,
	threadID *string,
) *Thread {
	var idCopy *string
	if threadID != nil {
		value := *threadID
		idCopy = &value
	}
	return &Thread{
		execClient:    execClient,
		options:       *cloneCodexOptions(&options),
		threadOptions: *cloneThreadOptions(&threadOptions),
		id:            idCopy,
	}
}

// ID returns the thread identifier used for resume operations.
func (t *Thread) ID() *string {
	if t == nil {
		return nil
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.id == nil {
		return nil
	}
	value := *t.id
	return &value
}

// RunStreamed starts a streamed turn and returns decoded thread events.
func (t *Thread) RunStreamed(ctx context.Context, input any, turnOptions any) (*StreamedTurn, error) {
	resolvedTurnOptions, err := CoerceTurnOptions(turnOptions)
	if err != nil {
		return nil, err
	}
	if resolvedTurnOptions == nil {
		resolvedTurnOptions = &TurnOptions{}
	}

	events, errs := t.runStreamedInternal(ctx, input, *resolvedTurnOptions)
	return &StreamedTurn{
		Events: events,
		Errors: errs,
	}, nil
}

// Run aggregates a streamed turn into final turn output.
func (t *Thread) Run(ctx context.Context, input any, turnOptions any) (*Turn, error) {
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	streamed, err := t.RunStreamed(runCtx, input, turnOptions)
	if err != nil {
		return nil, err
	}

	items := make([]ThreadItem, 0)
	finalResponse := ""
	var usage *Usage

	events := streamed.Events
	errs := streamed.Errors
	for events != nil || errs != nil {
		select {
		case event, ok := <-events:
			if !ok {
				events = nil
				continue
			}
			switch typed := event.(type) {
			case ItemCompletedEvent:
				item := typed.Item
				if agentMessageItem, ok := item.(AgentMessageItem); ok {
					finalResponse = agentMessageItem.Text
				}
				items = append(items, item)
			case TurnCompletedEvent:
				usage = typed.Usage
			case TurnFailedEvent:
				cancel()
				message := strings.TrimSpace(typed.Error.Message)
				if message == "" {
					message = "Codex turn failed"
				}
				return nil, fmt.Errorf("%s", message)
			case ThreadErrorEvent:
				cancel()
				return nil, fmt.Errorf("Codex stream error: %s", typed.Message)
			}
		case streamErr, ok := <-errs:
			if !ok {
				errs = nil
				continue
			}
			if streamErr != nil {
				cancel()
				return nil, streamErr
			}
		}
	}

	return &Turn{
		Items:         items,
		FinalResponse: finalResponse,
		Usage:         usage,
	}, nil
}

func (t *Thread) runStreamedInternal(
	ctx context.Context,
	input any,
	turnOptions TurnOptions,
) (<-chan ThreadEvent, <-chan error) {
	events := make(chan ThreadEvent, 8)
	errs := make(chan error, 1)

	outputSchemaFile, err := CreateOutputSchemaFile(turnOptions.OutputSchema)
	if err != nil {
		close(events)
		errs <- err
		close(errs)
		return events, errs
	}

	prompt, images, err := normalizeInput(input)
	if err != nil {
		outputSchemaFile.Cleanup()
		close(events)
		errs <- err
		close(errs)
		return events, errs
	}

	signal, triggerSignal := resolveSignal(turnOptions)
	execArgs := CodexExecArgs{
		Input:                 prompt,
		BaseURL:               t.options.BaseURL,
		APIKey:                t.options.APIKey,
		ThreadID:              t.ID(),
		Images:                images,
		Model:                 t.threadOptions.Model,
		SandboxMode:           t.threadOptions.SandboxMode,
		WorkingDirectory:      t.threadOptions.WorkingDirectory,
		AdditionalDirectories: cloneStringSlice(t.threadOptions.AdditionalDirectories),
		SkipGitRepoCheck:      t.threadOptions.SkipGitRepoCheck,
		OutputSchemaFile:      outputSchemaFile.SchemaPath,
		ModelReasoningEffort:  t.threadOptions.ModelReasoningEffort,
		Signal:                signal,
		IdleTimeoutSeconds:    turnOptions.IdleTimeoutSeconds,
		NetworkAccessEnabled:  t.threadOptions.NetworkAccessEnabled,
		WebSearchMode:         t.threadOptions.WebSearchMode,
		WebSearchEnabled:      t.threadOptions.WebSearchEnabled,
		ApprovalPolicy:        t.threadOptions.ApprovalPolicy,
	}
	rawLines, rawErrs := t.execClient.RunJSONL(ctx, execArgs)

	go func() {
		defer close(events)
		defer close(errs)
		defer outputSchemaFile.Cleanup()

		nonNativeTimeout := turnOptions.IdleTimeoutSeconds != nil && !isNativeCodexExec(t.execClient)
		timeoutDuration := durationFromIdleTimeout(turnOptions.IdleTimeoutSeconds)
		var idleTimer *time.Timer
		var idleTimeout <-chan time.Time
		if nonNativeTimeout && timeoutDuration > 0 {
			idleTimer = time.NewTimer(timeoutDuration)
			idleTimeout = idleTimer.C
		}
		stopIdleTimer := func() {
			if idleTimer == nil {
				return
			}
			if !idleTimer.Stop() {
				select {
				case <-idleTimer.C:
				default:
				}
			}
		}
		resetIdleTimer := func() {
			if idleTimer == nil {
				return
			}
			stopIdleTimer()
			idleTimer.Reset(timeoutDuration)
		}
		defer stopIdleTimer()

		lines := rawLines
		lineErrs := rawErrs
		for lines != nil || lineErrs != nil {
			select {
			case <-ctx.Done():
				trySendError(errs, ctx.Err())
				return
			case <-idleTimeout:
				triggerSignal()
				trySendError(errs, idleTimeoutError(turnOptions.IdleTimeoutSeconds))
				return
			case rawLine, ok := <-lines:
				if !ok {
					lines = nil
					continue
				}
				resetIdleTimer()
				parsedEvent, err := parseEvent(rawLine)
				if err != nil {
					trySendError(errs, fmt.Errorf("Failed to parse event: %s", rawLine))
					return
				}
				if startedEvent, ok := parsedEvent.(ThreadStartedEvent); ok {
					t.setID(startedEvent.ThreadID)
				}
				select {
				case events <- parsedEvent:
				case <-ctx.Done():
					trySendError(errs, ctx.Err())
					return
				}
			case lineErr, ok := <-lineErrs:
				if !ok {
					lineErrs = nil
					continue
				}
				resetIdleTimer()
				if lineErr != nil {
					trySendError(errs, lineErr)
					return
				}
			}
		}
	}()

	return events, errs
}

func (t *Thread) setID(threadID string) {
	if strings.TrimSpace(threadID) == "" {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	value := threadID
	t.id = &value
}

func durationFromIdleTimeout(idleTimeoutSeconds *float64) time.Duration {
	if idleTimeoutSeconds == nil {
		return 0
	}
	if *idleTimeoutSeconds <= 0 {
		return 0
	}
	return time.Duration(*idleTimeoutSeconds * float64(time.Second))
}

func resolveSignal(turnOptions TurnOptions) (<-chan struct{}, func()) {
	if turnOptions.Signal != nil {
		return turnOptions.Signal, func() {}
	}
	if turnOptions.IdleTimeoutSeconds == nil {
		return nil, func() {}
	}
	created := make(chan struct{})
	var once sync.Once
	return created, func() {
		once.Do(func() {
			close(created)
		})
	}
}

func parseEvent(raw string) (ThreadEvent, error) {
	var parsed map[string]any
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil, err
	}
	return CoerceThreadEvent(parsed)
}

func isNativeCodexExec(execClient CodexExecClient) bool {
	_, ok := execClient.(*CodexExec)
	return ok
}

func cloneStringSlice(value []string) []string {
	if value == nil {
		return nil
	}
	out := make([]string, len(value))
	copy(out, value)
	return out
}

func trySendError(ch chan<- error, err error) {
	if err == nil {
		return
	}
	select {
	case ch <- err:
	default:
	}
}

func idleTimeoutError(idleTimeoutSeconds *float64) error {
	if idleTimeoutSeconds == nil {
		return fmt.Errorf("Codex stream idle timeout exceeded")
	}
	return fmt.Errorf("Codex stream idle for %g seconds.", *idleTimeoutSeconds)
}

func normalizeInput(input any) (string, []string, error) {
	switch typed := input.(type) {
	case string:
		return typed, nil, nil
	case []map[string]any:
		return normalizeInputItems(typed)
	case []any:
		items := make([]map[string]any, 0, len(typed))
		for _, raw := range typed {
			item, ok := raw.(map[string]any)
			if !ok {
				return "", nil, agents.NewUserError("input items must be a list of objects")
			}
			items = append(items, item)
		}
		return normalizeInputItems(items)
	default:
		return "", nil, agents.NewUserError("input must be a string or a list of input items")
	}
}

func normalizeInputItems(items []map[string]any) (string, []string, error) {
	promptParts := make([]string, 0)
	images := make([]string, 0)

	for _, item := range items {
		itemType, _ := item["type"].(string)
		switch itemType {
		case "text":
			text, _ := item["text"].(string)
			promptParts = append(promptParts, text)
		case "local_image":
			path, _ := item["path"].(string)
			if strings.TrimSpace(path) != "" {
				images = append(images, path)
			}
		default:
			return "", nil, agents.UserErrorf("unsupported input item type %q", itemType)
		}
	}

	return strings.Join(promptParts, "\n\n"), images, nil
}
