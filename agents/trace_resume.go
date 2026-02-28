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

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"maps"
	"sync"

	"github.com/denggeng/openai-agents-go-plus/tracing"
)

var startedTraceIDs sync.Map

func hashTracingAPIKey(apiKey string) string {
	if apiKey == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(apiKey))
	return hex.EncodeToString(sum[:])
}

func currentTracingAPIKeyHash() string {
	return hashTracingAPIKey(tracing.DefaultExporter().APIKey())
}

func traceStartCacheKey(traceID string, apiKeyHash string) string {
	return traceID + "|" + apiKeyHash
}

func markTraceIDStarted(traceID string, apiKeyHash string) {
	if traceID == "" || traceID == "no-op" {
		return
	}
	startedTraceIDs.Store(traceStartCacheKey(traceID, apiKeyHash), struct{}{})
}

func traceIDWasStarted(traceID string, apiKeyHash string) bool {
	if traceID == "" || traceID == "no-op" {
		return false
	}
	_, ok := startedTraceIDs.Load(traceStartCacheKey(traceID, apiKeyHash))
	return ok
}

func mergeTraceMetadata(primary, secondary map[string]any) map[string]any {
	if len(primary) == 0 && len(secondary) == 0 {
		return nil
	}
	merged := maps.Clone(primary)
	if merged == nil {
		merged = make(map[string]any, len(secondary))
	}
	maps.Copy(merged, secondary)
	return merged
}

func applyResumeTraceDefaults(params tracing.TraceParams, resumeState *RunState) tracing.TraceParams {
	if resumeState == nil || resumeState.Trace == nil {
		return params
	}
	traceState := resumeState.Trace
	if params.TraceID == "" {
		params.TraceID = traceState.TraceID
	}
	if params.WorkflowName == "" {
		params.WorkflowName = traceState.WorkflowName
	}
	if params.GroupID == "" {
		params.GroupID = traceState.GroupID
	}
	if len(params.Metadata) == 0 && len(traceState.Metadata) > 0 {
		params.Metadata = maps.Clone(traceState.Metadata)
	}
	return params
}

func resolveReattachedTrace(
	params tracing.TraceParams,
	resumeState *RunState,
) tracing.Trace {
	if resumeState == nil || resumeState.Trace == nil {
		return nil
	}
	traceState := resumeState.Trace
	if traceState.TraceID == "" {
		return nil
	}
	if params.TraceID != "" && params.TraceID != traceState.TraceID {
		return nil
	}
	currentHash := currentTracingAPIKeyHash()
	if traceState.TracingAPIKeyHash != "" && currentHash != "" && traceState.TracingAPIKeyHash != currentHash {
		return nil
	}
	if !traceIDWasStarted(traceState.TraceID, currentHash) {
		return nil
	}
	return &reattachedTrace{
		traceID: traceState.TraceID,
		name:    cmp.Or(params.WorkflowName, traceState.WorkflowName),
		groupID: cmp.Or(params.GroupID, traceState.GroupID),
		metadata: mergeTraceMetadata(
			traceState.Metadata,
			params.Metadata,
		),
	}
}

type reattachedTrace struct {
	traceID  string
	name     string
	groupID  string
	metadata map[string]any

	prevContextTrace tracing.Trace
}

func (t *reattachedTrace) Run(ctx context.Context, fn func(context.Context, tracing.Trace) error) (err error) {
	ctx = tracing.ContextWithClonedOrNewScope(ctx)
	if err = t.Start(ctx, true); err != nil {
		return err
	}
	defer func() {
		if finishErr := t.Finish(ctx, true); finishErr != nil {
			err = errors.Join(err, finishErr)
		}
	}()
	return fn(ctx, t)
}

func (t *reattachedTrace) Start(ctx context.Context, markAsCurrent bool) error {
	if markAsCurrent {
		t.prevContextTrace = tracing.SetCurrentTraceToContextScope(ctx, t)
	}
	return nil
}

func (t *reattachedTrace) Finish(ctx context.Context, resetCurrent bool) error {
	if resetCurrent {
		tracing.SetCurrentTraceToContextScope(ctx, t.prevContextTrace)
		t.prevContextTrace = nil
	}
	return nil
}

func (t *reattachedTrace) TraceID() string { return t.traceID }
func (t *reattachedTrace) Name() string    { return t.name }
func (t *reattachedTrace) Export() map[string]any {
	return map[string]any{
		"object":        "trace",
		"id":            t.traceID,
		"workflow_name": t.name,
		"group_id":      t.groupID,
		"metadata":      maps.Clone(t.metadata),
	}
}
