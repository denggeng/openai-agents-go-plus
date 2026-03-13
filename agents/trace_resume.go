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
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"maps"
	"reflect"
	"sync"

	"github.com/denggeng/openai-agents-go-plus/tracing"
)

const maxStartedTraceIDs = 4096

var (
	startedTraceIDsMu   sync.Mutex
	startedTraceIDs     = make(map[string]struct{})
	startedTraceIDOrder []string
)

func hashTracingAPIKey(apiKey string) string {
	if apiKey == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(apiKey))
	return hex.EncodeToString(sum[:])
}

func currentTracingAPIKey() string {
	return tracing.DefaultExporter().APIKey()
}

func currentTracingAPIKeyHash() string {
	return hashTracingAPIKey(currentTracingAPIKey())
}

func markTraceIDStarted(traceID string) {
	if traceID == "" || traceID == "no-op" {
		return
	}
	startedTraceIDsMu.Lock()
	defer startedTraceIDsMu.Unlock()

	if _, exists := startedTraceIDs[traceID]; exists {
		for i, existing := range startedTraceIDOrder {
			if existing == traceID {
				copy(startedTraceIDOrder[i:], startedTraceIDOrder[i+1:])
				startedTraceIDOrder = startedTraceIDOrder[:len(startedTraceIDOrder)-1]
				break
			}
		}
	} else {
		startedTraceIDs[traceID] = struct{}{}
	}
	startedTraceIDOrder = append(startedTraceIDOrder, traceID)
	for len(startedTraceIDOrder) > maxStartedTraceIDs {
		evicted := startedTraceIDOrder[0]
		startedTraceIDOrder = startedTraceIDOrder[1:]
		delete(startedTraceIDs, evicted)
	}
}

func traceIDWasStarted(traceID string) bool {
	if traceID == "" || traceID == "no-op" {
		return false
	}
	startedTraceIDsMu.Lock()
	defer startedTraceIDsMu.Unlock()
	_, ok := startedTraceIDs[traceID]
	return ok
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
	if params.Disabled || resumeState == nil || resumeState.Trace == nil {
		return nil
	}
	traceState := resumeState.Trace
	if !traceStateMatchesEffectiveSettings(traceState, params) {
		return nil
	}
	if !traceIDWasStarted(traceState.TraceID) {
		return nil
	}
	currentAPIKey := currentTracingAPIKey()
	currentHash := hashTracingAPIKey(currentAPIKey)
	return &reattachedTrace{
		traceID:           traceState.TraceID,
		name:              traceState.WorkflowName,
		groupID:           traceState.GroupID,
		metadata:          maps.Clone(traceState.Metadata),
		tracingAPIKey:     currentAPIKey,
		tracingAPIKeyHash: currentHash,
	}
}

type reattachedTrace struct {
	traceID           string
	name              string
	groupID           string
	metadata          map[string]any
	tracingAPIKey     string
	tracingAPIKeyHash string

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
	payload := map[string]any{
		"object":        "trace",
		"id":            t.traceID,
		"workflow_name": t.name,
		"group_id":      t.groupID,
		"metadata":      maps.Clone(t.metadata),
	}
	if t.tracingAPIKey != "" {
		payload["tracing_api_key"] = t.tracingAPIKey
	}
	if t.tracingAPIKeyHash != "" {
		payload["tracing_api_key_hash"] = t.tracingAPIKeyHash
	}
	return payload
}

func traceStateMatchesEffectiveSettings(traceState *TraceState, params tracing.TraceParams) bool {
	if traceState == nil || traceState.TraceID == "" || params.TraceID != traceState.TraceID {
		return false
	}
	if traceState.WorkflowName != params.WorkflowName {
		return false
	}
	if traceState.GroupID != params.GroupID {
		return false
	}
	if !reflect.DeepEqual(traceState.Metadata, params.Metadata) {
		return false
	}

	currentAPIKey := currentTracingAPIKey()
	switch {
	case traceState.TracingAPIKey != "":
		return traceState.TracingAPIKey == currentAPIKey
	case traceState.TracingAPIKeyHash != "":
		return traceState.TracingAPIKeyHash == hashTracingAPIKey(currentAPIKey)
	default:
		return currentAPIKey == ""
	}
}
