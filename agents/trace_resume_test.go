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
	"fmt"
	"maps"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/tracing"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveReattachedTraceMatchesEffectiveSettings(t *testing.T) {
	clearStartedTraceIDsForTest()
	setTracingAPIKeyForTest(t, "trace-key")

	traceState := newStartedTraceState("trace_reattach_match", "workflow", "group-1", map[string]any{"key": "value"}, "trace-key")
	trace := resolveReattachedTrace(
		tracing.TraceParams{
			WorkflowName: "workflow",
			TraceID:      traceState.TraceID,
			GroupID:      "group-1",
			Metadata:     map[string]any{"key": "value"},
		},
		&RunState{Trace: traceState},
	)

	reattached, ok := trace.(*reattachedTrace)
	require.True(t, ok)
	assert.Equal(t, traceState.TraceID, reattached.TraceID())
	assert.Equal(t, "workflow", reattached.Name())
	assert.Equal(t, "trace-key", reattached.Export()["tracing_api_key"])
	assert.Equal(t, hashTracingAPIKey("trace-key"), reattached.Export()["tracing_api_key_hash"])
}

func TestResolveReattachedTraceDoesNotReattachAfterTraceStateReload(t *testing.T) {
	clearStartedTraceIDsForTest()
	setTracingAPIKeyForTest(t, "")

	traceState := newStartedTraceState("trace_reload", "workflow", "group-1", map[string]any{"key": "value"}, "")
	clearStartedTraceIDsForTest()

	trace := resolveReattachedTrace(
		tracing.TraceParams{
			WorkflowName: "workflow",
			TraceID:      traceState.TraceID,
			GroupID:      "group-1",
			Metadata:     map[string]any{"key": "value"},
		},
		&RunState{Trace: traceState},
	)

	assert.Nil(t, trace)
}

func TestResolveReattachedTraceSupportsStrippedTraceKeyWithMatchingResumeKey(t *testing.T) {
	clearStartedTraceIDsForTest()
	setTracingAPIKeyForTest(t, "trace-key")

	traceState := newStartedTraceState("trace_stripped_match", "workflow", "group-1", map[string]any{"key": "value"}, "trace-key")
	traceState.TracingAPIKey = ""

	trace := resolveReattachedTrace(
		tracing.TraceParams{
			WorkflowName: "workflow",
			TraceID:      traceState.TraceID,
			GroupID:      "group-1",
			Metadata:     map[string]any{"key": "value"},
		},
		&RunState{Trace: traceState},
	)

	reattached, ok := trace.(*reattachedTrace)
	require.True(t, ok)
	assert.Equal(t, "trace-key", reattached.Export()["tracing_api_key"])
}

func TestResolveReattachedTraceRejectsMismatchedSettings(t *testing.T) {
	clearStartedTraceIDsForTest()
	traceState := newStartedTraceState("trace_mismatch", "workflow", "group-1", map[string]any{"key": "value"}, "trace-key")

	testCases := []struct {
		name   string
		params tracing.TraceParams
		apiKey string
	}{
		{
			name: "workflow mismatch",
			params: tracing.TraceParams{
				WorkflowName: "workflow-override",
				TraceID:      traceState.TraceID,
				GroupID:      "group-1",
				Metadata:     map[string]any{"key": "value"},
			},
			apiKey: "trace-key",
		},
		{
			name: "group mismatch",
			params: tracing.TraceParams{
				WorkflowName: "workflow",
				TraceID:      traceState.TraceID,
				GroupID:      "group-override",
				Metadata:     map[string]any{"key": "value"},
			},
			apiKey: "trace-key",
		},
		{
			name: "metadata mismatch",
			params: tracing.TraceParams{
				WorkflowName: "workflow",
				TraceID:      traceState.TraceID,
				GroupID:      "group-1",
				Metadata:     map[string]any{"key": "override"},
			},
			apiKey: "trace-key",
		},
		{
			name: "api key mismatch",
			params: tracing.TraceParams{
				WorkflowName: "workflow",
				TraceID:      traceState.TraceID,
				GroupID:      "group-1",
				Metadata:     map[string]any{"key": "value"},
			},
			apiKey: "other-trace-key",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			setTracingAPIKeyForTest(t, tc.apiKey)
			trace := resolveReattachedTrace(tc.params, &RunState{Trace: traceState})
			assert.Nil(t, trace)
		})
	}
}

func TestResolveReattachedTraceRespectsDisabledFlag(t *testing.T) {
	clearStartedTraceIDsForTest()
	setTracingAPIKeyForTest(t, "")

	traceState := newStartedTraceState("trace_disabled", "workflow", "group-1", map[string]any{"key": "value"}, "")
	trace := resolveReattachedTrace(
		tracing.TraceParams{
			WorkflowName: "workflow",
			TraceID:      traceState.TraceID,
			GroupID:      "group-1",
			Metadata:     map[string]any{"key": "value"},
			Disabled:     true,
		},
		&RunState{Trace: traceState},
	)

	assert.Nil(t, trace)
}

func TestStartedTraceIDCacheIsBounded(t *testing.T) {
	clearStartedTraceIDsForTest()
	for i := 0; i < maxStartedTraceIDs+1; i++ {
		markTraceIDStarted(fmt.Sprintf("trace_%04d", i))
	}

	startedTraceIDsMu.Lock()
	defer startedTraceIDsMu.Unlock()

	assert.Len(t, startedTraceIDs, maxStartedTraceIDs)
	assert.Len(t, startedTraceIDOrder, maxStartedTraceIDs)
	_, exists := startedTraceIDs["trace_0000"]
	assert.False(t, exists)
	_, exists = startedTraceIDs[fmt.Sprintf("trace_%04d", maxStartedTraceIDs)]
	assert.True(t, exists)
}

func newStartedTraceState(
	traceID string,
	workflowName string,
	groupID string,
	metadata map[string]any,
	tracingAPIKey string,
) *TraceState {
	markTraceIDStarted(traceID)
	return &TraceState{
		ObjectType:        "trace",
		TraceID:           traceID,
		WorkflowName:      workflowName,
		GroupID:           groupID,
		Metadata:          maps.Clone(metadata),
		TracingAPIKey:     tracingAPIKey,
		TracingAPIKeyHash: hashTracingAPIKey(tracingAPIKey),
	}
}

func clearStartedTraceIDsForTest() {
	startedTraceIDsMu.Lock()
	defer startedTraceIDsMu.Unlock()
	clear(startedTraceIDs)
	startedTraceIDOrder = nil
}

func setTracingAPIKeyForTest(t *testing.T, apiKey string) {
	t.Helper()
	prev := tracing.DefaultExporter().APIKey()
	tracing.SetTracingExportAPIKey(apiKey)
	t.Cleanup(func() {
		tracing.SetTracingExportAPIKey(prev)
	})
}
