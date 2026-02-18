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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type brokenStringer struct{}

func (brokenStringer) String() string {
	panic("broken")
}

func TestRunContextToStrOrNoneHandlesErrors(t *testing.T) {
	value, ok := coerceStringValue("ok")
	require.True(t, ok)
	assert.Equal(t, "ok", value)

	value, ok = coerceStringValue(123)
	require.True(t, ok)
	assert.Equal(t, "123", value)

	value, ok = coerceStringValue(brokenStringer{})
	assert.False(t, ok)
	assert.Equal(t, "", value)

	value, ok = coerceStringValue(nil)
	assert.False(t, ok)
	assert.Equal(t, "", value)
}

func TestRunContextResolveToolNameAndCallIDFallbacks(t *testing.T) {
	raw := map[string]any{
		"name": "raw_tool",
		"id":   "raw-id",
	}
	item := ToolApprovalItem{RawItem: raw}

	assert.Equal(t, "raw_tool", resolveApprovalToolName(item))
	assert.Equal(t, "raw-id", resolveApprovalCallID(item))
}

func TestRunContextUnknownToolNameFallback(t *testing.T) {
	item := ToolApprovalItem{RawItem: map[string]any{}}
	assert.Equal(t, "unknown_tool", resolveApprovalToolName(item))
}

func TestRunContextScopesApprovalsToCallIDs(t *testing.T) {
	wrapper := NewRunContextWrapper(map[string]any{})
	approval := ToolApprovalItem{RawItem: map[string]any{"type": "tool_call", "call_id": "call-1"}}

	wrapper.ApproveTool(approval, false)
	approved, known := wrapper.IsToolApproved("tool_call", "call-1")
	require.True(t, known)
	assert.True(t, approved)

	_, known = wrapper.IsToolApproved("tool_call", "call-2")
	assert.False(t, known)
}

func TestRunContextScopesRejectionsToCallIDs(t *testing.T) {
	wrapper := NewRunContextWrapper(map[string]any{})
	approval := ToolApprovalItem{RawItem: map[string]any{"type": "tool_call", "call_id": "call-1"}}

	wrapper.RejectTool(approval, false)
	approved, known := wrapper.IsToolApproved("tool_call", "call-1")
	require.True(t, known)
	assert.False(t, approved)

	_, known = wrapper.IsToolApproved("tool_call", "call-2")
	assert.False(t, known)
}

func TestRunContextHonorsGlobalApprovalAndRejection(t *testing.T) {
	wrapper := NewRunContextWrapper(map[string]any{})
	approval := ToolApprovalItem{RawItem: map[string]any{"type": "tool_call", "call_id": "call-1"}}

	wrapper.ApproveTool(approval, true)
	approved, known := wrapper.IsToolApproved("tool_call", "call-2")
	require.True(t, known)
	assert.True(t, approved)

	wrapper.RejectTool(approval, true)
	approved, known = wrapper.IsToolApproved("tool_call", "call-3")
	require.True(t, known)
	assert.False(t, approved)
}
