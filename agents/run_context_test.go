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

package agents_test

import (
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunContextLatestApprovalDecisionWinsForCallID(t *testing.T) {
	ctx := agents.NewRunContextWrapper[any](nil)
	approvalItem := agents.ToolApprovalItem{
		ToolName: "test_tool",
		RawItem: map[string]any{
			"call_id": "call-1",
		},
	}

	ctx.ApproveTool(approvalItem, false)
	approved, known := ctx.IsToolApproved("test_tool", "call-1")
	require.True(t, known)
	assert.True(t, approved)

	ctx.RejectTool(approvalItem, false)
	approved, known = ctx.IsToolApproved("test_tool", "call-1")
	require.True(t, known)
	assert.False(t, approved)

	ctx.ApproveTool(approvalItem, false)
	approved, known = ctx.IsToolApproved("test_tool", "call-1")
	require.True(t, known)
	assert.True(t, approved)
}

func TestRunContextAlwaysApproveAndReject(t *testing.T) {
	ctx := agents.NewRunContextWrapper[any](nil)
	approvalItem := agents.ToolApprovalItem{
		ToolName: "test_tool",
		RawItem: map[string]any{
			"call_id": "call-1",
		},
	}

	ctx.ApproveTool(approvalItem, true)
	approved, known := ctx.IsToolApproved("test_tool", "another-call")
	require.True(t, known)
	assert.True(t, approved)

	ctx.RejectTool(approvalItem, true)
	approved, known = ctx.IsToolApproved("test_tool", "another-call")
	require.True(t, known)
	assert.False(t, approved)
}

func TestRunContextResolveToolNameAndCallIDFromRawItem(t *testing.T) {
	ctx := agents.NewRunContextWrapper[any](nil)

	approvalItem := agents.ToolApprovalItem{
		RawItem: map[string]any{
			"type": "mcp_tool",
			"provider_data": map[string]any{
				"type": "mcp_approval_request",
				"id":   "approval-1",
			},
		},
	}

	ctx.ApproveTool(approvalItem, false)
	approved, known := ctx.IsToolApproved("mcp_tool", "approval-1")
	require.True(t, known)
	assert.True(t, approved)
}

func TestRunContextGetApprovalStatusFallbackToolName(t *testing.T) {
	ctx := agents.NewRunContextWrapper[any](nil)
	pending := agents.ToolApprovalItem{
		ToolName: "actual_tool",
		RawItem: map[string]any{
			"call_id": "call-1",
		},
	}

	ctx.ApproveTool(pending, false)

	approved, known := ctx.GetApprovalStatus("mismatched_name", "call-1", &pending)
	require.True(t, known)
	assert.True(t, approved)
}

func TestRunContextSerializeAndRebuildApprovals(t *testing.T) {
	ctx := agents.NewRunContextWrapper[any](nil)
	ctx.ApproveTool(agents.ToolApprovalItem{
		ToolName: "tool_1",
		RawItem:  map[string]any{"call_id": "call-1"},
	}, false)
	ctx.RejectTool(agents.ToolApprovalItem{
		ToolName: "tool_2",
		RawItem:  map[string]any{"call_id": "call-2"},
	}, true)

	serialized := ctx.SerializeApprovals()
	require.NotEmpty(t, serialized)

	restored := agents.NewRunContextWrapper[any](nil)
	restored.RebuildApprovals(serialized)

	approved, known := restored.IsToolApproved("tool_1", "call-1")
	require.True(t, known)
	assert.True(t, approved)

	approved, known = restored.IsToolApproved("tool_2", "any")
	require.True(t, known)
	assert.False(t, approved)
}
