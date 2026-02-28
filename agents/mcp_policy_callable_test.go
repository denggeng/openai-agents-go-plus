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
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMCPRequireApprovalCallableDefaultsToApprovedWhenAgentMissing(t *testing.T) {
	called := false
	policy := normalizeMCPNeedsApprovalPolicy(MCPRequireApprovalFunc(
		func(context.Context, *RunContextWrapper[any], *Agent, *mcp.Tool) (bool, error) {
			called = true
			return false, nil
		},
	))

	needsApproval := policy.forTool(&mcp.Tool{Name: "search"}, nil)
	require.NotNil(t, needsApproval)

	approved, err := needsApproval.NeedsApproval(
		t.Context(),
		NewRunContextWrapper[any](nil),
		FunctionTool{Name: "search"},
		nil,
		"call_1",
	)
	require.NoError(t, err)
	assert.True(t, approved)
	assert.False(t, called)
}

func TestMCPRequireApprovalCallableUsesDynamicPolicyWhenAgentPresent(t *testing.T) {
	agent := New("test")
	var capturedToolName string
	var capturedAgent *Agent
	var capturedContext any

	policy := normalizeMCPNeedsApprovalPolicy(MCPRequireApprovalFunc(
		func(
			_ context.Context,
			runContext *RunContextWrapper[any],
			inAgent *Agent,
			inTool *mcp.Tool,
		) (bool, error) {
			capturedAgent = inAgent
			capturedToolName = inTool.Name
			capturedContext = runContext.Context
			return false, nil
		},
	))

	needsApproval := policy.forTool(&mcp.Tool{Name: "search"}, agent)
	require.NotNil(t, needsApproval)

	runContext := NewRunContextWrapper[any](map[string]any{"request_id": "req_1"})
	approved, err := needsApproval.NeedsApproval(
		t.Context(),
		runContext,
		FunctionTool{Name: "search"},
		nil,
		"call_1",
	)
	require.NoError(t, err)
	assert.False(t, approved)
	assert.Same(t, agent, capturedAgent)
	assert.Equal(t, "search", capturedToolName)
	assert.Equal(t, map[string]any{"request_id": "req_1"}, capturedContext)
}
