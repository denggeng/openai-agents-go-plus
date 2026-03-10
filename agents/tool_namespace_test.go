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

package agents_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolNamespaceCopiesToolsWithMetadata(t *testing.T) {
	tool := testFunctionTool("argless_function", "Lookup account")

	namespacedTools, err := agents.ToolNamespace("crm", "CRM tools", tool)
	require.NoError(t, err)
	require.Len(t, namespacedTools, 1)

	assert.Equal(t, "crm", namespacedTools[0].Namespace)
	assert.Equal(t, "CRM tools", namespacedTools[0].NamespaceDescription)
	assert.Equal(t, "crm.argless_function", namespacedTools[0].QualifiedName())
	assert.Empty(t, tool.Namespace)
	assert.Equal(t, "argless_function", tool.QualifiedName())
}

func TestToolNamespaceRequiresNonEmptyDescription(t *testing.T) {
	tool := testFunctionTool("argless_function", "Lookup account")

	_, err := agents.ToolNamespace("crm", "   ", tool)
	require.ErrorContains(t, err, "non-empty description")
}

func TestToolNamespaceRejectsReservedSameNameShape(t *testing.T) {
	tool := testFunctionTool("lookup_account", "Lookup account")

	_, err := agents.ToolNamespace("lookup_account", "Same-name namespace", tool)
	require.ErrorContains(t, err, "synthetic namespace `lookup_account.lookup_account`")
}

func TestProcessModelResponseUsesNamespacedFunctionTool(t *testing.T) {
	crmTools, err := agents.ToolNamespace("crm", "CRM tools", testFunctionTool("lookup_account", "CRM lookup"))
	require.NoError(t, err)
	billingTools, err := agents.ToolNamespace("billing", "Billing tools", testFunctionTool("lookup_account", "Billing lookup"))
	require.NoError(t, err)

	processed, err := agents.RunImpl().ProcessModelResponse(
		t.Context(),
		&agents.Agent{Name: "agent"},
		[]agents.Tool{crmTools[0], billingTools[0]},
		agents.ModelResponse{
			Output: []responses.ResponseOutputItemUnion{
				mustFunctionCallOutputItem(t, "lookup_account", "call-billing", "billing"),
			},
		},
		nil,
	)
	require.NoError(t, err)
	require.Len(t, processed.Functions, 1)
	assert.Equal(t, "billing", processed.Functions[0].FunctionTool.Namespace)
	assert.Equal(t, "billing.lookup_account", processed.Functions[0].FunctionTool.QualifiedName())
	assert.Equal(t, []string{"billing.lookup_account"}, processed.ToolsUsed)

	callItem, ok := processed.NewItems[0].(agents.ToolCallItem)
	require.True(t, ok)
	raw, err := json.Marshal(callItem.ToInputItem())
	require.NoError(t, err)
	assert.Contains(t, string(raw), `"namespace":"billing"`)
}

func TestProcessModelResponseDoesNotFallbackFromNamespacedCallToBareTool(t *testing.T) {
	_, err := agents.RunImpl().ProcessModelResponse(
		t.Context(),
		&agents.Agent{Name: "agent"},
		[]agents.Tool{testFunctionTool("lookup_account", "Bare lookup")},
		agents.ModelResponse{
			Output: []responses.ResponseOutputItemUnion{
				mustFunctionCallOutputItem(t, "lookup_account", "call-billing", "billing"),
			},
		},
		nil,
	)
	require.ErrorContains(t, err, "billing.lookup_account")
}

func testFunctionTool(name, description string) agents.FunctionTool {
	return agents.FunctionTool{
		Name:             name,
		Description:      description,
		ParamsJSONSchema: map[string]any{"type": "object", "properties": map[string]any{}},
		StrictJSONSchema: param.NewOpt(true),
		OnInvokeTool: func(context.Context, string) (any, error) {
			return name, nil
		},
	}
}

func mustFunctionCallOutputItem(
	t *testing.T,
	name string,
	callID string,
	namespace string,
) responses.ResponseOutputItemUnion {
	t.Helper()
	payload := map[string]any{
		"type":      "function_call",
		"name":      name,
		"call_id":   callID,
		"arguments": "{}",
		"status":    "completed",
	}
	if namespace != "" {
		payload["namespace"] = namespace
	}
	raw, err := json.Marshal(payload)
	require.NoError(t, err)
	var item responses.ResponseOutputItemUnion
	require.NoError(t, json.Unmarshal(raw, &item))
	return item
}
