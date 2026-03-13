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
	"encoding/json"
	"strings"
)

type ToolSearchExecution string

const (
	ToolSearchExecutionServer ToolSearchExecution = "server"
	ToolSearchExecutionClient ToolSearchExecution = "client"
)

// ToolSearchTool is a hosted Responses API tool that lets the model search deferred tools.
type ToolSearchTool struct {
	Description string
	Execution   ToolSearchExecution
	Parameters  any
}

func (t ToolSearchTool) ToolName() string { return "tool_search" }

func (ToolSearchTool) isTool() {}

func isResponsesToolSearchSurface(tool Tool) bool {
	functionTool, ok := asFunctionTool(tool)
	if ok {
		return functionTool.DeferLoading || strings.TrimSpace(functionTool.Namespace) != ""
	}

	hostedMCPTool, ok := asHostedMCPTool(tool)
	if ok {
		return hostedMCPTool.DeferLoading || hostedMCPToolConfigDeferLoading(hostedMCPTool.ToolConfig)
	}
	return false
}

func hasResponsesToolSearchSurface(tools []Tool) bool {
	for _, tool := range tools {
		if isResponsesToolSearchSurface(tool) {
			return true
		}
	}
	return false
}

func isRequiredToolSearchSurface(tool Tool) bool {
	functionTool, ok := asFunctionTool(tool)
	if ok {
		return functionTool.DeferLoading
	}

	hostedMCPTool, ok := asHostedMCPTool(tool)
	if ok {
		return hostedMCPTool.DeferLoading || hostedMCPToolConfigDeferLoading(hostedMCPTool.ToolConfig)
	}
	return false
}

func hasRequiredToolSearchSurface(tools []Tool) bool {
	for _, tool := range tools {
		if isRequiredToolSearchSurface(tool) {
			return true
		}
	}
	return false
}

func validateResponsesToolSearchConfiguration(
	tools []Tool,
	allowOpaqueSearchSurface bool,
) error {
	toolSearchCount := 0
	hasToolSearch := false
	for _, tool := range tools {
		switch typed := tool.(type) {
		case ToolSearchTool:
			toolSearchCount++
			hasToolSearch = true
		case *ToolSearchTool:
			if typed != nil {
				toolSearchCount++
				hasToolSearch = true
			}
		}
	}
	hasToolSearchSurface := hasResponsesToolSearchSurface(tools)
	hasRequiredToolSearch := hasRequiredToolSearchSurface(tools)

	if toolSearchCount > 1 {
		return NewUserError("Only one ToolSearchTool() is allowed when using OpenAI Responses models.")
	}
	if err := validateFunctionToolLookupConfiguration(tools); err != nil {
		return err
	}
	if hasRequiredToolSearch && !hasToolSearch {
		return NewUserError(
			"Deferred-loading Responses tools require ToolSearchTool() when using OpenAI Responses models.",
		)
	}
	if hasToolSearch && !hasToolSearchSurface && !allowOpaqueSearchSurface {
		return NewUserError(
			"ToolSearchTool() requires at least one searchable Responses surface: a tool_namespace(...) function tool, a deferred-loading function tool (`function_tool(..., defer_loading=True)`), or a deferred-loading hosted MCP server (`HostedMCPTool(tool_config={..., 'defer_loading': True})`).",
		)
	}
	return nil
}

func hostedMCPToolConfigDeferLoading(config any) bool {
	data, err := json.Marshal(config)
	if err != nil {
		return false
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return false
	}
	value, ok := payload["defer_loading"]
	if !ok {
		return false
	}
	flag, ok := value.(bool)
	return ok && flag
}

func asHostedMCPTool(tool Tool) (HostedMCPTool, bool) {
	switch typed := tool.(type) {
	case HostedMCPTool:
		return typed, true
	case *HostedMCPTool:
		if typed == nil {
			return HostedMCPTool{}, false
		}
		return *typed, true
	default:
		return HostedMCPTool{}, false
	}
}
