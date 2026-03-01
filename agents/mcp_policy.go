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
	"encoding/json"
	"log/slog"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// MCPRequireApprovalToolList configures approval policy with an explicit tool list.
type MCPRequireApprovalToolList struct {
	ToolNames []string
}

// MCPRequireApprovalObject configures approval policy with TS-style always/never lists.
type MCPRequireApprovalObject struct {
	Always *MCPRequireApprovalToolList
	Never  *MCPRequireApprovalToolList
}

// MCPToolMetaContext provides metadata resolver context for MCP tool calls.
type MCPToolMetaContext struct {
	RunContext *RunContextWrapper[any]
	ServerName string
	ToolName   string
	Arguments  map[string]any
}

// MCPToolMetaResolver computes request `_meta` values for MCP tool calls.
type MCPToolMetaResolver func(context.Context, MCPToolMetaContext) (map[string]any, error)

// MCPRequireApprovalFunc computes dynamic approval requirements for MCP tools.
type MCPRequireApprovalFunc func(
	ctx context.Context,
	runContext *RunContextWrapper[any],
	agent *Agent,
	tool *mcp.Tool,
) (bool, error)

type mcpNeedsApprovalPolicy struct {
	defaultNeedsApproval bool
	toolNeedsApproval    map[string]bool
	callable             MCPRequireApprovalFunc
}

func normalizeMCPNeedsApprovalPolicy(requireApproval any) mcpNeedsApprovalPolicy {
	switch value := requireApproval.(type) {
	case nil:
		return mcpNeedsApprovalPolicy{}
	case bool:
		return mcpNeedsApprovalPolicy{defaultNeedsApproval: value}
	case string:
		return mcpNeedsApprovalPolicy{defaultNeedsApproval: strings.EqualFold(value, "always")}
	case MCPRequireApprovalObject:
		return mcpNeedsApprovalPolicy{toolNeedsApproval: mapToolApprovalLists(value.Always, value.Never)}
	case *MCPRequireApprovalObject:
		if value == nil {
			return mcpNeedsApprovalPolicy{}
		}
		return mcpNeedsApprovalPolicy{toolNeedsApproval: mapToolApprovalLists(value.Always, value.Never)}
	case map[string]bool:
		mapped := make(map[string]bool, len(value))
		for name, v := range value {
			mapped[name] = v
		}
		return mcpNeedsApprovalPolicy{toolNeedsApproval: mapped}
	case map[string]string:
		mapped := make(map[string]bool, len(value))
		for name, v := range value {
			switch strings.ToLower(strings.TrimSpace(v)) {
			case "always":
				mapped[name] = true
			case "never":
				mapped[name] = false
			}
		}
		return mcpNeedsApprovalPolicy{toolNeedsApproval: mapped}
	case map[string]any:
		if mapped, ok := parseTSStyleToolApprovalObject(value); ok {
			return mcpNeedsApprovalPolicy{toolNeedsApproval: mapped}
		}
		mapped := make(map[string]bool)
		for name, raw := range value {
			switch v := raw.(type) {
			case bool:
				mapped[name] = v
			case string:
				switch strings.ToLower(strings.TrimSpace(v)) {
				case "always":
					mapped[name] = true
				case "never":
					mapped[name] = false
				}
			}
		}
		return mcpNeedsApprovalPolicy{toolNeedsApproval: mapped}
	case MCPRequireApprovalFunc:
		return mcpNeedsApprovalPolicy{callable: value}
	case *MCPRequireApprovalFunc:
		if value == nil {
			return mcpNeedsApprovalPolicy{}
		}
		return mcpNeedsApprovalPolicy{callable: *value}
	default:
		return mcpNeedsApprovalPolicy{}
	}
}

func parseTSStyleToolApprovalObject(raw map[string]any) (map[string]bool, bool) {
	alwaysRaw, hasAlways := raw["always"]
	neverRaw, hasNever := raw["never"]
	if !hasAlways && !hasNever {
		return nil, false
	}

	alwaysMap, alwaysOK := alwaysRaw.(map[string]any)
	neverMap, neverOK := neverRaw.(map[string]any)
	if hasAlways && (!alwaysOK || alwaysMap["tool_names"] == nil) {
		return nil, false
	}
	if hasNever && (!neverOK || neverMap["tool_names"] == nil) {
		return nil, false
	}

	alwaysTools := extractToolNames(alwaysMap["tool_names"])
	neverTools := extractToolNames(neverMap["tool_names"])
	return mapToolApprovalNames(alwaysTools, neverTools), true
}

func mapToolApprovalLists(always, never *MCPRequireApprovalToolList) map[string]bool {
	var alwaysNames, neverNames []string
	if always != nil {
		alwaysNames = always.ToolNames
	}
	if never != nil {
		neverNames = never.ToolNames
	}
	return mapToolApprovalNames(alwaysNames, neverNames)
}

func mapToolApprovalNames(always, never []string) map[string]bool {
	result := make(map[string]bool, len(always)+len(never))
	for _, name := range always {
		if strings.TrimSpace(name) == "" {
			continue
		}
		result[name] = true
	}
	for _, name := range never {
		if strings.TrimSpace(name) == "" {
			continue
		}
		result[name] = false
	}
	return result
}

func extractToolNames(raw any) []string {
	switch value := raw.(type) {
	case []string:
		return append([]string(nil), value...)
	case []any:
		tools := make([]string, 0, len(value))
		for _, item := range value {
			if s, ok := item.(string); ok {
				tools = append(tools, s)
			}
		}
		return tools
	default:
		return nil
	}
}

func (p mcpNeedsApprovalPolicy) forTool(tool *mcp.Tool, agent *Agent) FunctionToolNeedsApproval {
	if p.callable != nil {
		if agent == nil {
			// Keep approval conservative when dynamic policy context is unavailable.
			return FunctionToolNeedsApprovalEnabled()
		}
		return FunctionToolNeedsApprovalFunc(func(
			ctx context.Context,
			runContext *RunContextWrapper[any],
			_ FunctionTool,
			_ map[string]any,
			_ string,
		) (bool, error) {
			return p.callable(ctx, runContext, agent, tool)
		})
	}
	if tool != nil && len(p.toolNeedsApproval) > 0 {
		if p.toolNeedsApproval[tool.Name] {
			return FunctionToolNeedsApprovalEnabled()
		}
		return nil
	}
	if p.defaultNeedsApproval {
		return FunctionToolNeedsApprovalEnabled()
	}
	return nil
}

func resolveMCPRunContextFromContext(ctx context.Context) *RunContextWrapper[any] {
	if value, ok := RunContextValueFromContext(ctx); ok {
		if wrapper, ok := value.(*RunContextWrapper[any]); ok && wrapper != nil {
			return wrapper
		}
		return NewRunContextWrapper[any](value)
	}
	return NewRunContextWrapper[any](nil)
}

func mergeMCPMeta(resolvedMeta, explicitMeta map[string]any) map[string]any {
	if len(resolvedMeta) == 0 && len(explicitMeta) == 0 {
		return nil
	}
	merged := make(map[string]any, len(resolvedMeta)+len(explicitMeta))
	for key, value := range resolvedMeta {
		merged[key] = value
	}
	for key, value := range explicitMeta {
		merged[key] = value
	}
	return merged
}

func deepCopyMap(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}

	payload, err := json.Marshal(input)
	if err == nil {
		var output map[string]any
		unmarshalErr := json.Unmarshal(payload, &output)
		if unmarshalErr == nil {
			return output
		}
		Logger().Warn(
			"MCP meta deep copy fallback to shallow copy",
			slog.String("reason", "unmarshal"),
			slog.String("error", unmarshalErr.Error()),
		)
		return shallowCopyMap(input)
	}

	Logger().Warn(
		"MCP meta deep copy fallback to shallow copy",
		slog.String("reason", "marshal"),
		slog.String("error", err.Error()),
	)
	return shallowCopyMap(input)
}

func shallowCopyMap(input map[string]any) map[string]any {
	clone := make(map[string]any, len(input))
	for key, value := range input {
		clone[key] = value
	}
	return clone
}

func cloneToolErrorFunctionPointer(fn *ToolErrorFunction) *ToolErrorFunction {
	if fn == nil {
		return nil
	}
	cloned := *fn
	return &cloned
}
