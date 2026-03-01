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

package agentstesting

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type FakeMCPServer struct {
	name                      string
	Tools                     []*mcp.Tool
	ToolCalls                 []string
	ToolResults               []string
	ToolMetas                 []map[string]any
	ToolFilter                agents.MCPToolFilter
	RequireApproval           any
	ToolMetaResolver          agents.MCPToolMetaResolver
	FailureErrorFunction      *agents.ToolErrorFunction
	FailureErrorFunctionSet   bool
	UseStructuredContentValue bool
}

func NewFakeMCPServer(
	tools []*mcp.Tool,
	toolFilter agents.MCPToolFilter,
	name string,
) *FakeMCPServer {
	return &FakeMCPServer{
		name:       cmp.Or(name, "fake_mcp_server"),
		Tools:      tools,
		ToolFilter: toolFilter,
	}
}

func (s *FakeMCPServer) AddTool(name string, inputSchema *jsonschema.Schema) {
	s.Tools = append(s.Tools, &mcp.Tool{
		Name:        name,
		InputSchema: inputSchema,
	})
}

func (s *FakeMCPServer) Connect(context.Context) error { return nil }
func (s *FakeMCPServer) Cleanup(context.Context) error { return nil }
func (s *FakeMCPServer) Name() string                  { return s.name }
func (s *FakeMCPServer) UseStructuredContent() bool    { return s.UseStructuredContentValue }

func (s *FakeMCPServer) ListTools(ctx context.Context, agent *agents.Agent) ([]*mcp.Tool, error) {
	tools := s.Tools

	// Apply tool filtering using the REAL implementation
	if s.ToolFilter != nil {
		filterContext := agents.MCPToolFilterContext{
			Agent:      agent,
			ServerName: s.name,
		}
		tools = agents.ApplyMCPToolFilter(ctx, filterContext, s.ToolFilter, tools, agent)
	}
	return tools, nil
}

func (s *FakeMCPServer) CallTool(_ context.Context, toolName string, arguments map[string]any, meta map[string]any) (*mcp.CallToolResult, error) {
	s.ToolCalls = append(s.ToolCalls, toolName)
	if meta != nil {
		s.ToolMetas = append(s.ToolMetas, mapsClone(meta))
	} else {
		s.ToolMetas = append(s.ToolMetas, nil)
	}
	b, err := json.Marshal(arguments)
	if err != nil {
		return nil, err
	}
	result := fmt.Sprintf("result_%s_%s", toolName, string(b))
	s.ToolResults = append(s.ToolResults, result)
	return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: result}}}, nil
}

func (s *FakeMCPServer) MCPNeedsApprovalForTool(tool *mcp.Tool, _ *agents.Agent) agents.FunctionToolNeedsApproval {
	policy := normalizeApprovalForFakeServer(s.RequireApproval)
	return policy.forTool(tool)
}

func (s *FakeMCPServer) MCPFailureErrorFunctionOverride() (bool, *agents.ToolErrorFunction) {
	if !s.FailureErrorFunctionSet {
		return false, nil
	}
	if s.FailureErrorFunction == nil {
		var nilFn agents.ToolErrorFunction
		return true, &nilFn
	}
	cloned := *s.FailureErrorFunction
	return true, &cloned
}

func (s *FakeMCPServer) MCPResolveToolMeta(
	ctx context.Context,
	metaCtx agents.MCPToolMetaContext,
) (map[string]any, error) {
	if s.ToolMetaResolver == nil {
		return nil, nil
	}
	return s.ToolMetaResolver(ctx, metaCtx)
}

// ListPrompts returns empty list of prompts for fake server.
func (s *FakeMCPServer) ListPrompts(context.Context) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}

// GetPrompt returns a simple prompt result for fake server.
func (s *FakeMCPServer) GetPrompt(_ context.Context, name string, _ map[string]string) (*mcp.GetPromptResult, error) {
	content := fmt.Sprintf("Fake prompt content for %s", name)
	message := &mcp.PromptMessage{
		Content: &mcp.TextContent{Text: content},
		Role:    "user",
	}
	return &mcp.GetPromptResult{
		Description: fmt.Sprintf("Fake prompt: %s", name),
		Messages:    []*mcp.PromptMessage{message},
	}, nil
}

func mapsClone(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

type fakeMCPNeedsApprovalPolicy struct {
	defaultNeedsApproval bool
	toolNeedsApproval    map[string]bool
}

func normalizeApprovalForFakeServer(requireApproval any) fakeMCPNeedsApprovalPolicy {
	switch value := requireApproval.(type) {
	case nil:
		return fakeMCPNeedsApprovalPolicy{}
	case bool:
		return fakeMCPNeedsApprovalPolicy{defaultNeedsApproval: value}
	case string:
		return fakeMCPNeedsApprovalPolicy{defaultNeedsApproval: value == "always"}
	case map[string]bool:
		return fakeMCPNeedsApprovalPolicy{toolNeedsApproval: value}
	case map[string]string:
		mapped := make(map[string]bool, len(value))
		for name, policy := range value {
			switch policy {
			case "always":
				mapped[name] = true
			case "never":
				mapped[name] = false
			}
		}
		return fakeMCPNeedsApprovalPolicy{toolNeedsApproval: mapped}
	case map[string]any:
		if mapped, ok := parseTSStylePolicy(value); ok {
			return fakeMCPNeedsApprovalPolicy{toolNeedsApproval: mapped}
		}
		mapped := make(map[string]bool)
		for name, raw := range value {
			switch policy := raw.(type) {
			case bool:
				mapped[name] = policy
			case string:
				switch policy {
				case "always":
					mapped[name] = true
				case "never":
					mapped[name] = false
				}
			}
		}
		return fakeMCPNeedsApprovalPolicy{toolNeedsApproval: mapped}
	default:
		return fakeMCPNeedsApprovalPolicy{}
	}
}

func (p fakeMCPNeedsApprovalPolicy) forTool(tool *mcp.Tool) agents.FunctionToolNeedsApproval {
	if tool != nil && len(p.toolNeedsApproval) > 0 {
		if p.toolNeedsApproval[tool.Name] {
			return agents.FunctionToolNeedsApprovalEnabled()
		}
		return nil
	}
	if p.defaultNeedsApproval {
		return agents.FunctionToolNeedsApprovalEnabled()
	}
	return nil
}

func parseTSStylePolicy(raw map[string]any) (map[string]bool, bool) {
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
	always := anyToStringSlice(alwaysMap["tool_names"])
	never := anyToStringSlice(neverMap["tool_names"])
	mapped := make(map[string]bool, len(always)+len(never))
	for _, name := range always {
		if name == "" {
			continue
		}
		mapped[name] = true
	}
	for _, name := range never {
		if name == "" {
			continue
		}
		mapped[name] = false
	}
	return mapped, true
}

func anyToStringSlice(value any) []string {
	switch v := value.(type) {
	case []string:
		return slices.Clone(v)
	case []any:
		var out []string
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}
