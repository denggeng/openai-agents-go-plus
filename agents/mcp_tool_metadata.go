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

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3/responses"
)

type mcpToolMetadata struct {
	Description string
	Title       string
}

type mcpToolMetadataKey struct {
	ServerLabel string
	Name        string
}

type rawJSONProvider interface {
	RawJSON() string
}

func resolveMCPToolTitle(tool any) string {
	if payload := rawJSONPayload(tool); payload != nil {
		if title := nonEmptyString(payload["title"]); title != "" {
			return title
		}
		if annotations, ok := payload["annotations"].(map[string]any); ok {
			if title := nonEmptyString(annotations["title"]); title != "" {
				return title
			}
		}
	}

	switch typed := tool.(type) {
	case *mcp.Tool:
		if typed == nil {
			return ""
		}
		if typed.Title != "" {
			return typed.Title
		}
		if typed.Annotations != nil && typed.Annotations.Title != "" {
			return typed.Annotations.Title
		}
	case mcp.Tool:
		return resolveMCPToolTitle(&typed)
	case *responses.ResponseOutputItemMcpListToolsTool:
		if typed == nil {
			return ""
		}
		return titleFromAnnotationsValue(typed.Annotations)
	case responses.ResponseOutputItemMcpListToolsTool:
		return resolveMCPToolTitle(&typed)
	case map[string]any:
		if title := nonEmptyString(typed["title"]); title != "" {
			return title
		}
		if annotations, ok := typed["annotations"].(map[string]any); ok {
			return nonEmptyString(annotations["title"])
		}
	}

	return ""
}

func resolveMCPToolDescription(tool any) string {
	if payload := rawJSONPayload(tool); payload != nil {
		if description := nonEmptyString(payload["description"]); description != "" {
			return description
		}
	}

	switch typed := tool.(type) {
	case *mcp.Tool:
		if typed == nil {
			return ""
		}
		return typed.Description
	case mcp.Tool:
		return typed.Description
	case *responses.ResponseOutputItemMcpListToolsTool:
		if typed == nil {
			return ""
		}
		return typed.Description
	case responses.ResponseOutputItemMcpListToolsTool:
		return typed.Description
	case map[string]any:
		return nonEmptyString(typed["description"])
	}

	return ""
}

func resolveMCPToolDescriptionForModel(tool any) string {
	if description := resolveMCPToolDescription(tool); description != "" {
		return description
	}
	return resolveMCPToolTitle(tool)
}

func extractMCPToolMetadata(tool any) mcpToolMetadata {
	return mcpToolMetadata{
		Description: resolveMCPToolDescription(tool),
		Title:       resolveMCPToolTitle(tool),
	}
}

func collectMCPListToolsMetadata(items []RunItem) map[mcpToolMetadataKey]mcpToolMetadata {
	if len(items) == 0 {
		return nil
	}

	out := make(map[mcpToolMetadataKey]mcpToolMetadata)
	for _, item := range items {
		switch typed := item.(type) {
		case MCPListToolsItem:
			collectMCPListToolsMetadataFromRawItem(typed.RawItem, out)
		case *MCPListToolsItem:
			if typed != nil {
				collectMCPListToolsMetadataFromRawItem(typed.RawItem, out)
			}
		}
	}

	if len(out) == 0 {
		return nil
	}
	return out
}

func collectMCPListToolsMetadataFromInputItems(items []TResponseInputItem) map[mcpToolMetadataKey]mcpToolMetadata {
	if len(items) == 0 {
		return nil
	}

	out := make(map[mcpToolMetadataKey]mcpToolMetadata)
	for _, item := range items {
		raw := normalizeJSONValue(item)
		payload, ok := raw.(map[string]any)
		if !ok || payload["type"] != "mcp_list_tools" {
			continue
		}
		decoded, ok := decodeRawToResponseOutputItemMcpListTools(payload)
		if !ok {
			continue
		}
		collectMCPListToolsMetadataFromRawItem(decoded, out)
	}

	if len(out) == 0 {
		return nil
	}
	return out
}

func collectMCPListToolsMetadataFromRawItem(
	item responses.ResponseOutputItemMcpListTools,
	out map[mcpToolMetadataKey]mcpToolMetadata,
) {
	if item.ServerLabel == "" || len(item.Tools) == 0 {
		return
	}
	for _, tool := range item.Tools {
		if tool.Name == "" {
			continue
		}
		out[mcpToolMetadataKey{
			ServerLabel: item.ServerLabel,
			Name:        tool.Name,
		}] = extractMCPToolMetadata(tool)
	}
}

func rawJSONPayload(value any) map[string]any {
	provider, ok := value.(rawJSONProvider)
	if !ok {
		return nil
	}
	raw := provider.RawJSON()
	if raw == "" {
		return nil
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		return nil
	}
	return payload
}

func titleFromAnnotationsValue(value any) string {
	switch typed := value.(type) {
	case *mcp.ToolAnnotations:
		if typed == nil {
			return ""
		}
		return typed.Title
	case mcp.ToolAnnotations:
		return typed.Title
	case map[string]any:
		return nonEmptyString(typed["title"])
	default:
		if payload := rawJSONPayload(value); payload != nil {
			return nonEmptyString(payload["title"])
		}
	}
	return ""
}

func nonEmptyString(value any) string {
	text, _ := value.(string)
	return text
}
