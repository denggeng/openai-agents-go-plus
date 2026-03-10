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

	"github.com/openai/openai-go/v3/responses"
)

type FunctionToolLookupKeyKind string

const (
	functionToolLookupKeyBare       FunctionToolLookupKeyKind = "bare"
	functionToolLookupKeyNamespaced FunctionToolLookupKeyKind = "namespaced"
)

// FunctionToolLookupKey identifies a function tool without colliding with other
// tools that share the same public name.
type FunctionToolLookupKey struct {
	Kind      FunctionToolLookupKeyKind
	Name      string
	Namespace string
}

func (k FunctionToolLookupKey) IsZero() bool {
	return strings.TrimSpace(k.Name) == ""
}

func (k FunctionToolLookupKey) ApprovalKey() string {
	switch k.Kind {
	case functionToolLookupKeyNamespaced:
		return toolQualifiedName(k.Name, k.Namespace)
	case functionToolLookupKeyBare:
		return k.Name
	default:
		return ""
	}
}

func toolQualifiedName(name, namespace string) string {
	name = strings.TrimSpace(name)
	namespace = strings.TrimSpace(namespace)
	if name == "" {
		return ""
	}
	if namespace != "" {
		return namespace + "." + name
	}
	return name
}

func toolTraceName(name, namespace string) string {
	name = strings.TrimSpace(name)
	namespace = strings.TrimSpace(namespace)
	if name == "" {
		return ""
	}
	if isReservedSyntheticToolNamespace(name, namespace) {
		return name
	}
	return toolQualifiedName(name, namespace)
}

func isReservedSyntheticToolNamespace(name, namespace string) bool {
	name = strings.TrimSpace(name)
	namespace = strings.TrimSpace(namespace)
	return name != "" && namespace != "" && name == namespace
}

func asFunctionTool(tool Tool) (FunctionTool, bool) {
	switch typed := tool.(type) {
	case FunctionTool:
		return typed, true
	case *FunctionTool:
		if typed == nil {
			return FunctionTool{}, false
		}
		return *typed, true
	default:
		return FunctionTool{}, false
	}
}

func getFunctionToolLookupKey(toolName, namespace string) (FunctionToolLookupKey, bool) {
	toolName = strings.TrimSpace(toolName)
	namespace = strings.TrimSpace(namespace)
	if toolName == "" {
		return FunctionToolLookupKey{}, false
	}
	if namespace != "" {
		return FunctionToolLookupKey{
			Kind:      functionToolLookupKeyNamespaced,
			Name:      toolName,
			Namespace: namespace,
		}, true
	}
	return FunctionToolLookupKey{
		Kind: functionToolLookupKeyBare,
		Name: toolName,
	}, true
}

func getFunctionToolLookupKeyForTool(tool FunctionTool) (FunctionToolLookupKey, bool) {
	return getFunctionToolLookupKey(tool.Name, tool.Namespace)
}

func validateFunctionToolNamespaceShape(toolName, namespace string) error {
	if !isReservedSyntheticToolNamespace(toolName, namespace) {
		return nil
	}
	reservedKey := toolQualifiedName(toolName, namespace)
	if reservedKey == "" {
		reservedKey = strings.TrimSpace(toolName)
	}
	if reservedKey == "" {
		reservedKey = "unknown_tool"
	}
	return UserErrorf(
		"Responses tool-search reserves the synthetic namespace `%s` for deferred top-level function tools. Rename the namespace or tool name to avoid ambiguous dispatch.",
		reservedKey,
	)
}

func validateFunctionToolLookupConfiguration(tools []Tool) error {
	type qualifiedOwner struct {
		namespace string
	}

	qualifiedNameOwners := make(map[string]qualifiedOwner)
	for _, tool := range tools {
		functionTool, ok := asFunctionTool(tool)
		if !ok {
			continue
		}
		if err := validateFunctionToolNamespaceShape(functionTool.Name, functionTool.Namespace); err != nil {
			return err
		}

		qualifiedName := toolQualifiedName(functionTool.Name, functionTool.Namespace)
		if qualifiedName == "" {
			continue
		}

		priorOwner, ok := qualifiedNameOwners[qualifiedName]
		if !ok {
			qualifiedNameOwners[qualifiedName] = qualifiedOwner{namespace: functionTool.Namespace}
			continue
		}

		if strings.TrimSpace(priorOwner.namespace) == "" && strings.TrimSpace(functionTool.Namespace) == "" {
			// Bare duplicates are allowed; the last definition wins.
			continue
		}

		return UserErrorf(
			"Ambiguous function tool configuration: the qualified name `%s` is used by multiple tools. Rename the namespace-wrapped function or dotted top-level tool to avoid ambiguous dispatch.",
			qualifiedName,
		)
	}
	return nil
}

func buildFunctionToolLookupMap(tools []Tool) (map[string]FunctionTool, error) {
	if err := validateFunctionToolLookupConfiguration(tools); err != nil {
		return nil, err
	}

	out := make(map[string]FunctionTool)
	for _, tool := range tools {
		functionTool, ok := asFunctionTool(tool)
		if !ok {
			continue
		}
		lookupKey, ok := getFunctionToolLookupKeyForTool(functionTool)
		if !ok {
			continue
		}
		out[lookupKey.ApprovalKey()] = functionTool
	}
	return out, nil
}

func ensureFunctionToolSupportsResponsesOnlyFeatures(tool FunctionTool, backendName string) error {
	if strings.TrimSpace(tool.Namespace) == "" {
		return nil
	}
	toolName := tool.QualifiedName()
	if toolName == "" {
		toolName = tool.Name
	}
	return UserErrorf(
		"The following function-tool features are only supported with OpenAI Responses models: tool_namespace(). Tool `%s` cannot be used with %s.",
		toolName,
		backendName,
	)
}

// ToolNamespace copies function tools and attaches namespace metadata used by
// OpenAI Responses namespace wrappers.
func ToolNamespace(name, description string, tools ...FunctionTool) ([]FunctionTool, error) {
	namespaceName := strings.TrimSpace(name)
	if namespaceName == "" {
		return nil, NewUserError("tool_namespace() requires a non-empty namespace name.")
	}
	namespaceDescription := strings.TrimSpace(description)
	if namespaceDescription == "" {
		return nil, NewUserError("tool_namespace() requires a non-empty description.")
	}

	namespacedTools := make([]FunctionTool, 0, len(tools))
	for _, tool := range tools {
		if err := validateFunctionToolNamespaceShape(tool.Name, namespaceName); err != nil {
			return nil, err
		}
		toolCopy := tool
		toolCopy.Namespace = namespaceName
		toolCopy.NamespaceDescription = namespaceDescription
		namespacedTools = append(namespacedTools, toolCopy)
	}
	return namespacedTools, nil
}

func functionToolCallNamespace(raw any) string {
	if value, ok := stringFromMap(raw, "namespace"); ok {
		return strings.TrimSpace(value)
	}
	return rawJSONObjectFieldString(raw, "namespace")
}

func rawJSONObjectFieldString(value any, key string) string {
	raw := rawJSONFromValue(value)
	if raw == "" {
		return ""
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		return ""
	}
	field, ok := payload[key]
	if !ok {
		return ""
	}
	var out string
	if err := json.Unmarshal(field, &out); err != nil {
		return ""
	}
	return strings.TrimSpace(out)
}

func rawJSONFromValue(value any) string {
	switch typed := value.(type) {
	case interface{ RawJSON() string }:
		return typed.RawJSON()
	case ResponseFunctionToolCall:
		return responses.ResponseFunctionToolCall(typed).RawJSON()
	case *ResponseFunctionToolCall:
		if typed == nil {
			return ""
		}
		return responses.ResponseFunctionToolCall(*typed).RawJSON()
	case responses.ResponseFunctionToolCall:
		return typed.RawJSON()
	case *responses.ResponseFunctionToolCall:
		if typed == nil {
			return ""
		}
		return typed.RawJSON()
	default:
		return ""
	}
}
