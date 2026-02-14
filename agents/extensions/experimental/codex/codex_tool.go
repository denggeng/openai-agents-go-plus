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

package codex

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	usagepkg "github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

var jsonPrimitiveTypes = map[string]struct{}{
	"string":  {},
	"number":  {},
	"integer": {},
	"boolean": {},
}

const (
	DefaultCodexToolName         = "codex"
	CodexToolNamePrefix          = "codex_"
	DefaultRunContextThreadIDKey = "codex_thread_id"
	DefaultCodexToolDescription  = "Executes an agentic Codex task against the current workspace."
	defaultSpanDataMaxChars      = 8192
)

var spanTrimKeys = []string{
	"arguments",
	"command",
	"output",
	"result",
	"error",
	"text",
	"changes",
	"items",
}

// CodexToolInputItem is one structured input for Codex tool invocations.
type CodexToolInputItem struct {
	Type string  `json:"type"`
	Text *string `json:"text,omitempty"`
	Path *string `json:"path,omitempty"`
}

// CodexToolResult is the normalized result returned by Codex tool invocations.
type CodexToolResult struct {
	ThreadID *string `json:"thread_id"`
	Response string  `json:"response"`
	Usage    *Usage  `json:"usage"`
}

func (r CodexToolResult) AsMap() map[string]any {
	var usageMap any
	if r.Usage != nil {
		usageMap = map[string]any{
			"input_tokens":        r.Usage.InputTokens,
			"cached_input_tokens": r.Usage.CachedInputTokens,
			"output_tokens":       r.Usage.OutputTokens,
		}
	}
	return map[string]any{
		"thread_id": r.ThreadID,
		"response":  r.Response,
		"usage":     usageMap,
	}
}

func (r CodexToolResult) String() string {
	raw, err := json.Marshal(r.AsMap())
	if err != nil {
		return "{}"
	}
	return string(raw)
}

// CodexToolStreamEvent is emitted to the optional on_stream callback.
type CodexToolStreamEvent struct {
	Event    ThreadEvent
	Thread   *Thread
	ToolData *agents.ToolContextData
}

// CodexToolStreamHandler receives streamed Codex events.
type CodexToolStreamHandler func(context.Context, CodexToolStreamEvent) error

// CodexToolOptions configures NewCodexTool.
type CodexToolOptions struct {
	Name                  *string
	Description           *string
	Parameters            any
	SpanDataMaxChars      *int
	OutputSchema          any
	Codex                 *Codex
	CodexOptions          any
	DefaultThreadOptions  any
	ThreadID              *string
	SandboxMode           *string
	WorkingDirectory      *string
	SkipGitRepoCheck      *bool
	DefaultTurnOptions    any
	PersistSession        bool
	OnStream              CodexToolStreamHandler
	IsEnabled             agents.FunctionToolEnabler
	FailureErrorFunction  *agents.ToolErrorFunction
	UseRunContextThreadID bool
	RunContextThreadIDKey *string
}

type codexToolCallArguments struct {
	Inputs   []map[string]any
	ThreadID *string
}

type codexToolInputPayload struct {
	Inputs   []CodexToolInputItem `json:"inputs"`
	ThreadID *string              `json:"thread_id,omitempty"`
}

type codexToolRunContextInputPayload struct {
	Inputs []CodexToolInputItem `json:"inputs"`
}

// NewCodexTool builds a FunctionTool that executes Codex thread turns.
func NewCodexTool(options any) (agents.FunctionTool, error) {
	resolvedOptions, err := CoerceCodexToolOptions(options)
	if err != nil {
		return agents.FunctionTool{}, err
	}

	name, err := resolveCodexToolName(resolvedOptions.Name)
	if err != nil {
		return agents.FunctionTool{}, err
	}
	resolvedRunContextThreadIDKey, err := resolveRunContextThreadIDKey(
		name,
		resolvedOptions.RunContextThreadIDKey,
		resolvedOptions.UseRunContextThreadID,
	)
	if err != nil {
		return agents.FunctionTool{}, err
	}

	description := DefaultCodexToolDescription
	if resolvedOptions.Description != nil && strings.TrimSpace(*resolvedOptions.Description) != "" {
		description = strings.TrimSpace(*resolvedOptions.Description)
	}
	resolvedParametersSchema, err := resolveCodexToolParametersSchema(resolvedOptions.Parameters)
	if err != nil {
		return agents.FunctionTool{}, err
	}

	resolvedOutputSchema, err := resolveOutputSchema(resolvedOptions.OutputSchema)
	if err != nil {
		return agents.FunctionTool{}, err
	}

	resolvedThreadOptions, err := resolveThreadOptions(
		resolvedOptions.DefaultThreadOptions,
		resolvedOptions.SandboxMode,
		resolvedOptions.WorkingDirectory,
		resolvedOptions.SkipGitRepoCheck,
	)
	if err != nil {
		return agents.FunctionTool{}, err
	}

	resolvedTurnOptions, err := CoerceTurnOptions(resolvedOptions.DefaultTurnOptions)
	if err != nil {
		return agents.FunctionTool{}, err
	}
	spanDataMaxChars := resolvedOptions.SpanDataMaxChars
	if spanDataMaxChars == nil {
		value := defaultSpanDataMaxChars
		spanDataMaxChars = &value
	}

	toolSchema := buildCodexToolSchema(!resolvedOptions.UseRunContextThreadID)
	if resolvedParametersSchema != nil {
		toolSchema = resolvedParametersSchema
	}

	var codexMu sync.Mutex
	var codexInstance *Codex
	resolveCodex := func() (*Codex, error) {
		if resolvedOptions.Codex != nil {
			return resolvedOptions.Codex, nil
		}
		codexMu.Lock()
		defer codexMu.Unlock()
		if codexInstance != nil {
			return codexInstance, nil
		}

		optionsValue, err := resolveCodexOptions(resolvedOptions.CodexOptions)
		if err != nil {
			return nil, err
		}
		created, err := NewCodex(optionsValue)
		if err != nil {
			return nil, err
		}
		codexInstance = created
		return codexInstance, nil
	}

	var persistedMu sync.Mutex
	var persistedThread *Thread

	tool := agents.FunctionTool{
		Name:             name,
		Description:      description,
		ParamsJSONSchema: toolSchema,
		StrictJSONSchema: param.NewOpt(true),
		IsEnabled:        resolvedOptions.IsEnabled,
		IsCodexTool:      true,
		OnInvokeTool: func(ctx context.Context, arguments string) (any, error) {
			runContextValue, _ := agents.RunContextValueFromContext(ctx)
			if resolvedOptions.UseRunContextThreadID {
				if err := validateRunContextThreadIDContext(runContextValue, resolvedRunContextThreadIDKey); err != nil {
					return nil, err
				}
			}

			var parsed codexToolCallArguments
			if resolvedParametersSchema != nil {
				parsed, err = parseCodexToolInputWithCustomParameters(arguments)
			} else if resolvedOptions.UseRunContextThreadID {
				parsed, err = parseCodexToolInputWithoutThreadID(arguments)
			} else {
				parsed, err = parseCodexToolInput(arguments)
			}
			if err != nil {
				return nil, err
			}

			callThreadID, err := resolveCallThreadID(
				parsed.ThreadID,
				resolvedOptions.ThreadID,
				runContextValue,
				resolvedRunContextThreadIDKey,
				resolvedOptions.UseRunContextThreadID,
			)
			if err != nil {
				return nil, err
			}
			codexClient, err := resolveCodex()
			if err != nil {
				return nil, err
			}

			var thread *Thread
			if resolvedOptions.PersistSession {
				persistedMu.Lock()
				thread, err = getOrCreatePersistedThread(
					codexClient,
					callThreadID,
					resolvedThreadOptions,
					persistedThread,
				)
				if err == nil && persistedThread == nil {
					persistedThread = thread
				}
				persistedMu.Unlock()
			} else {
				thread, err = getThread(codexClient, callThreadID, resolvedThreadOptions)
			}
			if err != nil {
				return nil, err
			}

			turnOptions := buildTurnOptions(resolvedTurnOptions, resolvedOutputSchema)
			streamed, err := thread.RunStreamed(ctx, parsed.Inputs, turnOptions)
			if err != nil {
				return nil, err
			}

			response, usage, resolvedThreadID, err := consumeCodexToolEvents(
				ctx,
				streamed,
				parsed,
				thread,
				resolvedOptions.OnStream,
				spanDataMaxChars,
			)
			if resolvedThreadID == nil {
				resolvedThreadID = callThreadID
			}
			if err != nil {
				tryStoreThreadIDInRunContextAfterError(
					runContextValue,
					resolvedRunContextThreadIDKey,
					resolvedThreadID,
					resolvedOptions.UseRunContextThreadID,
				)
				return nil, err
			}

			if resolvedOptions.UseRunContextThreadID {
				if err := storeThreadIDInRunContext(
					runContextValue,
					resolvedRunContextThreadIDKey,
					resolvedThreadID,
				); err != nil {
					return nil, err
				}
			}

			return CodexToolResult{
				ThreadID: resolvedThreadID,
				Response: response,
				Usage:    usage,
			}, nil
		},
	}
	if resolvedOptions.FailureErrorFunction != nil {
		tool.FailureErrorFunction = resolvedOptions.FailureErrorFunction
	}
	return tool, nil
}

// MustNewCodexTool is the panic-on-error wrapper around NewCodexTool.
func MustNewCodexTool(options any) agents.FunctionTool {
	tool, err := NewCodexTool(options)
	if err != nil {
		panic(err)
	}
	return tool
}

// CoerceCodexToolOptions accepts nil, CodexToolOptions, or map-based options.
func CoerceCodexToolOptions(options any) (*CodexToolOptions, error) {
	var out *CodexToolOptions
	switch typed := options.(type) {
	case nil:
		out = &CodexToolOptions{}
	case CodexToolOptions:
		out = cloneCodexToolOptions(&typed)
	case *CodexToolOptions:
		if typed == nil {
			out = &CodexToolOptions{}
			break
		}
		out = cloneCodexToolOptions(typed)
	case map[string]any:
		resolved, err := coerceCodexToolOptionsMap(typed)
		if err != nil {
			return nil, err
		}
		out = resolved
	default:
		return nil, agents.NewUserError("Codex tool options must be a CodexToolOptions or a mapping.")
	}

	if out.RunContextThreadIDKey != nil {
		validated, err := validateRunContextThreadIDKey(*out.RunContextThreadIDKey)
		if err != nil {
			return nil, err
		}
		out.RunContextThreadIDKey = &validated
	}
	return out, nil
}

func coerceCodexToolOptionsMap(values map[string]any) (*CodexToolOptions, error) {
	allowed := map[string]struct{}{
		"name":                      {},
		"description":               {},
		"parameters":                {},
		"span_data_max_chars":       {},
		"output_schema":             {},
		"codex":                     {},
		"codex_options":             {},
		"default_thread_options":    {},
		"thread_id":                 {},
		"sandbox_mode":              {},
		"working_directory":         {},
		"skip_git_repo_check":       {},
		"default_turn_options":      {},
		"persist_session":           {},
		"on_stream":                 {},
		"is_enabled":                {},
		"failure_error_function":    {},
		"use_run_context_thread_id": {},
		"run_context_thread_id_key": {},
	}
	unknown := make([]string, 0)
	for key := range values {
		if _, ok := allowed[key]; !ok {
			unknown = append(unknown, key)
		}
	}
	if len(unknown) > 0 {
		return nil, agents.UserErrorf("Unknown Codex tool option(s): %v", unknown)
	}

	out := &CodexToolOptions{}
	for key, raw := range values {
		switch key {
		case "name":
			value, err := optionalString(raw, "name")
			if err != nil {
				return nil, err
			}
			out.Name = value
		case "description":
			value, err := optionalString(raw, "description")
			if err != nil {
				return nil, err
			}
			out.Description = value
		case "parameters":
			out.Parameters = raw
		case "span_data_max_chars":
			value, err := optionalInt(raw, "span_data_max_chars")
			if err != nil {
				return nil, err
			}
			out.SpanDataMaxChars = value
		case "output_schema":
			out.OutputSchema = raw
		case "codex":
			if raw == nil {
				out.Codex = nil
				continue
			}
			client, ok := raw.(*Codex)
			if !ok {
				return nil, agents.NewUserError("codex must be a *Codex or nil")
			}
			out.Codex = client
		case "codex_options":
			out.CodexOptions = raw
		case "default_thread_options":
			out.DefaultThreadOptions = raw
		case "thread_id":
			value, err := optionalString(raw, "thread_id")
			if err != nil {
				return nil, err
			}
			out.ThreadID = value
		case "sandbox_mode":
			value, err := optionalString(raw, "sandbox_mode")
			if err != nil {
				return nil, err
			}
			out.SandboxMode = value
		case "working_directory":
			value, err := optionalString(raw, "working_directory")
			if err != nil {
				return nil, err
			}
			out.WorkingDirectory = value
		case "skip_git_repo_check":
			value, err := optionalBool(raw, "skip_git_repo_check")
			if err != nil {
				return nil, err
			}
			out.SkipGitRepoCheck = value
		case "default_turn_options":
			out.DefaultTurnOptions = raw
		case "persist_session":
			value, ok := raw.(bool)
			if !ok {
				return nil, agents.NewUserError("persist_session must be a bool")
			}
			out.PersistSession = value
		case "on_stream":
			if raw == nil {
				out.OnStream = nil
				continue
			}
			handler, ok := raw.(CodexToolStreamHandler)
			if !ok {
				return nil, agents.NewUserError("on_stream must be a CodexToolStreamHandler")
			}
			out.OnStream = handler
		case "is_enabled":
			if raw == nil {
				out.IsEnabled = nil
				continue
			}
			switch typed := raw.(type) {
			case bool:
				out.IsEnabled = agents.NewFunctionToolEnabledFlag(typed)
			case agents.FunctionToolEnabler:
				out.IsEnabled = typed
			default:
				return nil, agents.NewUserError("is_enabled must be a bool or FunctionToolEnabler")
			}
		case "failure_error_function":
			if raw == nil {
				var fn agents.ToolErrorFunction
				out.FailureErrorFunction = &fn
				continue
			}
			switch typed := raw.(type) {
			case agents.ToolErrorFunction:
				fn := typed
				out.FailureErrorFunction = &fn
			case *agents.ToolErrorFunction:
				out.FailureErrorFunction = typed
			default:
				return nil, agents.NewUserError("failure_error_function must be a ToolErrorFunction")
			}
		case "use_run_context_thread_id":
			value, ok := raw.(bool)
			if !ok {
				return nil, agents.NewUserError("use_run_context_thread_id must be a bool")
			}
			out.UseRunContextThreadID = value
		case "run_context_thread_id_key":
			value, err := optionalString(raw, "run_context_thread_id_key")
			if err != nil {
				return nil, err
			}
			if value != nil {
				validated, err := validateRunContextThreadIDKey(*value)
				if err != nil {
					return nil, err
				}
				out.RunContextThreadIDKey = &validated
			}
		}
	}
	return out, nil
}

func cloneCodexToolOptions(options *CodexToolOptions) *CodexToolOptions {
	if options == nil {
		return &CodexToolOptions{}
	}
	clone := *options
	if options.Name != nil {
		value := *options.Name
		clone.Name = &value
	}
	if options.Description != nil {
		value := *options.Description
		clone.Description = &value
	}
	if options.SpanDataMaxChars != nil {
		value := *options.SpanDataMaxChars
		clone.SpanDataMaxChars = &value
	}
	if options.ThreadID != nil {
		value := *options.ThreadID
		clone.ThreadID = &value
	}
	if options.SandboxMode != nil {
		value := *options.SandboxMode
		clone.SandboxMode = &value
	}
	if options.WorkingDirectory != nil {
		value := *options.WorkingDirectory
		clone.WorkingDirectory = &value
	}
	if options.SkipGitRepoCheck != nil {
		value := *options.SkipGitRepoCheck
		clone.SkipGitRepoCheck = &value
	}
	if options.FailureErrorFunction != nil {
		value := *options.FailureErrorFunction
		clone.FailureErrorFunction = &value
	}
	if options.RunContextThreadIDKey != nil {
		value := *options.RunContextThreadIDKey
		clone.RunContextThreadIDKey = &value
	}
	return &clone
}

func resolveCodexToolName(configuredName *string) (string, error) {
	if configuredName == nil {
		return DefaultCodexToolName, nil
	}
	name := strings.TrimSpace(*configuredName)
	if name == "" {
		return "", agents.NewUserError("Codex tool name must be a non-empty string.")
	}
	if name != DefaultCodexToolName && !strings.HasPrefix(name, CodexToolNamePrefix) {
		return "", agents.UserErrorf(
			`Codex tool name must be %q or start with %q.`,
			DefaultCodexToolName,
			CodexToolNamePrefix,
		)
	}
	return name, nil
}

func buildCodexToolSchema(includeThreadID bool) map[string]any {
	properties := map[string]any{
		"inputs": map[string]any{
			"type":     "array",
			"minItems": 1,
			"items": map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []any{"type"},
				"properties": map[string]any{
					"type": map[string]any{
						"type": "string",
						"enum": []any{"text", "local_image"},
					},
					"text": map[string]any{
						"type": "string",
					},
					"path": map[string]any{
						"type": "string",
					},
				},
			},
		},
	}
	if includeThreadID {
		properties["thread_id"] = map[string]any{
			"type": "string",
		}
	}
	return map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"inputs"},
		"properties":           properties,
	}
}

func parseCodexToolInput(inputJSON string) (codexToolCallArguments, error) {
	payload := codexToolInputPayload{}
	if strings.TrimSpace(inputJSON) == "" {
		inputJSON = "{}"
	}
	decoder := json.NewDecoder(strings.NewReader(inputJSON))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&payload); err != nil {
		return codexToolCallArguments{}, agents.ModelBehaviorErrorf("Invalid JSON input for codex tool: %v", err)
	}

	if len(payload.Inputs) == 0 {
		return codexToolCallArguments{}, agents.NewUserError("Codex tool parameters must include an inputs field.")
	}

	normalizedInputs := make([]map[string]any, 0, len(payload.Inputs))
	for _, item := range payload.Inputs {
		normalized, err := normalizeCodexToolInputItem(item)
		if err != nil {
			return codexToolCallArguments{}, err
		}
		normalizedInputs = append(normalizedInputs, normalized)
	}

	normalizedThreadID, err := normalizeThreadID(payload.ThreadID)
	if err != nil {
		return codexToolCallArguments{}, err
	}

	return codexToolCallArguments{
		Inputs:   normalizedInputs,
		ThreadID: normalizedThreadID,
	}, nil
}

func parseCodexToolInputWithoutThreadID(inputJSON string) (codexToolCallArguments, error) {
	payload := codexToolRunContextInputPayload{}
	if strings.TrimSpace(inputJSON) == "" {
		inputJSON = "{}"
	}
	decoder := json.NewDecoder(strings.NewReader(inputJSON))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&payload); err != nil {
		return codexToolCallArguments{}, agents.ModelBehaviorErrorf("Invalid JSON input for codex tool: %v", err)
	}
	if len(payload.Inputs) == 0 {
		return codexToolCallArguments{}, agents.NewUserError("Codex tool parameters must include an inputs field.")
	}

	normalizedInputs := make([]map[string]any, 0, len(payload.Inputs))
	for _, item := range payload.Inputs {
		normalized, err := normalizeCodexToolInputItem(item)
		if err != nil {
			return codexToolCallArguments{}, err
		}
		normalizedInputs = append(normalizedInputs, normalized)
	}
	return codexToolCallArguments{
		Inputs:   normalizedInputs,
		ThreadID: nil,
	}, nil
}

func parseCodexToolInputWithCustomParameters(inputJSON string) (codexToolCallArguments, error) {
	if strings.TrimSpace(inputJSON) == "" {
		inputJSON = "{}"
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(inputJSON), &payload); err != nil {
		return codexToolCallArguments{}, agents.ModelBehaviorErrorf("Invalid JSON input for codex tool: %v", err)
	}

	inputsRaw, ok := payload["inputs"]
	if !ok || inputsRaw == nil {
		return codexToolCallArguments{}, agents.NewUserError("Codex tool parameters must include an inputs field.")
	}
	inputsList, ok := inputsRaw.([]any)
	if !ok || len(inputsList) == 0 {
		return codexToolCallArguments{}, agents.NewUserError("Codex tool parameters must include an inputs field.")
	}

	normalizedInputs := make([]map[string]any, 0, len(inputsList))
	for _, raw := range inputsList {
		inputMap, ok := toStringAnyMap(raw)
		if !ok {
			return codexToolCallArguments{}, agents.NewUserError("Codex tool inputs must be objects.")
		}
		item, err := coerceCodexToolInputItem(inputMap)
		if err != nil {
			return codexToolCallArguments{}, err
		}
		normalized, err := normalizeCodexToolInputItem(item)
		if err != nil {
			return codexToolCallArguments{}, err
		}
		normalizedInputs = append(normalizedInputs, normalized)
	}

	var threadID *string
	if rawThreadID, ok := payload["thread_id"]; ok {
		if rawThreadID == nil {
			threadID = nil
		} else {
			text, ok := rawThreadID.(string)
			if !ok {
				return codexToolCallArguments{}, agents.NewUserError(`When provided, "thread_id" must be a non-empty string.`)
			}
			normalizedThreadID, err := normalizeThreadID(&text)
			if err != nil {
				return codexToolCallArguments{}, err
			}
			threadID = normalizedThreadID
		}
	}

	return codexToolCallArguments{
		Inputs:   normalizedInputs,
		ThreadID: threadID,
	}, nil
}

func coerceCodexToolInputItem(mapping map[string]any) (CodexToolInputItem, error) {
	itemType, _ := mapping["type"].(string)

	var textValue *string
	if rawText, ok := mapping["text"]; ok && rawText != nil {
		text, ok := rawText.(string)
		if !ok {
			return CodexToolInputItem{}, agents.NewUserError(`"text" must be a string when provided.`)
		}
		textCopy := text
		textValue = &textCopy
	}

	var pathValue *string
	if rawPath, ok := mapping["path"]; ok && rawPath != nil {
		path, ok := rawPath.(string)
		if !ok {
			return CodexToolInputItem{}, agents.NewUserError(`"path" must be a string when provided.`)
		}
		pathCopy := path
		pathValue = &pathCopy
	}

	return CodexToolInputItem{
		Type: itemType,
		Text: textValue,
		Path: pathValue,
	}, nil
}

func normalizeCodexToolInputItem(item CodexToolInputItem) (map[string]any, error) {
	itemType := strings.TrimSpace(item.Type)
	textValue := ""
	if item.Text != nil {
		textValue = strings.TrimSpace(*item.Text)
	}
	pathValue := ""
	if item.Path != nil {
		pathValue = strings.TrimSpace(*item.Path)
	}

	switch itemType {
	case "text":
		if textValue == "" {
			return nil, agents.NewUserError(`Text inputs must include a non-empty "text" field.`)
		}
		if pathValue != "" {
			return nil, agents.NewUserError(`"path" is not allowed when type is "text".`)
		}
		return map[string]any{
			"type": "text",
			"text": textValue,
		}, nil
	case "local_image":
		if pathValue == "" {
			return nil, agents.NewUserError(`Local image inputs must include a non-empty "path" field.`)
		}
		if textValue != "" {
			return nil, agents.NewUserError(`"text" is not allowed when type is "local_image".`)
		}
		return map[string]any{
			"type": "local_image",
			"path": pathValue,
		}, nil
	default:
		return nil, agents.UserErrorf("unsupported input item type %q", itemType)
	}
}

func normalizeThreadID(value *string) (*string, error) {
	if value == nil {
		return nil, nil
	}
	normalized := strings.TrimSpace(*value)
	if normalized == "" {
		return nil, agents.NewUserError(`When provided, "thread_id" must be a non-empty string.`)
	}
	return &normalized, nil
}

func resolveCallThreadID(
	explicit *string,
	configured *string,
	runContextValue any,
	runContextThreadIDKey string,
	useRunContextThreadID bool,
) (*string, error) {
	if explicit != nil {
		value := strings.TrimSpace(*explicit)
		if value != "" {
			return &value, nil
		}
	}
	if useRunContextThreadID {
		contextThreadID, err := readThreadIDFromRunContext(runContextValue, runContextThreadIDKey)
		if err != nil {
			return nil, err
		}
		if contextThreadID != nil {
			return contextThreadID, nil
		}
	}
	if configured != nil {
		value := strings.TrimSpace(*configured)
		if value != "" {
			return &value, nil
		}
	}
	return nil, nil
}

func resolveCodexToolParametersSchema(parameters any) (map[string]any, error) {
	if parameters == nil {
		return nil, nil
	}
	mapping, ok := toStringAnyMap(parameters)
	if !ok {
		return nil, agents.NewUserError("parameters must be a JSON schema object.")
	}
	return agents.EnsureStrictJSONSchema(cloneStringAnyMap(mapping))
}

func validateRunContextThreadIDKey(value string) (string, error) {
	key := strings.TrimSpace(value)
	if key == "" {
		return "", agents.NewUserError("run_context_thread_id_key must be a non-empty string.")
	}
	return key, nil
}

func resolveRunContextThreadIDKey(
	toolName string,
	configuredKey *string,
	strictDefaultKey bool,
) (string, error) {
	if configuredKey != nil {
		return validateRunContextThreadIDKey(*configuredKey)
	}
	if toolName == DefaultCodexToolName {
		return DefaultRunContextThreadIDKey, nil
	}

	suffix := toolName[len(CodexToolNamePrefix):]
	if strictDefaultKey {
		validatedSuffix, err := validateDefaultRunContextThreadIDSuffix(suffix)
		if err != nil {
			return "", err
		}
		return DefaultRunContextThreadIDKey + "_" + validatedSuffix, nil
	}
	return DefaultRunContextThreadIDKey + "_" + normalizeNameForContextKey(suffix), nil
}

func normalizeNameForContextKey(value string) string {
	trimmed := strings.TrimSpace(strings.ToLower(value))
	if trimmed == "" {
		return "tool"
	}
	var b strings.Builder
	lastUnderscore := false
	for _, r := range trimmed {
		isAlphaNum := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if isAlphaNum || r == '_' {
			b.WriteRune(r)
			lastUnderscore = r == '_'
			continue
		}
		if !lastUnderscore {
			b.WriteByte('_')
			lastUnderscore = true
		}
	}
	normalized := strings.Trim(b.String(), "_")
	if normalized == "" {
		return "tool"
	}
	return normalized
}

func validateDefaultRunContextThreadIDSuffix(value string) (string, error) {
	suffix := strings.TrimSpace(value)
	if suffix == "" {
		return "", agents.NewUserError(
			`When use_run_context_thread_id=true and run_context_thread_id_key is omitted, ` +
				`codex tool names must include a non-empty suffix after "codex_".`,
		)
	}
	for _, r := range suffix {
		valid := (r >= 'A' && r <= 'Z') ||
			(r >= 'a' && r <= 'z') ||
			(r >= '0' && r <= '9') ||
			r == '_'
		if !valid {
			return "", agents.NewUserError(
				`When use_run_context_thread_id=true and run_context_thread_id_key is omitted, ` +
					`the codex tool name suffix (after "codex_") must match [A-Za-z0-9_]+. ` +
					`Use only letters, numbers, and underscores, or set run_context_thread_id_key explicitly.`,
			)
		}
	}
	return suffix, nil
}

func validateRunContextThreadIDContext(runContextValue any, key string) error {
	if runContextValue == nil {
		return agents.NewUserError(
			"use_run_context_thread_id=true requires a mutable run context object. " +
				"Pass a mutable map/object using agents.ContextWithRunContextValue(...).",
		)
	}
	root := reflect.ValueOf(runContextValue)
	if !root.IsValid() {
		return agents.NewUserError(
			"use_run_context_thread_id=true requires a mutable run context object. " +
				"Pass a mutable map/object using agents.ContextWithRunContextValue(...).",
		)
	}
	value := root
	for value.Kind() == reflect.Pointer || value.Kind() == reflect.Interface {
		if value.IsNil() {
			return agents.NewUserError(
				"use_run_context_thread_id=true requires a mutable run context object. " +
					"Pass a mutable map/object using agents.ContextWithRunContextValue(...).",
			)
		}
		value = value.Elem()
	}

	switch value.Kind() {
	case reflect.Map:
		if value.Type().Key().Kind() != reflect.String || value.IsNil() {
			return agents.NewUserError(
				"use_run_context_thread_id=true requires a non-nil mutable map run context with string keys.",
			)
		}
		return nil
	case reflect.Struct:
		if root.Kind() != reflect.Pointer {
			return agents.NewUserError(
				"use_run_context_thread_id=true requires a mutable run context object; " +
					"struct contexts must be passed by pointer.",
			)
		}
		field, ok := findStructFieldForContextKey(value, key)
		if !ok {
			return agents.NewUserError(
				`use_run_context_thread_id=true requires the run context to support field "` + key + `".`,
			)
		}
		if !field.CanSet() {
			return agents.NewUserError(
				`use_run_context_thread_id=true requires writable run context field "` + key + `".`,
			)
		}
		return nil
	default:
		return agents.NewUserError(
			"use_run_context_thread_id=true requires a mutable run context mapping " +
				"or a writable object context.",
		)
	}
}

func readThreadIDFromRunContext(runContextValue any, key string) (*string, error) {
	if runContextValue == nil {
		return nil, nil
	}
	value := reflect.ValueOf(runContextValue)
	for value.IsValid() && (value.Kind() == reflect.Pointer || value.Kind() == reflect.Interface) {
		if value.IsNil() {
			return nil, nil
		}
		value = value.Elem()
	}
	if !value.IsValid() {
		return nil, nil
	}
	switch value.Kind() {
	case reflect.Map:
		if value.Type().Key().Kind() != reflect.String {
			return nil, agents.NewUserError("run context must be a map with string keys.")
		}
		raw := value.MapIndex(reflect.ValueOf(key))
		if !raw.IsValid() {
			return nil, nil
		}
		text, err := readThreadIDFromContextValue(raw, key)
		if err != nil {
			return nil, err
		}
		return text, nil
	case reflect.Struct:
		field, ok := findStructFieldForContextKey(value, key)
		if !ok {
			return nil, nil
		}
		text, err := readThreadIDFromContextValue(field, key)
		if err != nil {
			return nil, err
		}
		return text, nil
	default:
		return nil, agents.NewUserError("run context must be a map with string keys or a struct pointer.")
	}
}

func storeThreadIDInRunContext(runContextValue any, key string, threadID *string) error {
	if threadID == nil {
		return nil
	}
	if runContextValue == nil {
		return agents.NewUserError("run context is nil and cannot store thread_id.")
	}
	value := reflect.ValueOf(runContextValue)
	if !value.IsValid() {
		return agents.NewUserError("run context is nil and cannot store thread_id.")
	}
	for value.Kind() == reflect.Pointer || value.Kind() == reflect.Interface {
		if value.IsNil() {
			return agents.NewUserError("run context is nil and cannot store thread_id.")
		}
		value = value.Elem()
	}

	switch value.Kind() {
	case reflect.Map:
		if value.Type().Key().Kind() != reflect.String {
			return agents.NewUserError("run context must be a mutable map with string keys.")
		}
		if value.IsNil() {
			return agents.NewUserError("run context map is nil and cannot store thread_id.")
		}
		mapValue, err := buildContextMapValueForThreadID(value.Type().Elem(), *threadID, key)
		if err != nil {
			return err
		}
		value.SetMapIndex(reflect.ValueOf(key).Convert(value.Type().Key()), mapValue)
		return nil
	case reflect.Struct:
		field, ok := findStructFieldForContextKey(value, key)
		if !ok {
			return agents.NewUserError(
				`use_run_context_thread_id=true requires the run context to support field "` + key + `".`,
			)
		}
		if !field.CanSet() {
			return agents.NewUserError(
				`use_run_context_thread_id=true requires writable run context field "` + key + `".`,
			)
		}
		return setContextFieldThreadID(field, *threadID, key)
	default:
		return agents.NewUserError("run context must be a mutable map with string keys or a writable struct pointer.")
	}
}

func tryStoreThreadIDInRunContextAfterError(
	runContextValue any,
	key string,
	threadID *string,
	enabled bool,
) {
	if !enabled || threadID == nil {
		return
	}
	_ = storeThreadIDInRunContext(runContextValue, key, threadID)
}

func readThreadIDFromContextValue(raw reflect.Value, key string) (*string, error) {
	for raw.Kind() == reflect.Interface || raw.Kind() == reflect.Pointer {
		if raw.IsNil() {
			return nil, nil
		}
		raw = raw.Elem()
	}
	if raw.Kind() != reflect.String {
		return nil, agents.UserErrorf(`Run context %q must be a string when provided.`, key)
	}
	text := strings.TrimSpace(raw.String())
	if text == "" {
		return nil, nil
	}
	return &text, nil
}

func buildContextMapValueForThreadID(elemType reflect.Type, threadID string, key string) (reflect.Value, error) {
	switch elemType.Kind() {
	case reflect.Interface:
		return reflect.ValueOf(threadID), nil
	case reflect.String:
		return reflect.ValueOf(threadID).Convert(elemType), nil
	case reflect.Pointer:
		if elemType.Elem().Kind() == reflect.String {
			v := reflect.New(elemType.Elem())
			v.Elem().SetString(threadID)
			return v, nil
		}
	}
	threadValue := reflect.ValueOf(threadID)
	if threadValue.Type().AssignableTo(elemType) {
		return threadValue, nil
	}
	if threadValue.Type().ConvertibleTo(elemType) {
		return threadValue.Convert(elemType), nil
	}
	return reflect.Value{}, agents.UserErrorf(
		`Unable to store Codex thread_id in run context field %q: incompatible map value type.`,
		key,
	)
}

func setContextFieldThreadID(field reflect.Value, threadID string, key string) error {
	switch field.Kind() {
	case reflect.String:
		field.SetString(threadID)
		return nil
	case reflect.Interface:
		field.Set(reflect.ValueOf(threadID))
		return nil
	case reflect.Pointer:
		if field.Type().Elem().Kind() != reflect.String {
			return agents.UserErrorf(
				`Unable to store Codex thread_id in run context field %q: incompatible field type.`,
				key,
			)
		}
		v := reflect.New(field.Type().Elem())
		v.Elem().SetString(threadID)
		field.Set(v)
		return nil
	default:
		return agents.UserErrorf(
			`Unable to store Codex thread_id in run context field %q: incompatible field type.`,
			key,
		)
	}
}

func findStructFieldForContextKey(structValue reflect.Value, key string) (reflect.Value, bool) {
	structType := structValue.Type()
	normalizedKey := normalizeContextIdentifier(key)
	for i := 0; i < structType.NumField(); i++ {
		fieldType := structType.Field(i)
		fieldValue := structValue.Field(i)

		tagName := strings.TrimSpace(strings.Split(fieldType.Tag.Get("json"), ",")[0])
		if tagName != "" && tagName != "-" {
			if tagName == key || normalizeContextIdentifier(tagName) == normalizedKey {
				return fieldValue, true
			}
		}

		if fieldType.Name == key ||
			strings.EqualFold(fieldType.Name, key) ||
			normalizeContextIdentifier(fieldType.Name) == normalizedKey {
			return fieldValue, true
		}
	}
	return reflect.Value{}, false
}

func normalizeContextIdentifier(value string) string {
	trimmed := strings.TrimSpace(strings.ToLower(value))
	if trimmed == "" {
		return ""
	}
	var b strings.Builder
	for _, r := range trimmed {
		isAlphaNum := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if isAlphaNum {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func resolveCodexOptions(raw any) (*CodexOptions, error) {
	options, err := CoerceCodexOptions(raw)
	if err != nil {
		return nil, err
	}

	if options == nil {
		options = &CodexOptions{}
	}
	if options.APIKey != nil && strings.TrimSpace(*options.APIKey) != "" {
		return options, nil
	}

	if envKey := resolveCodexAPIKeyFromEnvOverride(options.Env); envKey != "" {
		options.APIKey = &envKey
		return options, nil
	}
	if envKey := strings.TrimSpace(os.Getenv("CODEX_API_KEY")); envKey != "" {
		options.APIKey = &envKey
		return options, nil
	}
	if envKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")); envKey != "" {
		options.APIKey = &envKey
		return options, nil
	}
	if defaultKey := agents.GetDefaultOpenaiKey(); defaultKey.Valid() {
		value := defaultKey.Value
		options.APIKey = &value
	}
	return options, nil
}

func resolveCodexAPIKeyFromEnvOverride(env map[any]any) string {
	if env == nil {
		return ""
	}
	for key, value := range env {
		keyText := strings.TrimSpace(anyToString(key))
		valueText := strings.TrimSpace(anyToString(value))
		switch keyText {
		case "CODEX_API_KEY":
			if valueText != "" {
				return valueText
			}
		case "OPENAI_API_KEY":
			if valueText != "" {
				return valueText
			}
		}
	}
	return ""
}

func resolveThreadOptions(
	defaults any,
	sandboxMode *string,
	workingDirectory *string,
	skipGitRepoCheck *bool,
) (*ThreadOptions, error) {
	resolvedDefaults, err := CoerceThreadOptions(defaults)
	if err != nil {
		return nil, err
	}
	if resolvedDefaults == nil {
		resolvedDefaults = &ThreadOptions{}
	}

	resolved := cloneThreadOptions(resolvedDefaults)
	if sandboxMode != nil {
		value := strings.TrimSpace(*sandboxMode)
		resolved.SandboxMode = &value
	}
	if workingDirectory != nil {
		value := strings.TrimSpace(*workingDirectory)
		resolved.WorkingDirectory = &value
	}
	if skipGitRepoCheck != nil {
		value := *skipGitRepoCheck
		resolved.SkipGitRepoCheck = &value
	}
	if isThreadOptionsEmpty(*resolved) {
		return nil, nil
	}
	return resolved, nil
}

func isThreadOptionsEmpty(value ThreadOptions) bool {
	return value.Model == nil &&
		value.SandboxMode == nil &&
		value.WorkingDirectory == nil &&
		value.SkipGitRepoCheck == nil &&
		value.ModelReasoningEffort == nil &&
		value.NetworkAccessEnabled == nil &&
		value.WebSearchMode == nil &&
		value.WebSearchEnabled == nil &&
		value.ApprovalPolicy == nil &&
		len(value.AdditionalDirectories) == 0
}

func buildTurnOptions(defaults *TurnOptions, outputSchema map[string]any) TurnOptions {
	if defaults == nil {
		defaults = &TurnOptions{}
	}
	turn := *cloneTurnOptions(defaults)
	if outputSchema != nil {
		turn.OutputSchema = cloneStringAnyMap(outputSchema)
	}
	return turn
}

func resolveOutputSchema(option any) (map[string]any, error) {
	if option == nil {
		return nil, nil
	}

	mapping, ok := toStringAnyMap(option)
	if !ok {
		return nil, agents.NewUserError("Codex output schema must be a JSON schema or descriptor.")
	}

	if looksLikeDescriptor(mapping) {
		descriptor, err := validateDescriptor(mapping)
		if err != nil {
			return nil, err
		}
		return buildCodexOutputSchema(descriptor)
	}

	if schemaType, ok := mapping["type"].(string); ok && schemaType != "object" {
		return nil, agents.NewUserError(`Codex output schema must be a JSON object schema with type "object".`)
	}

	schemaCopy := cloneStringAnyMap(mapping)
	return agents.EnsureStrictJSONSchema(schemaCopy)
}

func looksLikeDescriptor(option map[string]any) bool {
	properties, ok := option["properties"].([]any)
	if !ok {
		return false
	}
	for _, each := range properties {
		propertyMap, ok := toStringAnyMap(each)
		if !ok {
			return false
		}
		if _, ok := propertyMap["name"].(string); !ok {
			return false
		}
	}
	return true
}

func validateDescriptor(option map[string]any) (map[string]any, error) {
	propertiesRaw, ok := option["properties"].([]any)
	if !ok || len(propertiesRaw) == 0 {
		return nil, agents.NewUserError("Codex output schema descriptor must include properties.")
	}

	seen := make(map[string]struct{}, len(propertiesRaw))
	for _, rawProperty := range propertiesRaw {
		property, ok := toStringAnyMap(rawProperty)
		if !ok {
			return nil, agents.NewUserError("Codex output schema properties must include non-empty names.")
		}

		name, _ := property["name"].(string)
		name = strings.TrimSpace(name)
		if name == "" {
			return nil, agents.NewUserError("Codex output schema properties must include non-empty names.")
		}
		if _, exists := seen[name]; exists {
			return nil, agents.UserErrorf(`Duplicate property name %q in output_schema.`, name)
		}
		seen[name] = struct{}{}

		if !isValidOutputSchemaField(property["schema"]) {
			return nil, agents.UserErrorf(`Invalid schema for output property %q.`, name)
		}
	}

	if requiredRaw, ok := option["required"]; ok {
		requiredList, ok := requiredRaw.([]any)
		if !ok {
			return nil, agents.NewUserError("output_schema.required must be a list of strings.")
		}
		for _, rawName := range requiredList {
			name, ok := rawName.(string)
			if !ok {
				return nil, agents.NewUserError("output_schema.required must be a list of strings.")
			}
			if _, exists := seen[name]; !exists {
				return nil, agents.UserErrorf(`Required property %q must also be defined in "properties".`, name)
			}
		}
	}

	return option, nil
}

func isValidOutputSchemaField(field any) bool {
	fieldMap, ok := toStringAnyMap(field)
	if !ok {
		return false
	}
	fieldType, _ := fieldMap["type"].(string)
	if _, ok := jsonPrimitiveTypes[fieldType]; ok {
		if enumRaw, ok := fieldMap["enum"]; ok {
			enumValues, ok := enumRaw.([]any)
			if !ok {
				return false
			}
			for _, each := range enumValues {
				if _, ok := each.(string); !ok {
					return false
				}
			}
		}
		return true
	}
	if fieldType == "array" {
		return isValidOutputSchemaField(fieldMap["items"])
	}
	return false
}

func buildCodexOutputSchema(descriptor map[string]any) (map[string]any, error) {
	propertiesRaw, _ := descriptor["properties"].([]any)
	properties := make(map[string]any, len(propertiesRaw))
	for _, rawProperty := range propertiesRaw {
		property, _ := toStringAnyMap(rawProperty)
		name, _ := property["name"].(string)

		schemaField, err := buildCodexOutputSchemaField(property["schema"])
		if err != nil {
			return nil, err
		}
		if description, ok := property["description"].(string); ok && strings.TrimSpace(description) != "" {
			schemaField["description"] = description
		}
		properties[name] = schemaField
	}

	required := make([]any, 0)
	if requiredRaw, ok := descriptor["required"].([]any); ok {
		required = append(required, requiredRaw...)
	}

	schema := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"properties":           properties,
		"required":             required,
	}
	if title, ok := descriptor["title"].(string); ok && strings.TrimSpace(title) != "" {
		schema["title"] = title
	}
	if description, ok := descriptor["description"].(string); ok && strings.TrimSpace(description) != "" {
		schema["description"] = description
	}
	return schema, nil
}

func buildCodexOutputSchemaField(field any) (map[string]any, error) {
	fieldMap, ok := toStringAnyMap(field)
	if !ok {
		return nil, agents.NewUserError("Invalid output schema field")
	}
	fieldType, _ := fieldMap["type"].(string)

	if fieldType == "array" {
		items, err := buildCodexOutputSchemaField(fieldMap["items"])
		if err != nil {
			return nil, err
		}
		schema := map[string]any{
			"type":  "array",
			"items": items,
		}
		if description, ok := fieldMap["description"].(string); ok && strings.TrimSpace(description) != "" {
			schema["description"] = description
		}
		return schema, nil
	}

	schema := map[string]any{"type": fieldType}
	if description, ok := fieldMap["description"].(string); ok && strings.TrimSpace(description) != "" {
		schema["description"] = description
	}
	if enumValues, ok := fieldMap["enum"].([]any); ok && len(enumValues) > 0 {
		schema["enum"] = enumValues
	}
	return schema, nil
}

func getThread(codexClient *Codex, threadID *string, defaults *ThreadOptions) (*Thread, error) {
	if threadID != nil {
		return codexClient.ResumeThread(*threadID, defaults)
	}
	return codexClient.StartThread(defaults)
}

func getOrCreatePersistedThread(
	codexClient *Codex,
	threadID *string,
	threadOptions *ThreadOptions,
	existingThread *Thread,
) (*Thread, error) {
	if existingThread != nil {
		if threadID != nil {
			existingID := existingThread.ID()
			if existingID != nil && *existingID != *threadID {
				return nil, agents.NewUserError(
					"Codex tool is configured with persist_session=true and already has an active thread.",
				)
			}
		}
		return existingThread, nil
	}
	return getThread(codexClient, threadID, threadOptions)
}

func consumeCodexToolEvents(
	ctx context.Context,
	streamed *StreamedTurn,
	args codexToolCallArguments,
	thread *Thread,
	onStream CodexToolStreamHandler,
	spanDataMaxChars *int,
) (string, *Usage, *string, error) {
	finalResponse := ""
	var usage *Usage
	resolvedThreadID := thread.ID()
	toolData := agents.ToolDataFromContext(ctx)
	activeSpans := make(map[string]tracing.Span)
	defer closeAllCodexSpans(ctx, activeSpans)

	events := streamed.Events
	errs := streamed.Errors
	for events != nil || errs != nil {
		select {
		case event, ok := <-events:
			if !ok {
				events = nil
				continue
			}

			dispatchCodexToolStreamEvent(ctx, onStream, CodexToolStreamEvent{
				Event:    event,
				Thread:   thread,
				ToolData: toolData,
			})

			switch typed := event.(type) {
			case ItemStartedEvent:
				handleCodexItemStarted(ctx, typed.Item, activeSpans, spanDataMaxChars)
			case ItemUpdatedEvent:
				handleCodexItemUpdated(typed.Item, activeSpans, spanDataMaxChars)
			case ItemCompletedEvent:
				handleCodexItemCompleted(ctx, typed.Item, activeSpans, spanDataMaxChars)
				if agentMessage, ok := typed.Item.(AgentMessageItem); ok {
					finalResponse = agentMessage.Text
				}
			case TurnCompletedEvent:
				usage = typed.Usage
				if typed.Usage != nil {
					addCodexUsageToContext(ctx, *typed.Usage)
				}
			case ThreadStartedEvent:
				id := typed.ThreadID
				resolvedThreadID = &id
			case TurnFailedEvent:
				message := strings.TrimSpace(typed.Error.Message)
				if message == "" {
					return "", usage, resolvedThreadID, agents.NewUserError("Codex turn failed")
				}
				return "", usage, resolvedThreadID, agents.UserErrorf("Codex turn failed: %s", message)
			case ThreadErrorEvent:
				return "", usage, resolvedThreadID, agents.UserErrorf("Codex stream error: %s", typed.Message)
			}
		case err, ok := <-errs:
			if !ok {
				errs = nil
				continue
			}
			if err != nil {
				return "", usage, resolvedThreadID, err
			}
		}
	}

	if strings.TrimSpace(finalResponse) == "" {
		finalResponse = buildDefaultCodexToolResponse(args)
	}
	return finalResponse, usage, resolvedThreadID, nil
}

func dispatchCodexToolStreamEvent(
	ctx context.Context,
	handler CodexToolStreamHandler,
	payload CodexToolStreamEvent,
) {
	if handler == nil {
		return
	}
	defer func() {
		_ = recover()
	}()
	_ = handler(ctx, payload)
}

func addCodexUsageToContext(ctx context.Context, usage Usage) {
	contextUsage, ok := usagepkg.FromContext(ctx)
	if !ok || contextUsage == nil {
		return
	}
	contextUsage.Add(&usagepkg.Usage{
		Requests:     1,
		InputTokens:  uint64(usage.InputTokens),
		OutputTokens: uint64(usage.OutputTokens),
		TotalTokens:  uint64(usage.InputTokens + usage.OutputTokens),
		InputTokensDetails: responses.ResponseUsageInputTokensDetails{
			CachedTokens: int64(usage.CachedInputTokens),
		},
	})
}

func closeAllCodexSpans(ctx context.Context, spans map[string]tracing.Span) {
	for _, span := range spans {
		_ = span.Finish(ctx, false)
	}
	clear(spans)
}

func handleCodexItemStarted(
	ctx context.Context,
	item ThreadItem,
	spans map[string]tracing.Span,
	spanDataMaxChars *int,
) {
	itemID, ok := codexThreadItemID(item)
	if !ok || itemID == "" {
		return
	}

	switch typed := item.(type) {
	case CommandExecutionItem:
		updates := map[string]any{
			"command":   typed.Command,
			"status":    typed.Status,
			"exit_code": commandExitCodeValue(typed.ExitCode),
		}
		if strings.TrimSpace(typed.AggregatedOutput) != "" {
			updates["output"] = truncateSpanValue(typed.AggregatedOutput, spanDataMaxChars)
		}
		span := tracing.NewCustomSpan(ctx, tracing.CustomSpanParams{
			Name: "Codex command execution",
			Data: mergeSpanData(nil, updates, spanDataMaxChars),
		})
		_ = span.Start(ctx, false)
		spans[itemID] = span
	case McpToolCallItem:
		span := tracing.NewCustomSpan(ctx, tracing.CustomSpanParams{
			Name: "Codex MCP tool call",
			Data: mergeSpanData(nil, map[string]any{
				"server":    typed.Server,
				"tool":      typed.Tool,
				"status":    typed.Status,
				"arguments": truncateSpanValue(maybeAsMapLike(typed.Arguments), spanDataMaxChars),
			}, spanDataMaxChars),
		})
		_ = span.Start(ctx, false)
		spans[itemID] = span
	case ReasoningItem:
		span := tracing.NewCustomSpan(ctx, tracing.CustomSpanParams{
			Name: "Codex reasoning",
			Data: mergeSpanData(nil, map[string]any{
				"text": truncateSpanValue(typed.Text, spanDataMaxChars),
			}, spanDataMaxChars),
		})
		_ = span.Start(ctx, false)
		spans[itemID] = span
	}
}

func handleCodexItemUpdated(item ThreadItem, spans map[string]tracing.Span, spanDataMaxChars *int) {
	itemID, ok := codexThreadItemID(item)
	if !ok || itemID == "" {
		return
	}
	span, ok := spans[itemID]
	if !ok || span == nil {
		return
	}
	switch typed := item.(type) {
	case CommandExecutionItem:
		updateCommandSpan(span, typed, spanDataMaxChars)
	case McpToolCallItem:
		updateMCPToolSpan(span, typed, spanDataMaxChars)
	case ReasoningItem:
		updateReasoningSpan(span, typed, spanDataMaxChars)
	}
}

func handleCodexItemCompleted(
	ctx context.Context,
	item ThreadItem,
	spans map[string]tracing.Span,
	spanDataMaxChars *int,
) {
	itemID, ok := codexThreadItemID(item)
	if !ok || itemID == "" {
		return
	}
	span, ok := spans[itemID]
	if !ok || span == nil {
		return
	}

	switch typed := item.(type) {
	case CommandExecutionItem:
		updateCommandSpan(span, typed, spanDataMaxChars)
		if typed.Status == "failed" {
			data := map[string]any{"exit_code": commandExitCodeValue(typed.ExitCode)}
			if strings.TrimSpace(typed.AggregatedOutput) != "" {
				data["output"] = truncateSpanValue(typed.AggregatedOutput, spanDataMaxChars)
			}
			span.SetError(tracing.SpanError{
				Message: "Codex command execution failed.",
				Data:    data,
			})
		}
	case McpToolCallItem:
		updateMCPToolSpan(span, typed, spanDataMaxChars)
		if typed.Status == "failed" && typed.Error != nil && strings.TrimSpace(typed.Error.Message) != "" {
			span.SetError(tracing.SpanError{
				Message: typed.Error.Message,
			})
		}
	case ReasoningItem:
		updateReasoningSpan(span, typed, spanDataMaxChars)
	}

	_ = span.Finish(ctx, false)
	delete(spans, itemID)
}

func updateCommandSpan(span tracing.Span, item CommandExecutionItem, spanDataMaxChars *int) {
	updates := map[string]any{
		"command":   item.Command,
		"status":    item.Status,
		"exit_code": commandExitCodeValue(item.ExitCode),
	}
	if strings.TrimSpace(item.AggregatedOutput) != "" {
		updates["output"] = truncateSpanValue(item.AggregatedOutput, spanDataMaxChars)
	}
	applySpanUpdates(span, updates, spanDataMaxChars)
}

func updateMCPToolSpan(span tracing.Span, item McpToolCallItem, spanDataMaxChars *int) {
	applySpanUpdates(span, map[string]any{
		"server":    item.Server,
		"tool":      item.Tool,
		"status":    item.Status,
		"arguments": truncateSpanValue(maybeAsMapLike(item.Arguments), spanDataMaxChars),
		"result":    truncateSpanValue(maybeAsMapLike(item.Result), spanDataMaxChars),
		"error":     truncateSpanValue(maybeAsMapLike(item.Error), spanDataMaxChars),
	}, spanDataMaxChars)
}

func updateReasoningSpan(span tracing.Span, item ReasoningItem, spanDataMaxChars *int) {
	applySpanUpdates(span, map[string]any{
		"text": truncateSpanValue(item.Text, spanDataMaxChars),
	}, spanDataMaxChars)
}

func commandExitCodeValue(exitCode *int) any {
	if exitCode == nil {
		return nil
	}
	return *exitCode
}

func applySpanUpdates(span tracing.Span, updates map[string]any, spanDataMaxChars *int) {
	customData, ok := span.SpanData().(*tracing.CustomSpanData)
	if !ok || customData == nil {
		return
	}
	current := customData.Data
	if current == nil {
		current = make(map[string]any)
		customData.Data = current
	}

	merged := mergeSpanData(current, updates, spanDataMaxChars)
	clear(current)
	for key, value := range merged {
		current[key] = value
	}
}

func codexThreadItemID(item ThreadItem) (string, bool) {
	switch typed := item.(type) {
	case CommandExecutionItem:
		return typed.ID, typed.ID != ""
	case FileChangeItem:
		return typed.ID, typed.ID != ""
	case McpToolCallItem:
		return typed.ID, typed.ID != ""
	case AgentMessageItem:
		return typed.ID, typed.ID != ""
	case ReasoningItem:
		return typed.ID, typed.ID != ""
	case WebSearchItem:
		return typed.ID, typed.ID != ""
	case ErrorItem:
		return typed.ID, typed.ID != ""
	case TodoListItem:
		return typed.ID, typed.ID != ""
	case UnknownThreadItem:
		if typed.ID == nil {
			return "", false
		}
		return *typed.ID, *typed.ID != ""
	default:
		return "", false
	}
}

func truncateSpanString(value string, maxChars *int) string {
	if maxChars == nil {
		return value
	}
	limit := *maxChars
	if limit <= 0 {
		return ""
	}
	if len(value) <= limit {
		return value
	}
	suffix := fmt.Sprintf("... [truncated, %d chars]", len(value))
	maxPrefix := limit - len(suffix)
	if maxPrefix <= 0 {
		if limit > len(value) {
			limit = len(value)
		}
		return value[:limit]
	}
	return value[:maxPrefix] + suffix
}

func stringifySpanValue(value any) string {
	if value == nil {
		return ""
	}
	if typed, ok := value.(string); ok {
		return typed
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(raw)
}

func jsonCharSize(value any) int {
	raw, err := json.Marshal(value)
	if err != nil {
		return len(fmt.Sprint(value))
	}
	return len(raw)
}

func truncateSpanValue(value any, maxChars *int) any {
	if maxChars == nil {
		return value
	}
	if value == nil {
		return nil
	}
	switch value.(type) {
	case bool, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
		return value
	case string:
		return truncateSpanString(value.(string), maxChars)
	}

	encoded := stringifySpanValue(value)
	if len(encoded) <= *maxChars {
		return value
	}
	return map[string]any{
		"preview":         truncateSpanString(encoded, maxChars),
		"truncated":       true,
		"original_length": len(encoded),
	}
}

func dropEmptyStringFields(data map[string]any) map[string]any {
	out := make(map[string]any, len(data))
	for key, value := range data {
		if text, ok := value.(string); ok && text == "" {
			continue
		}
		out[key] = value
	}
	return out
}

func mergeSpanData(current map[string]any, updates map[string]any, maxChars *int) map[string]any {
	merged := make(map[string]any, len(current)+len(updates))
	for key, value := range current {
		merged[key] = value
	}
	for key, value := range updates {
		merged[key] = value
	}
	return enforceSpanDataBudget(merged, maxChars)
}

func enforceSpanDataBudget(data map[string]any, maxChars *int) map[string]any {
	trimmed := dropEmptyStringFields(data)
	if maxChars == nil {
		return trimmed
	}
	limit := *maxChars
	if limit <= 0 {
		return map[string]any{}
	}
	if jsonCharSize(trimmed) <= limit {
		return trimmed
	}

	keptKeys := make([]string, 0, len(spanTrimKeys))
	for _, key := range spanTrimKeys {
		if _, ok := trimmed[key]; ok {
			keptKeys = append(keptKeys, key)
		}
	}
	if len(keptKeys) == 0 {
		return trimmed
	}

	base := cloneStringAnyMap(trimmed)
	for _, key := range keptKeys {
		base[key] = ""
	}
	baseSize := jsonCharSize(base)
	for baseSize > limit && len(keptKeys) > 0 {
		dropKey := keptKeys[len(keptKeys)-1]
		keptKeys = keptKeys[:len(keptKeys)-1]
		delete(base, dropKey)
		delete(trimmed, dropKey)
		baseSize = jsonCharSize(base)
	}
	if baseSize > limit {
		return dropEmptyStringFields(base)
	}

	values := make(map[string]string, len(keptKeys))
	for _, key := range keptKeys {
		value, ok := trimmed[key]
		if !ok || value == nil {
			trimmed[key] = ""
			continue
		}
		text := stringifySpanValue(value)
		if text == "" {
			trimmed[key] = ""
			continue
		}
		values[key] = text
	}

	filteredKeys := keptKeys[:0]
	for _, key := range keptKeys {
		if _, ok := values[key]; ok {
			filteredKeys = append(filteredKeys, key)
			continue
		}
		if _, ok := trimmed[key]; ok {
			filteredKeys = append(filteredKeys, key)
		}
	}
	keptKeys = filteredKeys
	if len(keptKeys) == 0 {
		return dropEmptyStringFields(base)
	}

	baseSize = jsonCharSize(base)
	available := limit - baseSize
	if available <= 0 {
		return dropEmptyStringFields(base)
	}

	const minBudget = 1
	budgets := make(map[string]int, len(values))
	for key := range values {
		budgets[key] = 0
	}

	remaining := 0
	if available >= len(values) {
		for key := range values {
			budgets[key] = minBudget
		}
		remaining = available - len(values)
	} else {
		orderedKeys := make([]string, 0, len(spanTrimKeys))
		for _, key := range spanTrimKeys {
			if _, ok := values[key]; ok {
				orderedKeys = append(orderedKeys, key)
			}
		}
		for i := 0; i < available && i < len(orderedKeys); i++ {
			budgets[orderedKeys[i]] = minBudget
		}
	}

	if current, ok := values["arguments"]; ok && remaining > 0 {
		needed := len(current) - budgets["arguments"]
		if needed > 0 {
			grant := needed
			if grant > remaining {
				grant = remaining
			}
			budgets["arguments"] += grant
			remaining -= grant
		}
	}

	if remaining > 0 {
		weights := make(map[string]int, len(values))
		weightTotal := 0
		for key, value := range values {
			weight := len(value) - budgets[key]
			if weight < 0 {
				weight = 0
			}
			weights[key] = weight
			weightTotal += weight
		}
		if weightTotal > 0 {
			for key, weight := range weights {
				if weight == 0 {
					continue
				}
				budgets[key] += int(float64(remaining) * (float64(weight) / float64(weightTotal)))
			}
		}
		allocated := 0
		for key, value := range values {
			if budgets[key] > len(value) {
				budgets[key] = len(value)
			}
			allocated += budgets[key]
		}
		leftover := available - allocated
		if leftover > 0 {
			ordered := make([]string, 0, len(values))
			for key := range values {
				ordered = append(ordered, key)
			}
			slices.SortStableFunc(ordered, func(a, b string) int {
				return cmp.Compare(weights[b], weights[a])
			})
			for idx := 0; leftover > 0; idx++ {
				expandable := make([]string, 0, len(ordered))
				for _, key := range ordered {
					if budgets[key] < len(values[key]) {
						expandable = append(expandable, key)
					}
				}
				if len(expandable) == 0 {
					break
				}
				key := expandable[idx%len(expandable)]
				budgets[key]++
				leftover--
			}
		}
	}

	for _, key := range keptKeys {
		value, ok := values[key]
		if !ok {
			trimmed[key] = ""
			continue
		}
		budget := budgets[key]
		trimmed[key] = truncateSpanString(value, &budget)
	}

	size := jsonCharSize(trimmed)
	for size > limit && len(keptKeys) > 0 {
		// Trim the currently largest value first.
		maxIdx := 0
		maxLen := len(fmt.Sprint(trimmed[keptKeys[0]]))
		for i := 1; i < len(keptKeys); i++ {
			currentLen := len(fmt.Sprint(trimmed[keptKeys[i]]))
			if currentLen > maxLen {
				maxLen = currentLen
				maxIdx = i
			}
		}
		key := keptKeys[maxIdx]
		current := fmt.Sprint(trimmed[key])
		if len(current) > 0 {
			source := values[key]
			nextLimit := len(current) - 1
			trimmed[key] = truncateSpanString(source, &nextLimit)
		} else {
			keptKeys = append(keptKeys[:maxIdx], keptKeys[maxIdx+1:]...)
		}
		size = jsonCharSize(trimmed)
	}

	if jsonCharSize(trimmed) <= limit {
		return dropEmptyStringFields(trimmed)
	}
	return dropEmptyStringFields(base)
}

func maybeAsMapLike(value any) any {
	switch typed := value.(type) {
	case nil:
		return nil
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, each := range typed {
			out[key] = maybeAsMapLike(each)
		}
		return out
	case []any:
		out := make([]any, 0, len(typed))
		for _, each := range typed {
			out = append(out, maybeAsMapLike(each))
		}
		return out
	case *McpToolCallResult:
		if typed == nil {
			return nil
		}
		return map[string]any{
			"content":            maybeAsMapLike(typed.Content),
			"structured_content": maybeAsMapLike(typed.StructuredContent),
		}
	case *McpToolCallError:
		if typed == nil {
			return nil
		}
		return map[string]any{"message": typed.Message}
	default:
		return value
	}
}

func ptrInt(value int) *int {
	return &value
}

func buildDefaultCodexToolResponse(args codexToolCallArguments) string {
	if len(args.Inputs) == 0 {
		return "Codex task completed with no inputs."
	}
	return "Codex task completed with inputs."
}
