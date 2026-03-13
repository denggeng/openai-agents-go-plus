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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"reflect"
	"slices"
	"strings"

	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/denggeng/openai-agents-go-plus/tracing"
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

// OpenAIResponsesModel is an implementation of Model that uses the OpenAI Responses API.
type OpenAIResponsesModel struct {
	Model  openai.ChatModel
	client OpenaiClient
}

func NewOpenAIResponsesModel(model openai.ChatModel, client OpenaiClient) OpenAIResponsesModel {
	return OpenAIResponsesModel{
		Model:  model,
		client: client,
	}
}

func (m OpenAIResponsesModel) GetResponse(
	ctx context.Context,
	params ModelResponseParams,
) (*ModelResponse, error) {
	var u *usage.Usage
	var response *responses.Response
	var rawResponse *http.Response

	err := tracing.ResponseSpan(
		ctx, tracing.ResponseSpanParams{Disabled: params.Tracing.IsDisabled()},
		func(ctx context.Context, spanResponse tracing.Span) (err error) {
			defer func() {
				if err != nil {
					var v string
					if params.Tracing.IncludeData() {
						v = err.Error()
					} else {
						v = fmt.Sprintf("%T", err)
					}
					spanResponse.SetError(tracing.SpanError{
						Message: "Error getting response",
						Data:    map[string]any{"error": v},
					})
				}
			}()

			body, opts, err := m.prepareRequest(
				ctx,
				params.SystemInstructions,
				params.Input,
				params.ModelSettings,
				params.Tools,
				params.OutputType,
				params.Handoffs,
				params.PreviousResponseID,
				params.ConversationID,
				false,
				params.Prompt,
			)
			if err != nil {
				return err
			}
			opts = append(opts, option.WithResponseInto(&rawResponse))

			response, err = m.client.Responses.New(ctx, *body, opts...)
			if err != nil {
				Logger().Error("error getting response", slog.String("error", err.Error()))
				return err
			}

			if DontLogModelData {
				Logger().Debug("LLM responded")
			} else {
				Logger().Debug("LLM responded", slog.String("output", SimplePrettyJSONMarshal(response.Output)))
			}

			u = usage.NewUsage()
			if !reflect.ValueOf(response.Usage).IsZero() {
				*u = usage.Usage{
					Requests:            1,
					InputTokens:         uint64(response.Usage.InputTokens),
					InputTokensDetails:  response.Usage.InputTokensDetails,
					OutputTokens:        uint64(response.Usage.OutputTokens),
					OutputTokensDetails: response.Usage.OutputTokensDetails,
					TotalTokens:         uint64(response.Usage.TotalTokens),
				}
			}

			if params.Tracing.IncludeData() {
				spanData := spanResponse.SpanData().(*tracing.ResponseSpanData)
				spanData.Response = response
				spanData.Input = params.Input
			}

			return nil
		},
	)
	if err != nil {
		return nil, err
	}

	return &ModelResponse{
		Output:     response.Output,
		Usage:      u,
		ResponseID: response.ID,
		RequestID:  requestIDFromHeaders(rawResponse),
	}, nil
}

// StreamResponse yields a partial message as it is generated, as well as the usage information.
func (m OpenAIResponsesModel) StreamResponse(
	ctx context.Context,
	params ModelResponseParams,
	yield ModelStreamResponseCallback,
) error {
	return tracing.ResponseSpan(
		ctx, tracing.ResponseSpanParams{Disabled: params.Tracing.IsDisabled()},
		func(ctx context.Context, spanResponse tracing.Span) (err error) {
			defer func() {
				if err != nil {
					var v string
					if params.Tracing.IncludeData() {
						v = err.Error()
					} else {
						v = fmt.Sprintf("%T", err)
					}
					spanResponse.SetError(tracing.SpanError{
						Message: "Error streaming response",
						Data:    map[string]any{"error": v},
					})
					Logger().Error("error streaming response", slog.String("error", err.Error()))
				}
			}()

			body, opts, err := m.prepareRequest(
				ctx,
				params.SystemInstructions,
				params.Input,
				params.ModelSettings,
				params.Tools,
				params.OutputType,
				params.Handoffs,
				params.PreviousResponseID,
				params.ConversationID,
				true,
				params.Prompt,
			)
			if err != nil {
				return err
			}
			var rawResponse *http.Response
			opts = append(opts, option.WithResponseInto(&rawResponse))

			stream := m.client.Responses.NewStreaming(ctx, *body, opts...)
			defer func() {
				if e := stream.Close(); e != nil {
					err = errors.Join(err, fmt.Errorf("error closing stream: %w", e))
				}
			}()

			streamCtx := contextWithModelRequestID(ctx, requestIDFromHeaders(rawResponse))
			var finalResponse *responses.Response
			for stream.Next() {
				chunk := stream.Current()
				if chunk.Type == "response.completed" {
					finalResponse = &chunk.Response
				}
				if err = yield(streamCtx, chunk); err != nil {
					return err
				}
			}

			if err = stream.Err(); err != nil {
				return fmt.Errorf("error streaming response: %w", err)
			}

			if finalResponse != nil && params.Tracing.IncludeData() {
				spanData := spanResponse.SpanData().(*tracing.ResponseSpanData)
				spanData.Response = finalResponse
				spanData.Input = params.Input
			}
			return nil
		})
}

func (m OpenAIResponsesModel) prepareRequest(
	ctx context.Context,
	systemInstructions param.Opt[string],
	input Input,
	modelSettings modelsettings.ModelSettings,
	tools []Tool,
	outputType OutputTypeInterface,
	handoffs []Handoff,
	previousResponseID string,
	conversationID string,
	stream bool,
	prompt responses.ResponsePromptParam,
) (*responses.ResponseNewParams, []option.RequestOption, error) {
	listInput := ItemHelpers().InputToNewInputList(input)

	var parallelToolCalls param.Opt[bool]
	if modelSettings.ParallelToolCalls.Valid() {
		if modelSettings.ParallelToolCalls.Value && len(tools) > 0 {
			parallelToolCalls = param.NewOpt(true)
		} else if !modelSettings.ParallelToolCalls.Value {
			parallelToolCalls = param.NewOpt(false)
		}
	}

	toolChoice, err := ResponsesConverter().ConvertToolChoiceForRequest(
		modelSettings.ToolChoice,
		tools,
		handoffs,
		m.Model,
	)
	if err != nil {
		return nil, nil, err
	}
	convertedTools, err := ResponsesConverter().convertTools(
		ctx,
		tools,
		handoffs,
		responsesConvertToolsOptions{
			allowOpaqueToolSearchSurface: !reflect.ValueOf(prompt).IsZero(),
			model:                        m.Model,
			toolChoice:                   modelSettings.ToolChoice,
		},
	)
	if err != nil {
		return nil, nil, err
	}
	responseFormat, err := ResponsesConverter().GetResponseFormat(outputType)
	if err != nil {
		return nil, nil, err
	}

	include := slices.Concat(convertedTools.Includes, modelSettings.ResponseInclude)
	if modelSettings.TopLogprobs.Valid() {
		include = append(include, "message.output_text.logprobs")
	}

	// Remove duplicates
	slices.Sort(include)
	include = slices.Compact(include)

	if modelSettings.Verbosity.Valid() {
		responseFormat.Verbosity = responses.ResponseTextConfigVerbosity(modelSettings.Verbosity.Value)
	}

	if DontLogModelData {
		Logger().Debug("Calling LLM")
	} else {
		Logger().Debug(
			"Calling LLM",
			slog.String("Input", SimplePrettyJSONMarshal(listInput)),
			slog.String("Tools", SimplePrettyJSONMarshal(convertedTools.Tools)),
			slog.Bool("Stream", stream),
			slog.String("Tool choice", SimplePrettyJSONMarshal(toolChoice)),
			slog.String("Response format", SimplePrettyJSONMarshal(responseFormat)),
			slog.String("Previous response ID", previousResponseID),
			slog.String("Conversation ID", conversationID),
		)
	}

	var prevRespIDParam param.Opt[string]
	if previousResponseID != "" {
		prevRespIDParam = param.NewOpt(previousResponseID)
	}
	var conversationParam responses.ResponseNewParamsConversationUnion
	if conversationID != "" {
		conversationParam = responses.ResponseNewParamsConversationUnion{
			OfString: param.NewOpt(conversationID),
		}
	}

	params := &responses.ResponseNewParams{
		PreviousResponseID: prevRespIDParam,
		Conversation:       conversationParam,
		Instructions:       systemInstructions,
		Model:              m.Model,
		Input:              responses.ResponseNewParamsInputUnion{OfInputItemList: listInput},
		Include:            include,
		Tools:              convertedTools.Tools,
		Prompt:             prompt,
		Temperature:        modelSettings.Temperature,
		TopP:               modelSettings.TopP,
		Truncation:         responses.ResponseNewParamsTruncation(modelSettings.Truncation.Or("")),
		MaxOutputTokens:    modelSettings.MaxTokens,
		ToolChoice:         toolChoice,
		ParallelToolCalls:  parallelToolCalls,
		Text:               responseFormat,
		Store:              modelSettings.Store,
		Reasoning:          modelSettings.Reasoning,
		TopLogprobs:        modelSettings.TopLogprobs,
		Metadata:           modelSettings.Metadata,
	}

	headers := map[string]string{
		"User-Agent": DefaultUserAgent(),
	}
	for k, v := range modelSettings.ExtraHeaders {
		headers[k] = v
	}
	if override := ResponsesHeadersOverride.Get(); len(override) > 0 {
		for k, v := range override {
			headers[k] = v
		}
	}

	var opts []option.RequestOption
	for k, v := range headers {
		opts = append(opts, option.WithHeader(k, v))
	}
	for k, v := range modelSettings.ExtraQuery {
		opts = append(opts, option.WithQuery(k, v))
	}
	for k, v := range mergedModelExtraJSON(modelSettings) {
		opts = append(opts, option.WithJSONSet(k, v))
	}

	if modelSettings.CustomizeResponsesRequest != nil {
		return modelSettings.CustomizeResponsesRequest(ctx, params, opts)
	}

	return params, opts, nil
}

func (m OpenAIResponsesModel) removeOpenAIResponsesAPIIncompatibleFields(listInput []map[string]any) []map[string]any {
	hasProviderData := false
	for _, item := range listInput {
		if hasTruthyProviderData(item) {
			hasProviderData = true
			break
		}
	}
	if !hasProviderData {
		return listInput
	}

	result := make([]map[string]any, 0, len(listInput))
	for _, item := range listInput {
		cleaned := m.cleanItemForOpenAI(item)
		if cleaned != nil {
			result = append(result, cleaned)
		}
	}
	return result
}

func (m OpenAIResponsesModel) cleanItemForOpenAI(item map[string]any) map[string]any {
	if item == nil {
		return item
	}
	if itemType, _ := item["type"].(string); itemType == "reasoning" && hasTruthyProviderData(item) {
		return nil
	}
	if id, ok := item["id"].(string); ok && id == FakeResponsesID {
		delete(item, "id")
	}
	if _, ok := item["provider_data"]; ok {
		delete(item, "provider_data")
	}
	return item
}

func hasTruthyProviderData(item map[string]any) bool {
	if item == nil {
		return false
	}
	providerData, ok := item["provider_data"]
	if !ok {
		return false
	}
	switch v := providerData.(type) {
	case nil:
		return false
	case map[string]any:
		return len(v) > 0
	case []any:
		return len(v) > 0
	case string:
		return v != ""
	case bool:
		return v
	default:
		return true
	}
}

type ConvertedTools struct {
	Tools    []responses.ToolUnionParam
	Includes []responses.ResponseIncludable
}

type responsesConvertToolsOptions struct {
	allowOpaqueToolSearchSurface bool
	model                        openai.ChatModel
	toolChoice                   modelsettings.ToolChoice
}

type responsesConverter struct{}

func ResponsesConverter() responsesConverter { return responsesConverter{} }

func (responsesConverter) ConvertToolChoice(toolChoice modelsettings.ToolChoice) responses.ResponseNewParamsToolChoiceUnion {
	out, _ := responsesConverter{}.convertToolChoice(toolChoice, nil, nil, "")
	return out
}

func (responsesConverter) ConvertToolChoiceForRequest(
	toolChoice modelsettings.ToolChoice,
	tools []Tool,
	handoffs []Handoff,
	model openai.ChatModel,
) (responses.ResponseNewParamsToolChoiceUnion, error) {
	return responsesConverter{}.convertToolChoice(toolChoice, tools, handoffs, model)
}

func (responsesConverter) convertToolChoice(
	toolChoice modelsettings.ToolChoice,
	tools []Tool,
	handoffs []Handoff,
	model openai.ChatModel,
) (responses.ResponseNewParamsToolChoiceUnion, error) {
	switch toolChoice := toolChoice.(type) {
	case nil:
		return responses.ResponseNewParamsToolChoiceUnion{}, nil
	case modelsettings.ToolChoiceString:
		if err := validateResponsesNamedToolChoice(toolChoice.String(), tools, handoffs); err != nil {
			return responses.ResponseNewParamsToolChoiceUnion{}, err
		}
		switch toolChoice {
		case "none", "auto", "required":
			if toolChoice == modelsettings.ToolChoiceRequired {
				if err := validateResponsesRequiredToolChoice(tools); err != nil {
					return responses.ResponseNewParamsToolChoiceUnion{}, err
				}
			}
			return responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptions(toolChoice)),
			}, nil
		case "file_search", "web_search_preview", "web_search_preview_2025_03_11",
			"image_generation", "code_interpreter":
			return responses.ResponseNewParamsToolChoiceUnion{
				OfHostedTool: &responses.ToolChoiceTypesParam{
					Type: responses.ToolChoiceTypesType(toolChoice),
				},
			}, nil
		case "computer", "computer_use", "computer_use_preview":
			if hasComputerTool(tools) {
				return convertBuiltinComputerToolChoice(model, toolChoice), nil
			}
			return responses.ResponseNewParamsToolChoiceUnion{
				OfFunctionTool: &responses.ToolChoiceFunctionParam{
					Name: toolChoice.String(),
					Type: constant.ValueOf[constant.Function](),
				},
			}, nil
		case "mcp":
			// Note that this is still here for backwards compatibility,
			// but migrating to ToolChoiceMCP is recommended.
			return responses.ResponseNewParamsToolChoiceUnion{
				OfMcpTool: &responses.ToolChoiceMcpParam{
					Type: constant.ValueOf[constant.Mcp](),
				},
			}, nil
		default:
			return responses.ResponseNewParamsToolChoiceUnion{
				OfFunctionTool: &responses.ToolChoiceFunctionParam{
					Name: toolChoice.String(),
					Type: constant.ValueOf[constant.Function](),
				},
			}, nil
		}
	case modelsettings.ToolChoiceMCP:
		return responses.ResponseNewParamsToolChoiceUnion{
			OfMcpTool: &responses.ToolChoiceMcpParam{
				ServerLabel: toolChoice.ServerLabel,
				Name:        param.NewOpt(toolChoice.Name),
				Type:        constant.ValueOf[constant.Mcp](),
			},
		}, nil
	default:
		// This would be an unrecoverable implementation bug, so a panic is appropriate.
		panic(fmt.Errorf("unexpected ToolChoice type %T", toolChoice))
	}
}

func (responsesConverter) GetResponseFormat(
	outputType OutputTypeInterface,
) (responses.ResponseTextConfigParam, error) {
	if outputType == nil || outputType.IsPlainText() {
		return responses.ResponseTextConfigParam{}, nil
	}
	schema, err := outputType.JSONSchema()
	if err != nil {
		return responses.ResponseTextConfigParam{}, err
	}
	return responses.ResponseTextConfigParam{
		Format: responses.ResponseFormatTextConfigUnionParam{
			OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "final_output",
				Schema: schema,
				Strict: param.NewOpt(outputType.IsStrictJSONSchema()),
				Type:   constant.ValueOf[constant.JSONSchema](),
			},
		},
	}, nil
}

func (conv responsesConverter) ConvertTools(ctx context.Context, ts []Tool, handoffs []Handoff) (*ConvertedTools, error) {
	return conv.convertTools(ctx, ts, handoffs, responsesConvertToolsOptions{})
}

func (conv responsesConverter) convertTools(
	ctx context.Context,
	ts []Tool,
	handoffs []Handoff,
	opts responsesConvertToolsOptions,
) (*ConvertedTools, error) {
	var convertedTools []responses.ToolUnionParam
	var includes []responses.ResponseIncludable

	if err := validateResponsesToolSearchConfiguration(ts, opts.allowOpaqueToolSearchSurface); err != nil {
		return nil, err
	}

	var computerTools []ComputerTool
	for _, tool := range ts {
		switch ct := tool.(type) {
		case ComputerTool:
			computerTools = append(computerTools, ct)
		case *ComputerTool:
			if ct != nil {
				computerTools = append(computerTools, *ct)
			}
		}
	}
	if len(computerTools) > 1 {
		return nil, UserErrorf("you can only provide one computer tool, got %d", len(computerTools))
	}
	usePreviewComputerTool := shouldUsePreviewComputerTool(opts.model, opts.toolChoice)

	namespaceIndexByName := make(map[string]int)
	namespaceToolsByName := make(map[string][]map[string]any)
	namespaceDescriptions := make(map[string]string)

	for _, tool := range ts {
		functionTool, ok := asFunctionTool(tool)
		if ok && functionTool.Namespace != "" {
			if _, exists := namespaceIndexByName[functionTool.Namespace]; exists {
				expectedDescription := namespaceDescriptions[functionTool.Namespace]
				if expectedDescription != functionTool.NamespaceDescription {
					return nil, UserErrorf(
						"all tools in namespace %q must share the same description",
						functionTool.Namespace,
					)
				}
				namespaceToolsByName[functionTool.Namespace] = append(
					namespaceToolsByName[functionTool.Namespace],
					functionToolResponsesPayload(functionTool, true),
				)
				continue
			}

			namespaceIndexByName[functionTool.Namespace] = len(convertedTools)
			namespaceDescriptions[functionTool.Namespace] = functionTool.NamespaceDescription
			namespaceToolsByName[functionTool.Namespace] = []map[string]any{
				functionToolResponsesPayload(functionTool, true),
			}
			convertedTools = append(convertedTools, responses.ToolUnionParam{})
			continue
		}

		convertedTool, include, err := conv.convertTool(ctx, tool, usePreviewComputerTool)
		if err != nil {
			return nil, err
		}
		convertedTools = append(convertedTools, *convertedTool)
		if include != nil {
			includes = append(includes, *include)
		}
	}

	for namespaceName, index := range namespaceIndexByName {
		namespaceTool, err := responsesNamespaceToolParam(
			namespaceName,
			namespaceDescriptions[namespaceName],
			namespaceToolsByName[namespaceName],
		)
		if err != nil {
			return nil, err
		}
		convertedTools[index] = namespaceTool
	}

	for _, handoff := range handoffs {
		convertedTools = append(convertedTools, conv.convertHandoffTool(handoff))
	}

	return &ConvertedTools{
		Tools:    convertedTools,
		Includes: includes,
	}, nil
}

// convertTool returns converted tool and includes.
func (conv responsesConverter) convertTool(
	ctx context.Context,
	tool Tool,
	usePreviewComputerTool bool,
) (*responses.ToolUnionParam, *responses.ResponseIncludable, error) {
	var convertedTool *responses.ToolUnionParam
	var includes *responses.ResponseIncludable

	switch t := tool.(type) {
	case FunctionTool:
		convertedTool = &responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        t.Name,
				Parameters:  materializeJSONMap(t.ParamsJSONSchema),
				Strict:      param.NewOpt(t.StrictJSONSchema.Or(true)),
				Description: param.NewOpt(t.Description),
				Type:        constant.ValueOf[constant.Function](),
			},
		}
		if t.DeferLoading {
			raw, err := rawToolUnionParam(functionToolResponsesPayload(t, true))
			if err != nil {
				return nil, nil, err
			}
			convertedTool = &raw
		}
		includes = nil
	case *FunctionTool:
		if t == nil {
			return nil, nil, NewUserError("function tool is nil")
		}
		return conv.convertTool(ctx, *t, usePreviewComputerTool)
	case WebSearchTool:
		convertedTool = &responses.ToolUnionParam{
			OfWebSearch: &responses.WebSearchToolParam{
				Type:              responses.WebSearchToolTypeWebSearch,
				Filters:           t.Filters,
				UserLocation:      t.UserLocation,
				SearchContextSize: t.SearchContextSize,
			},
		}
		includes = nil
	case FileSearchTool:
		convertedTool = &responses.ToolUnionParam{
			OfFileSearch: &responses.FileSearchToolParam{
				VectorStoreIDs: t.VectorStoreIDs,
				MaxNumResults:  t.MaxNumResults,
				Filters:        t.Filters,
				RankingOptions: t.RankingOptions,
				Type:           constant.ValueOf[constant.FileSearch](),
			},
		}
		if t.IncludeSearchResults {
			includes = new(responses.ResponseIncludable)
			*includes = responses.ResponseIncludableFileSearchCallResults
		}
	case ComputerTool:
		if t.Computer == nil {
			return nil, nil, NewUserError(
				"computer tool has no resolved computer. Call ResolveComputer/InitializeComputerTools before model conversion",
			)
		}
		environment, err := t.Computer.Environment(ctx)
		if err != nil {
			return nil, nil, err
		}

		dimensions, err := t.Computer.Dimensions(ctx)
		if err != nil {
			return nil, nil, err
		}

		if usePreviewComputerTool {
			convertedTool = &responses.ToolUnionParam{
				OfComputerUsePreview: &responses.ComputerToolParam{
					DisplayHeight: dimensions.Height,
					DisplayWidth:  dimensions.Width,
					Environment:   responses.ComputerToolEnvironment(environment),
					Type:          constant.ValueOf[constant.ComputerUsePreview](),
				},
			}
		} else {
			raw, err := rawToolUnionParam(map[string]any{"type": "computer"})
			if err != nil {
				return nil, nil, err
			}
			convertedTool = &raw
		}
		includes = nil
	case *ComputerTool:
		if t == nil {
			return nil, nil, NewUserError("computer tool is nil")
		}
		return conv.convertTool(ctx, *t, usePreviewComputerTool)
	case HostedMCPTool:
		if t.DeferLoading || hostedMCPToolConfigDeferLoading(t.ToolConfig) {
			payload, err := hostedMCPToolPayload(t)
			if err != nil {
				return nil, nil, err
			}
			raw, err := rawToolUnionParam(payload)
			if err != nil {
				return nil, nil, err
			}
			convertedTool = &raw
		} else {
			convertedTool = &responses.ToolUnionParam{
				OfMcp: &t.ToolConfig,
			}
		}
		includes = nil
	case ImageGenerationTool:
		convertedTool = &responses.ToolUnionParam{
			OfImageGeneration: &t.ToolConfig,
		}
		includes = nil
	case CodeInterpreterTool:
		convertedTool = &responses.ToolUnionParam{
			OfCodeInterpreter: &t.ToolConfig,
		}
		includes = nil
	case LocalShellTool:
		convertedTool = &responses.ToolUnionParam{
			OfLocalShell: &responses.ToolLocalShellParam{
				Type: constant.ValueOf[constant.LocalShell](),
			},
		}
		includes = nil
	case ShellTool:
		environment := t.Environment
		if normalized, err := normalizeShellToolEnvironment(environment); err == nil {
			environment = normalized
		}
		var envParam responses.FunctionShellToolEnvironmentUnionParam
		if environment != nil {
			raw, err := json.Marshal(environment)
			if err != nil {
				return nil, nil, err
			}
			envParam = param.Override[responses.FunctionShellToolEnvironmentUnionParam](json.RawMessage(raw))
		}
		convertedTool = &responses.ToolUnionParam{
			OfShell: &responses.FunctionShellToolParam{
				Type:        constant.ValueOf[constant.Shell](),
				Environment: envParam,
			},
		}
		includes = nil
	case ApplyPatchTool:
		convertedTool = &responses.ToolUnionParam{
			OfApplyPatch: &responses.ApplyPatchToolParam{
				Type: constant.ValueOf[constant.ApplyPatch](),
			},
		}
		includes = nil
	case ToolSearchTool:
		payload := map[string]any{"type": "tool_search"}
		if strings.TrimSpace(t.Description) != "" {
			payload["description"] = t.Description
		}
		if t.Execution != "" {
			payload["execution"] = string(t.Execution)
		}
		if t.Parameters != nil {
			payload["parameters"] = normalizeJSONValue(t.Parameters)
		}
		raw, err := rawToolUnionParam(payload)
		if err != nil {
			return nil, nil, err
		}
		convertedTool = &raw
		includes = nil
	case *ToolSearchTool:
		if t == nil {
			return nil, nil, NewUserError("tool search tool is nil")
		}
		return conv.convertTool(ctx, *t, usePreviewComputerTool)
	default:
		return nil, nil, UserErrorf("Unknown tool type: %T", tool)
	}

	return convertedTool, includes, nil
}

func (responsesConverter) convertHandoffTool(handoff Handoff) responses.ToolUnionParam {
	return responses.ToolUnionParam{
		OfFunction: &responses.FunctionToolParam{
			Name:        handoff.ToolName,
			Parameters:  materializeJSONMap(handoff.InputJSONSchema),
			Strict:      param.NewOpt(handoff.StrictJSONSchema.Or(true)),
			Description: param.NewOpt(handoff.ToolDescription),
			Type:        constant.ValueOf[constant.Function](),
		},
	}
}

func responsesNamespaceToolParam(
	name string,
	description string,
	tools []map[string]any,
) (responses.ToolUnionParam, error) {
	payload := map[string]any{
		"type":        "namespace",
		"name":        name,
		"description": description,
		"tools":       tools,
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return responses.ToolUnionParam{}, fmt.Errorf("marshal namespace tool payload: %w", err)
	}
	return param.Override[responses.ToolUnionParam](json.RawMessage(data)), nil
}

func functionToolResponsesPayload(tool FunctionTool, includeDeferLoading bool) map[string]any {
	payload := map[string]any{
		"type":        "function",
		"name":        tool.Name,
		"description": tool.Description,
		"parameters":  materializeJSONMap(tool.ParamsJSONSchema),
		"strict":      tool.StrictJSONSchema.Or(true),
	}
	if includeDeferLoading && tool.DeferLoading {
		payload["defer_loading"] = true
	}
	return payload
}

func rawToolUnionParam(payload map[string]any) (responses.ToolUnionParam, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return responses.ToolUnionParam{}, err
	}
	return param.Override[responses.ToolUnionParam](json.RawMessage(data)), nil
}

func hostedMCPToolPayload(tool HostedMCPTool) (map[string]any, error) {
	data, err := json.Marshal(tool.ToolConfig)
	if err != nil {
		return nil, err
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, err
	}
	if payload == nil {
		payload = map[string]any{}
	}
	if payload["type"] == nil {
		payload["type"] = "mcp"
	}
	if tool.DeferLoading {
		payload["defer_loading"] = true
	}
	return payload, nil
}

func hasComputerTool(tools []Tool) bool {
	for _, tool := range tools {
		switch typed := tool.(type) {
		case ComputerTool:
			return true
		case *ComputerTool:
			if typed != nil {
				return true
			}
		}
	}
	return false
}

func isPreviewComputerModel(model openai.ChatModel) bool {
	return strings.HasPrefix(string(model), "computer-use-preview")
}

func shouldUsePreviewComputerTool(
	model openai.ChatModel,
	toolChoice modelsettings.ToolChoice,
) bool {
	if isPreviewComputerModel(model) {
		return true
	}
	if model != "" {
		return false
	}
	choiceString, ok := toolChoice.(modelsettings.ToolChoiceString)
	if ok && (choiceString == "computer" || choiceString == "computer_use") {
		return false
	}
	return true
}

func convertBuiltinComputerToolChoice(
	model openai.ChatModel,
	toolChoice modelsettings.ToolChoiceString,
) responses.ResponseNewParamsToolChoiceUnion {
	if isPreviewComputerModel(model) || shouldUsePreviewComputerTool(model, toolChoice) {
		return responses.ResponseNewParamsToolChoiceUnion{
			OfHostedTool: &responses.ToolChoiceTypesParam{
				Type: responses.ToolChoiceTypesTypeComputerUsePreview,
			},
		}
	}
	raw := json.RawMessage(`{"type":"computer"}`)
	return param.Override[responses.ResponseNewParamsToolChoiceUnion](raw)
}

func validateResponsesRequiredToolChoice(tools []Tool) error {
	if len(tools) == 0 {
		return nil
	}
	for _, tool := range tools {
		switch tool.(type) {
		case ToolSearchTool, *ToolSearchTool:
			return nil
		}
	}
	if hasRequiredToolSearchSurface(tools) {
		return NewUserError(
			"tool_choice='required' is not currently supported when deferred-loading Responses tools are configured without ToolSearchTool() on the OpenAI Responses API. Add ToolSearchTool() or use `auto`.",
		)
	}
	return nil
}

func validateResponsesNamedToolChoice(toolChoice string, tools []Tool, handoffs []Handoff) error {
	if toolChoice == "" || len(tools) == 0 && len(handoffs) == 0 {
		return nil
	}

	topLevelFunctionNames := make(map[string]struct{})
	allLocalFunctionNames := make(map[string]struct{})
	deferredOnlyFunctionNames := make(map[string]struct{})
	namespacedFunctionNames := make(map[string]struct{})
	namespaceNames := make(map[string]struct{})
	hasHostedToolSearch := false

	for _, handoff := range handoffs {
		topLevelFunctionNames[handoff.ToolName] = struct{}{}
		allLocalFunctionNames[handoff.ToolName] = struct{}{}
	}

	for _, tool := range tools {
		switch tool.(type) {
		case ToolSearchTool, *ToolSearchTool:
			hasHostedToolSearch = true
		}
		functionTool, ok := asFunctionTool(tool)
		if !ok {
			continue
		}
		allLocalFunctionNames[functionTool.Name] = struct{}{}
		if strings.TrimSpace(functionTool.Namespace) == "" {
			if functionTool.DeferLoading {
				deferredOnlyFunctionNames[functionTool.Name] = struct{}{}
			} else {
				topLevelFunctionNames[functionTool.Name] = struct{}{}
			}
			continue
		}
		namespacedFunctionNames[functionTool.Name] = struct{}{}
		namespaceNames[functionTool.Namespace] = struct{}{}
	}

	_, isRealTopLevelFunction := topLevelFunctionNames[toolChoice]
	_, isLocalFunction := allLocalFunctionNames[toolChoice]
	_, isNamespacedFunction := namespacedFunctionNames[toolChoice]
	_, isNamespaceName := namespaceNames[toolChoice]
	_, isDeferredOnlyFunction := deferredOnlyFunctionNames[toolChoice]

	if toolChoice == "tool_search" && hasHostedToolSearch && !isLocalFunction {
		return NewUserError(
			"tool_choice='tool_search' is not supported for ToolSearchTool() on the OpenAI Responses API. Use `auto` or `required`, or target a real top-level function tool named `tool_search`.",
		)
	}
	if toolChoice == "tool_search" && !hasHostedToolSearch && !isLocalFunction {
		return NewUserError(
			"tool_choice='tool_search' requires ToolSearchTool() or a real top-level function tool named `tool_search` on the OpenAI Responses API.",
		)
	}
	if (isNamespacedFunction && !isRealTopLevelFunction) || (isNamespaceName && !isRealTopLevelFunction) {
		return NewUserError(
			"Named tool_choice must target a callable tool, not a namespace wrapper or bare inner name from tool_namespace(), on the OpenAI Responses API. Use `auto`, `required`, `none`, or target a top-level or qualified namespaced function tool.",
		)
	}
	if isDeferredOnlyFunction && !isRealTopLevelFunction {
		return NewUserError(
			"Named tool_choice is not currently supported for deferred-loading function tools on the OpenAI Responses API. Use `auto`, `required`, `none`, or load the tool via ToolSearchTool() first.",
		)
	}
	return nil
}
