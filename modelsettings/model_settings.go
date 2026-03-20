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

package modelsettings

import (
	"context"
	"maps"
	"reflect"

	"github.com/denggeng/openai-agents-go-plus/retry"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

// ModelSettings holds settings to use when calling an LLM.
//
// This type holds optional model configuration parameters (e.g. temperature,
// top-p, penalties, truncation, etc.).
//
// Not all models/providers support all of these parameters, so please check
// the API documentation for the specific model and provider you are using.
type ModelSettings struct {
	// The temperature to use when calling the model.
	Temperature param.Opt[float64] `json:"temperature"`

	// The top_p to use when calling the model.
	TopP param.Opt[float64] `json:"top_p"`

	// The frequency penalty to use when calling the model.
	FrequencyPenalty param.Opt[float64] `json:"frequency_penalty"`

	// The presence penalty to use when calling the model.
	PresencePenalty param.Opt[float64] `json:"presence_penalty"`

	// Optional tool choice to use when calling the model.
	ToolChoice ToolChoice `json:"tool_choice"`

	// Controls whether the model can make multiple parallel tool calls in a single turn.
	// If not provided, this behavior defers to the underlying model provider's default.
	// For most current providers (e.g., OpenAI), this typically means parallel tool calls
	// are enabled (true).
	// Set to true to explicitly enable parallel tool calls, or false to restrict the
	// model to at most one tool call per turn.
	ParallelToolCalls param.Opt[bool] `json:"parallel_tool_calls"`

	// The truncation strategy to use when calling the model.
	// For more details, see Responses API documentation:
	// https://platform.openai.com/docs/api-reference/responses/create#responses_create-truncation
	Truncation param.Opt[Truncation] `json:"truncation"`

	// The maximum number of output tokens to generate.
	MaxTokens param.Opt[int64] `json:"max_tokens"`

	// Optional configuration options for reasoning models
	// (see https://platform.openai.com/docs/guides/reasoning).
	Reasoning openai.ReasoningParam `json:"reasoning"`

	// Constrains the verbosity of the model's response.
	Verbosity param.Opt[Verbosity] `json:"verbosity"`

	// Optional metadata to include with the model response call.
	Metadata map[string]string `json:"metadata"`

	// Whether to store the generated model response for later retrieval.
	// For Responses API: automatically enabled when not specified.
	// For Chat Completions API: disabled when not specified.
	Store param.Opt[bool] `json:"store"`

	// The retention policy for the prompt cache. Set to `24h` to enable extended
	// prompt caching, which keeps cached prefixes active for longer, up to a maximum
	// of 24 hours.
	PromptCacheRetention param.Opt[PromptCacheRetention] `json:"prompt_cache_retention"`

	// Whether to include usage chunk.
	//Only available for Chat Completions API.
	IncludeUsage param.Opt[bool] `json:"include_usage"`

	// Optional additional output data to include in the model response
	// (see https://platform.openai.com/docs/api-reference/responses/create#responses-create-include).
	ResponseInclude []responses.ResponseIncludable `json:"response_include"`

	// Number of top tokens to return logprobs for.
	// Setting this will automatically include ``"message.output_text.logprobs"`` in the response.
	TopLogprobs param.Opt[int64] `json:"top_logprobs"`

	// Optional additional query fields to provide with the request.
	ExtraQuery map[string]string `json:"extra_query"`

	// Optional additional headers to provide with the request.
	ExtraHeaders map[string]string `json:"extra_headers"`

	// Optional additional JSON body fields to provide with the request.
	// These are applied after typed request params and can be useful for
	// provider-specific extensions.
	ExtraBody map[string]any `json:"extra_body"`

	// Optional additional request args for provider-specific integrations.
	// These are merged with ExtraBody; for duplicate keys, ExtraArgs wins,
	// except `reasoning_effort` where ExtraBody wins for compatibility.
	ExtraArgs map[string]any `json:"extra_args"`

	// Optional runner-managed retry settings for model calls.
	Retry *retry.ModelRetrySettings `json:"retry,omitempty"`

	// Optional function which allows you to fully customize parameters and options
	// for a call to the responses API. Pre-built parameters and options are given.
	// You should return the final parameters and options that will be passed
	// directly to the underlying model provider's API.
	// Use with caution as not all models support all parameters.
	CustomizeResponsesRequest func(context.Context, *responses.ResponseNewParams, []option.RequestOption) (*responses.ResponseNewParams, []option.RequestOption, error) `json:"-"`

	// Optional function which allows you to fully customize parameters and options
	// for a call to the chat-completion API. Pre-built parameters and options are given.
	// You should return the final parameters and options that will be passed
	// directly to the underlying model provider's API.
	// Use with caution as not all models support all parameters.
	CustomizeChatCompletionsRequest func(context.Context, *openai.ChatCompletionNewParams, []option.RequestOption) (*openai.ChatCompletionNewParams, []option.RequestOption, error) `json:"-"`
}

type Verbosity string

const (
	VerbosityLow    Verbosity = "low"
	VerbosityMedium Verbosity = "medium"
	VerbosityHigh   Verbosity = "high"
)

type ToolChoice interface {
	isToolChoice()
}

type ToolChoiceString string

func (ToolChoiceString) isToolChoice()     {}
func (tc ToolChoiceString) String() string { return string(tc) }

const (
	ToolChoiceAuto     ToolChoiceString = "auto"
	ToolChoiceRequired ToolChoiceString = "required"
	ToolChoiceNone     ToolChoiceString = "none"
)

type ToolChoiceMCP struct {
	ServerLabel string `json:"server_label"`
	Name        string `json:"name"`
}

func (ToolChoiceMCP) isToolChoice() {}

type Truncation string

const (
	TruncationAuto     Truncation = "auto"
	TruncationDisabled Truncation = "disabled"
)

type PromptCacheRetention string

const (
	PromptCacheRetentionInMemory PromptCacheRetention = "in_memory"
	PromptCacheRetention24h      PromptCacheRetention = "24h"
)

// Resolve produces a new ModelSettings by overlaying any present values from
// the override on top of this instance.
func (ms ModelSettings) Resolve(override ModelSettings) ModelSettings {
	newSettings := ms
	resolveOpt(&newSettings.Temperature, override.Temperature)
	resolveOpt(&newSettings.TopP, override.TopP)
	resolveOpt(&newSettings.FrequencyPenalty, override.FrequencyPenalty)
	resolveOpt(&newSettings.PresencePenalty, override.PresencePenalty)
	resolveAny(&newSettings.ToolChoice, override.ToolChoice)
	resolveOpt(&newSettings.ParallelToolCalls, override.ParallelToolCalls)
	resolveOpt(&newSettings.Truncation, override.Truncation)
	resolveOpt(&newSettings.MaxTokens, override.MaxTokens)
	resolveAny(&newSettings.Reasoning, override.Reasoning)
	resolveOpt(&newSettings.Verbosity, override.Verbosity)
	resolveMap(&newSettings.Metadata, override.Metadata)
	resolveOpt(&newSettings.Store, override.Store)
	resolveOpt(&newSettings.PromptCacheRetention, override.PromptCacheRetention)
	resolveOpt(&newSettings.IncludeUsage, override.IncludeUsage)
	resolveAny(&newSettings.ResponseInclude, override.ResponseInclude)
	resolveOpt(&newSettings.TopLogprobs, override.TopLogprobs)
	resolveMap(&newSettings.ExtraQuery, override.ExtraQuery)
	resolveMap(&newSettings.ExtraHeaders, override.ExtraHeaders)
	resolveMap(&newSettings.ExtraBody, override.ExtraBody)
	newSettings.ExtraArgs = resolveMergedMap(ms.ExtraArgs, override.ExtraArgs)
	newSettings.Retry = resolveRetrySettings(ms.Retry, override.Retry)
	resolveAny(&newSettings.CustomizeResponsesRequest, override.CustomizeResponsesRequest)
	resolveAny(&newSettings.CustomizeChatCompletionsRequest, override.CustomizeChatCompletionsRequest)
	return newSettings
}

func resolveOpt[T comparable](base *param.Opt[T], override param.Opt[T]) {
	if override.Valid() {
		*base = override
	}
}

func resolveAny[T any](base *T, override T) {
	v := reflect.ValueOf(override)
	if v.Kind() != reflect.Invalid && !v.IsZero() {
		*base = override
	}
}

func resolveMap[M ~map[K]V, K comparable, V any](base *M, override M) {
	if len(override) > 0 {
		*base = maps.Clone(override)
	}
}

func resolveMergedMap[M ~map[K]V, K comparable, V any](base M, override M) M {
	switch {
	case len(base) == 0 && len(override) == 0:
		var zero M
		return zero
	case len(base) == 0:
		return maps.Clone(override)
	case len(override) == 0:
		return maps.Clone(base)
	}

	merged := maps.Clone(base)
	for key, value := range override {
		merged[key] = value
	}
	return merged
}

func resolveRetrySettings(
	base *retry.ModelRetrySettings,
	override *retry.ModelRetrySettings,
) *retry.ModelRetrySettings {
	switch {
	case base == nil && override == nil:
		return nil
	case base == nil:
		return cloneRetrySettings(override)
	case override == nil:
		return cloneRetrySettings(base)
	}

	merged := cloneRetrySettings(base)
	if override.MaxRetries != nil {
		merged.MaxRetries = clonePointer(override.MaxRetries)
	}
	if override.Policy != nil {
		merged.Policy = override.Policy
	}
	merged.Backoff = resolveRetryBackoffSettings(base.Backoff, override.Backoff)
	return merged
}

func resolveRetryBackoffSettings(
	base *retry.ModelRetryBackoffSettings,
	override *retry.ModelRetryBackoffSettings,
) *retry.ModelRetryBackoffSettings {
	switch {
	case base == nil && override == nil:
		return nil
	case base == nil:
		return cloneRetryBackoffSettings(override)
	case override == nil:
		return cloneRetryBackoffSettings(base)
	}

	merged := cloneRetryBackoffSettings(base)
	if override.InitialDelay != nil {
		merged.InitialDelay = clonePointer(override.InitialDelay)
	}
	if override.MaxDelay != nil {
		merged.MaxDelay = clonePointer(override.MaxDelay)
	}
	if override.Multiplier != nil {
		merged.Multiplier = clonePointer(override.Multiplier)
	}
	if override.Jitter != nil {
		merged.Jitter = clonePointer(override.Jitter)
	}
	return merged
}

func cloneRetrySettings(settings *retry.ModelRetrySettings) *retry.ModelRetrySettings {
	if settings == nil {
		return nil
	}
	return &retry.ModelRetrySettings{
		MaxRetries: clonePointer(settings.MaxRetries),
		Backoff:    cloneRetryBackoffSettings(settings.Backoff),
		Policy:     settings.Policy,
	}
}

func cloneRetryBackoffSettings(settings *retry.ModelRetryBackoffSettings) *retry.ModelRetryBackoffSettings {
	if settings == nil {
		return nil
	}
	return &retry.ModelRetryBackoffSettings{
		InitialDelay: clonePointer(settings.InitialDelay),
		MaxDelay:     clonePointer(settings.MaxDelay),
		Multiplier:   clonePointer(settings.Multiplier),
		Jitter:       clonePointer(settings.Jitter),
	}
}

func clonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
