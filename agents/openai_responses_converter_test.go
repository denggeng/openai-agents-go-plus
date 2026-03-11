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

package agents_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/computer"
	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConvertToolChoiceStandardValues(t *testing.T) {
	// Make sure that the standard ToolChoice values map to themselves or
	// to "auto"/"required"/"none" as appropriate, and that special string
	// values map to the appropriate items.

	type R = responses.ResponseNewParamsToolChoiceUnion

	testCases := []struct {
		toolChoice modelsettings.ToolChoice
		want       R
	}{
		{nil, R{}},
		{
			modelsettings.ToolChoiceAuto,
			R{OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsAuto)},
		},
		{
			modelsettings.ToolChoiceRequired,
			R{OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired)},
		},
		{
			modelsettings.ToolChoiceNone,
			R{OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone)},
		},
		{
			modelsettings.ToolChoiceString("file_search"),
			R{OfHostedTool: &responses.ToolChoiceTypesParam{Type: responses.ToolChoiceTypesTypeFileSearch}},
		},
		{
			modelsettings.ToolChoiceString("web_search_preview"),
			R{OfHostedTool: &responses.ToolChoiceTypesParam{Type: responses.ToolChoiceTypesTypeWebSearchPreview}},
		},
		{
			modelsettings.ToolChoiceString("web_search_preview_2025_03_11"),
			R{OfHostedTool: &responses.ToolChoiceTypesParam{Type: responses.ToolChoiceTypesTypeWebSearchPreview2025_03_11}},
		},
		{
			modelsettings.ToolChoiceString("computer_use_preview"),
			R{
				OfFunctionTool: &responses.ToolChoiceFunctionParam{
					Name: "computer_use_preview",
					Type: constant.ValueOf[constant.Function](),
				},
			},
		},
		{
			modelsettings.ToolChoiceString("image_generation"),
			R{OfHostedTool: &responses.ToolChoiceTypesParam{Type: responses.ToolChoiceTypesTypeImageGeneration}},
		},
		{
			modelsettings.ToolChoiceString("code_interpreter"),
			R{OfHostedTool: &responses.ToolChoiceTypesParam{Type: responses.ToolChoiceTypesTypeCodeInterpreter}},
		},
		{
			modelsettings.ToolChoiceString("my_function"),
			R{ // Arbitrary string should be interpreted as a function name.
				OfFunctionTool: &responses.ToolChoiceFunctionParam{
					Name: "my_function",
					Type: constant.ValueOf[constant.Function](),
				},
			},
		},
		{
			modelsettings.ToolChoiceString("mcp"),
			R{OfMcpTool: &responses.ToolChoiceMcpParam{Type: constant.ValueOf[constant.Mcp]()}},
		},
		{
			modelsettings.ToolChoiceMCP{ServerLabel: "foo", Name: "bar"},
			R{OfMcpTool: &responses.ToolChoiceMcpParam{
				ServerLabel: "foo",
				Name:        param.NewOpt("bar"),
				Type:        constant.ValueOf[constant.Mcp](),
			}},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("toolChoice %#v", tc.toolChoice), func(t *testing.T) {
			v := agents.ResponsesConverter().ConvertToolChoice(tc.toolChoice)
			assert.Equal(t, tc.want, v)
		})
	}
}

func TestGetResponseFormatPlainTextAndJsonSchema(t *testing.T) {
	// For plain text output, the converter should return a zero-value,
	// indicating no special response format constraint.
	// If an output type is provided for a structured value, the converter
	// should return a ResponseTextConfigParam with the schema and strictness.

	// Default output (None) should be considered plain text.
	v, err := agents.ResponsesConverter().GetResponseFormat(nil)
	require.NoError(t, err)
	assert.Zero(t, v)

	// An explicit plain-text schema (string) should also yield zero-value.
	v, err = agents.ResponsesConverter().GetResponseFormat(agents.OutputType[string]())
	require.NoError(t, err)
	assert.Zero(t, v)

	// A model-based schema should produce a format object.
	type OutputModel struct {
		Foo int    `json:"foo"`
		Bar string `json:"bar"`
	}
	outputType := agents.OutputType[OutputModel]()
	schema, err := outputType.JSONSchema()
	require.NoError(t, err)
	v, err = agents.ResponsesConverter().GetResponseFormat(outputType)
	require.NoError(t, err)
	assert.Equal(t, responses.ResponseTextConfigParam{
		Format: responses.ResponseFormatTextConfigUnionParam{
			OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "final_output",
				Schema: schema,
				Strict: param.NewOpt(true),
				Type:   constant.ValueOf[constant.JSONSchema](),
			},
		},
	}, v)
}

// DummyComputer tool implements a computer.Computer with minimal methods.
type DummyComputer struct{}

func (DummyComputer) Environment(context.Context) (computer.Environment, error) {
	return computer.EnvironmentLinux, nil
}
func (DummyComputer) Dimensions(context.Context) (computer.Dimensions, error) {
	return computer.Dimensions{Width: 800, Height: 600}, nil
}
func (DummyComputer) Screenshot(context.Context) (string, error) {
	return "", errors.New("not implemented")
}
func (DummyComputer) Click(context.Context, int64, int64, computer.Button) error {
	return errors.New("not implemented")
}
func (DummyComputer) DoubleClick(context.Context, int64, int64) error {
	return errors.New("not implemented")
}
func (DummyComputer) Scroll(context.Context, int64, int64, int64, int64) error {
	return errors.New("not implemented")
}
func (DummyComputer) Type(context.Context, string) error {
	return errors.New("not implemented")
}
func (DummyComputer) Wait(context.Context) error {
	return errors.New("not implemented")
}
func (DummyComputer) Move(context.Context, int64, int64) error {
	return errors.New("not implemented")
}
func (DummyComputer) Keypress(context.Context, []string) error {
	return errors.New("not implemented")
}
func (DummyComputer) Drag(context.Context, []computer.Position) error {
	return errors.New("not implemented")
}

func TestConvertToolsBasicTypesAndIncludes(t *testing.T) {
	// Construct a variety of tool types and make sure `ConvertTools` returns
	// a matching list of tool params and the expected includes.

	// Simple function tool
	toolFn := agents.FunctionTool{
		Name:             "fn",
		Description:      "...",
		ParamsJSONSchema: map[string]any{"title": "Fn"},
		OnInvokeTool: func(context.Context, string) (any, error) {
			return nil, errors.New("not implemented")
		},
	}

	// File search tool with IncludeSearchResults set
	fileTool := agents.FileSearchTool{
		MaxNumResults:        param.NewOpt[int64](3),
		VectorStoreIDs:       []string{"vs1"},
		IncludeSearchResults: true,
	}

	// Web search tool with custom params
	webTool := agents.WebSearchTool{SearchContextSize: responses.WebSearchToolSearchContextSizeHigh}

	// Wrap our concrete computer in a tools.ComputerTool for conversion.
	compTool := agents.ComputerTool{Computer: DummyComputer{}}
	allTools := []agents.Tool{toolFn, fileTool, webTool, compTool}
	converted, err := agents.ResponsesConverter().ConvertTools(t.Context(), allTools, nil)
	require.NoError(t, err)
	assert.Equal(t, &agents.ConvertedTools{
		Tools: []responses.ToolUnionParam{
			{
				OfFunction: &responses.FunctionToolParam{
					Name:        "fn",
					Parameters:  toolFn.ParamsJSONSchema,
					Strict:      param.NewOpt(true),
					Description: param.NewOpt("..."),
					Type:        constant.ValueOf[constant.Function](),
				},
			},
			{
				OfFileSearch: &responses.FileSearchToolParam{
					VectorStoreIDs: []string{"vs1"},
					MaxNumResults:  param.NewOpt[int64](3),
					Type:           constant.ValueOf[constant.FileSearch](),
				},
			},
			{
				OfWebSearch: &responses.WebSearchToolParam{
					Type:              responses.WebSearchToolTypeWebSearch,
					UserLocation:      responses.WebSearchToolUserLocationParam{},
					SearchContextSize: responses.WebSearchToolSearchContextSizeHigh,
				},
			},
			{
				OfComputerUsePreview: &responses.ComputerToolParam{
					DisplayHeight: 600,
					DisplayWidth:  800,
					Environment:   responses.ComputerToolEnvironmentLinux,
					Type:          constant.ValueOf[constant.ComputerUsePreview](),
				},
			},
		},
		// The Includes list should have exactly the include for file search
		// when IncludeSearchResults is true.
		Includes: []responses.ResponseIncludable{
			responses.ResponseIncludableFileSearchCallResults,
		},
	}, converted)

	t.Run("only one computer tool should be allowed", func(t *testing.T) {
		_, err = agents.ResponsesConverter().ConvertTools(t.Context(), []agents.Tool{compTool, compTool}, nil)
		assert.ErrorAs(t, err, &agents.UserError{})
	})
}

func TestConvertToolsGroupsNamespacedFunctionTools(t *testing.T) {
	crmTools, err := agents.ToolNamespace(
		"crm",
		"CRM tools",
		testFunctionTool("lookup_account", "Lookup account"),
		testFunctionTool("update_account", "Update account"),
	)
	require.NoError(t, err)

	converted, err := agents.ResponsesConverter().ConvertTools(
		t.Context(),
		[]agents.Tool{crmTools[0], crmTools[1], testFunctionTool("bare_tool", "Bare tool")},
		nil,
	)
	require.NoError(t, err)
	require.Empty(t, converted.Includes)

	raw, err := json.Marshal(converted.Tools)
	require.NoError(t, err)
	assert.JSONEq(t, `[
		{
			"type": "namespace",
			"name": "crm",
			"description": "CRM tools",
			"tools": [
				{
					"type": "function",
					"name": "lookup_account",
					"description": "Lookup account",
					"parameters": {"type":"object","properties":{}},
					"strict": true
				},
				{
					"type": "function",
					"name": "update_account",
					"description": "Update account",
					"parameters": {"type":"object","properties":{}},
					"strict": true
				}
			]
		},
		{
			"type": "function",
			"name": "bare_tool",
			"description": "Bare tool",
			"parameters": {"type":"object","properties":{}},
			"strict": true
		}
	]`, string(raw))
}

func TestConvertToolsIncludesHandoffs(t *testing.T) {
	//  When handoff objects are included, `ConvertTools` should append their
	//  tool param items after tools and include appropriate descriptions.

	agent := &agents.Agent{
		Name:               "support",
		HandoffDescription: "Handles support",
	}
	handoff, err := agents.SafeHandoffFromAgent(agents.HandoffFromAgentParams{Agent: agent})
	require.NoError(t, err)
	require.NotNil(t, handoff)

	converted, err := agents.ResponsesConverter().ConvertTools(t.Context(), nil, []agents.Handoff{*handoff})
	require.NoError(t, err)
	assert.Equal(t, &agents.ConvertedTools{
		Tools: []responses.ToolUnionParam{
			{
				OfFunction: &responses.FunctionToolParam{
					Name: agents.DefaultHandoffToolName(agent),
					Parameters: map[string]any{
						"type":                 "object",
						"additionalProperties": false,
						"properties":           map[string]any{},
						"required":             []string{},
					},
					Strict:      param.NewOpt(true),
					Description: param.NewOpt(agents.DefaultHandoffToolDescription(agent)),
					Type:        constant.ValueOf[constant.Function](),
				},
			},
		},
		Includes: nil,
	}, converted)
}

func TestConvertToolChoiceComputerVariantsFollowEffectiveModel(t *testing.T) {
	compTool := agents.ComputerTool{Computer: DummyComputer{}}

	testCases := []struct {
		name       string
		toolChoice modelsettings.ToolChoice
		tools      []agents.Tool
		model      openai.ChatModel
		wantJSON   string
	}{
		{
			name:       "ga model uses ga selector for computer",
			toolChoice: modelsettings.ToolChoiceString("computer"),
			tools:      []agents.Tool{compTool},
			model:      "gpt-5.4",
			wantJSON:   `{"type":"computer"}`,
		},
		{
			name:       "ga model uses ga selector for preview alias",
			toolChoice: modelsettings.ToolChoiceString("computer_use_preview"),
			tools:      []agents.Tool{compTool},
			model:      "gpt-5.4",
			wantJSON:   `{"type":"computer"}`,
		},
		{
			name:       "preview model keeps preview selector",
			toolChoice: modelsettings.ToolChoiceString("computer"),
			tools:      []agents.Tool{compTool},
			model:      "computer-use-preview",
			wantJSON:   `{"type":"computer_use_preview"}`,
		},
		{
			name:       "prompt managed explicit ga alias uses ga selector",
			toolChoice: modelsettings.ToolChoiceString("computer_use"),
			tools:      []agents.Tool{compTool},
			model:      "",
			wantJSON:   `{"type":"computer"}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := agents.ResponsesConverter().ConvertToolChoiceForRequest(
				tc.toolChoice,
				tc.tools,
				nil,
				tc.model,
			)
			require.NoError(t, err)
			assertToolChoiceJSONEq(t, tc.wantJSON, got)
		})
	}
}

func TestConvertToolChoiceAllowsFunctionNamedBuiltinAliases(t *testing.T) {
	toolCases := []struct {
		name       string
		toolChoice modelsettings.ToolChoice
		tool       agents.FunctionTool
		wantJSON   string
	}{
		{
			name:       "computer",
			toolChoice: modelsettings.ToolChoiceString("computer"),
			tool:       testFunctionTool("computer", "Computer alias"),
			wantJSON:   `{"type":"function","name":"computer"}`,
		},
		{
			name:       "computer_use",
			toolChoice: modelsettings.ToolChoiceString("computer_use"),
			tool:       testFunctionTool("computer_use", "Computer use alias"),
			wantJSON:   `{"type":"function","name":"computer_use"}`,
		},
		{
			name:       "tool_search",
			toolChoice: modelsettings.ToolChoiceString("tool_search"),
			tool:       testFunctionTool("tool_search", "Tool search alias"),
			wantJSON:   `{"type":"function","name":"tool_search"}`,
		},
	}

	for _, tc := range toolCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := agents.ResponsesConverter().ConvertToolChoiceForRequest(
				tc.toolChoice,
				[]agents.Tool{tc.tool},
				nil,
				"",
			)
			require.NoError(t, err)
			assertToolChoiceJSONEq(t, tc.wantJSON, got)
		})
	}
}

func TestConvertToolChoiceValidatesToolSearchAndNamespaceRules(t *testing.T) {
	deferredTool := testFunctionTool("lookup_weather", "Deferred weather")
	deferredTool.DeferLoading = true

	t.Run("hosted tool search choice is rejected", func(t *testing.T) {
		_, err := agents.ResponsesConverter().ConvertToolChoiceForRequest(
			modelsettings.ToolChoiceString("tool_search"),
			[]agents.Tool{deferredTool, agents.ToolSearchTool{}},
			nil,
			"",
		)
		require.ErrorContains(t, err, "ToolSearchTool()")
	})

	t.Run("tool_search without matching local or hosted definition is rejected", func(t *testing.T) {
		namespacedTools, err := agents.ToolNamespace(
			"crm",
			"CRM tools",
			testFunctionTool("lookup_weather", "Lookup weather"),
		)
		require.NoError(t, err)

		_, err = agents.ResponsesConverter().ConvertToolChoiceForRequest(
			modelsettings.ToolChoiceString("tool_search"),
			[]agents.Tool{namespacedTools[0]},
			nil,
			"",
		)
		require.ErrorContains(t, err, "requires ToolSearchTool() or a real top-level function tool named `tool_search`")
	})

	t.Run("required rejects deferred tools without tool search", func(t *testing.T) {
		_, err := agents.ResponsesConverter().ConvertToolChoiceForRequest(
			modelsettings.ToolChoiceRequired,
			[]agents.Tool{deferredTool},
			nil,
			"",
		)
		require.ErrorContains(t, err, "ToolSearchTool()")
	})

	t.Run("required allows deferred tools with tool search", func(t *testing.T) {
		got, err := agents.ResponsesConverter().ConvertToolChoiceForRequest(
			modelsettings.ToolChoiceRequired,
			[]agents.Tool{deferredTool, agents.ToolSearchTool{}},
			nil,
			"",
		)
		require.NoError(t, err)
		assertToolChoiceJSONEq(t, `"required"`, got)
	})

	t.Run("qualified namespaced function is allowed", func(t *testing.T) {
		namespacedTools, err := agents.ToolNamespace(
			"crm",
			"CRM tools",
			testFunctionTool("lookup_account", "Lookup account"),
		)
		require.NoError(t, err)

		got, err := agents.ResponsesConverter().ConvertToolChoiceForRequest(
			modelsettings.ToolChoiceString("crm.lookup_account"),
			[]agents.Tool{namespacedTools[0]},
			nil,
			"",
		)
		require.NoError(t, err)
		assertToolChoiceJSONEq(t, `{"type":"function","name":"crm.lookup_account"}`, got)

		_, err = agents.ResponsesConverter().ConvertToolChoiceForRequest(
			modelsettings.ToolChoiceString("lookup_account"),
			[]agents.Tool{namespacedTools[0]},
			nil,
			"",
		)
		require.ErrorContains(t, err, "tool_namespace()")

		_, err = agents.ResponsesConverter().ConvertToolChoiceForRequest(
			modelsettings.ToolChoiceString("crm"),
			[]agents.Tool{namespacedTools[0]},
			nil,
			"",
		)
		require.ErrorContains(t, err, "tool_namespace()")
	})

	t.Run("deferred only top level function name is rejected", func(t *testing.T) {
		_, err := agents.ResponsesConverter().ConvertToolChoiceForRequest(
			modelsettings.ToolChoiceString("lookup_weather"),
			[]agents.Tool{deferredTool},
			nil,
			"",
		)
		require.ErrorContains(t, err, "deferred-loading function tools")
	})
}

func TestConvertToolsSupportsToolSearchNamespacesAndDeferredSurfaces(t *testing.T) {
	eagerTool := testFunctionTool("get_customer_profile", "Get customer profile")
	deferredTool := testFunctionTool("list_open_orders", "List open orders")
	deferredTool.DeferLoading = true

	namespacedTools, err := agents.ToolNamespace(
		"crm",
		"CRM tools for customer lookups.",
		eagerTool,
		deferredTool,
	)
	require.NoError(t, err)

	converted, err := agents.ResponsesConverter().ConvertTools(
		t.Context(),
		[]agents.Tool{namespacedTools[0], namespacedTools[1], agents.ToolSearchTool{}},
		nil,
	)
	require.NoError(t, err)
	require.Empty(t, converted.Includes)

	assertConvertedToolsJSONEq(t, `[
		{
			"type": "namespace",
			"name": "crm",
			"description": "CRM tools for customer lookups.",
			"tools": [
				{
					"type": "function",
					"name": "get_customer_profile",
					"description": "Get customer profile",
					"parameters": {"type":"object","properties":{}},
					"strict": true
				},
				{
					"type": "function",
					"name": "list_open_orders",
					"description": "List open orders",
					"parameters": {"type":"object","properties":{}},
					"strict": true,
					"defer_loading": true
				}
			]
		},
		{"type": "tool_search"}
	]`, converted)
}

func TestConvertToolsValidatesToolSearchAndDeferredHostedMCP(t *testing.T) {
	deferredTool := testFunctionTool("get_weather", "Get weather")
	deferredTool.DeferLoading = true

	t.Run("deferred top level function requires tool search", func(t *testing.T) {
		_, err := agents.ResponsesConverter().ConvertTools(
			t.Context(),
			[]agents.Tool{deferredTool},
			nil,
		)
		require.ErrorContains(t, err, "ToolSearchTool()")
	})

	t.Run("tool search without a searchable surface is rejected", func(t *testing.T) {
		_, err := agents.ResponsesConverter().ConvertTools(
			t.Context(),
			[]agents.Tool{testFunctionTool("get_weather", "Get weather"), agents.ToolSearchTool{}},
			nil,
		)
		require.ErrorContains(t, err, "requires at least one searchable Responses surface")
	})

	t.Run("deferred hosted mcp is serialized with defer_loading", func(t *testing.T) {
		hostedMCP := agents.HostedMCPTool{
			ToolConfig: responses.ToolMcpParam{
				ServerLabel: "crm_server",
				ServerURL:   param.NewOpt("https://example.com/mcp"),
				Type:        constant.ValueOf[constant.Mcp](),
			},
			DeferLoading: true,
		}

		converted, err := agents.ResponsesConverter().ConvertTools(
			t.Context(),
			[]agents.Tool{
				hostedMCP,
				agents.ToolSearchTool{
					Description: "Search deferred tools on the server.",
					Execution:   agents.ToolSearchExecutionServer,
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"query": map[string]any{"type": "string"},
						},
						"required": []string{"query"},
					},
				},
			},
			nil,
		)
		require.NoError(t, err)

		assertConvertedToolsJSONEq(t, `[
			{
				"type": "mcp",
				"server_label": "crm_server",
				"server_url": "https://example.com/mcp",
				"defer_loading": true
			},
			{
				"type": "tool_search",
				"description": "Search deferred tools on the server.",
				"execution": "server",
				"parameters": {
					"type": "object",
					"properties": {"query": {"type": "string"}},
					"required": ["query"]
				}
			}
		]`, converted)
	})
}

func assertToolChoiceJSONEq(
	t *testing.T,
	want string,
	got responses.ResponseNewParamsToolChoiceUnion,
) {
	t.Helper()
	raw, err := json.Marshal(got)
	require.NoError(t, err)
	assert.JSONEq(t, want, string(raw))
}

func assertConvertedToolsJSONEq(t *testing.T, want string, got *agents.ConvertedTools) {
	t.Helper()
	require.NotNil(t, got)
	raw, err := json.Marshal(got.Tools)
	require.NoError(t, err)
	assert.JSONEq(t, want, string(raw))
}
