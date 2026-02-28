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
	"slices"
	"strings"
	"sync"

	"github.com/denggeng/openai-agents-go-plus/asyncqueue"
	"github.com/denggeng/openai-agents-go-plus/computer"
	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/denggeng/openai-agents-go-plus/openaitypes"
	"github.com/denggeng/openai-agents-go-plus/tracing"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

type queueCompleteSentinel struct{}

func (queueCompleteSentinel) isStreamEvent() {}

type AgentToolUseTracker struct {
	AgentToTools []AgentToToolsItem
	NameToTools  map[string][]string
}

func NewAgentToolUseTracker() *AgentToolUseTracker {
	return &AgentToolUseTracker{
		NameToTools: make(map[string][]string),
	}
}

type AgentToToolsItem struct {
	Agent     *Agent
	ToolNames []string
}

func (item *AgentToToolsItem) AppendToolNames(toolNames []string) {
	item.ToolNames = append(item.ToolNames, toolNames...)
}

func (t *AgentToolUseTracker) AddToolUse(agent *Agent, toolNames []string) {
	if agent != nil && agent.Name != "" {
		if t.NameToTools == nil {
			t.NameToTools = make(map[string][]string)
		}
		t.NameToTools[agent.Name] = append(t.NameToTools[agent.Name], toolNames...)
	}
	index := t.agentIndex(agent)
	if index == -1 {
		t.AgentToTools = append(t.AgentToTools, AgentToToolsItem{
			Agent:     agent,
			ToolNames: toolNames,
		})
	} else {
		t.AgentToTools[index].AppendToolNames(toolNames)
	}
}

func (t *AgentToolUseTracker) HasUsedTools(agent *Agent) bool {
	index := t.agentIndex(agent)
	if index != -1 && len(t.AgentToTools[index].ToolNames) > 0 {
		return true
	}
	if agent == nil || agent.Name == "" {
		return false
	}
	if t.NameToTools == nil {
		return false
	}
	return len(t.NameToTools[agent.Name]) > 0
}

func (t *AgentToolUseTracker) agentIndex(agent *Agent) int {
	return slices.IndexFunc(t.AgentToTools, func(item AgentToToolsItem) bool {
		return item.Agent == agent
	})
}

// LoadSnapshot restores tool usage from a serialized snapshot keyed by agent name.
func (t *AgentToolUseTracker) LoadSnapshot(snapshot map[string][]string) {
	if t == nil || len(snapshot) == 0 {
		return
	}
	if t.NameToTools == nil {
		t.NameToTools = make(map[string][]string, len(snapshot))
	}
	for agentName, tools := range snapshot {
		if agentName == "" {
			continue
		}
		t.NameToTools[agentName] = append([]string(nil), tools...)
	}
}

// Snapshot returns a copy of tool usage keyed by agent name.
func (t *AgentToolUseTracker) Snapshot() map[string][]string {
	return t.AsSerializable()
}

type ToolRunHandoff struct {
	Handoff  Handoff
	ToolCall ResponseFunctionToolCall
}

type ToolRunFunction struct {
	ToolCall     ResponseFunctionToolCall
	FunctionTool FunctionTool
}

type ToolRunComputerAction struct {
	ToolCall     responses.ResponseComputerToolCall
	ComputerTool ComputerTool
}

type ToolRunMCPApprovalRequest struct {
	RequestItem responses.ResponseOutputItemMcpApprovalRequest
	MCPTool     HostedMCPTool
}

type ToolRunLocalShellCall struct {
	ToolCall       responses.ResponseOutputItemLocalShellCall
	LocalShellTool LocalShellTool
}

type ToolRunShellCall struct {
	ToolCall  any
	ShellTool ShellTool
}

type ToolRunApplyPatchCall struct {
	ToolCall       any
	ApplyPatchTool ApplyPatchTool
}

type ProcessedResponse struct {
	NewItems        []RunItem
	Handoffs        []ToolRunHandoff
	Functions       []ToolRunFunction
	ComputerActions []ToolRunComputerAction
	LocalShellCalls []ToolRunLocalShellCall
	ShellCalls      []ToolRunShellCall
	ApplyPatchCalls []ToolRunApplyPatchCall
	Interruptions   []ToolApprovalItem
	// Names of all tools used, including hosted tools
	ToolsUsed []string
	// Only requests with callbacks
	MCPApprovalRequests []ToolRunMCPApprovalRequest
}

func (pr *ProcessedResponse) HasToolsOrApprovalsToRun() bool {
	// Handoffs, functions and computer actions need local processing.
	// Hosted tools have already run, so there's nothing to do.
	return len(pr.Handoffs) > 0 || len(pr.Functions) > 0 ||
		len(pr.ComputerActions) > 0 || len(pr.LocalShellCalls) > 0 ||
		len(pr.ShellCalls) > 0 || len(pr.ApplyPatchCalls) > 0 ||
		len(pr.MCPApprovalRequests) > 0
}

type NextStep interface {
	isNextStep()
}

type NextStepHandoff struct {
	NewAgent *Agent
}

func (NextStepHandoff) isNextStep() {}

type NextStepFinalOutput struct {
	Output any
}

func (NextStepFinalOutput) isNextStep() {}

type NextStepRunAgain struct{}

func (NextStepRunAgain) isNextStep() {}

type NextStepInterruption struct {
	Interruptions []ToolApprovalItem
}

func (NextStepInterruption) isNextStep() {}

type SingleStepResult struct {
	// The input items i.e. the items before Run() was called. May be mutated by handoff input filters.
	OriginalInput Input

	// The model response for the current step.
	ModelResponse ModelResponse

	// Items generated before the current step.
	PreStepItems []RunItem

	// Items generated during this current step.
	NewStepItems []RunItem

	// Full unfiltered items for session history. When set, these are used instead of
	// NewStepItems for session persistence and observability.
	SessionStepItems []RunItem

	// Results of tool input guardrails run during this step.
	ToolInputGuardrailResults []ToolInputGuardrailResult

	// Results of tool output guardrails run during this step.
	ToolOutputGuardrailResults []ToolOutputGuardrailResult

	// The next step to take.
	NextStep NextStep
}

// GeneratedItems returns the items generated during the agent run (i.e. everything generated after `OriginalInput`).
func (result SingleStepResult) GeneratedItems() []RunItem {
	return slices.Concat(result.PreStepItems, result.NewStepItems)
}

// StepSessionItems returns the items to use for session persistence and streaming.
func (result SingleStepResult) StepSessionItems() []RunItem {
	if result.SessionStepItems != nil {
		return result.SessionStepItems
	}
	return result.NewStepItems
}

func GetModelTracingImpl(tracingDisabled, traceIncludeSensitiveData bool) ModelTracing {
	switch {
	case tracingDisabled:
		return ModelTracingDisabled
	case traceIncludeSensitiveData:
		return ModelTracingEnabled
	default:
		return ModelTracingEnabledWithoutData
	}
}

type runImpl struct{}

func RunImpl() runImpl { return runImpl{} }

func (ri runImpl) ExecuteToolsAndSideEffects(
	ctx context.Context,
	agent *Agent,
	// The original input to the Runner
	originalInput Input,
	// Everything generated by Runner since the original input, but before the current step
	preStepItems []RunItem,
	newResponse ModelResponse,
	processedResponse ProcessedResponse,
	outputType OutputTypeInterface,
	hooks RunHooks,
	runConfig RunConfig,
	contextWrapper *RunContextWrapper[any],
) (*SingleStepResult, error) {
	// Make a copy of the generated items
	preStepItems = slices.Clone(preStepItems)

	var newStepItems []RunItem
	newStepItems = append(newStepItems, processedResponse.NewItems...)

	// First, let's run the tool calls - function tools and computer actions
	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	var (
		functionResults            []FunctionToolResult
		toolInputGuardrailResults  []ToolInputGuardrailResult
		toolOutputGuardrailResults []ToolOutputGuardrailResult
		functionInterruptions      []ToolApprovalItem
		computerResults            []RunItem
		localShellResults          []RunItem
		shellResults               []RunItem
		applyPatchResults          []RunItem
		shellInterruptions         []ToolApprovalItem
		applyPatchInterruptions    []ToolApprovalItem
		toolErrors                 [2]error
		wg                         sync.WaitGroup
	)
	wg.Add(2)
	go func() {
		defer wg.Done()
		functionResults, toolInputGuardrailResults, toolOutputGuardrailResults, functionInterruptions, toolErrors[0] = ri.ExecuteFunctionToolCalls(
			childCtx,
			agent,
			processedResponse.Functions,
			hooks,
			contextWrapper,
			runConfig,
		)
	}()
	go func() {
		defer wg.Done()
		computerResults, toolErrors[1] = ri.ExecuteComputerActions(
			childCtx,
			agent,
			processedResponse.ComputerActions,
			hooks,
			contextWrapper,
		)
	}()
	wg.Wait()
	if err := errors.Join(toolErrors[:]...); err != nil {
		return nil, err
	}

	for _, result := range functionResults {
		if result.RunItem != nil {
			newStepItems = append(newStepItems, result.RunItem)
		}
	}
	newStepItems = append(newStepItems, computerResults...)

	if len(processedResponse.LocalShellCalls) > 0 {
		var err error
		localShellResults, err = ri.ExecuteLocalShellCalls(
			ctx,
			agent,
			processedResponse.LocalShellCalls,
			hooks,
		)
		if err != nil {
			return nil, err
		}
		newStepItems = append(newStepItems, localShellResults...)
	}

	if len(processedResponse.ShellCalls) > 0 {
		var err error
		shellResults, shellInterruptions, err = ri.ExecuteShellCalls(
			ctx,
			agent,
			processedResponse.ShellCalls,
			hooks,
			contextWrapper,
			runConfig,
		)
		if err != nil {
			return nil, err
		}
		newStepItems = append(newStepItems, shellResults...)
	}

	if len(processedResponse.ApplyPatchCalls) > 0 {
		var err error
		applyPatchResults, applyPatchInterruptions, err = ri.ExecuteApplyPatchCalls(
			ctx,
			agent,
			processedResponse.ApplyPatchCalls,
			hooks,
			contextWrapper,
			runConfig,
		)
		if err != nil {
			return nil, err
		}
		newStepItems = append(newStepItems, applyPatchResults...)
	}

	// Next, run the MCP approval requests
	if mcpApprovalRequests := processedResponse.MCPApprovalRequests; len(mcpApprovalRequests) > 0 {
		approvalResults, err := ri.ExecuteMCPApprovalRequests(ctx, agent, processedResponse.MCPApprovalRequests)
		if err != nil {
			return nil, err
		}
		newStepItems = append(newStepItems, approvalResults...)
	}

	interruptions := slices.Clone(processedResponse.Interruptions)
	if len(functionInterruptions) > 0 {
		interruptions = append(interruptions, functionInterruptions...)
	}
	if len(shellInterruptions) > 0 {
		interruptions = append(interruptions, shellInterruptions...)
	}
	if len(applyPatchInterruptions) > 0 {
		interruptions = append(interruptions, applyPatchInterruptions...)
	}

	if len(interruptions) > 0 {
		return &SingleStepResult{
			OriginalInput:              originalInput,
			ModelResponse:              newResponse,
			PreStepItems:               preStepItems,
			NewStepItems:               newStepItems,
			ToolInputGuardrailResults:  toolInputGuardrailResults,
			ToolOutputGuardrailResults: toolOutputGuardrailResults,
			NextStep: NextStepInterruption{
				Interruptions: slices.Clone(interruptions),
			},
		}, nil
	}

	// Next, check if there are any handoffs
	if runHandoffs := processedResponse.Handoffs; len(runHandoffs) > 0 {
		stepResult, err := ri.ExecuteHandoffs(
			ctx,
			agent,
			originalInput,
			preStepItems,
			newStepItems,
			newResponse,
			runHandoffs,
			hooks,
			runConfig,
			contextWrapper,
		)
		if err != nil {
			return nil, err
		}
		stepResult.ToolInputGuardrailResults = toolInputGuardrailResults
		stepResult.ToolOutputGuardrailResults = toolOutputGuardrailResults
		return stepResult, nil
	}

	// Next, we'll check if the tool use should result in a final output
	checkToolUse, err := ri.checkForFinalOutputFromTools(ctx, agent, functionResults)
	if err != nil {
		return nil, err
	}

	if checkToolUse.IsFinalOutput {
		if !checkToolUse.FinalOutput.Valid() {
			Logger().Error("Model returned a final output of None. Not raising an error because we assume you know what you're doing.")
		}

		// If the output type is string, then let's just stringify the result
		if agent.OutputType == nil || agent.OutputType.IsPlainText() {
			if _, ok := checkToolUse.FinalOutput.Value.(string); !ok {
				checkToolUse.FinalOutput = param.NewOpt[any](fmt.Sprintf("%v", checkToolUse.FinalOutput.Value))
			}
		}

		stepResult, err := ri.ExecuteFinalOutput(
			ctx,
			agent,
			originalInput,
			newResponse,
			preStepItems,
			newStepItems,
			checkToolUse.FinalOutput.Or(nil),
			hooks,
		)
		if err != nil {
			return nil, err
		}
		stepResult.ToolInputGuardrailResults = toolInputGuardrailResults
		stepResult.ToolOutputGuardrailResults = toolOutputGuardrailResults
		return stepResult, nil
	}

	// Now we can check if the model also produced a final output
	messageItems := make([]MessageOutputItem, 0)
	for _, item := range newStepItems {
		if messageItem, ok := item.(MessageOutputItem); ok {
			messageItems = append(messageItems, messageItem)
		}
	}

	// We'll use the last content output as the final output
	potentialFinalOutputText := ""
	if len(messageItems) > 0 {
		rawItem := messageItems[len(messageItems)-1].RawItem
		potentialFinalOutputText, _ = ItemHelpers().ExtractLastText(
			openaitypes.ResponseOutputItemUnionFromResponseOutputMessage(rawItem))
	}

	// There are two possibilities that lead to a final output:
	// 1. Structured output type => always leads to a final output
	// 2. Plain text output type => only leads to a final output if there are no tool calls
	if outputType != nil && !outputType.IsPlainText() && potentialFinalOutputText != "" {
		finalOutput, err := outputType.ValidateJSON(ctx, potentialFinalOutputText)
		if err != nil {
			return nil, fmt.Errorf("final output type JSON validation failed: %w", err)
		}
		stepResult, err := ri.ExecuteFinalOutput(
			ctx,
			agent,
			originalInput,
			newResponse,
			preStepItems,
			newStepItems,
			finalOutput,
			hooks,
		)
		if err != nil {
			return nil, err
		}
		stepResult.ToolInputGuardrailResults = toolInputGuardrailResults
		stepResult.ToolOutputGuardrailResults = toolOutputGuardrailResults
		return stepResult, nil
	} else if (outputType == nil || outputType.IsPlainText()) && !processedResponse.HasToolsOrApprovalsToRun() {
		stepResult, err := ri.ExecuteFinalOutput(
			ctx,
			agent,
			originalInput,
			newResponse,
			preStepItems,
			newStepItems,
			potentialFinalOutputText,
			hooks,
		)
		if err != nil {
			return nil, err
		}
		stepResult.ToolInputGuardrailResults = toolInputGuardrailResults
		stepResult.ToolOutputGuardrailResults = toolOutputGuardrailResults
		return stepResult, nil
	} else {
		// If there's no final output, we can just run again
		return &SingleStepResult{
			OriginalInput:              originalInput,
			ModelResponse:              newResponse,
			PreStepItems:               preStepItems,
			NewStepItems:               newStepItems,
			ToolInputGuardrailResults:  toolInputGuardrailResults,
			ToolOutputGuardrailResults: toolOutputGuardrailResults,
			NextStep:                   NextStepRunAgain{},
		}, nil
	}
}

// MaybeResetToolChoice resets tool choice to nil if the agent has used tools
// and the agent's ResetToolChoice flag is true.
func (runImpl) MaybeResetToolChoice(
	agent *Agent,
	toolUseTracker *AgentToolUseTracker,
	modelSettings modelsettings.ModelSettings,
) modelsettings.ModelSettings {
	resetToolChoice := agent.ResetToolChoice.Or(true)
	if resetToolChoice && toolUseTracker.HasUsedTools(agent) {
		newSettings := modelSettings
		newSettings.ToolChoice = nil
		return newSettings
	}
	return modelSettings
}

func (runImpl) ProcessModelResponse(
	ctx context.Context,
	agent *Agent,
	allTools []Tool,
	response ModelResponse,
	handoffs []Handoff,
) (*ProcessedResponse, error) {
	var (
		items               []RunItem
		runHandoffs         []ToolRunHandoff
		functions           []ToolRunFunction
		computerActions     []ToolRunComputerAction
		localShellCalls     []ToolRunLocalShellCall
		shellCalls          []ToolRunShellCall
		applyPatchCalls     []ToolRunApplyPatchCall
		interruptions       []ToolApprovalItem
		mcpApprovalRequests []ToolRunMCPApprovalRequest
		computerTool        *ComputerTool
		localShellTool      *LocalShellTool
		shellTool           *ShellTool
		applyPatchTool      *ApplyPatchTool
		toolsUsed           []string
	)

	handoffMap := make(map[string]Handoff, len(handoffs))
	for _, handoff := range handoffs {
		handoffMap[handoff.ToolName] = handoff
	}

	functionMap := make(map[string]FunctionTool)
	hostedMCPServerMap := make(map[string]HostedMCPTool)

	for _, tool := range allTools {
		switch t := tool.(type) {
		case FunctionTool:
			functionMap[t.Name] = t
		case ComputerTool:
			computerTool = &t
		case LocalShellTool:
			localShellTool = &t
		case ShellTool:
			shellTool = &t
		case ApplyPatchTool:
			toolCopy := t
			applyPatchTool = &toolCopy
		case HostedMCPTool:
			hostedMCPServerMap[t.ToolConfig.ServerLabel] = t
		}
	}

	for _, outputUnion := range response.Output {
		switch outputUnion.Type {
		case "message":
			output := responses.ResponseOutputMessage{
				ID:      outputUnion.ID,
				Content: outputUnion.Content,
				Role:    outputUnion.Role,
				Status:  responses.ResponseOutputMessageStatus(outputUnion.Status),
				Type:    constant.ValueOf[constant.Message](),
			}
			items = append(items, MessageOutputItem{
				Agent:   agent,
				RawItem: output,
				Type:    "message_output_item",
			})
		case "file_search_call":
			output := responses.ResponseFileSearchToolCall{
				ID:      outputUnion.ID,
				Queries: outputUnion.Queries,
				Status:  responses.ResponseFileSearchToolCallStatus(outputUnion.Status),
				Type:    constant.ValueOf[constant.FileSearchCall](),
				Results: outputUnion.Results,
			}
			items = append(items, ToolCallItem{
				Agent:   agent,
				RawItem: ResponseFileSearchToolCall(output),
				Type:    "tool_call_item",
			})
			toolsUsed = append(toolsUsed, "file_search")
		case "web_search_call":
			output := responses.ResponseFunctionWebSearch{
				ID:     outputUnion.ID,
				Action: openaitypes.ResponseFunctionWebSearchActionUnionFromResponseOutputItemUnionAction(outputUnion.Action),
				Status: responses.ResponseFunctionWebSearchStatus(outputUnion.Status),
				Type:   constant.ValueOf[constant.WebSearchCall](),
			}
			items = append(items, ToolCallItem{
				Agent:   agent,
				RawItem: ResponseFunctionWebSearch(output),
				Type:    "tool_call_item",
			})
			toolsUsed = append(toolsUsed, "web_search")
		case "reasoning":
			output := responses.ResponseReasoningItem{
				ID:               outputUnion.ID,
				Summary:          outputUnion.Summary,
				Type:             constant.ValueOf[constant.Reasoning](),
				EncryptedContent: outputUnion.EncryptedContent,
				Status:           responses.ResponseReasoningItemStatus(outputUnion.Status),
			}
			items = append(items, ReasoningItem{
				Agent:   agent,
				RawItem: output,
				Type:    "reasoning_item",
			})
		case "compaction":
			var rawItem map[string]any
			if outputUnion.RawJSON() != "" {
				if err := json.Unmarshal([]byte(outputUnion.RawJSON()), &rawItem); err != nil {
					rawItem = nil
				}
			}
			if rawItem == nil {
				rawItem = map[string]any{
					"type":              "compaction",
					"id":                outputUnion.ID,
					"encrypted_content": outputUnion.EncryptedContent,
				}
			}
			delete(rawItem, "created_by")
			if _, ok := rawItem["type"]; !ok {
				rawItem["type"] = "compaction"
			}
			items = append(items, CompactionItem{
				Agent:   agent,
				RawItem: CompactionItemRawItem(rawItem),
				Type:    "compaction_item",
			})
		case "computer_call":
			output := responses.ResponseComputerToolCall{
				ID:                  outputUnion.ID,
				Action:              openaitypes.ResponseComputerToolCallActionUnionFromResponseOutputItemUnionAction(outputUnion.Action),
				CallID:              outputUnion.CallID,
				PendingSafetyChecks: outputUnion.PendingSafetyChecks,
				Status:              responses.ResponseComputerToolCallStatus(outputUnion.Status),
				Type:                responses.ResponseComputerToolCallTypeComputerCall,
			}
			items = append(items, ToolCallItem{
				Agent:   agent,
				RawItem: ResponseComputerToolCall(output),
				Type:    "tool_call_item",
			})
			toolsUsed = append(toolsUsed, "computer_use")
			if computerTool == nil {
				AttachErrorToCurrentSpan(ctx, tracing.SpanError{Message: "Computer tool not found"})
				return nil, NewModelBehaviorError("model produced computer action without a computer tool")
			}
			computerActions = append(computerActions, ToolRunComputerAction{
				ToolCall:     output,
				ComputerTool: *computerTool,
			})
		case "mcp_approval_request":
			output := responses.ResponseOutputItemMcpApprovalRequest{
				ID:          outputUnion.ID,
				Arguments:   outputUnion.Arguments,
				Name:        outputUnion.Name,
				ServerLabel: outputUnion.ServerLabel,
				Type:        constant.ValueOf[constant.McpApprovalRequest](),
			}
			items = append(items, MCPApprovalRequestItem{
				Agent:   agent,
				RawItem: output,
				Type:    "mcp_approval_request_item",
			})
			if server, ok := hostedMCPServerMap[output.ServerLabel]; !ok {
				AttachErrorToCurrentSpan(ctx, tracing.SpanError{
					Message: "MCP server label not found",
					Data:    map[string]any{"server_label": output.ServerLabel},
				})
				return nil, ModelBehaviorErrorf("MCP server label %q not found", output.ServerLabel)
			} else if server.OnApprovalRequest != nil {
				mcpApprovalRequests = append(mcpApprovalRequests, ToolRunMCPApprovalRequest{
					RequestItem: output,
					MCPTool:     server,
				})
			} else {
				Logger().Warn("MCP server has no OnApprovalRequest hook",
					slog.String("serverLabel", output.ServerLabel))
				interruptions = append(interruptions, ToolApprovalItem{
					ToolName: output.Name,
					RawItem:  output,
				})
			}
		case "mcp_list_tools":
			output := responses.ResponseOutputItemMcpListTools{
				ID:          outputUnion.ID,
				ServerLabel: outputUnion.ServerLabel,
				Tools:       outputUnion.Tools,
				Type:        constant.ValueOf[constant.McpListTools](),
				Error:       outputUnion.Error,
			}
			items = append(items, MCPListToolsItem{
				Agent:   agent,
				RawItem: output,
				Type:    "mcp_list_tools_item",
			})
		case "mcp_call":
			output := responses.ResponseOutputItemMcpCall{
				ID:          outputUnion.ID,
				Arguments:   outputUnion.Arguments,
				Name:        outputUnion.Name,
				ServerLabel: outputUnion.ServerLabel,
				Type:        constant.ValueOf[constant.McpCall](),
				Error:       outputUnion.Error,
				Output:      outputUnion.Output.OfString,
			}
			items = append(items, ToolCallItem{
				Agent:   agent,
				RawItem: ResponseOutputItemMcpCall(output),
				Type:    "tool_call_item",
			})
			toolsUsed = append(toolsUsed, "mcp")
		case "image_generation_call":
			output := responses.ResponseOutputItemImageGenerationCall{
				ID:     outputUnion.ID,
				Result: outputUnion.Result,
				Status: outputUnion.Status,
				Type:   constant.ValueOf[constant.ImageGenerationCall](),
			}
			items = append(items, ToolCallItem{
				Agent:   agent,
				RawItem: ResponseOutputItemImageGenerationCall(output),
				Type:    "tool_call_item",
			})
			toolsUsed = append(toolsUsed, "image_generation")
		case "code_interpreter_call":
			output := responses.ResponseCodeInterpreterToolCall{
				ID:          outputUnion.ID,
				Code:        outputUnion.Code,
				Outputs:     outputUnion.Outputs,
				Status:      responses.ResponseCodeInterpreterToolCallStatus(outputUnion.Status),
				Type:        constant.ValueOf[constant.CodeInterpreterCall](),
				ContainerID: outputUnion.ContainerID,
			}
			items = append(items, ToolCallItem{
				Agent:   agent,
				RawItem: ResponseCodeInterpreterToolCall(output),
				Type:    "tool_call_item",
			})
			toolsUsed = append(toolsUsed, "code_interpreter")
		case "shell_call":
			var rawCall map[string]any
			if outputUnion.RawJSON() != "" {
				if err := json.Unmarshal([]byte(outputUnion.RawJSON()), &rawCall); err != nil {
					rawCall = nil
				}
			}
			if rawCall == nil {
				rawCall = map[string]any{
					"type":        "shell_call",
					"id":          outputUnion.ID,
					"call_id":     outputUnion.CallID,
					"status":      outputUnion.Status,
					"action":      openaitypes.ResponseFunctionShellToolCallActionFromResponseOutputItemUnionAction(outputUnion.Action),
					"environment": outputUnion.Environment,
				}
			}
			delete(rawCall, "created_by")
			if _, ok := rawCall["type"]; !ok {
				rawCall["type"] = "shell_call"
			}
			items = append(items, ToolCallItem{
				Agent:   agent,
				RawItem: ShellToolCallRawItem(rawCall),
				Type:    "tool_call_item",
			})
			if shellTool == nil {
				toolsUsed = append(toolsUsed, "shell")
				AttachErrorToCurrentSpan(ctx, tracing.SpanError{Message: "Shell tool not found"})
				return nil, NewModelBehaviorError("model produced shell call without a shell tool")
			}
			toolsUsed = append(toolsUsed, shellTool.ToolName())
			environment := shellTool.Environment
			if normalized, err := normalizeShellToolEnvironment(environment); err == nil {
				environment = normalized
			}
			envType := "local"
			if environment != nil {
				if value, ok := environment["type"]; ok {
					if str, ok := coerceStringValue(value); ok && str != "" {
						envType = strings.ToLower(str)
					}
				}
			}
			if envType != "local" {
				continue
			}
			if shellTool.Executor == nil {
				AttachErrorToCurrentSpan(ctx, tracing.SpanError{Message: "Local shell executor not found"})
				return nil, NewModelBehaviorError("model produced local shell call without a local shell executor")
			}
			shellCalls = append(shellCalls, ToolRunShellCall{
				ToolCall:  rawCall,
				ShellTool: *shellTool,
			})
		case "shell_call_output":
			toolName := "shell"
			if shellTool != nil {
				toolName = shellTool.ToolName()
			}
			toolsUsed = append(toolsUsed, toolName)

			rawItem := ShellCallOutputRawItem{
				"type":    "shell_call_output",
				"call_id": outputUnion.CallID,
			}
			if outputUnion.ID != "" {
				rawItem["id"] = outputUnion.ID
			}
			if outputUnion.Status != "" {
				rawItem["status"] = outputUnion.Status
			}
			if outputUnion.JSON.MaxOutputLength.Valid() {
				rawItem["max_output_length"] = outputUnion.MaxOutputLength
			}
			if len(outputUnion.Output.OfResponseFunctionShellToolCallOutputOutputArray) > 0 {
				outputEntries := make([]any, 0, len(outputUnion.Output.OfResponseFunctionShellToolCallOutputOutputArray))
				for _, entry := range outputUnion.Output.OfResponseFunctionShellToolCallOutputOutputArray {
					payload := map[string]any{}
					if data, err := json.Marshal(entry); err == nil {
						if err := json.Unmarshal(data, &payload); err != nil {
							payload = map[string]any{}
						}
					}
					delete(payload, "created_by")
					outputEntries = append(outputEntries, payload)
				}
				rawItem["output"] = outputEntries
			} else if outputUnion.Output.OfString != "" {
				rawItem["output"] = outputUnion.Output.OfString
			}
			items = append(items, ToolCallOutputItem{
				Agent:   agent,
				RawItem: rawItem,
				Output:  rawItem["output"],
				Type:    "tool_call_output_item",
			})
		case "local_shell_call":
			output := responses.ResponseOutputItemLocalShellCall{
				ID:     outputUnion.ID,
				Action: openaitypes.ResponseOutputItemLocalShellCallActionFromResponseOutputItemUnionAction(outputUnion.Action),
				CallID: outputUnion.CallID,
				Status: outputUnion.Status,
				Type:   constant.ValueOf[constant.LocalShellCall](),
			}
			items = append(items, ToolCallItem{
				Agent:   agent,
				RawItem: ResponseOutputItemLocalShellCall(output),
				Type:    "tool_call_item",
			})
			if localShellTool != nil {
				toolsUsed = append(toolsUsed, "local_shell")
				localShellCalls = append(localShellCalls, ToolRunLocalShellCall{
					ToolCall:       output,
					LocalShellTool: *localShellTool,
				})
				continue
			}
			if shellTool != nil {
				toolsUsed = append(toolsUsed, shellTool.ToolName())
				shellCalls = append(shellCalls, ToolRunShellCall{
					ToolCall:  output,
					ShellTool: *shellTool,
				})
				continue
			}
			toolsUsed = append(toolsUsed, "local_shell")
			AttachErrorToCurrentSpan(ctx, tracing.SpanError{Message: "Local shell tool not found"})
			return nil, NewModelBehaviorError("model produced local shell call without a local shell tool")
		case "apply_patch_call":
			var rawCall map[string]any
			if outputUnion.RawJSON() != "" {
				if err := json.Unmarshal([]byte(outputUnion.RawJSON()), &rawCall); err != nil {
					rawCall = nil
				}
			}
			if rawCall == nil {
				rawCall = map[string]any{
					"type":      "apply_patch_call",
					"id":        outputUnion.ID,
					"call_id":   outputUnion.CallID,
					"status":    outputUnion.Status,
					"operation": outputUnion.Operation,
				}
			}
			delete(rawCall, "created_by")
			if _, ok := rawCall["type"]; !ok {
				rawCall["type"] = "apply_patch_call"
			}
			items = append(items, ToolCallItem{
				Agent:   agent,
				RawItem: ApplyPatchToolCallRawItem(rawCall),
				Type:    "tool_call_item",
			})
			if applyPatchTool == nil {
				AttachErrorToCurrentSpan(ctx, tracing.SpanError{Message: "Apply patch tool not found"})
				return nil, NewModelBehaviorError("model produced apply_patch call without an apply_patch tool")
			}
			toolsUsed = append(toolsUsed, applyPatchTool.ToolName())
			applyPatchCalls = append(applyPatchCalls, ToolRunApplyPatchCall{
				ToolCall:       rawCall,
				ApplyPatchTool: *applyPatchTool,
			})
		case "custom_tool_call":
			output := outputUnion.AsCustomToolCall()
			if isApplyPatchName(output.Name, applyPatchTool) {
				parsedOperation, err := parseApplyPatchCustomInput(output.Input)
				if err != nil {
					return nil, err
				}
				operationUnion, err := applyPatchOperationUnionFromMap(parsedOperation)
				if err != nil {
					return nil, err
				}
				pseudoCall := responses.ResponseApplyPatchToolCall{
					CallID:    output.CallID,
					Operation: operationUnion,
					Status:    responses.ResponseApplyPatchToolCallStatusInProgress,
					Type:      constant.ValueOf[constant.ApplyPatchCall](),
				}
				items = append(items, ToolCallItem{
					Agent:   agent,
					RawItem: ResponseApplyPatchToolCall(pseudoCall),
					Type:    "tool_call_item",
				})
				if applyPatchTool == nil {
					AttachErrorToCurrentSpan(ctx, tracing.SpanError{Message: "Apply patch tool not found"})
					return nil, NewModelBehaviorError("model produced apply_patch call without an apply_patch tool")
				}
				toolsUsed = append(toolsUsed, applyPatchTool.ToolName())
				applyPatchCalls = append(applyPatchCalls, ToolRunApplyPatchCall{
					ToolCall:       pseudoCall,
					ApplyPatchTool: *applyPatchTool,
				})
			} else {
				Logger().Warn(fmt.Sprintf("unexpected custom tool call %q, ignoring", output.Name))
			}
		case "function_call":
			var output responses.ResponseFunctionToolCall
			if outputUnion.RawJSON() != "" {
				output = outputUnion.AsFunctionCall()
			} else {
				output = responses.ResponseFunctionToolCall{
					Arguments: outputUnion.Arguments,
					CallID:    outputUnion.CallID,
					Name:      outputUnion.Name,
					Type:      constant.ValueOf[constant.FunctionCall](),
					ID:        outputUnion.ID,
					Status:    responses.ResponseFunctionToolCallStatus(outputUnion.Status),
				}
			}
			if output.Type == "" {
				output.Type = constant.ValueOf[constant.FunctionCall]()
			}

			if isApplyPatchName(output.Name, applyPatchTool) {
				if _, ok := functionMap[output.Name]; !ok {
					parsedOperation, err := parseApplyPatchFunctionArgs(output.Arguments)
					if err != nil {
						return nil, err
					}
					operationUnion, err := applyPatchOperationUnionFromMap(parsedOperation)
					if err != nil {
						return nil, err
					}
					pseudoCall := responses.ResponseApplyPatchToolCall{
						CallID:    output.CallID,
						Operation: operationUnion,
						Status:    responses.ResponseApplyPatchToolCallStatusInProgress,
						Type:      constant.ValueOf[constant.ApplyPatchCall](),
					}
					items = append(items, ToolCallItem{
						Agent:   agent,
						RawItem: ResponseApplyPatchToolCall(pseudoCall),
						Type:    "tool_call_item",
					})
					if applyPatchTool == nil {
						AttachErrorToCurrentSpan(ctx, tracing.SpanError{Message: "Apply patch tool not found"})
						return nil, NewModelBehaviorError("model produced apply_patch call without an apply_patch tool")
					}
					toolsUsed = append(toolsUsed, applyPatchTool.ToolName())
					applyPatchCalls = append(applyPatchCalls, ToolRunApplyPatchCall{
						ToolCall:       pseudoCall,
						ApplyPatchTool: *applyPatchTool,
					})
					continue
				}
			}

			toolsUsed = append(toolsUsed, output.Name)

			// Handoffs
			if handoff, ok := handoffMap[output.Name]; ok {
				items = append(items, HandoffCallItem{
					Agent:   agent,
					RawItem: output,
					Type:    "handoff_call_item",
				})
				runHandoffs = append(runHandoffs, ToolRunHandoff{
					Handoff:  handoff,
					ToolCall: ResponseFunctionToolCall(output),
				})
			} else { // Regular function tool call
				functionTool, ok := functionMap[output.Name]
				if !ok {
					AttachErrorToCurrentSpan(ctx, tracing.SpanError{
						Message: "Tool not found",
						Data:    map[string]any{"tool_name": output.Name},
					})
					return nil, ModelBehaviorErrorf("tool %s not found in agent %s", output.Name, agent.Name)
				}
				items = append(items, ToolCallItem{
					Agent:   agent,
					RawItem: ResponseFunctionToolCall(output),
					Type:    "tool_call_item",
				})
				functions = append(functions, ToolRunFunction{
					ToolCall:     ResponseFunctionToolCall(output),
					FunctionTool: functionTool,
				})
			}
		default:
			Logger().Warn(fmt.Sprintf("unexpected output type, ignoring %q", outputUnion.Type))
		}
	}

	return &ProcessedResponse{
		NewItems:            items,
		Handoffs:            runHandoffs,
		Functions:           functions,
		ComputerActions:     computerActions,
		LocalShellCalls:     localShellCalls,
		ShellCalls:          shellCalls,
		ApplyPatchCalls:     applyPatchCalls,
		Interruptions:       interruptions,
		ToolsUsed:           toolsUsed,
		MCPApprovalRequests: mcpApprovalRequests,
	}, nil
}

type FunctionToolResult struct {
	// The tool that was run.
	Tool FunctionTool

	// The output of the tool.
	Output any

	// The run item that was produced as a result of the tool call.
	RunItem RunItem
}

func (ri runImpl) ExecuteFunctionToolCalls(
	ctx context.Context,
	agent *Agent,
	toolRuns []ToolRunFunction,
	hooks RunHooks,
	contextWrapper *RunContextWrapper[any],
	config RunConfig,
) ([]FunctionToolResult, []ToolInputGuardrailResult, []ToolOutputGuardrailResult, []ToolApprovalItem, error) {
	if len(toolRuns) == 0 {
		return nil, nil, nil, nil, nil
	}
	if contextWrapper == nil {
		if value, ok := RunContextValueFromContext(ctx); ok {
			contextWrapper = NewRunContextWrapper[any](value)
		} else {
			contextWrapper = NewRunContextWrapper[any](nil)
		}
	}

	runSingleTool := func(
		ctx context.Context,
		funcTool FunctionTool,
		toolCall ResponseFunctionToolCall,
	) (any, []ToolInputGuardrailResult, []ToolOutputGuardrailResult, []ToolApprovalItem, error) {
		var result any
		var approvalItems []ToolApprovalItem
		var toolInputGuardrailResults []ToolInputGuardrailResult
		var toolOutputGuardrailResults []ToolOutputGuardrailResult

		traceIncludeSensitiveData := config.TraceIncludeSensitiveData.Or(defaultTraceIncludeSensitiveData())

		errorFn := DefaultToolErrorFunction // non-fatal
		if funcTool.FailureErrorFunction != nil {
			errorFn = *funcTool.FailureErrorFunction
		}

		err := tracing.FunctionSpan(
			ctx, tracing.FunctionSpanParams{Name: funcTool.Name},
			func(ctx context.Context, spanFn tracing.Span) (err error) {
				ctx = ContextWithToolData(ctx, toolCall.CallID, responses.ResponseFunctionToolCall(toolCall))
				toolContextData := ToolContextData{
					ToolName:      toolCall.Name,
					ToolCallID:    toolCall.CallID,
					ToolArguments: toolCall.Arguments,
				}
				if traceIncludeSensitiveData {
					spanFn.SpanData().(*tracing.FunctionSpanData).Input = toolCall.Arguments
				}

				defer func() {
					if err != nil {
						AttachErrorToCurrentSpan(ctx, tracing.SpanError{
							Message: "Error running tool",
							Data:    map[string]any{"tool_name": funcTool.Name, "error": err.Error()},
						})
					}
				}()

				var hooksErrors [2]error
				var toolError error

				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				defer cancel()

				if funcTool.NeedsApproval != nil {
					parsedArgs := map[string]any{}
					if toolCall.Arguments != "" {
						if err := json.Unmarshal([]byte(toolCall.Arguments), &parsedArgs); err != nil {
							parsedArgs = map[string]any{}
						}
					}
					needsApproval, err := funcTool.NeedsApproval.NeedsApproval(
						ctx,
						contextWrapper,
						funcTool,
						parsedArgs,
						toolCall.CallID,
					)
					if err != nil {
						return err
					}
					if needsApproval {
						pending := ToolApprovalItem{
							ToolName: funcTool.Name,
							RawItem:  responses.ResponseFunctionToolCall(toolCall),
						}
						approved, known := contextWrapper.GetApprovalStatus(
							funcTool.Name,
							toolCall.CallID,
							&pending,
						)
						if known && !approved {
							rejectionMessage := resolveApprovalRejectionMessage(
								contextWrapper,
								config,
								"function",
								funcTool.Name,
								toolCall.CallID,
							)
							AttachErrorToCurrentSpan(ctx, tracing.SpanError{
								Message: rejectionMessage,
								Data: map[string]any{
									"tool_name": funcTool.Name,
									"error": fmt.Sprintf(
										"Tool execution for %s was manually rejected by user.",
										toolCall.CallID,
									),
								},
							})
							result = rejectionMessage
							if traceIncludeSensitiveData {
								spanFn.SpanData().(*tracing.FunctionSpanData).Output = result
							}
							return nil
						}
						if !known {
							approvalItems = append(approvalItems, pending)
							return nil
						}
					}
				}

				rejectionMessage, err := ri.executeToolInputGuardrails(
					ctx,
					agent,
					funcTool,
					toolContextData,
					&toolInputGuardrailResults,
				)
				if err != nil {
					return err
				}
				if rejectionMessage != nil {
					result = *rejectionMessage
					if traceIncludeSensitiveData {
						spanFn.SpanData().(*tracing.FunctionSpanData).Output = result
					}
					return nil
				}

				var wg sync.WaitGroup

				wg.Add(1)
				go func() {
					defer wg.Done()
					err := hooks.OnToolStart(ctx, agent, funcTool)
					if err != nil {
						cancel()
						hooksErrors[0] = fmt.Errorf("RunHooks.OnToolStart failed: %w", err)
					}
				}()

				if agent.Hooks != nil {
					wg.Add(1)
					go func() {
						defer wg.Done()
						err := agent.Hooks.OnToolStart(ctx, agent, funcTool, toolCall.Arguments)
						if err != nil {
							cancel()
							hooksErrors[1] = fmt.Errorf("AgentHooks.OnToolStart failed: %w", err)
						}
					}()
				}

				wg.Add(1)
				go func() {
					defer wg.Done()
					result, toolError = funcTool.OnInvokeTool(ctx, toolCall.Arguments)
					if toolError != nil && errorFn == nil {
						cancel()
					}
				}()

				wg.Wait()

				if err = errors.Join(hooksErrors[:]...); err != nil {
					return err
				}

				if toolError != nil {
					if errorFn == nil {
						return fmt.Errorf("error running tool %s: %w", funcTool.Name, toolError)
					}
					result, err = errorFn(ctx, toolError)
					if err != nil {
						return fmt.Errorf("error running tool %s: %w", funcTool.Name, err)
					}
					AttachErrorToCurrentSpan(ctx, tracing.SpanError{
						Message: "Error running tool (non-fatal)",
						Data: map[string]any{
							"tool_name": funcTool.Name,
							"error":     toolError.Error(),
						},
					})
				}

				if agentResult, ok := asAgentToolRunResult(result); ok {
					if agentResult.Result != nil && len(agentResult.Result.Interruptions) > 0 {
						parentSig := agentToolSignatureFromResponseToolCall(&toolCall)
						approvalItems = approvalItems[:0]
						for _, interruption := range agentResult.Result.Interruptions {
							approvalItems = append(approvalItems, wrapAgentToolInterruption(interruption, parentSig))
						}
						return nil
					}
					result = agentResult.Output
				}

				result, err = ri.executeToolOutputGuardrails(
					ctx,
					agent,
					funcTool,
					toolContextData,
					result,
					&toolOutputGuardrailResults,
				)
				if err != nil {
					return err
				}

				wg.Add(1)
				go func() {
					defer wg.Done()
					err := hooks.OnToolEnd(ctx, agent, funcTool, result)
					if err != nil {
						cancel()
						hooksErrors[0] = fmt.Errorf("RunHooks.OnToolEnd failed: %w", err)
					}
				}()

				if agent.Hooks != nil {
					wg.Add(1)
					go func() {
						defer wg.Done()
						err := agent.Hooks.OnToolEnd(ctx, agent, funcTool, result)
						if err != nil {
							cancel()
							hooksErrors[1] = fmt.Errorf("AgentHooks.OnToolEnd failed: %w", err)
						}
					}()
				}

				wg.Wait()
				if err = errors.Join(hooksErrors[:]...); err != nil {
					return err
				}

				if traceIncludeSensitiveData {
					spanFn.SpanData().(*tracing.FunctionSpanData).Output = result
				}

				return nil
			})

		if err != nil {
			return nil, nil, nil, nil, err
		}
		return result, toolInputGuardrailResults, toolOutputGuardrailResults, approvalItems, nil
	}

	results := make([]any, len(toolRuns))
	perToolInputGuardrailResults := make([][]ToolInputGuardrailResult, len(toolRuns))
	perToolOutputGuardrailResults := make([][]ToolOutputGuardrailResult, len(toolRuns))
	perToolApprovalItems := make([][]ToolApprovalItem, len(toolRuns))
	resultErrors := make([]error, len(toolRuns))

	var cancel context.CancelFunc
	ctx, cancel = context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(len(toolRuns))

	for i, toolRun := range toolRuns {
		go func(i int, toolRun ToolRunFunction) {
			defer wg.Done()
			results[i],
				perToolInputGuardrailResults[i],
				perToolOutputGuardrailResults[i],
				perToolApprovalItems[i],
				resultErrors[i] = runSingleTool(ctx, toolRun.FunctionTool, toolRun.ToolCall)
			if resultErrors[i] != nil {
				cancel()
			}
		}(i, toolRun)
	}

	wg.Wait()
	if err := errors.Join(resultErrors...); err != nil {
		return nil, nil, nil, nil, err
	}

	functionToolResults := make([]FunctionToolResult, len(results))
	functionInterruptions := make([]ToolApprovalItem, 0)
	for i, result := range results {
		toolRun := toolRuns[i]
		if approvalItems := perToolApprovalItems[i]; len(approvalItems) > 0 {
			functionInterruptions = append(functionInterruptions, approvalItems...)
			functionToolResults[i] = FunctionToolResult{
				Tool:    toolRun.FunctionTool,
				Output:  nil,
				RunItem: nil,
			}
			continue
		}

		functionToolResults[i] = FunctionToolResult{
			Tool:   toolRun.FunctionTool,
			Output: result,
			RunItem: ToolCallOutputItem{
				Agent: agent,
				RawItem: ResponseInputItemFunctionCallOutputParam(
					ItemHelpers().ToolCallOutputItem(toolRun.ToolCall, result)),
				Output: result,
				Type:   "tool_call_output_item",
			},
		}
	}

	var toolInputGuardrailResults []ToolInputGuardrailResult
	var toolOutputGuardrailResults []ToolOutputGuardrailResult
	for i := range toolRuns {
		toolInputGuardrailResults = append(toolInputGuardrailResults, perToolInputGuardrailResults[i]...)
		toolOutputGuardrailResults = append(toolOutputGuardrailResults, perToolOutputGuardrailResults[i]...)
	}

	return functionToolResults, toolInputGuardrailResults, toolOutputGuardrailResults, functionInterruptions, nil
}

func asAgentToolRunResult(result any) (AgentToolRunResult, bool) {
	switch v := result.(type) {
	case AgentToolRunResult:
		return v, true
	case *AgentToolRunResult:
		if v == nil {
			return AgentToolRunResult{}, false
		}
		return *v, true
	default:
		return AgentToolRunResult{}, false
	}
}

func (runImpl) executeToolInputGuardrails(
	ctx context.Context,
	agent *Agent,
	funcTool FunctionTool,
	toolContextData ToolContextData,
	results *[]ToolInputGuardrailResult,
) (*string, error) {
	if len(funcTool.ToolInputGuardrails) == 0 {
		return nil, nil
	}

	for _, guardrail := range funcTool.ToolInputGuardrails {
		guardrailResult, err := guardrail.Run(ctx, ToolInputGuardrailData{
			Context: toolContextData,
			Agent:   agent,
		})
		if err != nil {
			return nil, err
		}

		*results = append(*results, guardrailResult)

		switch behaviorType := guardrailResult.Output.BehaviorType(); behaviorType {
		case ToolGuardrailBehaviorTypeAllow:
			// Continue to the next guardrail.
		case ToolGuardrailBehaviorTypeRaiseException:
			err := NewToolInputGuardrailTripwireTriggeredError(guardrail, guardrailResult.Output)
			return nil, err
		case ToolGuardrailBehaviorTypeRejectContent:
			message := guardrailResult.Output.BehaviorMessage()
			return &message, nil
		default:
			return nil, UserErrorf("unknown tool input guardrail behavior type %q", behaviorType)
		}
	}

	return nil, nil
}

func (runImpl) executeToolOutputGuardrails(
	ctx context.Context,
	agent *Agent,
	funcTool FunctionTool,
	toolContextData ToolContextData,
	realResult any,
	results *[]ToolOutputGuardrailResult,
) (any, error) {
	if len(funcTool.ToolOutputGuardrails) == 0 {
		return realResult, nil
	}

	finalResult := realResult
	for _, guardrail := range funcTool.ToolOutputGuardrails {
		guardrailResult, err := guardrail.Run(ctx, ToolOutputGuardrailData{
			ToolInputGuardrailData: ToolInputGuardrailData{
				Context: toolContextData,
				Agent:   agent,
			},
			Output: realResult,
		})
		if err != nil {
			return nil, err
		}

		*results = append(*results, guardrailResult)

		switch behaviorType := guardrailResult.Output.BehaviorType(); behaviorType {
		case ToolGuardrailBehaviorTypeAllow:
			// Continue to the next guardrail.
		case ToolGuardrailBehaviorTypeRaiseException:
			err := NewToolOutputGuardrailTripwireTriggeredError(guardrail, guardrailResult.Output)
			return nil, err
		case ToolGuardrailBehaviorTypeRejectContent:
			finalResult = guardrailResult.Output.BehaviorMessage()
			return finalResult, nil
		default:
			return nil, UserErrorf("unknown tool output guardrail behavior type %q", behaviorType)
		}
	}

	return finalResult, nil
}

func (runImpl) ExecuteLocalShellCalls(
	ctx context.Context,
	agent *Agent,
	calls []ToolRunLocalShellCall,
	hooks RunHooks,
) ([]RunItem, error) {
	results := make([]RunItem, len(calls))

	// Need to run these serially, because each call can affect the local shell state
	for i, call := range calls {
		result, err := LocalShellAction().Execute(ctx, agent, call, hooks)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}

	return results, nil
}

func (runImpl) ExecuteShellCalls(
	ctx context.Context,
	agent *Agent,
	calls []ToolRunShellCall,
	hooks RunHooks,
	contextWrapper *RunContextWrapper[any],
	config RunConfig,
) ([]RunItem, []ToolApprovalItem, error) {
	if len(calls) == 0 {
		return nil, nil, nil
	}
	if contextWrapper == nil {
		if value, ok := RunContextValueFromContext(ctx); ok {
			contextWrapper = NewRunContextWrapper[any](value)
		} else {
			contextWrapper = NewRunContextWrapper[any](nil)
		}
	}

	results := make([]RunItem, 0, len(calls))
	interruptions := make([]ToolApprovalItem, 0)

	for _, call := range calls {
		result, err := ShellAction().Execute(ctx, agent, call, hooks, contextWrapper, config)
		if err != nil {
			return nil, nil, err
		}
		switch v := result.(type) {
		case ToolCallOutputItem:
			results = append(results, v)
		case *ToolCallOutputItem:
			results = append(results, *v)
		case ToolApprovalItem:
			interruptions = append(interruptions, v)
		case *ToolApprovalItem:
			interruptions = append(interruptions, *v)
		default:
			return nil, nil, fmt.Errorf("unexpected shell action result type %T", result)
		}
	}

	return results, interruptions, nil
}

func (runImpl) ExecuteApplyPatchCalls(
	ctx context.Context,
	agent *Agent,
	calls []ToolRunApplyPatchCall,
	hooks RunHooks,
	contextWrapper *RunContextWrapper[any],
	config RunConfig,
) ([]RunItem, []ToolApprovalItem, error) {
	if len(calls) == 0 {
		return nil, nil, nil
	}
	if contextWrapper == nil {
		if value, ok := RunContextValueFromContext(ctx); ok {
			contextWrapper = NewRunContextWrapper[any](value)
		} else {
			contextWrapper = NewRunContextWrapper[any](nil)
		}
	}

	results := make([]RunItem, 0, len(calls))
	interruptions := make([]ToolApprovalItem, 0)

	for _, call := range calls {
		result, err := ApplyPatchAction().Execute(ctx, agent, call, hooks, contextWrapper, config)
		if err != nil {
			return nil, nil, err
		}
		switch v := result.(type) {
		case ToolCallOutputItem:
			results = append(results, v)
		case *ToolCallOutputItem:
			results = append(results, *v)
		case ToolApprovalItem:
			interruptions = append(interruptions, v)
		case *ToolApprovalItem:
			interruptions = append(interruptions, *v)
		default:
			return nil, nil, fmt.Errorf("unexpected apply_patch action result type %T", result)
		}
	}

	return results, interruptions, nil
}

func (runImpl) ExecuteComputerActions(
	ctx context.Context,
	agent *Agent,
	actions []ToolRunComputerAction,
	hooks RunHooks,
	contextWrapper *RunContextWrapper[any],
) ([]RunItem, error) {
	results := make([]RunItem, len(actions))

	// Need to run these serially, because each action can affect the computer state
	for i, action := range actions {
		if contextWrapper != nil {
			resolved, err := ResolveComputer(ctx, &action.ComputerTool, contextWrapper)
			if err != nil {
				return nil, err
			}
			action.ComputerTool.Computer = resolved
		}
		if action.ComputerTool.Computer == nil {
			return nil, NewUserError("computer tool has no resolved computer")
		}

		var acknowledged []responses.ResponseInputItemComputerCallOutputAcknowledgedSafetyCheckParam
		if len(action.ToolCall.PendingSafetyChecks) > 0 && action.ComputerTool.OnSafetyCheck != nil {
			for _, check := range action.ToolCall.PendingSafetyChecks {
				data := ComputerToolSafetyCheckData{
					Agent:       agent,
					ToolCall:    action.ToolCall,
					SafetyCheck: check,
				}
				ack, err := action.ComputerTool.OnSafetyCheck(ctx, data)
				if err != nil {
					return nil, err
				}
				if ack {
					acknowledged = append(acknowledged, responses.ResponseInputItemComputerCallOutputAcknowledgedSafetyCheckParam{
						ID:      check.ID,
						Code:    param.NewOpt(check.Code),
						Message: param.NewOpt(check.Message),
					})
				} else {
					return nil, NewUserError("computer tool safety check was not acknowledged")
				}
			}
		}

		result, err := ComputerAction().Execute(ctx, agent, action, hooks, acknowledged)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}

	return results, nil
}

func (runImpl) ExecuteHandoffs(
	ctx context.Context,
	agent *Agent,
	originalInput Input,
	preStepItems []RunItem,
	newStepItems []RunItem,
	newResponse ModelResponse,
	runHandoffs []ToolRunHandoff,
	hooks RunHooks,
	runConfig RunConfig,
	contextWrapper *RunContextWrapper[any],
) (*SingleStepResult, error) {
	// If there is more than one handoff, add tool responses that reject those handoffs
	multipleHandoffs := len(runHandoffs) > 1

	if multipleHandoffs {
		const outputMessage = "Multiple handoffs detected, ignoring this one."
		for _, handoff := range runHandoffs[1:] {
			newStepItems = append(newStepItems, ToolCallOutputItem{
				Agent: agent,
				RawItem: ResponseInputItemFunctionCallOutputParam(
					ItemHelpers().ToolCallOutputItem(handoff.ToolCall, outputMessage)),
				Output: outputMessage,
				Type:   "tool_call_output_item",
			})
		}
	}

	actualHandoff := runHandoffs[0]
	var handoff Handoff
	var newAgent *Agent

	err := tracing.HandoffSpan(
		ctx, tracing.HandoffSpanParams{FromAgent: agent.Name},
		func(ctx context.Context, spanHandoff tracing.Span) error {
			handoff = actualHandoff.Handoff
			var err error
			newAgent, err = handoff.OnInvokeHandoff(ctx, actualHandoff.ToolCall.Arguments)
			if err != nil {
				return fmt.Errorf("failed to invoke handoff: %w", err)
			}

			spanHandoff.SpanData().(*tracing.HandoffSpanData).ToAgent = newAgent.Name
			if multipleHandoffs {
				requestedAgents := make([]string, len(runHandoffs))
				for i, h := range runHandoffs {
					requestedAgents[i] = h.Handoff.AgentName
				}
				spanHandoff.SetError(tracing.SpanError{
					Message: "Multiple handoffs requested",
					Data:    map[string]any{"requested_agents": requestedAgents},
				})
			}

			return nil
		})
	if err != nil {
		return nil, err
	}

	// Append a tool output item for the handoff
	toolCallOutputItem := ItemHelpers().ToolCallOutputItem(
		actualHandoff.ToolCall,
		handoff.GetTransferMessage(newAgent),
	)
	newStepItems = append(newStepItems, HandoffOutputItem{
		Agent: agent,
		RawItem: TResponseInputItem{
			OfFunctionCallOutput: &toolCallOutputItem,
		},
		SourceAgent: agent,
		TargetAgent: newAgent,
		Type:        "handoff_output_item",
	})

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Execute handoff hooks
	var wg sync.WaitGroup
	var handoffErrors [2]error

	wg.Add(1)
	go func() {
		defer wg.Done()
		err := hooks.OnHandoff(childCtx, agent, newAgent)
		if err != nil {
			cancel()
			handoffErrors[0] = fmt.Errorf("RunHooks.OnHandoff failed: %w", err)
		}
	}()

	if agent.Hooks != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := agent.Hooks.OnHandoff(childCtx, newAgent, agent)
			if err != nil {
				cancel()
				handoffErrors[1] = fmt.Errorf("AgentHooks.OnHandoff failed: %w", err)
			}
		}()
	}

	wg.Wait()
	if err = errors.Join(handoffErrors[:]...); err != nil {
		return nil, err
	}

	// If there's an input filter, filter the input for the next agent
	inputFilter := handoff.InputFilter
	if inputFilter == nil {
		inputFilter = runConfig.HandoffInputFilter
	}
	shouldNestHistory := runConfig.NestHandoffHistory
	if handoff.NestHandoffHistory.Valid() {
		shouldNestHistory = handoff.NestHandoffHistory.Value
	}

	var sessionStepItems []RunItem
	if inputFilter != nil || shouldNestHistory {
		Logger().Debug("Filtering inputs for handoff")
		handoffInputData := HandoffInputData{
			InputHistory:    CopyInput(originalInput),
			PreHandoffItems: slices.Clone(preStepItems),
			NewItems:        slices.Clone(newStepItems),
			RunContext:      contextWrapper,
		}

		if inputFilter != nil {
			filtered, err := inputFilter(ctx, handoffInputData)
			if err != nil {
				return nil, fmt.Errorf("handoff input filter error: %w", err)
			}

			if filtered.InputHistory != nil {
				originalInput = CopyInput(filtered.InputHistory)
			} else {
				originalInput = InputItems{}
			}
			preStepItems = slices.Clone(filtered.PreHandoffItems)
			newStepItems = slices.Clone(filtered.NewItems)

			if filtered.InputItems != nil {
				sessionStepItems = slices.Clone(filtered.NewItems)
				newStepItems = slices.Clone(filtered.InputItems)
			}
		} else if shouldNestHistory {
			nested := NestHandoffHistory(handoffInputData, runConfig.HandoffHistoryMapper)
			if nested.InputHistory != nil {
				originalInput = CopyInput(nested.InputHistory)
			} else {
				originalInput = InputItems{}
			}
			preStepItems = slices.Clone(nested.PreHandoffItems)
			sessionStepItems = slices.Clone(nested.NewItems)
			if nested.InputItems != nil {
				newStepItems = slices.Clone(nested.InputItems)
			} else {
				newStepItems = slices.Clone(nested.NewItems)
			}
		}
	}

	return &SingleStepResult{
		OriginalInput:    originalInput,
		ModelResponse:    newResponse,
		PreStepItems:     preStepItems,
		NewStepItems:     newStepItems,
		SessionStepItems: sessionStepItems,
		NextStep:         NextStepHandoff{NewAgent: newAgent},
	}, nil
}

func (ri runImpl) ExecuteMCPApprovalRequests(
	ctx context.Context,
	agent *Agent,
	approvalRequests []ToolRunMCPApprovalRequest,
) ([]RunItem, error) {
	if len(approvalRequests) == 0 {
		return nil, nil
	}

	var cancel context.CancelFunc
	ctx, cancel = context.WithCancel(ctx)
	defer cancel()

	errs := make([]error, len(approvalRequests))
	results := make([]RunItem, len(approvalRequests))

	var wg sync.WaitGroup
	wg.Add(len(approvalRequests))

	for i, approvalRequest := range approvalRequests {
		go func() {
			defer wg.Done()
			results[i], errs[i] = ri.runSingleMCPApproval(ctx, agent, approvalRequest)
			if errs[i] != nil {
				cancel()
			}
		}()
	}

	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return results, nil
}

func (ri runImpl) runSingleMCPApproval(
	ctx context.Context,
	agent *Agent,
	approvalRequest ToolRunMCPApprovalRequest,
) (RunItem, error) {
	callback := approvalRequest.MCPTool.OnApprovalRequest
	if callback == nil {
		return nil, errors.New("callback is required for MCP approval requests")
	}

	result, err := callback(ctx, approvalRequest.RequestItem)
	if err != nil {
		return nil, err
	}

	var reason param.Opt[string]
	if !result.Approve && result.Reason != "" {
		reason = param.NewOpt(result.Reason)
	}

	rawItem := responses.ResponseInputItemMcpApprovalResponseParam{
		ApprovalRequestID: approvalRequest.RequestItem.ID,
		Approve:           result.Approve,
		ID:                param.Opt[string]{},
		Reason:            reason,
		Type:              constant.ValueOf[constant.McpApprovalResponse](),
	}

	return MCPApprovalResponseItem{
		Agent:   agent,
		RawItem: rawItem,
		Type:    "mcp_approval_response_item",
	}, nil
}

func (ri runImpl) ExecuteFinalOutput(
	ctx context.Context,
	agent *Agent,
	originalInput Input,
	newResponse ModelResponse,
	preStepItems []RunItem,
	newStepItems []RunItem,
	finalOutput any,
	hooks RunHooks,
) (*SingleStepResult, error) {
	// Run the onEnd hooks
	err := ri.RunFinalOutputHooks(ctx, agent, hooks, finalOutput)
	if err != nil {
		return nil, err
	}

	return &SingleStepResult{
		OriginalInput: originalInput,
		ModelResponse: newResponse,
		PreStepItems:  preStepItems,
		NewStepItems:  newStepItems,
		NextStep:      NextStepFinalOutput{Output: finalOutput},
	}, nil
}

func (ri runImpl) RunFinalOutputHooks(
	ctx context.Context,
	agent *Agent,
	hooks RunHooks,
	finalOutput any,
) error {
	var hooksErrors [2]error

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		err := hooks.OnAgentEnd(childCtx, agent, finalOutput)
		if err != nil {
			cancel()
			hooksErrors[0] = fmt.Errorf("RunHooks.OnAgentEnd failed: %w", err)
		}
	}()

	if agent.Hooks != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := agent.Hooks.OnEnd(childCtx, agent, finalOutput)
			if err != nil {
				cancel()
				hooksErrors[1] = fmt.Errorf("AgentHooks.OnEnd failed: %w", err)
			}
		}()
	}

	wg.Wait()
	return errors.Join(hooksErrors[:]...)
}

func (runImpl) RunSingleInputGuardrail(
	ctx context.Context,
	agent *Agent,
	guardrail InputGuardrail,
	input Input,
) (InputGuardrailResult, error) {
	var result InputGuardrailResult

	err := tracing.GuardrailSpan(
		ctx, tracing.GuardrailSpanParams{Name: guardrail.Name},
		func(ctx context.Context, spanGuardrail tracing.Span) error {
			var err error
			result, err = guardrail.Run(ctx, agent, input)
			if err != nil {
				return err
			}
			spanGuardrail.SpanData().(*tracing.GuardrailSpanData).Triggered = result.Output.TripwireTriggered
			return nil
		},
	)

	return result, err
}

func (runImpl) RunSingleOutputGuardrail(
	ctx context.Context,
	guardrail OutputGuardrail,
	agent *Agent,
	agentOutput any,
) (OutputGuardrailResult, error) {
	var result OutputGuardrailResult

	err := tracing.GuardrailSpan(
		ctx, tracing.GuardrailSpanParams{Name: guardrail.Name},
		func(ctx context.Context, spanGuardrail tracing.Span) error {
			var err error
			result, err = guardrail.Run(ctx, agent, agentOutput)
			if err != nil {
				return err
			}
			spanGuardrail.SpanData().(*tracing.GuardrailSpanData).Triggered = result.Output.TripwireTriggered
			return nil
		},
	)

	return result, err
}

func (runImpl) StreamStepResultToQueue(stepResult SingleStepResult, queue *asyncqueue.Queue[StreamEvent]) {
	for _, item := range stepResult.StepSessionItems() {
		var event StreamEvent

		switch item.(type) {
		case MessageOutputItem:
			event = NewRunItemStreamEvent(StreamEventMessageOutputCreated, item)
		case HandoffCallItem:
			event = NewRunItemStreamEvent(StreamEventHandoffRequested, item)
		case HandoffOutputItem:
			event = NewRunItemStreamEvent(StreamEventHandoffOccurred, item)
		case ToolCallItem:
			event = NewRunItemStreamEvent(StreamEventToolCalled, item)
		case ToolCallOutputItem:
			event = NewRunItemStreamEvent(StreamEventToolOutput, item)
		case ReasoningItem:
			event = NewRunItemStreamEvent(StreamEventReasoningItemCreated, item)
		case MCPApprovalRequestItem:
			event = NewRunItemStreamEvent(StreamEventMCPApprovalRequested, item)
		case MCPListToolsItem:
			event = NewRunItemStreamEvent(StreamEventMCPListTools, item)
		// TODO: is it right not to handle MCPApprovalResponseItem here?
		default:
			Logger().Warn(fmt.Sprintf("Unexpected RunItem type %T", item))
			event = nil
		}

		if event != nil {
			queue.Put(event)
		}
	}
}

// checkForFinalOutputFromTools determines if tool results should produce a final output.
// The returned ToolsToFinalOutputResult indicates whether final output is ready, and the output value.
func (runImpl) checkForFinalOutputFromTools(
	ctx context.Context,
	agent *Agent,
	toolResults []FunctionToolResult,
) (ToolsToFinalOutputResult, error) {
	if len(toolResults) == 0 {
		return notFinalOutput, nil
	}

	toolUseBehavior := agent.ToolUseBehavior
	if toolUseBehavior == nil {
		toolUseBehavior = RunLLMAgain()
	}

	return toolUseBehavior.ToolsToFinalOutput(ctx, toolResults)
}

// ManageTraceCtx creates a trace only if there is no current trace, and manages the trace lifecycle around the given function.
func ManageTraceCtx(ctx context.Context, params tracing.TraceParams, fn func(context.Context) error) error {
	if ct := tracing.GetCurrentTrace(ctx); ct != nil {
		return fn(ctx)
	}
	return tracing.RunTrace(ctx, params, func(ctx context.Context, _ tracing.Trace) error {
		return fn(ctx)
	})
}

type computerAction struct{}

func ComputerAction() computerAction { return computerAction{} }

func (ca computerAction) Execute(
	ctx context.Context,
	agent *Agent,
	action ToolRunComputerAction,
	hooks RunHooks,
	acknowledgedSafetyChecks []responses.ResponseInputItemComputerCallOutputAcknowledgedSafetyCheckParam,
) (RunItem, error) {
	var (
		hooksErrors [2]error
		toolError   error
		output      string
	)

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		err := hooks.OnToolStart(childCtx, agent, action.ComputerTool)
		if err != nil {
			cancel()
			hooksErrors[0] = fmt.Errorf("RunHooks.OnToolStart failed: %w", err)
		}
	}()

	if agent.Hooks != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := agent.Hooks.OnToolStart(childCtx, agent, action.ComputerTool, nil)
			if err != nil {
				cancel()
				hooksErrors[1] = fmt.Errorf("AgentHooks.OnToolStart failed: %w", err)
			}
		}()
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		output, toolError = ca.getScreenshot(ctx, action.ComputerTool.Computer, action.ToolCall)
		if toolError != nil {
			cancel()
		}
	}()

	wg.Wait()

	if err := errors.Join(hooksErrors[:]...); err != nil {
		return nil, err
	}
	if toolError != nil {
		return nil, fmt.Errorf("error running computer tool: %w", toolError)
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		err := hooks.OnToolEnd(childCtx, agent, action.ComputerTool, output)
		if err != nil {
			cancel()
			hooksErrors[0] = fmt.Errorf("RunHooks.OnToolEnd failed: %w", err)
		}
	}()

	if agent.Hooks != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := agent.Hooks.OnToolEnd(childCtx, agent, action.ComputerTool, output)
			if err != nil {
				cancel()
				hooksErrors[1] = fmt.Errorf("AgentHooks.OnToolEnd failed: %w", err)
			}
		}()
	}

	wg.Wait()
	if err := errors.Join(hooksErrors[:]...); err != nil {
		return nil, err
	}

	// TODO: don't send a screenshot every single time, use references
	imageURL := "data:image/png;base64," + output
	return ToolCallOutputItem{
		Agent: agent,
		RawItem: ResponseInputItemComputerCallOutputParam{
			CallID: action.ToolCall.CallID,
			Output: responses.ResponseComputerToolCallOutputScreenshotParam{
				ImageURL: param.NewOpt(imageURL),
				Type:     constant.ValueOf[constant.ComputerScreenshot](),
			},
			Type:                     constant.ValueOf[constant.ComputerCallOutput](),
			AcknowledgedSafetyChecks: acknowledgedSafetyChecks,
		},
		Output: imageURL,
		Type:   "tool_call_output_item",
	}, nil
}

func (computerAction) getScreenshot(
	ctx context.Context,
	comp computer.Computer,
	toolCall responses.ResponseComputerToolCall,
) (string, error) {
	action := toolCall.Action

	var err error
	switch action.Type {
	case "click":
		err = comp.Click(ctx, action.X, action.Y, computer.Button(action.Button))
	case "double_click":
		err = comp.DoubleClick(ctx, action.X, action.Y)
	case "drag":
		path := make([]computer.Position, len(action.Path))
		for i, p := range action.Path {
			path[i] = computer.Position{X: p.X, Y: p.Y}
		}
		err = comp.Drag(ctx, path)
	case "keypress":
		err = comp.Keypress(ctx, action.Keys)
	case "move":
		err = comp.Move(ctx, action.X, action.Y)
	case "screenshot":
		_, err = comp.Screenshot(ctx)
	case "scroll":
		err = comp.Scroll(ctx, action.X, action.Y, action.ScrollX, action.ScrollY)
	case "type":
		err = comp.Type(ctx, action.Text)
	case "wait":
		err = comp.Wait(ctx)
	default:
		err = fmt.Errorf("unexpected ResponseComputerToolCallActionUnion type %q", action.Type)
	}
	if err != nil {
		return "", err
	}

	return comp.Screenshot(ctx)
}

type localShellAction struct{}

func LocalShellAction() localShellAction { return localShellAction{} }

func (localShellAction) Execute(
	ctx context.Context,
	agent *Agent,
	call ToolRunLocalShellCall,
	hooks RunHooks,
) (RunItem, error) {
	var hooksErrors [2]error

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		err := hooks.OnToolStart(childCtx, agent, call.LocalShellTool)
		if err != nil {
			cancel()
			hooksErrors[0] = fmt.Errorf("RunHooks.OnToolStart failed: %w", err)
		}
	}()

	if agent.Hooks != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := agent.Hooks.OnToolStart(childCtx, agent, call.LocalShellTool, nil)
			if err != nil {
				cancel()
				hooksErrors[1] = fmt.Errorf("AgentHooks.OnToolStart failed: %w", err)
			}
		}()
	}

	wg.Wait()
	if err := errors.Join(hooksErrors[:]...); err != nil {
		return nil, err
	}

	// TODO: why this does not run concurrently with the hooks, as for other tools?
	request := LocalShellCommandRequest{Data: call.ToolCall}
	result, err := call.LocalShellTool.Executor(ctx, request)
	if err != nil {
		return nil, err
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		err := hooks.OnToolEnd(childCtx, agent, call.LocalShellTool, result)
		if err != nil {
			cancel()
			hooksErrors[0] = fmt.Errorf("RunHooks.OnToolEnd failed: %w", err)
		}
	}()

	if agent.Hooks != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := agent.Hooks.OnToolEnd(childCtx, agent, call.LocalShellTool, result)
			if err != nil {
				cancel()
				hooksErrors[1] = fmt.Errorf("AgentHooks.OnToolEnd failed: %w", err)
			}
		}()
	}

	wg.Wait()
	if err = errors.Join(hooksErrors[:]...); err != nil {
		return nil, err
	}

	return ToolCallOutputItem{
		Agent: agent,
		RawItem: ResponseInputItemLocalShellCallOutputParam{
			ID:     call.ToolCall.CallID,
			Output: result,
			Status: "",
			Type:   constant.ValueOf[constant.LocalShellCallOutput](),
		},
		Output: result,
		Type:   "tool_call_output_item",
	}, nil
}
