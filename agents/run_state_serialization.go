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
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/denggeng/openai-agents-go-plus/tracing"
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3/responses"
)

// RunStateSerializeOptions controls RunState serialization behavior.
type RunStateSerializeOptions struct {
	// ContextSerializer serializes non-mapping context values into a mapping.
	ContextSerializer func(any) (map[string]any, error)
	// StrictContext requires mapping contexts or an explicit serializer.
	StrictContext bool
	// IncludeTracingAPIKey includes tracing API keys in the trace payload when present.
	IncludeTracingAPIKey bool
}

// RunStateDeserializeOptions controls RunState deserialization behavior.
type RunStateDeserializeOptions struct {
	// ContextOverride replaces the serialized context value (mapping or custom type).
	ContextOverride any
	// ContextDeserializer rebuilds custom contexts from serialized mappings.
	ContextDeserializer func(map[string]any) (any, error)
	// StrictContext requires a deserializer or override when metadata indicates it is needed.
	StrictContext bool
}

// RunStateContextState stores serialized run-context metadata.
type RunStateContextState struct {
	Usage       *usage.Usage                       `json:"usage,omitempty"`
	Approvals   map[string]ToolApprovalRecordState `json:"approvals,omitempty"`
	Context     any                                `json:"context,omitempty"`
	ContextMeta *RunStateContextMeta               `json:"context_meta,omitempty"`
	ToolInput   any                                `json:"tool_input,omitempty"`
}

// RunStateContextMeta describes how the context was serialized.
type RunStateContextMeta struct {
	OriginalType         string `json:"original_type,omitempty"`
	SerializedVia        string `json:"serialized_via,omitempty"`
	RequiresDeserializer bool   `json:"requires_deserializer,omitempty"`
	Omitted              bool   `json:"omitted,omitempty"`
	ClassPath            string `json:"class_path,omitempty"`
}

// TraceState stores trace metadata for run-state persistence.
type TraceState struct {
	ObjectType    string         `json:"object,omitempty"`
	TraceID       string         `json:"id,omitempty"`
	WorkflowName  string         `json:"workflow_name,omitempty"`
	GroupID       string         `json:"group_id,omitempty"`
	Metadata      map[string]any `json:"metadata,omitempty"`
	TracingAPIKey string         `json:"tracing_api_key,omitempty"`
}

// RunStateCurrentStepState captures interruption state for run resumption.
type RunStateCurrentStepState struct {
	Type string                   `json:"type,omitempty"`
	Data *RunStateCurrentStepData `json:"data,omitempty"`
}

// RunStateCurrentStepData stores the interruptions data for current step.
type RunStateCurrentStepData struct {
	Interruptions []RunStateInterruptionState `json:"interruptions,omitempty"`
}

// RunStateInterruptionState stores serialized interruption details.
type RunStateInterruptionState struct {
	Type     string `json:"type,omitempty"`
	RawItem  any    `json:"raw_item,omitempty"`
	ToolName string `json:"tool_name,omitempty"`
}

// RunStateAgentState stores serialized agent metadata.
type RunStateAgentState struct {
	Name string `json:"name,omitempty"`
}

// RunStateRunItemState stores serialized run-item data.
type RunStateRunItemState struct {
	Type        string              `json:"type,omitempty"`
	RawItem     any                 `json:"raw_item,omitempty"`
	Agent       *RunStateAgentState `json:"agent,omitempty"`
	Output      any                 `json:"output,omitempty"`
	SourceAgent *RunStateAgentState `json:"source_agent,omitempty"`
	TargetAgent *RunStateAgentState `json:"target_agent,omitempty"`
	ToolName    string              `json:"tool_name,omitempty"`
	Description string              `json:"description,omitempty"`
}

// RunStateProcessedResponseState stores serialized processed-response data.
type RunStateProcessedResponseState struct {
	NewItems            []RunStateRunItemState      `json:"new_items,omitempty"`
	ToolsUsed           []string                    `json:"tools_used,omitempty"`
	Functions           []map[string]any            `json:"functions,omitempty"`
	ComputerActions     []map[string]any            `json:"computer_actions,omitempty"`
	LocalShellActions   []map[string]any            `json:"local_shell_actions,omitempty"`
	ShellActions        []map[string]any            `json:"shell_actions,omitempty"`
	ApplyPatchActions   []map[string]any            `json:"apply_patch_actions,omitempty"`
	Handoffs            []map[string]any            `json:"handoffs,omitempty"`
	MCPApprovalRequests []map[string]any            `json:"mcp_approval_requests,omitempty"`
	Interruptions       []RunStateInterruptionState `json:"interruptions,omitempty"`
}

type runStateWire struct {
	SchemaVersion                 string                             `json:"$schemaVersion"`
	CurrentTurn                   uint64                             `json:"current_turn"`
	MaxTurns                      uint64                             `json:"max_turns"`
	CurrentAgentName              string                             `json:"current_agent_name,omitempty"`
	CurrentTurnPersistedItemCount uint64                             `json:"current_turn_persisted_item_count,omitempty"`
	OriginalInput                 []TResponseInputItem               `json:"original_input,omitempty"`
	GeneratedItems                json.RawMessage                    `json:"generated_items,omitempty"`
	ModelResponses                []ModelResponse                    `json:"model_responses,omitempty"`
	SessionItems                  json.RawMessage                    `json:"session_items,omitempty"`
	LastProcessedResponse         json.RawMessage                    `json:"last_processed_response,omitempty"`
	PreviousResponseID            string                             `json:"previous_response_id,omitempty"`
	Interruptions                 []ToolApprovalItem                 `json:"interruptions,omitempty"`
	CurrentStep                   *RunStateCurrentStepState          `json:"current_step,omitempty"`
	InputGuardrailResults         []GuardrailResultState             `json:"input_guardrail_results,omitempty"`
	OutputGuardrailResults        []GuardrailResultState             `json:"output_guardrail_results,omitempty"`
	ToolInputGuardrailResults     []ToolGuardrailResultState         `json:"tool_input_guardrail_results,omitempty"`
	ToolOutputGuardrailResults    []ToolGuardrailResultState         `json:"tool_output_guardrail_results,omitempty"`
	ToolApprovals                 map[string]ToolApprovalRecordState `json:"tool_approvals,omitempty"`
	Context                       *RunStateContextState              `json:"context,omitempty"`
	ToolUseTracker                map[string][]string                `json:"tool_use_tracker,omitempty"`
	ConversationID                string                             `json:"conversation_id,omitempty"`
	AutoPreviousResponseID        bool                               `json:"auto_previous_response_id,omitempty"`
	Trace                         *TraceState                        `json:"trace,omitempty"`
}

// SetTrace captures trace metadata for serialization.
func (s *RunState) SetTrace(trace tracing.Trace) {
	if s == nil {
		return
	}
	s.Trace = TraceStateFromTrace(trace)
}

// TraceStateFromTrace builds a TraceState from an active trace.
func TraceStateFromTrace(trace tracing.Trace) *TraceState {
	if trace == nil {
		return nil
	}
	payload := trace.Export()
	return TraceStateFromMap(payload)
}

// TraceStateFromMap builds a TraceState from an exported trace payload.
func TraceStateFromMap(payload map[string]any) *TraceState {
	if len(payload) == 0 {
		return nil
	}
	state := &TraceState{}
	if v, ok := payload["object"].(string); ok {
		state.ObjectType = v
	}
	if v, ok := payload["id"].(string); ok {
		state.TraceID = v
	}
	if v, ok := payload["workflow_name"].(string); ok {
		state.WorkflowName = v
	}
	if v, ok := payload["group_id"].(string); ok {
		state.GroupID = v
	}
	if v, ok := payload["metadata"].(map[string]any); ok {
		state.Metadata = v
	}
	if v, ok := payload["tracing_api_key"].(string); ok {
		state.TracingAPIKey = v
	}
	return state
}

// ToJSONWithOptions encodes RunState to JSON bytes with options.
func (s RunState) ToJSONWithOptions(opts RunStateSerializeOptions) ([]byte, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}

	clone := s
	if clone.SchemaVersion == "" {
		clone.SchemaVersion = CurrentRunStateSchemaVersion
	}

	serializedContext, err := serializeRunStateContext(clone.Context, clone.ToolApprovals, opts)
	if err != nil {
		return nil, err
	}
	clone.Context = serializedContext

	if clone.CurrentStep == nil && len(clone.Interruptions) > 0 {
		clone.CurrentStep = buildCurrentStepState(clone.Interruptions)
	} else if clone.CurrentStep != nil {
		clone.CurrentStep = sanitizeCurrentStepState(clone.CurrentStep)
	}

	if !opts.IncludeTracingAPIKey && clone.Trace != nil {
		clean := *clone.Trace
		clean.TracingAPIKey = ""
		clone.Trace = &clean
	}

	payload, err := buildRunStateWire(clone)
	if err != nil {
		return nil, err
	}

	return json.Marshal(payload)
}

// ToJSONStringWithOptions encodes RunState to a JSON string with options.
func (s RunState) ToJSONStringWithOptions(opts RunStateSerializeOptions) (string, error) {
	b, err := s.ToJSONWithOptions(opts)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// RunStateFromJSONWithOptions decodes RunState from JSON bytes with options.
func RunStateFromJSONWithOptions(data []byte, opts RunStateDeserializeOptions) (RunState, error) {
	schemaVersion, err := extractSchemaVersion(data)
	if err != nil {
		return RunState{}, err
	}

	var wire runStateWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return RunState{}, err
	}

	state := RunState{
		SchemaVersion:                 wire.SchemaVersion,
		CurrentTurn:                   wire.CurrentTurn,
		MaxTurns:                      wire.MaxTurns,
		CurrentAgentName:              wire.CurrentAgentName,
		CurrentTurnPersistedItemCount: wire.CurrentTurnPersistedItemCount,
		OriginalInput:                 wire.OriginalInput,
		ModelResponses:                wire.ModelResponses,
		PreviousResponseID:            wire.PreviousResponseID,
		Interruptions:                 wire.Interruptions,
		CurrentStep:                   wire.CurrentStep,
		InputGuardrailResults:         wire.InputGuardrailResults,
		OutputGuardrailResults:        wire.OutputGuardrailResults,
		ToolInputGuardrailResults:     wire.ToolInputGuardrailResults,
		ToolOutputGuardrailResults:    wire.ToolOutputGuardrailResults,
		ToolApprovals:                 wire.ToolApprovals,
		Context:                       wire.Context,
		ToolUseTracker:                wire.ToolUseTracker,
		ConversationID:                wire.ConversationID,
		AutoPreviousResponseID:        wire.AutoPreviousResponseID,
		Trace:                         wire.Trace,
	}
	if state.SchemaVersion == "" {
		state.SchemaVersion = schemaVersion
	}
	if err := state.Validate(); err != nil {
		return RunState{}, err
	}

	parsedGenerated, generatedInput := parseRunStateGeneratedItems(wire.GeneratedItems)
	if len(parsedGenerated) > 0 {
		state.GeneratedRunItems = parsedGenerated
		state.GeneratedItems = runItemsToInputItems(parsedGenerated)
	} else if len(generatedInput) > 0 {
		state.GeneratedItems = generatedInput
	}

	lastProcessed := parseRunStateProcessedResponse(wire.LastProcessedResponse)
	if lastProcessed != nil {
		state.LastProcessedResponse = lastProcessed
	}

	if len(wire.SessionItems) > 0 {
		state.SessionItems = parseRunStateSessionItems(wire.SessionItems)
	} else if wire.SessionItems == nil {
		state.SessionItems = mergeGeneratedItemsWithProcessed(state.GeneratedRunItems, state.LastProcessedResponse)
	}

	if err := applyRunStateContextOverrides(&state, opts); err != nil {
		return RunState{}, err
	}

	applyRunStateCurrentStep(&state)

	return state, nil
}

// RunStateFromJSONStringWithOptions decodes RunState from a JSON string with options.
func RunStateFromJSONStringWithOptions(data string, opts RunStateDeserializeOptions) (RunState, error) {
	return RunStateFromJSONWithOptions([]byte(data), opts)
}

func buildRunStateWire(state RunState) (*runStateWire, error) {
	mergedGenerated := mergeGeneratedItemsWithProcessed(state.GeneratedRunItems, state.LastProcessedResponse)
	var generatedPayload json.RawMessage
	if len(mergedGenerated) > 0 {
		serialized := serializeRunItems(mergedGenerated)
		raw, err := json.Marshal(serialized)
		if err != nil {
			return nil, err
		}
		generatedPayload = raw
	} else if len(state.GeneratedItems) > 0 {
		raw, err := json.Marshal(state.GeneratedItems)
		if err != nil {
			return nil, err
		}
		generatedPayload = raw
	}

	var sessionPayload json.RawMessage
	if len(state.SessionItems) > 0 {
		serialized := serializeRunItems(state.SessionItems)
		raw, err := json.Marshal(serialized)
		if err != nil {
			return nil, err
		}
		sessionPayload = raw
	}

	var lastProcessedPayload json.RawMessage
	if state.LastProcessedResponse != nil {
		serialized := serializeProcessedResponse(*state.LastProcessedResponse)
		raw, err := json.Marshal(serialized)
		if err != nil {
			return nil, err
		}
		lastProcessedPayload = raw
	}

	return &runStateWire{
		SchemaVersion:                 state.SchemaVersion,
		CurrentTurn:                   state.CurrentTurn,
		MaxTurns:                      state.MaxTurns,
		CurrentAgentName:              state.CurrentAgentName,
		CurrentTurnPersistedItemCount: state.CurrentTurnPersistedItemCount,
		OriginalInput:                 state.OriginalInput,
		GeneratedItems:                generatedPayload,
		ModelResponses:                state.ModelResponses,
		SessionItems:                  sessionPayload,
		LastProcessedResponse:         lastProcessedPayload,
		PreviousResponseID:            state.PreviousResponseID,
		Interruptions:                 state.Interruptions,
		CurrentStep:                   state.CurrentStep,
		InputGuardrailResults:         state.InputGuardrailResults,
		OutputGuardrailResults:        state.OutputGuardrailResults,
		ToolInputGuardrailResults:     state.ToolInputGuardrailResults,
		ToolOutputGuardrailResults:    state.ToolOutputGuardrailResults,
		ToolApprovals:                 state.ToolApprovals,
		Context:                       state.Context,
		ToolUseTracker:                state.ToolUseTracker,
		ConversationID:                state.ConversationID,
		AutoPreviousResponseID:        state.AutoPreviousResponseID,
		Trace:                         state.Trace,
	}, nil
}

func parseRunStateGeneratedItems(raw json.RawMessage) ([]RunItem, []TResponseInputItem) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	if looksLikeRunItemStates(raw) {
		var states []RunStateRunItemState
		if err := json.Unmarshal(raw, &states); err != nil {
			return nil, nil
		}
		items := deserializeRunItems(states)
		return items, nil
	}

	var inputItems []TResponseInputItem
	if err := json.Unmarshal(raw, &inputItems); err != nil {
		return nil, nil
	}
	return nil, inputItems
}

func parseRunStateSessionItems(raw json.RawMessage) []RunItem {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var states []RunStateRunItemState
	if err := json.Unmarshal(raw, &states); err != nil {
		return nil
	}
	return deserializeRunItems(states)
}

func parseRunStateProcessedResponse(raw json.RawMessage) *ProcessedResponse {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var state RunStateProcessedResponseState
	if err := json.Unmarshal(raw, &state); err != nil {
		return nil
	}
	processed := deserializeProcessedResponse(state)
	return processed
}

// SetToolUseTrackerSnapshot stores a sanitized snapshot of tool usage.
func (s *RunState) SetToolUseTrackerSnapshot(snapshot any) {
	if s == nil {
		return
	}
	if snapshot == nil {
		s.ToolUseTracker = nil
		return
	}

	out := make(map[string][]string)
	switch typed := snapshot.(type) {
	case map[string][]string:
		for agent, tools := range typed {
			if agent == "" {
				continue
			}
			out[agent] = append([]string(nil), tools...)
		}
	case map[string]any:
		for agent, tools := range typed {
			if agent == "" {
				continue
			}
			out[agent] = filterStringSlice(tools)
		}
	case map[any]any:
		for agentKey, tools := range typed {
			agent, ok := agentKey.(string)
			if !ok || agent == "" {
				continue
			}
			out[agent] = filterStringSlice(tools)
		}
	default:
		// unsupported snapshot type; ignore
	}

	if len(out) == 0 {
		s.ToolUseTracker = nil
		return
	}
	s.ToolUseTracker = out
}

// GetToolUseTrackerSnapshot returns a defensive copy of tool usage snapshot.
func (s RunState) GetToolUseTrackerSnapshot() map[string][]string {
	if len(s.ToolUseTracker) == 0 {
		return map[string][]string{}
	}
	out := make(map[string][]string, len(s.ToolUseTracker))
	for agent, tools := range s.ToolUseTracker {
		out[agent] = append([]string(nil), tools...)
	}
	return out
}

func filterStringSlice(value any) []string {
	switch typed := value.(type) {
	case []string:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if item != "" {
				out = append(out, item)
			}
		}
		return out
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if s, ok := item.(string); ok && s != "" {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func looksLikeRunItemStates(raw json.RawMessage) bool {
	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil {
		return false
	}
	for _, item := range items {
		if item == nil {
			continue
		}
		if _, ok := item["raw_item"]; ok {
			return true
		}
		if _, ok := item["agent"]; ok {
			return true
		}
		if _, ok := item["source_agent"]; ok {
			return true
		}
		if _, ok := item["target_agent"]; ok {
			return true
		}
	}
	return false
}

func serializeRunItems(items []RunItem) []RunStateRunItemState {
	if len(items) == 0 {
		return nil
	}
	out := make([]RunStateRunItemState, 0, len(items))
	for _, item := range items {
		if item == nil {
			continue
		}
		state, ok := runItemToState(item)
		if !ok {
			continue
		}
		out = append(out, state)
	}
	return out
}

func deserializeRunItems(states []RunStateRunItemState) []RunItem {
	if len(states) == 0 {
		return nil
	}
	out := make([]RunItem, 0, len(states))
	for _, state := range states {
		item, ok := runItemFromState(state)
		if !ok {
			continue
		}
		out = append(out, item)
	}
	return out
}

func runItemsToInputItems(items []RunItem) []TResponseInputItem {
	if len(items) == 0 {
		return nil
	}
	out := make([]TResponseInputItem, 0, len(items))
	for _, item := range items {
		if item == nil {
			continue
		}
		out = append(out, item.ToInputItem())
	}
	return out
}

func runItemToState(item RunItem) (RunStateRunItemState, bool) {
	switch v := item.(type) {
	case MessageOutputItem:
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case *MessageOutputItem:
		if v == nil {
			return RunStateRunItemState{}, false
		}
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case ToolCallItem:
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case *ToolCallItem:
		if v == nil {
			return RunStateRunItemState{}, false
		}
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case ToolCallOutputItem:
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, v.Output, nil, nil, "", ""), true
	case *ToolCallOutputItem:
		if v == nil {
			return RunStateRunItemState{}, false
		}
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, v.Output, nil, nil, "", ""), true
	case HandoffCallItem:
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case *HandoffCallItem:
		if v == nil {
			return RunStateRunItemState{}, false
		}
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case HandoffOutputItem:
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, v.SourceAgent, v.TargetAgent, "", ""), true
	case *HandoffOutputItem:
		if v == nil {
			return RunStateRunItemState{}, false
		}
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, v.SourceAgent, v.TargetAgent, "", ""), true
	case ReasoningItem:
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case *ReasoningItem:
		if v == nil {
			return RunStateRunItemState{}, false
		}
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case CompactionItem:
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case *CompactionItem:
		if v == nil {
			return RunStateRunItemState{}, false
		}
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case MCPListToolsItem:
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case *MCPListToolsItem:
		if v == nil {
			return RunStateRunItemState{}, false
		}
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case MCPApprovalRequestItem:
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case *MCPApprovalRequestItem:
		if v == nil {
			return RunStateRunItemState{}, false
		}
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case MCPApprovalResponseItem:
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	case *MCPApprovalResponseItem:
		if v == nil {
			return RunStateRunItemState{}, false
		}
		return runItemStateFrom(v.Agent, v.RawItem, v.Type, nil, nil, nil, "", ""), true
	default:
		raw, itemType := runItemRawAndType(item)
		if raw == nil && itemType == "" {
			return RunStateRunItemState{}, false
		}
		state := RunStateRunItemState{
			Type:    itemType,
			RawItem: normalizeJSONValue(raw),
		}
		return state, true
	}
}

func runItemStateFrom(agent *Agent, raw any, itemType string, output any, source *Agent, target *Agent, toolName string, description string) RunStateRunItemState {
	state := RunStateRunItemState{
		Type:        itemType,
		RawItem:     normalizeJSONValue(raw),
		Agent:       agentStateFromAgent(agent),
		Output:      normalizeJSONValue(output),
		SourceAgent: agentStateFromAgent(source),
		TargetAgent: agentStateFromAgent(target),
		ToolName:    toolName,
		Description: description,
	}
	if output == nil {
		state.Output = nil
	}
	if toolName == "" {
		state.ToolName = ""
	}
	if description == "" {
		state.Description = ""
	}
	return state
}

func agentStateFromAgent(agent *Agent) *RunStateAgentState {
	if agent == nil || agent.Name == "" {
		return nil
	}
	return &RunStateAgentState{Name: agent.Name}
}

func runItemFromState(state RunStateRunItemState) (RunItem, bool) {
	agent := resolveAgentFromState(state.Agent)
	switch state.Type {
	case "message_output_item":
		raw, ok := decodeRawToResponseOutputMessage(state.RawItem)
		if !ok {
			return nil, false
		}
		return MessageOutputItem{Agent: agent, RawItem: raw, Type: state.Type}, true
	case "tool_call_item":
		raw, ok := decodeToolCallItemRaw(state.RawItem)
		if !ok {
			return nil, false
		}
		return ToolCallItem{Agent: agent, RawItem: raw, Type: state.Type}, true
	case "tool_call_output_item":
		raw, ok := decodeToolCallOutputRaw(state.RawItem)
		if !ok {
			return nil, false
		}
		return ToolCallOutputItem{Agent: agent, RawItem: raw, Output: state.Output, Type: state.Type}, true
	case "reasoning_item":
		raw, ok := decodeRawToResponseReasoningItem(state.RawItem)
		if !ok {
			return nil, false
		}
		return ReasoningItem{Agent: agent, RawItem: raw, Type: state.Type}, true
	case "handoff_call_item":
		raw, ok := decodeRawToResponseFunctionToolCall(state.RawItem)
		if !ok {
			return nil, false
		}
		return HandoffCallItem{Agent: agent, RawItem: responses.ResponseFunctionToolCall(raw), Type: state.Type}, true
	case "handoff_output_item":
		raw, ok := decodeRawToResponseInputItemUnion(state.RawItem)
		if !ok {
			raw = responses.ResponseInputItemUnionParam{}
		}
		source := resolveAgentFromState(state.SourceAgent)
		target := resolveAgentFromState(state.TargetAgent)
		return HandoffOutputItem{
			Agent:       agent,
			RawItem:     raw,
			SourceAgent: source,
			TargetAgent: target,
			Type:        state.Type,
		}, true
	case "compaction_item":
		raw := decodeRawToCompactionRawItem(state.RawItem)
		return CompactionItem{Agent: agent, RawItem: raw, Type: state.Type}, true
	case "mcp_list_tools_item":
		raw, ok := decodeRawToResponseOutputItemMcpListTools(state.RawItem)
		if !ok {
			return nil, false
		}
		return MCPListToolsItem{Agent: agent, RawItem: raw, Type: state.Type}, true
	case "mcp_approval_request_item":
		raw, ok := decodeRawToResponseOutputItemMcpApprovalRequest(state.RawItem)
		if !ok {
			return nil, false
		}
		return MCPApprovalRequestItem{Agent: agent, RawItem: raw, Type: state.Type}, true
	case "mcp_approval_response_item":
		raw, ok := decodeRawToResponseInputItemMcpApprovalResponse(state.RawItem)
		if !ok {
			return nil, false
		}
		return MCPApprovalResponseItem{Agent: agent, RawItem: raw, Type: state.Type}, true
	default:
		return nil, false
	}
}

func resolveAgentFromState(state *RunStateAgentState) *Agent {
	if state == nil || state.Name == "" {
		return nil
	}
	return &Agent{Name: state.Name}
}

func decodeRawToResponseOutputMessage(raw any) (responses.ResponseOutputMessage, bool) {
	if raw == nil {
		return responses.ResponseOutputMessage{}, false
	}
	if typed, ok := raw.(responses.ResponseOutputMessage); ok {
		return typed, true
	}
	var out responses.ResponseOutputMessage
	if !decodeRawToStruct(raw, &out) {
		return responses.ResponseOutputMessage{}, false
	}
	return out, true
}

func decodeRawToResponseReasoningItem(raw any) (responses.ResponseReasoningItem, bool) {
	if raw == nil {
		return responses.ResponseReasoningItem{}, false
	}
	if typed, ok := raw.(responses.ResponseReasoningItem); ok {
		return typed, true
	}
	var out responses.ResponseReasoningItem
	if !decodeRawToStruct(raw, &out) {
		return responses.ResponseReasoningItem{}, false
	}
	return out, true
}

func decodeRawToResponseFunctionToolCall(raw any) (ResponseFunctionToolCall, bool) {
	if raw == nil {
		return ResponseFunctionToolCall{}, false
	}
	if typed, ok := raw.(ResponseFunctionToolCall); ok {
		return typed, true
	}
	if typed, ok := raw.(responses.ResponseFunctionToolCall); ok {
		return ResponseFunctionToolCall(typed), true
	}
	var out responses.ResponseFunctionToolCall
	if !decodeRawToStruct(raw, &out) {
		return ResponseFunctionToolCall{}, false
	}
	return ResponseFunctionToolCall(out), true
}

func decodeRawToResponseOutputItemMcpListTools(raw any) (responses.ResponseOutputItemMcpListTools, bool) {
	if raw == nil {
		return responses.ResponseOutputItemMcpListTools{}, false
	}
	if typed, ok := raw.(responses.ResponseOutputItemMcpListTools); ok {
		return typed, true
	}
	var out responses.ResponseOutputItemMcpListTools
	if !decodeRawToStruct(raw, &out) {
		return responses.ResponseOutputItemMcpListTools{}, false
	}
	return out, true
}

func decodeRawToResponseOutputItemMcpApprovalRequest(raw any) (responses.ResponseOutputItemMcpApprovalRequest, bool) {
	if raw == nil {
		return responses.ResponseOutputItemMcpApprovalRequest{}, false
	}
	if typed, ok := raw.(responses.ResponseOutputItemMcpApprovalRequest); ok {
		return typed, true
	}
	var out responses.ResponseOutputItemMcpApprovalRequest
	if !decodeRawToStruct(raw, &out) {
		return responses.ResponseOutputItemMcpApprovalRequest{}, false
	}
	return out, true
}

func decodeRawToResponseInputItemMcpApprovalResponse(raw any) (responses.ResponseInputItemMcpApprovalResponseParam, bool) {
	if raw == nil {
		return responses.ResponseInputItemMcpApprovalResponseParam{}, false
	}
	if typed, ok := raw.(responses.ResponseInputItemMcpApprovalResponseParam); ok {
		return typed, true
	}
	var out responses.ResponseInputItemMcpApprovalResponseParam
	if !decodeRawToStruct(raw, &out) {
		return responses.ResponseInputItemMcpApprovalResponseParam{}, false
	}
	return out, true
}

func decodeRawToResponseInputItemUnion(raw any) (responses.ResponseInputItemUnionParam, bool) {
	if raw == nil {
		return responses.ResponseInputItemUnionParam{}, false
	}
	if typed, ok := raw.(responses.ResponseInputItemUnionParam); ok {
		return typed, true
	}
	var out responses.ResponseInputItemUnionParam
	if !decodeRawToStruct(raw, &out) {
		return responses.ResponseInputItemUnionParam{}, false
	}
	return out, true
}

func decodeRawToStruct(raw any, out any) bool {
	data, err := json.Marshal(raw)
	if err != nil {
		return false
	}
	if err := json.Unmarshal(data, out); err != nil {
		return false
	}
	return true
}

func decodeRawToCompactionRawItem(raw any) CompactionItemRawItem {
	if raw == nil {
		return CompactionItemRawItem{}
	}
	if typed, ok := raw.(CompactionItemRawItem); ok {
		return typed
	}
	if m, ok := raw.(map[string]any); ok {
		return CompactionItemRawItem(m)
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return CompactionItemRawItem{}
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return CompactionItemRawItem{}
	}
	return CompactionItemRawItem(out)
}

func decodeToolCallItemRaw(raw any) (ToolCallItemType, bool) {
	if raw == nil {
		return nil, false
	}
	if typed, ok := raw.(ToolCallItemType); ok {
		return typed, true
	}

	itemType := stringFieldFromRaw(raw, "type")
	switch itemType {
	case "function_call":
		if parsed, ok := decodeRawToResponseFunctionToolCall(raw); ok {
			return parsed, true
		}
	case "computer_call":
		var out responses.ResponseComputerToolCall
		if decodeRawToStruct(raw, &out) {
			return ResponseComputerToolCall(out), true
		}
	case "local_shell_call":
		var out responses.ResponseOutputItemLocalShellCall
		if decodeRawToStruct(raw, &out) {
			return ResponseOutputItemLocalShellCall(out), true
		}
	case "shell_call":
		var out responses.ResponseFunctionShellToolCall
		if decodeRawToStruct(raw, &out) {
			return ResponseFunctionShellToolCall(out), true
		}
		if m, ok := raw.(map[string]any); ok {
			return ShellToolCallRawItem(m), true
		}
	case "apply_patch_call":
		var out responses.ResponseApplyPatchToolCall
		if decodeRawToStruct(raw, &out) {
			return ResponseApplyPatchToolCall(out), true
		}
		if m, ok := raw.(map[string]any); ok {
			return ApplyPatchToolCallRawItem(m), true
		}
	case "mcp_call":
		var out responses.ResponseOutputItemMcpCall
		if decodeRawToStruct(raw, &out) {
			return ResponseOutputItemMcpCall(out), true
		}
	}

	if parsed, ok := decodeRawToResponseFunctionToolCall(raw); ok {
		return parsed, true
	}
	return nil, false
}

func decodeToolCallOutputRaw(raw any) (ToolCallOutputRawItem, bool) {
	if raw == nil {
		return nil, false
	}
	if typed, ok := raw.(ToolCallOutputRawItem); ok {
		return typed, true
	}
	itemType := stringFieldFromRaw(raw, "type")
	switch itemType {
	case "function_call_output":
		var out responses.ResponseInputItemFunctionCallOutputParam
		if decodeRawToStruct(raw, &out) {
			return ResponseInputItemFunctionCallOutputParam(out), true
		}
	case "computer_call_output":
		var out responses.ResponseInputItemComputerCallOutputParam
		if decodeRawToStruct(raw, &out) {
			return ResponseInputItemComputerCallOutputParam(out), true
		}
	case "local_shell_call_output":
		var out responses.ResponseInputItemLocalShellCallOutputParam
		if decodeRawToStruct(raw, &out) {
			return ResponseInputItemLocalShellCallOutputParam(out), true
		}
	case "shell_call_output":
		var out responses.ResponseInputItemShellCallOutputParam
		if decodeRawToStruct(raw, &out) {
			return ResponseInputItemShellCallOutputParam(out), true
		}
		if m, ok := raw.(map[string]any); ok {
			return ShellCallOutputRawItem(m), true
		}
	case "apply_patch_call_output":
		var out responses.ResponseInputItemApplyPatchCallOutputParam
		if decodeRawToStruct(raw, &out) {
			return ResponseInputItemApplyPatchCallOutputParam(out), true
		}
		if m, ok := raw.(map[string]any); ok {
			return ShellCallOutputRawItem(m), true
		}
	}

	var out responses.ResponseInputItemFunctionCallOutputParam
	if decodeRawToStruct(raw, &out) {
		return ResponseInputItemFunctionCallOutputParam(out), true
	}
	return nil, false
}

func mergeGeneratedItemsWithProcessed(generated []RunItem, processed *ProcessedResponse) []RunItem {
	if len(generated) == 0 {
		if processed == nil || len(processed.NewItems) == 0 {
			return nil
		}
		return slices.Clone(processed.NewItems)
	}

	out := slices.Clone(generated)
	if processed == nil || len(processed.NewItems) == 0 {
		return out
	}

	seenIDTypes := make(map[string]struct{})
	seenCallIDs := make(map[string]struct{})
	seenCallIDTypes := make(map[string]struct{})

	for _, item := range out {
		itemID, itemType, callID := runItemIDTypeCall(item)
		if itemID != "" && itemType != "" {
			seenIDTypes[itemID+"|"+itemType] = struct{}{}
		}
		if callID != "" && itemType != "" {
			seenCallIDTypes[callID+"|"+itemType] = struct{}{}
		} else if callID != "" {
			seenCallIDs[callID] = struct{}{}
		}
	}

	for _, item := range processed.NewItems {
		itemID, itemType, callID := runItemIDTypeCall(item)
		if callID != "" && itemType != "" {
			if _, exists := seenCallIDTypes[callID+"|"+itemType]; exists {
				continue
			}
		} else if callID != "" {
			if _, exists := seenCallIDs[callID]; exists {
				continue
			}
		}
		if itemID != "" && itemType != "" {
			if _, exists := seenIDTypes[itemID+"|"+itemType]; exists {
				continue
			}
		}

		if itemID != "" && itemType != "" {
			seenIDTypes[itemID+"|"+itemType] = struct{}{}
		}
		if callID != "" && itemType != "" {
			seenCallIDTypes[callID+"|"+itemType] = struct{}{}
		} else if callID != "" {
			seenCallIDs[callID] = struct{}{}
		}
		out = append(out, item)
	}

	return out
}

func runItemIDTypeCall(item RunItem) (string, string, string) {
	raw, itemType := runItemRawAndType(item)
	itemID := stringFieldFromRaw(raw, "id")
	rawType := stringFieldFromRaw(raw, "type")
	callID := stringFieldFromRaw(raw, "call_id")

	if itemID == "" {
		itemID = stringFieldFromRunItem(item, "ID")
	}
	if rawType == "" {
		rawType = stringFieldFromRunItem(item, "Type")
		if rawType == "" {
			rawType = itemType
		}
	}
	if callID == "" {
		callID = stringFieldFromRunItem(item, "CallID")
	}

	return itemID, rawType, callID
}

func stringFieldFromRunItem(item RunItem, field string) string {
	if item == nil {
		return ""
	}
	v := reflect.ValueOf(item)
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return ""
		}
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		return ""
	}
	f := v.FieldByName(field)
	if !f.IsValid() || f.Kind() != reflect.String {
		return ""
	}
	return f.String()
}

func serializeProcessedResponse(processed ProcessedResponse) RunStateProcessedResponseState {
	out := RunStateProcessedResponseState{
		NewItems:  serializeRunItems(processed.NewItems),
		ToolsUsed: append([]string(nil), processed.ToolsUsed...),
	}
	out.Functions = serializeToolActionsFunction(processed.Functions)
	out.ComputerActions = serializeToolActionsComputer(processed.ComputerActions)
	out.LocalShellActions = serializeToolActionsLocalShell(processed.LocalShellCalls)
	out.ShellActions = serializeToolActionsShell(processed.ShellCalls)
	out.ApplyPatchActions = serializeToolActionsApplyPatch(processed.ApplyPatchCalls)
	out.Handoffs = serializeToolActionsHandoff(processed.Handoffs)
	out.MCPApprovalRequests = serializeMCPApprovalRequests(processed.MCPApprovalRequests)
	out.Interruptions = serializeProcessedInterruptions(processed.Interruptions)
	return out
}

func serializeToolActionsFunction(actions []ToolRunFunction) []map[string]any {
	if len(actions) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(actions))
	for _, action := range actions {
		tool := action.FunctionTool
		meta := map[string]any{
			"name": tool.Name,
		}
		if tool.Description != "" {
			meta["description"] = tool.Description
		}
		if len(tool.ParamsJSONSchema) > 0 {
			meta["paramsJsonSchema"] = tool.ParamsJSONSchema
		}
		out = append(out, map[string]any{
			"tool_call": normalizeJSONValue(action.ToolCall),
			"tool":      meta,
		})
	}
	return out
}

func serializeToolActionsComputer(actions []ToolRunComputerAction) []map[string]any {
	if len(actions) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(actions))
	for _, action := range actions {
		meta := map[string]any{
			"name": action.ComputerTool.ToolName(),
		}
		out = append(out, map[string]any{
			"tool_call": normalizeJSONValue(action.ToolCall),
			"computer":  meta,
		})
	}
	return out
}

func serializeToolActionsLocalShell(actions []ToolRunLocalShellCall) []map[string]any {
	if len(actions) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(actions))
	for _, action := range actions {
		meta := map[string]any{
			"name": action.LocalShellTool.ToolName(),
		}
		out = append(out, map[string]any{
			"tool_call":   normalizeJSONValue(action.ToolCall),
			"local_shell": meta,
		})
	}
	return out
}

func serializeToolActionsShell(actions []ToolRunShellCall) []map[string]any {
	if len(actions) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(actions))
	for _, action := range actions {
		meta := map[string]any{
			"name": action.ShellTool.ToolName(),
		}
		out = append(out, map[string]any{
			"tool_call": normalizeJSONValue(action.ToolCall),
			"shell":     meta,
		})
	}
	return out
}

func serializeToolActionsApplyPatch(actions []ToolRunApplyPatchCall) []map[string]any {
	if len(actions) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(actions))
	for _, action := range actions {
		meta := map[string]any{
			"name": action.ApplyPatchTool.ToolName(),
		}
		out = append(out, map[string]any{
			"tool_call":   normalizeJSONValue(action.ToolCall),
			"apply_patch": meta,
		})
	}
	return out
}

func serializeToolActionsHandoff(actions []ToolRunHandoff) []map[string]any {
	if len(actions) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(actions))
	for _, action := range actions {
		toolName := action.Handoff.ToolName
		if toolName == "" && action.Handoff.AgentName != "" {
			toolName = DefaultHandoffToolName(&Agent{Name: action.Handoff.AgentName})
		}
		out = append(out, map[string]any{
			"tool_call": normalizeJSONValue(action.ToolCall),
			"handoff": map[string]any{
				"tool_name": toolName,
			},
		})
	}
	return out
}

func serializeMCPApprovalRequests(actions []ToolRunMCPApprovalRequest) []map[string]any {
	if len(actions) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(actions))
	for _, action := range actions {
		entry := map[string]any{
			"request_item": map[string]any{
				"raw_item": normalizeJSONValue(action.RequestItem),
			},
			"mcp_tool": serializeMCPTool(action.MCPTool),
		}
		out = append(out, entry)
	}
	return out
}

func serializeMCPTool(tool HostedMCPTool) map[string]any {
	meta := map[string]any{
		"name": tool.ToolName(),
	}
	if !reflect.ValueOf(tool.ToolConfig).IsZero() {
		meta["tool_config"] = normalizeJSONValue(tool.ToolConfig)
	}
	return meta
}

func serializeProcessedInterruptions(interruptions []ToolApprovalItem) []RunStateInterruptionState {
	if len(interruptions) == 0 {
		return nil
	}
	out := make([]RunStateInterruptionState, 0, len(interruptions))
	for _, item := range interruptions {
		state := RunStateInterruptionState{
			Type:    "tool_approval_item",
			RawItem: normalizeJSONValue(item.RawItem),
		}
		if item.ToolName != "" {
			state.ToolName = item.ToolName
		}
		out = append(out, state)
	}
	return out
}

func deserializeProcessedResponse(state RunStateProcessedResponseState) *ProcessedResponse {
	processed := &ProcessedResponse{
		NewItems:            deserializeRunItems(state.NewItems),
		ToolsUsed:           append([]string(nil), state.ToolsUsed...),
		Interruptions:       deserializeProcessedInterruptions(state.Interruptions),
		Functions:           deserializeFunctionActions(state.Functions),
		ComputerActions:     deserializeComputerActions(state.ComputerActions),
		LocalShellCalls:     deserializeLocalShellActions(state.LocalShellActions),
		ShellCalls:          deserializeShellActions(state.ShellActions),
		ApplyPatchCalls:     deserializeApplyPatchActions(state.ApplyPatchActions),
		Handoffs:            deserializeHandoffActions(state.Handoffs),
		MCPApprovalRequests: deserializeMCPApprovalRequests(state.MCPApprovalRequests),
	}
	return processed
}

func deserializeFunctionActions(entries []map[string]any) []ToolRunFunction {
	if len(entries) == 0 {
		return nil
	}
	out := make([]ToolRunFunction, 0, len(entries))
	for _, entry := range entries {
		toolName, description, schema := extractToolMetadata(entry, "tool")
		if toolName == "" {
			continue
		}
		tool := FunctionTool{Name: toolName, Description: description, ParamsJSONSchema: schema}
		if call, ok := decodeRawToResponseFunctionToolCall(entry["tool_call"]); ok {
			out = append(out, ToolRunFunction{ToolCall: call, FunctionTool: tool})
		}
	}
	return out
}

func deserializeComputerActions(entries []map[string]any) []ToolRunComputerAction {
	if len(entries) == 0 {
		return nil
	}
	out := make([]ToolRunComputerAction, 0, len(entries))
	for _, entry := range entries {
		var call responses.ResponseComputerToolCall
		if !decodeRawToStruct(entry["tool_call"], &call) {
			continue
		}
		out = append(out, ToolRunComputerAction{ToolCall: call, ComputerTool: ComputerTool{}})
	}
	return out
}

func deserializeLocalShellActions(entries []map[string]any) []ToolRunLocalShellCall {
	if len(entries) == 0 {
		return nil
	}
	out := make([]ToolRunLocalShellCall, 0, len(entries))
	for _, entry := range entries {
		var call responses.ResponseOutputItemLocalShellCall
		if !decodeRawToStruct(entry["tool_call"], &call) {
			continue
		}
		out = append(out, ToolRunLocalShellCall{ToolCall: call, LocalShellTool: LocalShellTool{}})
	}
	return out
}

func deserializeShellActions(entries []map[string]any) []ToolRunShellCall {
	if len(entries) == 0 {
		return nil
	}
	out := make([]ToolRunShellCall, 0, len(entries))
	for _, entry := range entries {
		call := entry["tool_call"]
		var parsed responses.ResponseFunctionShellToolCall
		if decodeRawToStruct(call, &parsed) {
			call = responses.ResponseFunctionShellToolCall(parsed)
		}
		toolName, _, _ := extractToolMetadata(entry, "shell")
		tool := ShellTool{Name: toolName}
		out = append(out, ToolRunShellCall{ToolCall: call, ShellTool: tool})
	}
	return out
}

func deserializeApplyPatchActions(entries []map[string]any) []ToolRunApplyPatchCall {
	if len(entries) == 0 {
		return nil
	}
	out := make([]ToolRunApplyPatchCall, 0, len(entries))
	for _, entry := range entries {
		call := entry["tool_call"]
		var parsed responses.ResponseApplyPatchToolCall
		if decodeRawToStruct(call, &parsed) {
			call = responses.ResponseApplyPatchToolCall(parsed)
		}
		toolName, _, _ := extractToolMetadata(entry, "apply_patch")
		tool := ApplyPatchTool{Name: toolName}
		out = append(out, ToolRunApplyPatchCall{ToolCall: call, ApplyPatchTool: tool})
	}
	return out
}

func deserializeHandoffActions(entries []map[string]any) []ToolRunHandoff {
	if len(entries) == 0 {
		return nil
	}
	out := make([]ToolRunHandoff, 0, len(entries))
	for _, entry := range entries {
		handoffMeta, ok := entry["handoff"].(map[string]any)
		if !ok {
			continue
		}
		toolName, _ := handoffMeta["tool_name"].(string)
		if toolName == "" {
			continue
		}
		handoff := Handoff{ToolName: toolName}
		call, ok := decodeRawToResponseFunctionToolCall(entry["tool_call"])
		if !ok {
			continue
		}
		out = append(out, ToolRunHandoff{ToolCall: call, Handoff: handoff})
	}
	return out
}

func deserializeMCPApprovalRequests(entries []map[string]any) []ToolRunMCPApprovalRequest {
	if len(entries) == 0 {
		return nil
	}
	out := make([]ToolRunMCPApprovalRequest, 0, len(entries))
	for _, entry := range entries {
		requestEntry, ok := entry["request_item"].(map[string]any)
		if !ok {
			continue
		}
		rawItem := requestEntry["raw_item"]
		requestItem, ok := decodeRawToResponseOutputItemMcpApprovalRequest(rawItem)
		if !ok {
			continue
		}
		mcpTool := HostedMCPTool{}
		out = append(out, ToolRunMCPApprovalRequest{RequestItem: requestItem, MCPTool: mcpTool})
	}
	return out
}

func deserializeProcessedInterruptions(entries []RunStateInterruptionState) []ToolApprovalItem {
	if len(entries) == 0 {
		return nil
	}
	out := make([]ToolApprovalItem, 0, len(entries))
	for _, entry := range entries {
		out = append(out, ToolApprovalItem{
			ToolName: entry.ToolName,
			RawItem:  entry.RawItem,
		})
	}
	return out
}

func extractToolMetadata(entry map[string]any, key string) (string, string, map[string]any) {
	toolEntry, ok := entry[key].(map[string]any)
	if !ok {
		return "", "", nil
	}
	name, _ := toolEntry["name"].(string)
	desc, _ := toolEntry["description"].(string)
	schema := map[string]any{}
	if rawSchema, ok := toolEntry["paramsJsonSchema"].(map[string]any); ok {
		schema = rawSchema
	}
	return name, desc, schema
}

func extractSchemaVersionFromRaw(raw map[string]json.RawMessage) (string, error) {
	versionRaw, ok := raw["$schemaVersion"]
	if !ok {
		return "", UserErrorf("run state is missing schema version")
	}
	var version string
	if err := json.Unmarshal(versionRaw, &version); err != nil || version == "" {
		return "", UserErrorf("run state is missing schema version")
	}
	return version, nil
}

func extractSchemaVersion(data []byte) (string, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return "", err
	}
	return extractSchemaVersionFromRaw(raw)
}

func serializeRunStateContext(
	contextState *RunStateContextState,
	approvals map[string]ToolApprovalRecordState,
	opts RunStateSerializeOptions,
) (*RunStateContextState, error) {
	if contextState == nil && len(approvals) == 0 {
		return nil, nil
	}

	out := &RunStateContextState{}
	if contextState != nil {
		out.Usage = cloneUsage(contextState.Usage)
		out.ToolInput = normalizeJSONValue(contextState.ToolInput)
		out.Approvals = cloneToolApprovalStates(contextState.Approvals)
		out.Context = contextState.Context
	}
	if len(out.Approvals) == 0 && len(approvals) > 0 {
		out.Approvals = cloneToolApprovalStates(approvals)
	}
	if out.Usage == nil {
		out.Usage = usage.NewUsage()
	}

	serializedContext, meta, err := serializeContextValue(out.Context, opts)
	if err != nil {
		return nil, err
	}
	out.Context = serializedContext
	out.ContextMeta = meta

	return out, nil
}

func applyRunStateContextOverrides(state *RunState, opts RunStateDeserializeOptions) error {
	if state == nil {
		return nil
	}

	if state.Context == nil && opts.ContextOverride == nil {
		return nil
	}

	if state.Context == nil {
		state.Context = &RunStateContextState{}
	}

	if state.Context.Usage == nil {
		state.Context.Usage = usage.NewUsage()
	}

	if len(state.ToolApprovals) == 0 && len(state.Context.Approvals) > 0 {
		state.ToolApprovals = cloneToolApprovalStates(state.Context.Approvals)
	}

	if opts.ContextOverride != nil {
		if wrapper, ok := opts.ContextOverride.(*RunContextWrapper[any]); ok {
			state.Context.Context = wrapper.Context
			state.Context.Usage = cloneUsage(wrapper.Usage)
			state.Context.Approvals = wrapper.SerializeApprovals()
			state.Context.ToolInput = wrapper.ToolInput
		} else {
			state.Context.Context = opts.ContextOverride
			state.Context.ToolInput = nil
		}
		return nil
	}

	meta := state.Context.ContextMeta
	if opts.ContextDeserializer == nil && contextMetaRequiresDeserializer(meta) {
		warn := contextMetaWarningMessage(meta)
		if opts.StrictContext {
			return UserErrorf("%s", warn)
		}
		Logger().Warn(warn)
	}

	if opts.ContextDeserializer != nil && state.Context.Context != nil {
		ctxMap, ok := state.Context.Context.(map[string]any)
		if !ok {
			return UserErrorf("serialized run state context must be a mapping to use context_deserializer")
		}
		rebuilt, err := opts.ContextDeserializer(ctxMap)
		if err != nil {
			return UserErrorf("context deserializer failed while rebuilding RunState context")
		}
		state.Context.Context = rebuilt
	}

	return nil
}

func serializeContextValue(value any, opts RunStateSerializeOptions) (any, *RunStateContextMeta, error) {
	if value == nil {
		return nil, buildContextMeta(nil, "none", false, false), nil
	}

	if mapping, ok := normalizeMapping(value); ok {
		return mapping, buildContextMeta(value, "mapping", false, false), nil
	}

	if opts.ContextSerializer != nil {
		serialized, err := opts.ContextSerializer(value)
		if err != nil {
			return nil, nil, UserErrorf("context serializer failed while serializing RunState context")
		}
		if serialized == nil {
			serialized = map[string]any{}
		}
		return serialized, buildContextMeta(value, "context_serializer", true, false), nil
	}

	if opts.StrictContext {
		return nil, nil, UserErrorf(
			"RunState serialization requires context to be a mapping when strict_context is true. " +
				"Provide context_serializer to serialize custom contexts.",
		)
	}

	serialized, ok := marshalToMapping(value)
	if ok {
		Logger().Warn(
			"RunState context was serialized from a custom type; provide context_deserializer " +
				"or context_override to restore it.",
		)
		return serialized, buildContextMeta(value, "json", true, false), nil
	}

	Logger().Warn(fmt.Sprintf(
		"RunState context of type %s is not serializable; storing empty context. "+
			"Provide context_serializer to preserve it.",
		contextTypeLabel(value),
	))
	return map[string]any{}, buildContextMeta(value, "omitted", true, true), nil
}

func normalizeMapping(value any) (map[string]any, bool) {
	switch v := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(v))
		for key, val := range v {
			out[key] = val
		}
		return out, true
	case map[string]string:
		out := make(map[string]any, len(v))
		for key, val := range v {
			out[key] = val
		}
		return out, true
	default:
		return nil, false
	}
}

func marshalToMapping(value any) (map[string]any, bool) {
	b, err := json.Marshal(value)
	if err != nil {
		return nil, false
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, false
	}
	return out, true
}

func normalizeJSONValue(value any) any {
	if value == nil {
		return nil
	}
	b, err := json.Marshal(value)
	if err != nil {
		return value
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		return value
	}
	return out
}

func cloneUsage(u *usage.Usage) *usage.Usage {
	if u == nil {
		return nil
	}
	copied := *u
	return &copied
}

func buildContextMeta(value any, serializedVia string, requiresDeserializer bool, omitted bool) *RunStateContextMeta {
	meta := &RunStateContextMeta{
		OriginalType:         contextTypeLabel(value),
		SerializedVia:        serializedVia,
		RequiresDeserializer: requiresDeserializer,
		Omitted:              omitted,
	}
	if classPath := contextClassPath(value); classPath != "" && meta.OriginalType != "mapping" && meta.OriginalType != "none" {
		meta.ClassPath = classPath
	}
	return meta
}

func contextTypeLabel(value any) string {
	if value == nil {
		return "none"
	}
	switch value.(type) {
	case map[string]any, map[string]string:
		return "mapping"
	}
	rt := reflect.TypeOf(value)
	if rt.Kind() == reflect.Pointer {
		rt = rt.Elem()
	}
	if rt.Kind() == reflect.Struct {
		return "struct"
	}
	return "custom"
}

func contextClassPath(value any) string {
	if value == nil {
		return ""
	}
	rt := reflect.TypeOf(value)
	if rt.Kind() == reflect.Pointer {
		rt = rt.Elem()
	}
	if rt.Kind() != reflect.Struct || rt.Name() == "" {
		return ""
	}
	if rt.PkgPath() == "" {
		return rt.Name()
	}
	return fmt.Sprintf("%s.%s", rt.PkgPath(), rt.Name())
}

func contextMetaRequiresDeserializer(meta *RunStateContextMeta) bool {
	if meta == nil {
		return false
	}
	if meta.Omitted {
		return true
	}
	return meta.RequiresDeserializer
}

func contextMetaWarningMessage(meta *RunStateContextMeta) string {
	if meta == nil {
		return "RunState context was serialized from a custom type; provide context_deserializer or context_override to restore it."
	}
	typeLabel := meta.OriginalType
	if meta.ClassPath != "" {
		typeLabel = fmt.Sprintf("%s (%s)", typeLabel, meta.ClassPath)
	}
	if meta.Omitted {
		return fmt.Sprintf(
			"RunState context was omitted during serialization for %s; provide context_override to supply it.",
			strings.TrimSpace(typeLabel),
		)
	}
	return fmt.Sprintf(
		"RunState context was serialized from %s; provide context_deserializer or context_override to restore it.",
		strings.TrimSpace(typeLabel),
	)
}

func buildCurrentStepState(interruptions []ToolApprovalItem) *RunStateCurrentStepState {
	if len(interruptions) == 0 {
		return nil
	}
	out := make([]RunStateInterruptionState, 0, len(interruptions))
	for _, item := range interruptions {
		state := RunStateInterruptionState{
			Type:    "tool_approval_item",
			RawItem: normalizeJSONValue(item.RawItem),
		}
		if item.ToolName != "" {
			state.ToolName = item.ToolName
		}
		out = append(out, state)
	}
	return &RunStateCurrentStepState{
		Type: "next_step_interruption",
		Data: &RunStateCurrentStepData{Interruptions: out},
	}
}

func sanitizeCurrentStepState(state *RunStateCurrentStepState) *RunStateCurrentStepState {
	if state == nil {
		return nil
	}
	if state.Type != "next_step_interruption" {
		return state
	}
	if state.Data == nil || len(state.Data.Interruptions) == 0 {
		return &RunStateCurrentStepState{Type: state.Type}
	}
	out := make([]RunStateInterruptionState, 0, len(state.Data.Interruptions))
	for _, item := range state.Data.Interruptions {
		out = append(out, RunStateInterruptionState{
			Type:     item.Type,
			RawItem:  normalizeJSONValue(item.RawItem),
			ToolName: item.ToolName,
		})
	}
	return &RunStateCurrentStepState{
		Type: state.Type,
		Data: &RunStateCurrentStepData{Interruptions: out},
	}
}

func applyRunStateCurrentStep(state *RunState) {
	if state == nil || state.CurrentStep == nil {
		return
	}
	if state.CurrentStep.Type != "next_step_interruption" {
		return
	}
	if len(state.Interruptions) > 0 {
		return
	}
	if state.CurrentStep.Data == nil {
		return
	}
	for _, interruption := range state.CurrentStep.Data.Interruptions {
		state.Interruptions = append(state.Interruptions, ToolApprovalItem{
			ToolName: interruption.ToolName,
			RawItem:  interruption.RawItem,
		})
	}
}
