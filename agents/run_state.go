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
	"encoding/json"
	"fmt"
	"slices"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

type toolApprovalStateSerializer interface {
	SerializeApprovals() map[string]ToolApprovalRecordState
}

type toolApprovalStateRebuilder interface {
	RebuildApprovals(map[string]ToolApprovalRecordState)
}

// CurrentRunStateSchemaVersion is the serialization schema version for RunState.
const CurrentRunStateSchemaVersion = "1.0"

// GuardrailFunctionOutputState is a JSON-friendly representation of GuardrailFunctionOutput.
type GuardrailFunctionOutputState struct {
	OutputInfo        any  `json:"output_info,omitempty"`
	TripwireTriggered bool `json:"tripwire_triggered"`
}

// GuardrailResultState is a JSON-friendly representation of an input/output guardrail result.
type GuardrailResultState struct {
	Name   string                       `json:"name"`
	Output GuardrailFunctionOutputState `json:"output"`
}

// ToolGuardrailFunctionOutputState is a JSON-friendly representation of ToolGuardrailFunctionOutput.
type ToolGuardrailFunctionOutputState struct {
	OutputInfo any                        `json:"output_info,omitempty"`
	Behavior   ToolGuardrailBehaviorState `json:"behavior"`
}

// ToolGuardrailBehaviorState is a JSON-friendly representation of ToolGuardrailBehavior.
type ToolGuardrailBehaviorState struct {
	Type    ToolGuardrailBehaviorType `json:"type"`
	Message string                    `json:"message,omitempty"`
}

// ToolGuardrailResultState is a JSON-friendly representation of a tool guardrail result.
type ToolGuardrailResultState struct {
	Name   string                           `json:"name"`
	Output ToolGuardrailFunctionOutputState `json:"output"`
}

// RunState is a serializable snapshot for basic run resumption.
type RunState struct {
	SchemaVersion string `json:"$schemaVersion"`

	CurrentTurn      uint64 `json:"current_turn"`
	MaxTurns         uint64 `json:"max_turns"`
	CurrentAgentName string `json:"current_agent_name,omitempty"`

	OriginalInput  []TResponseInputItem `json:"original_input,omitempty"`
	GeneratedItems []TResponseInputItem `json:"generated_items,omitempty"`
	ModelResponses []ModelResponse      `json:"model_responses,omitempty"`

	PreviousResponseID string             `json:"previous_response_id,omitempty"`
	Interruptions      []ToolApprovalItem `json:"interruptions,omitempty"`

	InputGuardrailResults      []GuardrailResultState             `json:"input_guardrail_results,omitempty"`
	OutputGuardrailResults     []GuardrailResultState             `json:"output_guardrail_results,omitempty"`
	ToolInputGuardrailResults  []ToolGuardrailResultState         `json:"tool_input_guardrail_results,omitempty"`
	ToolOutputGuardrailResults []ToolGuardrailResultState         `json:"tool_output_guardrail_results,omitempty"`
	ToolApprovals              map[string]ToolApprovalRecordState `json:"tool_approvals,omitempty"`
}

// NewRunStateFromResult builds a serializable RunState from a completed RunResult.
func NewRunStateFromResult(result RunResult, currentTurn uint64, maxTurns uint64) RunState {
	return RunState{
		SchemaVersion:              CurrentRunStateSchemaVersion,
		CurrentTurn:                currentTurn,
		MaxTurns:                   maxTurns,
		CurrentAgentName:           displayAgentName(result.LastAgent),
		OriginalInput:              slices.Clone(ItemHelpers().InputToNewInputList(result.Input)),
		GeneratedItems:             toInputList(InputItems{}, result.NewItems),
		ModelResponses:             slices.Clone(result.RawResponses),
		PreviousResponseID:         result.LastResponseID(),
		Interruptions:              slices.Clone(result.Interruptions),
		InputGuardrailResults:      guardrailResultStatesFromInput(result.InputGuardrailResults),
		OutputGuardrailResults:     guardrailResultStatesFromOutput(result.OutputGuardrailResults),
		ToolInputGuardrailResults:  toolGuardrailResultStatesFromInput(result.ToolInputGuardrailResults),
		ToolOutputGuardrailResults: toolGuardrailResultStatesFromOutput(result.ToolOutputGuardrailResults),
	}
}

// NewRunStateFromStreaming builds a serializable RunState from a RunResultStreaming snapshot.
func NewRunStateFromStreaming(result *RunResultStreaming) RunState {
	if result == nil {
		return RunState{SchemaVersion: CurrentRunStateSchemaVersion}
	}

	return RunState{
		SchemaVersion:              CurrentRunStateSchemaVersion,
		CurrentTurn:                result.CurrentTurn(),
		MaxTurns:                   result.MaxTurns(),
		CurrentAgentName:           displayAgentName(result.LastAgent()),
		OriginalInput:              slices.Clone(ItemHelpers().InputToNewInputList(result.Input())),
		GeneratedItems:             toInputList(InputItems{}, result.NewItems()),
		ModelResponses:             slices.Clone(result.RawResponses()),
		PreviousResponseID:         result.LastResponseID(),
		Interruptions:              slices.Clone(result.Interruptions()),
		InputGuardrailResults:      guardrailResultStatesFromInput(result.InputGuardrailResults()),
		OutputGuardrailResults:     guardrailResultStatesFromOutput(result.OutputGuardrailResults()),
		ToolInputGuardrailResults:  toolGuardrailResultStatesFromInput(result.ToolInputGuardrailResults()),
		ToolOutputGuardrailResults: toolGuardrailResultStatesFromOutput(result.ToolOutputGuardrailResults()),
	}
}

// Validate checks schema compatibility.
func (s RunState) Validate() error {
	switch s.SchemaVersion {
	case "", CurrentRunStateSchemaVersion:
		return nil
	default:
		return fmt.Errorf(
			"unsupported run state schema version %q (supported: %q)",
			s.SchemaVersion,
			CurrentRunStateSchemaVersion,
		)
	}
}

// ToJSON encodes RunState to JSON bytes.
func (s RunState) ToJSON() ([]byte, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	withDefaultVersion := s
	if withDefaultVersion.SchemaVersion == "" {
		withDefaultVersion.SchemaVersion = CurrentRunStateSchemaVersion
	}
	return json.Marshal(withDefaultVersion)
}

// ToJSONString encodes RunState to a JSON string.
func (s RunState) ToJSONString() (string, error) {
	b, err := s.ToJSON()
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// RunStateFromJSON decodes RunState from JSON bytes.
func RunStateFromJSON(data []byte) (RunState, error) {
	var state RunState
	if err := json.Unmarshal(data, &state); err != nil {
		return RunState{}, err
	}
	if err := state.Validate(); err != nil {
		return RunState{}, err
	}
	if state.SchemaVersion == "" {
		state.SchemaVersion = CurrentRunStateSchemaVersion
	}
	return state, nil
}

// RunStateFromJSONString decodes RunState from a JSON string.
func RunStateFromJSONString(data string) (RunState, error) {
	return RunStateFromJSON([]byte(data))
}

// SetToolApprovalsFromContext snapshots approval state from context into RunState.
func (s *RunState) SetToolApprovalsFromContext(ctx toolApprovalStateSerializer) {
	if s == nil {
		return
	}
	if ctx == nil {
		s.ToolApprovals = nil
		return
	}
	s.ToolApprovals = cloneToolApprovalStates(ctx.SerializeApprovals())
}

// ApplyToolApprovalsToContext restores RunState tool approvals into the given context.
func (s RunState) ApplyToolApprovalsToContext(ctx toolApprovalStateRebuilder) {
	if ctx == nil {
		return
	}
	ctx.RebuildApprovals(cloneToolApprovalStates(s.ToolApprovals))
}

// ApproveTool appends an approval response input item for the given interruption.
func (s *RunState) ApproveTool(approvalItem ToolApprovalItem) error {
	if s == nil {
		return nil
	}
	item, err := buildMCPApprovalResponseInputItem(approvalItem, true, "")
	if err != nil {
		return err
	}
	s.GeneratedItems = append(s.GeneratedItems, item)
	s.applyDecisionToToolApprovals(approvalItem, true)
	return nil
}

// RejectTool appends a rejection response input item for the given interruption.
func (s *RunState) RejectTool(approvalItem ToolApprovalItem, reason string) error {
	if s == nil {
		return nil
	}
	item, err := buildMCPApprovalResponseInputItem(approvalItem, false, reason)
	if err != nil {
		return err
	}
	s.GeneratedItems = append(s.GeneratedItems, item)
	s.applyDecisionToToolApprovals(approvalItem, false)
	return nil
}

// ResumeInputItems returns a merged input list for continuing the run.
func (s RunState) ResumeInputItems() []TResponseInputItem {
	return slices.Concat(
		slices.Clone(s.OriginalInput),
		slices.Clone(s.GeneratedItems),
	)
}

// ResumeInput wraps ResumeInputItems as InputItems.
func (s RunState) ResumeInput() InputItems {
	return InputItems(s.ResumeInputItems())
}

// ResumeRunConfig applies resumable options to a base RunConfig.
func (s RunState) ResumeRunConfig(base RunConfig) RunConfig {
	cfg := base
	if cfg.PreviousResponseID == "" && s.PreviousResponseID != "" {
		cfg.PreviousResponseID = s.PreviousResponseID
	}
	if cfg.MaxTurns == 0 && s.MaxTurns > 0 {
		cfg.MaxTurns = s.MaxTurns
	}
	return cfg
}

// ResumeInputGuardrailResults reconstructs input guardrail results saved in RunState.
func (s RunState) ResumeInputGuardrailResults() []InputGuardrailResult {
	return inputGuardrailResultsFromStates(s.InputGuardrailResults)
}

// ResumeOutputGuardrailResults reconstructs output guardrail results saved in RunState.
func (s RunState) ResumeOutputGuardrailResults() []OutputGuardrailResult {
	return outputGuardrailResultsFromStates(s.OutputGuardrailResults)
}

// ResumeToolInputGuardrailResults reconstructs tool-input guardrail results saved in RunState.
func (s RunState) ResumeToolInputGuardrailResults() []ToolInputGuardrailResult {
	return toolInputGuardrailResultsFromStates(s.ToolInputGuardrailResults)
}

// ResumeToolOutputGuardrailResults reconstructs tool-output guardrail results saved in RunState.
func (s RunState) ResumeToolOutputGuardrailResults() []ToolOutputGuardrailResult {
	return toolOutputGuardrailResultsFromStates(s.ToolOutputGuardrailResults)
}

// ApplyStoredToolApprovals appends missing MCP approval response items for pending interruptions,
// based on decisions persisted in ToolApprovals.
func (s *RunState) ApplyStoredToolApprovals() error {
	if s == nil || len(s.Interruptions) == 0 || len(s.ToolApprovals) == 0 {
		return nil
	}

	approvalContext := NewRunContextWrapper[any](nil)
	s.ApplyToolApprovalsToContext(approvalContext)

	existingResponses := existingApprovalResponseIDs(s.GeneratedItems)
	for _, interruption := range s.Interruptions {
		approvalRequestID := resolveApprovalCallID(interruption)
		if approvalRequestID == "" {
			continue
		}
		if _, exists := existingResponses[approvalRequestID]; exists {
			continue
		}

		approved, known := approvalContext.GetApprovalStatus(
			resolveApprovalToolName(interruption),
			approvalRequestID,
			&interruption,
		)
		if !known {
			continue
		}

		item, err := buildMCPApprovalResponseInputItem(interruption, approved, "")
		if err != nil {
			return err
		}
		s.GeneratedItems = append(s.GeneratedItems, item)
		existingResponses[approvalRequestID] = struct{}{}
	}

	return nil
}

func guardrailResultStatesFromInput(results []InputGuardrailResult) []GuardrailResultState {
	if len(results) == 0 {
		return nil
	}
	out := make([]GuardrailResultState, len(results))
	for i, result := range results {
		out[i] = GuardrailResultState{
			Name: result.Guardrail.Name,
			Output: GuardrailFunctionOutputState{
				OutputInfo:        result.Output.OutputInfo,
				TripwireTriggered: result.Output.TripwireTriggered,
			},
		}
	}
	return out
}

func inputGuardrailResultsFromStates(results []GuardrailResultState) []InputGuardrailResult {
	if len(results) == 0 {
		return nil
	}
	out := make([]InputGuardrailResult, len(results))
	for i, result := range results {
		out[i] = InputGuardrailResult{
			Guardrail: InputGuardrail{
				Name: result.Name,
			},
			Output: GuardrailFunctionOutput{
				OutputInfo:        result.Output.OutputInfo,
				TripwireTriggered: result.Output.TripwireTriggered,
			},
		}
	}
	return out
}

func guardrailResultStatesFromOutput(results []OutputGuardrailResult) []GuardrailResultState {
	if len(results) == 0 {
		return nil
	}
	out := make([]GuardrailResultState, len(results))
	for i, result := range results {
		out[i] = GuardrailResultState{
			Name: result.Guardrail.Name,
			Output: GuardrailFunctionOutputState{
				OutputInfo:        result.Output.OutputInfo,
				TripwireTriggered: result.Output.TripwireTriggered,
			},
		}
	}
	return out
}

func outputGuardrailResultsFromStates(results []GuardrailResultState) []OutputGuardrailResult {
	if len(results) == 0 {
		return nil
	}
	out := make([]OutputGuardrailResult, len(results))
	for i, result := range results {
		out[i] = OutputGuardrailResult{
			Guardrail: OutputGuardrail{
				Name: result.Name,
			},
			Output: GuardrailFunctionOutput{
				OutputInfo:        result.Output.OutputInfo,
				TripwireTriggered: result.Output.TripwireTriggered,
			},
		}
	}
	return out
}

func toolGuardrailResultStatesFromInput(results []ToolInputGuardrailResult) []ToolGuardrailResultState {
	if len(results) == 0 {
		return nil
	}
	out := make([]ToolGuardrailResultState, len(results))
	for i, result := range results {
		out[i] = ToolGuardrailResultState{
			Name: result.Guardrail.GetName(),
			Output: ToolGuardrailFunctionOutputState{
				OutputInfo: result.Output.OutputInfo,
				Behavior: ToolGuardrailBehaviorState{
					Type:    result.Output.BehaviorType(),
					Message: result.Output.BehaviorMessage(),
				},
			},
		}
	}
	return out
}

func toolInputGuardrailResultsFromStates(results []ToolGuardrailResultState) []ToolInputGuardrailResult {
	if len(results) == 0 {
		return nil
	}
	out := make([]ToolInputGuardrailResult, len(results))
	for i, result := range results {
		out[i] = ToolInputGuardrailResult{
			Guardrail: ToolInputGuardrail{
				Name: result.Name,
			},
			Output: toolGuardrailFunctionOutputFromState(result.Output),
		}
	}
	return out
}

func toolGuardrailResultStatesFromOutput(results []ToolOutputGuardrailResult) []ToolGuardrailResultState {
	if len(results) == 0 {
		return nil
	}
	out := make([]ToolGuardrailResultState, len(results))
	for i, result := range results {
		out[i] = ToolGuardrailResultState{
			Name: result.Guardrail.GetName(),
			Output: ToolGuardrailFunctionOutputState{
				OutputInfo: result.Output.OutputInfo,
				Behavior: ToolGuardrailBehaviorState{
					Type:    result.Output.BehaviorType(),
					Message: result.Output.BehaviorMessage(),
				},
			},
		}
	}
	return out
}

func toolOutputGuardrailResultsFromStates(results []ToolGuardrailResultState) []ToolOutputGuardrailResult {
	if len(results) == 0 {
		return nil
	}
	out := make([]ToolOutputGuardrailResult, len(results))
	for i, result := range results {
		out[i] = ToolOutputGuardrailResult{
			Guardrail: ToolOutputGuardrail{
				Name: result.Name,
			},
			Output: toolGuardrailFunctionOutputFromState(result.Output),
		}
	}
	return out
}

func toolGuardrailFunctionOutputFromState(state ToolGuardrailFunctionOutputState) ToolGuardrailFunctionOutput {
	return ToolGuardrailFunctionOutput{
		OutputInfo: state.OutputInfo,
		Behavior: ToolGuardrailBehavior{
			Type:    state.Behavior.Type,
			Message: state.Behavior.Message,
		},
	}
}

func displayAgentName(agent *Agent) string {
	if agent == nil {
		return ""
	}
	return agent.Name
}

func cloneToolApprovalStates(in map[string]ToolApprovalRecordState) map[string]ToolApprovalRecordState {
	if len(in) == 0 {
		return nil
	}

	out := make(map[string]ToolApprovalRecordState, len(in))
	for toolName, state := range in {
		out[toolName] = ToolApprovalRecordState{
			Approved: cloneToolApprovalValue(state.Approved),
			Rejected: cloneToolApprovalValue(state.Rejected),
		}
	}
	return out
}

func cloneToolApprovalValue(v any) any {
	switch typed := v.(type) {
	case []string:
		return slices.Clone(typed)
	case []any:
		return slices.Clone(typed)
	default:
		return v
	}
}

func existingApprovalResponseIDs(items []TResponseInputItem) map[string]struct{} {
	if len(items) == 0 {
		return map[string]struct{}{}
	}

	out := make(map[string]struct{}, len(items))
	for _, item := range items {
		itemType := item.GetType()
		if itemType == nil || *itemType != "mcp_approval_response" {
			continue
		}
		approvalRequestID := item.GetApprovalRequestID()
		if approvalRequestID == nil || *approvalRequestID == "" {
			continue
		}
		out[*approvalRequestID] = struct{}{}
	}
	return out
}

func (s *RunState) applyDecisionToToolApprovals(approvalItem ToolApprovalItem, approve bool) {
	if s == nil {
		return
	}

	approvalContext := NewRunContextWrapper[any](nil)
	s.ApplyToolApprovalsToContext(approvalContext)
	if approve {
		approvalContext.ApproveTool(approvalItem, false)
	} else {
		approvalContext.RejectTool(approvalItem, false)
	}
	s.SetToolApprovalsFromContext(approvalContext)
}

func buildMCPApprovalResponseInputItem(approvalItem ToolApprovalItem, approve bool, reason string) (TResponseInputItem, error) {
	approvalRequestID := resolveApprovalCallID(approvalItem)
	if approvalRequestID == "" {
		return TResponseInputItem{}, UserErrorf("approval item has no approval request id")
	}

	rawItem := responses.ResponseInputItemMcpApprovalResponseParam{
		ApprovalRequestID: approvalRequestID,
		Approve:           approve,
		ID:                param.Opt[string]{},
		Reason:            param.Opt[string]{},
		Type:              constant.ValueOf[constant.McpApprovalResponse](),
	}
	if !approve && reason != "" {
		rawItem.Reason = param.NewOpt(reason)
	}

	return responses.ResponseInputItemUnionParam{
		OfMcpApprovalResponse: &rawItem,
	}, nil
}
