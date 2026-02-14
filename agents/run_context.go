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
	"reflect"

	"github.com/nlpodyssey/openai-agents-go/usage"
)

type toolApprovalRecord struct {
	ApprovedAll     bool
	RejectedAll     bool
	ApprovedCallIDs map[string]struct{}
	RejectedCallIDs map[string]struct{}
}

// ToolApprovalRecordState is a JSON-friendly approval state snapshot.
type ToolApprovalRecordState struct {
	Approved any `json:"approved"`
	Rejected any `json:"rejected"`
}

// ToolApprovalItem stores tool identity data used to approve or reject tool calls.
type ToolApprovalItem struct {
	ToolName string
	RawItem  any
}

// RunContextWrapper wraps caller context and tracks usage and approval decisions.
type RunContextWrapper[T any] struct {
	Context   T
	Usage     *usage.Usage
	TurnInput []TResponseInputItem
	ToolInput any

	approvals map[string]*toolApprovalRecord
}

// NewRunContextWrapper creates a new RunContextWrapper.
func NewRunContextWrapper[T any](ctx T) *RunContextWrapper[T] {
	return &RunContextWrapper[T]{
		Context:   ctx,
		Usage:     usage.NewUsage(),
		approvals: make(map[string]*toolApprovalRecord),
	}
}

func (c *RunContextWrapper[T]) getOrCreateApprovalRecord(toolName string) *toolApprovalRecord {
	if c.approvals == nil {
		c.approvals = make(map[string]*toolApprovalRecord)
	}
	record, ok := c.approvals[toolName]
	if !ok {
		record = &toolApprovalRecord{
			ApprovedCallIDs: make(map[string]struct{}),
			RejectedCallIDs: make(map[string]struct{}),
		}
		c.approvals[toolName] = record
	}
	return record
}

// IsToolApproved returns (approved, known).
// If known is false, there is no explicit decision for this tool+call_id yet.
func (c *RunContextWrapper[T]) IsToolApproved(toolName, callID string) (bool, bool) {
	record, ok := c.approvals[toolName]
	if !ok || record == nil {
		return false, false
	}

	// Approval takes precedence if both flags are somehow true.
	if record.ApprovedAll {
		return true, true
	}
	if record.RejectedAll {
		return false, true
	}
	if _, ok := record.ApprovedCallIDs[callID]; ok {
		return true, true
	}
	if _, ok := record.RejectedCallIDs[callID]; ok {
		return false, true
	}
	return false, false
}

func (c *RunContextWrapper[T]) applyApprovalDecision(
	approvalItem ToolApprovalItem,
	always bool,
	approve bool,
) {
	toolName := resolveApprovalToolName(approvalItem)
	callID := resolveApprovalCallID(approvalItem)
	record := c.getOrCreateApprovalRecord(toolName)

	if always || callID == "" {
		record.ApprovedAll = approve
		record.RejectedAll = !approve
		clear(record.ApprovedCallIDs)
		clear(record.RejectedCallIDs)
		return
	}

	if approve {
		delete(record.RejectedCallIDs, callID)
		record.ApprovedCallIDs[callID] = struct{}{}
	} else {
		delete(record.ApprovedCallIDs, callID)
		record.RejectedCallIDs[callID] = struct{}{}
	}
}

// ApproveTool approves a tool call, optionally for all future calls of that tool.
func (c *RunContextWrapper[T]) ApproveTool(approvalItem ToolApprovalItem, alwaysApprove bool) {
	c.applyApprovalDecision(approvalItem, alwaysApprove, true)
}

// RejectTool rejects a tool call, optionally for all future calls of that tool.
func (c *RunContextWrapper[T]) RejectTool(approvalItem ToolApprovalItem, alwaysReject bool) {
	c.applyApprovalDecision(approvalItem, alwaysReject, false)
}

// GetApprovalStatus returns (approved, known).
// If known is false, there is no explicit decision for this call.
// When existingPending is set, we also retry lookup using its resolved tool name.
func (c *RunContextWrapper[T]) GetApprovalStatus(
	toolName string,
	callID string,
	existingPending *ToolApprovalItem,
) (bool, bool) {
	approved, known := c.IsToolApproved(toolName, callID)
	if known || existingPending == nil {
		return approved, known
	}
	return c.IsToolApproved(resolveApprovalToolName(*existingPending), callID)
}

// SerializeApprovals exports approval state as JSON-friendly data.
func (c *RunContextWrapper[T]) SerializeApprovals() map[string]ToolApprovalRecordState {
	if len(c.approvals) == 0 {
		return map[string]ToolApprovalRecordState{}
	}

	out := make(map[string]ToolApprovalRecordState, len(c.approvals))
	for toolName, record := range c.approvals {
		var approved any
		var rejected any
		if record.ApprovedAll {
			approved = true
		} else {
			approvedIDs := make([]string, 0, len(record.ApprovedCallIDs))
			for id := range record.ApprovedCallIDs {
				approvedIDs = append(approvedIDs, id)
			}
			approved = approvedIDs
		}
		if record.RejectedAll {
			rejected = true
		} else {
			rejectedIDs := make([]string, 0, len(record.RejectedCallIDs))
			for id := range record.RejectedCallIDs {
				rejectedIDs = append(rejectedIDs, id)
			}
			rejected = rejectedIDs
		}
		out[toolName] = ToolApprovalRecordState{
			Approved: approved,
			Rejected: rejected,
		}
	}
	return out
}

// RebuildApprovals restores approval state from serialized data.
func (c *RunContextWrapper[T]) RebuildApprovals(approvals map[string]ToolApprovalRecordState) {
	c.approvals = make(map[string]*toolApprovalRecord, len(approvals))
	for toolName, state := range approvals {
		record := &toolApprovalRecord{
			ApprovedCallIDs: make(map[string]struct{}),
			RejectedCallIDs: make(map[string]struct{}),
		}

		switch approved := state.Approved.(type) {
		case bool:
			record.ApprovedAll = approved
		case []string:
			for _, id := range approved {
				if id != "" {
					record.ApprovedCallIDs[id] = struct{}{}
				}
			}
		case []any:
			for _, v := range approved {
				if id, ok := v.(string); ok && id != "" {
					record.ApprovedCallIDs[id] = struct{}{}
				}
			}
		}

		switch rejected := state.Rejected.(type) {
		case bool:
			record.RejectedAll = rejected
		case []string:
			for _, id := range rejected {
				if id != "" {
					record.RejectedCallIDs[id] = struct{}{}
				}
			}
		case []any:
			for _, v := range rejected {
				if id, ok := v.(string); ok && id != "" {
					record.RejectedCallIDs[id] = struct{}{}
				}
			}
		}

		c.approvals[toolName] = record
	}
}

// ForkWithToolInput creates a child context that shares approvals and usage and has ToolInput set.
func (c *RunContextWrapper[T]) ForkWithToolInput(toolInput any) *RunContextWrapper[T] {
	return &RunContextWrapper[T]{
		Context:   c.Context,
		Usage:     c.Usage,
		TurnInput: c.TurnInput,
		ToolInput: toolInput,
		approvals: c.approvals,
	}
}

// ForkWithoutToolInput creates a child context that shares approvals and usage.
func (c *RunContextWrapper[T]) ForkWithoutToolInput() *RunContextWrapper[T] {
	return &RunContextWrapper[T]{
		Context:   c.Context,
		Usage:     c.Usage,
		TurnInput: c.TurnInput,
		approvals: c.approvals,
	}
}

func resolveApprovalToolName(item ToolApprovalItem) string {
	if item.ToolName != "" {
		return item.ToolName
	}
	if v, ok := stringFromMap(item.RawItem, "name"); ok && v != "" {
		return v
	}
	if v, ok := stringFromMap(item.RawItem, "type"); ok && v != "" {
		return v
	}
	if v, ok := stringFromField(item.RawItem, "Name"); ok && v != "" {
		return v
	}
	if v, ok := stringFromField(item.RawItem, "Type"); ok && v != "" {
		return v
	}
	return "unknown_tool"
}

func resolveApprovalCallID(item ToolApprovalItem) string {
	if id, ok := providerDataApprovalID(item.RawItem); ok && id != "" {
		return id
	}
	if v, ok := stringFromMap(item.RawItem, "call_id"); ok && v != "" {
		return v
	}
	if v, ok := stringFromMap(item.RawItem, "id"); ok && v != "" {
		return v
	}
	if v, ok := stringFromField(item.RawItem, "CallID"); ok && v != "" {
		return v
	}
	if v, ok := stringFromField(item.RawItem, "ID"); ok && v != "" {
		return v
	}
	return ""
}

func providerDataApprovalID(rawItem any) (string, bool) {
	providerData, ok := anyFromMap(rawItem, "provider_data")
	if !ok {
		providerData, ok = anyFromField(rawItem, "ProviderData")
		if !ok {
			return "", false
		}
	}

	if providerType, ok := stringFromMap(providerData, "type"); !ok || providerType != "mcp_approval_request" {
		return "", false
	}
	id, ok := stringFromMap(providerData, "id")
	return id, ok
}

func stringFromMap(rawItem any, key string) (string, bool) {
	v, ok := anyFromMap(rawItem, key)
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	if !ok || s == "" {
		return "", false
	}
	return s, true
}

func anyFromMap(rawItem any, key string) (any, bool) {
	switch v := rawItem.(type) {
	case map[string]any:
		value, ok := v[key]
		return value, ok
	case map[string]string:
		value, ok := v[key]
		if !ok {
			return nil, false
		}
		return value, true
	default:
		return nil, false
	}
}

func stringFromField(rawItem any, fieldName string) (string, bool) {
	value, ok := anyFromField(rawItem, fieldName)
	if !ok {
		return "", false
	}
	s, ok := value.(string)
	if !ok || s == "" {
		return "", false
	}
	return s, true
}

func anyFromField(rawItem any, fieldName string) (any, bool) {
	if rawItem == nil {
		return nil, false
	}
	v := reflect.ValueOf(rawItem)
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return nil, false
		}
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		return nil, false
	}
	field := v.FieldByName(fieldName)
	if !field.IsValid() || !field.CanInterface() {
		return nil, false
	}
	return field.Interface(), true
}
