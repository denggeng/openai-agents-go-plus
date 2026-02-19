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
	"reflect"
	"runtime"
	"sync"

	"github.com/openai/openai-go/v3/responses"
)

type toolCallSignature struct {
	CallID    string
	Name      string
	Arguments string
	Type      string
	ID        string
	Status    string
}

type agentToolSignature struct {
	CallID    string
	Name      string
	Arguments string
}

var (
	agentToolStateMu sync.Mutex

	agentToolRunResultsByObj         = make(map[uintptr]any)
	agentToolRunResultsBySignature   = make(map[toolCallSignature]map[uintptr]struct{})
	agentToolRunResultSignatureByObj = make(map[uintptr]toolCallSignature)
	agentToolCallRefsByObj           = make(map[uintptr]struct{})
	agentToolRunStatesBySignature    = make(map[agentToolSignature]*RunState)
)

const agentToolParentKey = "__agent_tool_parent"

func toolCallSignatureFromCall(call *responses.ResponseFunctionToolCall) toolCallSignature {
	if call == nil {
		return toolCallSignature{}
	}
	return toolCallSignature{
		CallID:    call.CallID,
		Name:      call.Name,
		Arguments: normalizeToolArguments(call.Arguments),
		Type:      string(call.Type),
		ID:        call.ID,
		Status:    string(call.Status),
	}
}

func agentToolSignatureFromCall(call *responses.ResponseFunctionToolCall) agentToolSignature {
	if call == nil {
		return agentToolSignature{}
	}
	return agentToolSignature{
		CallID:    call.CallID,
		Name:      call.Name,
		Arguments: normalizeToolArguments(call.Arguments),
	}
}

func agentToolSignatureFromResponseToolCall(call *ResponseFunctionToolCall) agentToolSignature {
	if call == nil {
		return agentToolSignature{}
	}
	return agentToolSignature{
		CallID:    call.CallID,
		Name:      call.Name,
		Arguments: normalizeToolArguments(call.Arguments),
	}
}

func agentToolSignatureFromToolData(data *ToolContextData) agentToolSignature {
	if data == nil {
		return agentToolSignature{}
	}
	return agentToolSignature{
		CallID:    data.ToolCallID,
		Name:      data.ToolName,
		Arguments: normalizeToolArguments(data.ToolArguments),
	}
}

func toolCallObjectID(call *responses.ResponseFunctionToolCall) uintptr {
	if call == nil {
		return 0
	}
	return reflect.ValueOf(call).Pointer()
}

func ensureAgentToolStateMapsLocked() {
	if agentToolRunResultsByObj == nil {
		agentToolRunResultsByObj = make(map[uintptr]any)
	}
	if agentToolRunResultsBySignature == nil {
		agentToolRunResultsBySignature = make(map[toolCallSignature]map[uintptr]struct{})
	}
	if agentToolRunResultSignatureByObj == nil {
		agentToolRunResultSignatureByObj = make(map[uintptr]toolCallSignature)
	}
	if agentToolCallRefsByObj == nil {
		agentToolCallRefsByObj = make(map[uintptr]struct{})
	}
	if agentToolRunStatesBySignature == nil {
		agentToolRunStatesBySignature = make(map[agentToolSignature]*RunState)
	}
}

func recordAgentToolRunResult(
	toolCall *responses.ResponseFunctionToolCall,
	runResult any,
) {
	if toolCall == nil {
		return
	}
	objID := toolCallObjectID(toolCall)
	if objID == 0 {
		return
	}

	agentToolStateMu.Lock()
	defer agentToolStateMu.Unlock()

	ensureAgentToolStateMapsLocked()

	agentToolRunResultsByObj[objID] = runResult
	signature := toolCallSignatureFromCall(toolCall)
	agentToolRunResultSignatureByObj[objID] = signature
	candidateIDs := agentToolRunResultsBySignature[signature]
	if candidateIDs == nil {
		candidateIDs = make(map[uintptr]struct{})
		agentToolRunResultsBySignature[signature] = candidateIDs
	}
	candidateIDs[objID] = struct{}{}
	agentToolCallRefsByObj[objID] = struct{}{}

	runtime.SetFinalizer(toolCall, func(*responses.ResponseFunctionToolCall) {
		dropAgentToolRunResultByID(objID)
	})
}

func consumeAgentToolRunResult(toolCall *responses.ResponseFunctionToolCall) any {
	if toolCall == nil {
		return nil
	}
	objID := toolCallObjectID(toolCall)
	if objID == 0 {
		return nil
	}

	agentToolStateMu.Lock()
	defer agentToolStateMu.Unlock()

	if agentToolRunResultsByObj == nil {
		return nil
	}
	if runResult, ok := agentToolRunResultsByObj[objID]; ok {
		delete(agentToolRunResultsByObj, objID)
		dropAgentToolRunResultByIDLocked(objID)
		return runResult
	}

	signature := toolCallSignatureFromCall(toolCall)
	if agentToolRunResultsBySignature == nil {
		return nil
	}
	candidateIDs := agentToolRunResultsBySignature[signature]
	if len(candidateIDs) != 1 {
		return nil
	}
	var candidateID uintptr
	for id := range candidateIDs {
		candidateID = id
		break
	}
	delete(agentToolRunResultsBySignature, signature)
	if agentToolRunResultSignatureByObj != nil {
		delete(agentToolRunResultSignatureByObj, candidateID)
	}
	if agentToolCallRefsByObj != nil {
		delete(agentToolCallRefsByObj, candidateID)
	}
	if agentToolRunResultsByObj == nil {
		return nil
	}
	runResult := agentToolRunResultsByObj[candidateID]
	delete(agentToolRunResultsByObj, candidateID)
	return runResult
}

func peekAgentToolRunResult(toolCall *responses.ResponseFunctionToolCall) any {
	if toolCall == nil {
		return nil
	}
	objID := toolCallObjectID(toolCall)
	if objID == 0 {
		return nil
	}

	agentToolStateMu.Lock()
	defer agentToolStateMu.Unlock()

	if agentToolRunResultsByObj == nil {
		return nil
	}
	if runResult, ok := agentToolRunResultsByObj[objID]; ok {
		return runResult
	}

	signature := toolCallSignatureFromCall(toolCall)
	if agentToolRunResultsBySignature == nil {
		return nil
	}
	candidateIDs := agentToolRunResultsBySignature[signature]
	if len(candidateIDs) != 1 {
		return nil
	}
	var candidateID uintptr
	for id := range candidateIDs {
		candidateID = id
		break
	}
	return agentToolRunResultsByObj[candidateID]
}

func dropAgentToolRunResult(toolCall *responses.ResponseFunctionToolCall) {
	if toolCall == nil {
		return
	}
	objID := toolCallObjectID(toolCall)
	if objID == 0 {
		return
	}

	agentToolStateMu.Lock()
	defer agentToolStateMu.Unlock()

	if agentToolRunResultsByObj == nil {
		return
	}
	if _, ok := agentToolRunResultsByObj[objID]; ok {
		delete(agentToolRunResultsByObj, objID)
		dropAgentToolRunResultByIDLocked(objID)
		return
	}

	signature := toolCallSignatureFromCall(toolCall)
	if agentToolRunResultsBySignature == nil {
		return
	}
	candidateIDs := agentToolRunResultsBySignature[signature]
	if len(candidateIDs) != 1 {
		return
	}
	var candidateID uintptr
	for id := range candidateIDs {
		candidateID = id
		break
	}
	delete(agentToolRunResultsBySignature, signature)
	if agentToolRunResultSignatureByObj != nil {
		delete(agentToolRunResultSignatureByObj, candidateID)
	}
	if agentToolCallRefsByObj != nil {
		delete(agentToolCallRefsByObj, candidateID)
	}
	if agentToolRunResultsByObj != nil {
		delete(agentToolRunResultsByObj, candidateID)
	}
}

func recordAgentToolRunState(signature agentToolSignature, state *RunState) {
	if signature == (agentToolSignature{}) {
		return
	}
	agentToolStateMu.Lock()
	defer agentToolStateMu.Unlock()

	ensureAgentToolStateMapsLocked()
	agentToolRunStatesBySignature[signature] = state
}

func peekAgentToolRunState(signature agentToolSignature) *RunState {
	if signature == (agentToolSignature{}) {
		return nil
	}
	agentToolStateMu.Lock()
	defer agentToolStateMu.Unlock()

	if agentToolRunStatesBySignature == nil {
		return nil
	}
	if state := agentToolRunStatesBySignature[signature]; state != nil {
		return state
	}
	if state, _, ok := findAgentToolRunStateByCallIDLocked(signature); ok {
		return state
	}
	return nil
}

func consumeAgentToolRunState(signature agentToolSignature) *RunState {
	if signature == (agentToolSignature{}) {
		return nil
	}
	agentToolStateMu.Lock()
	defer agentToolStateMu.Unlock()

	if agentToolRunStatesBySignature == nil {
		return nil
	}
	state := agentToolRunStatesBySignature[signature]
	if state == nil {
		var matchedSignature agentToolSignature
		var ok bool
		state, matchedSignature, ok = findAgentToolRunStateByCallIDLocked(signature)
		if !ok {
			return nil
		}
		signature = matchedSignature
	}
	delete(agentToolRunStatesBySignature, signature)
	return state
}

func findAgentToolRunStateByCallIDLocked(signature agentToolSignature) (*RunState, agentToolSignature, bool) {
	if agentToolRunStatesBySignature == nil {
		return nil, agentToolSignature{}, false
	}
	if signature.CallID == "" || signature.Name == "" {
		return nil, agentToolSignature{}, false
	}
	var match *RunState
	var matchSignature agentToolSignature
	for candidate, state := range agentToolRunStatesBySignature {
		if candidate.CallID != signature.CallID || candidate.Name != signature.Name {
			continue
		}
		if match != nil {
			return nil, agentToolSignature{}, false
		}
		match = state
		matchSignature = candidate
	}
	if match == nil {
		return nil, agentToolSignature{}, false
	}
	return match, matchSignature, true
}

func wrapAgentToolInterruption(item ToolApprovalItem, parent agentToolSignature) ToolApprovalItem {
	raw := normalizeJSONValue(item.RawItem)
	rawMap, ok := raw.(map[string]any)
	if !ok {
		rawMap = map[string]any{}
	}
	rawMap[agentToolParentKey] = map[string]any{
		"call_id":   parent.CallID,
		"name":      parent.Name,
		"arguments": parent.Arguments,
	}
	return ToolApprovalItem{
		ToolName: item.ToolName,
		RawItem:  rawMap,
	}
}

func agentToolParentSignatureFromRaw(raw any) (agentToolSignature, bool) {
	rawMap, ok := raw.(map[string]any)
	if !ok {
		return agentToolSignature{}, false
	}
	parentAny, ok := rawMap[agentToolParentKey]
	if !ok {
		return agentToolSignature{}, false
	}
	parentMap, ok := parentAny.(map[string]any)
	if !ok {
		return agentToolSignature{}, false
	}
	callID, _ := coerceStringValue(parentMap["call_id"])
	name, _ := coerceStringValue(parentMap["name"])
	arguments := normalizeToolArgumentsValue(parentMap["arguments"])
	if callID == "" && name == "" && arguments == "" {
		return agentToolSignature{}, false
	}
	return agentToolSignature{
		CallID:    callID,
		Name:      name,
		Arguments: arguments,
	}, true
}

func normalizeToolArgumentsValue(value any) string {
	if value == nil {
		return ""
	}
	if v, ok := value.(string); ok {
		return normalizeToolArguments(v)
	}
	if b, err := json.Marshal(value); err == nil {
		return string(b)
	}
	if v, ok := coerceStringValue(value); ok {
		return normalizeToolArguments(v)
	}
	return ""
}

func normalizeToolArguments(args string) string {
	if args == "" {
		return ""
	}
	var decoded any
	if err := json.Unmarshal([]byte(args), &decoded); err != nil {
		return args
	}
	normalized, err := json.Marshal(decoded)
	if err != nil {
		return args
	}
	return string(normalized)
}

func dropAgentToolRunResultByID(objID uintptr) {
	agentToolStateMu.Lock()
	defer agentToolStateMu.Unlock()
	dropAgentToolRunResultByIDLocked(objID)
}

func dropAgentToolRunResultByIDLocked(objID uintptr) {
	if agentToolCallRefsByObj != nil {
		delete(agentToolCallRefsByObj, objID)
	}
	signatureByObj := agentToolRunResultSignatureByObj
	if signatureByObj == nil {
		return
	}
	signature, ok := signatureByObj[objID]
	if !ok {
		return
	}
	delete(signatureByObj, objID)
	resultsBySignature := agentToolRunResultsBySignature
	if resultsBySignature == nil {
		return
	}
	candidateIDs := resultsBySignature[signature]
	if len(candidateIDs) == 0 {
		return
	}
	delete(candidateIDs, objID)
	if len(candidateIDs) == 0 {
		delete(resultsBySignature, signature)
	}
}
