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

package realtime

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"

	"github.com/denggeng/openai-agents-go-plus/agents"
)

const defaultRealtimeApprovalRejectionMessage = "Tool call rejected because approval was not granted."

type pendingRealtimeToolCall struct {
	event        RealtimeModelToolCallEvent
	agent        *RealtimeAgent[any]
	functionTool agents.FunctionTool
	approvalItem agents.ToolApprovalItem
}

// RealtimeSession is a basic runtime session over a realtime model transport.
type RealtimeSession struct {
	model                  RealtimeModel
	currentAgent           *RealtimeAgent[any]
	contextWrapper         *agents.RunContextWrapper[any]
	eventInfo              RealtimeEventInfo
	history                []any
	modelConfig            RealtimeModelConfig
	runConfig              RealtimeRunConfig
	baseModelSetting       RealtimeSessionModelSettings
	asyncToolCalls         bool
	debounceTextLength     int
	itemTranscripts        map[string]string
	itemGuardrailRunCounts map[string]int
	interruptedResponseIDs map[string]struct{}
	pendingToolCalls       map[string]pendingRealtimeToolCall
	eventQueue             chan RealtimeSessionEvent
	closed                 bool
	mutex                  sync.Mutex
}

// NewRealtimeSession creates a session bound to a model and initial agent.
func NewRealtimeSession(
	model RealtimeModel,
	agent *RealtimeAgent[any],
	contextValue any,
	modelConfig RealtimeModelConfig,
	runConfig RealtimeRunConfig,
) *RealtimeSession {
	baseModelSettings := RealtimeSessionModelSettings{}
	if runSettings, ok := toRealtimeSettings(runConfig["model_settings"]); ok {
		for key, value := range runSettings {
			baseModelSettings[key] = value
		}
	}
	for key, value := range modelConfig.InitialSettings {
		baseModelSettings[key] = value
	}
	asyncToolCalls, hasAsync := runConfig["async_tool_calls"].(bool)
	if !hasAsync {
		asyncToolCalls = true
	}
	debounceTextLength := readDebounceTextLength(runConfig)

	contextWrapper := agents.NewRunContextWrapper[any](contextValue)
	return &RealtimeSession{
		model:                  model,
		currentAgent:           agent,
		contextWrapper:         contextWrapper,
		eventInfo:              RealtimeEventInfo{Context: contextWrapper},
		modelConfig:            modelConfig,
		runConfig:              runConfig,
		baseModelSetting:       baseModelSettings,
		asyncToolCalls:         asyncToolCalls,
		debounceTextLength:     debounceTextLength,
		itemTranscripts:        make(map[string]string),
		itemGuardrailRunCounts: make(map[string]int),
		interruptedResponseIDs: make(map[string]struct{}),
		pendingToolCalls:       make(map[string]pendingRealtimeToolCall),
		eventQueue:             make(chan RealtimeSessionEvent, 128),
	}
}

// Model returns the underlying realtime model transport.
func (s *RealtimeSession) Model() RealtimeModel {
	return s.model
}

// Enter connects the underlying model and emits an initial history event.
func (s *RealtimeSession) Enter(ctx context.Context) error {
	s.model.AddListener(s)

	updatedSettings, err := BuildModelSettingsFromAgent(
		s.currentAgent,
		s.contextWrapper,
		s.baseModelSetting,
		s.modelConfig.InitialSettings,
		s.runConfig,
	)
	if err != nil {
		return err
	}

	connectConfig := s.modelConfig
	connectConfig.InitialSettings = updatedSettings
	if err := s.model.Connect(ctx, connectConfig); err != nil {
		return err
	}

	s.putEvent(RealtimeHistoryUpdatedEvent{
		History: s.History(),
		Info:    s.eventInfo,
	})
	return nil
}

// Close shuts down the session and underlying model transport.
func (s *RealtimeSession) Close(ctx context.Context) error {
	s.mutex.Lock()
	if s.closed {
		s.mutex.Unlock()
		return nil
	}
	s.closed = true
	clear(s.pendingToolCalls)
	s.mutex.Unlock()

	s.model.RemoveListener(s)
	err := s.model.Close(ctx)
	close(s.eventQueue)
	return err
}

// Events returns the session event stream channel.
func (s *RealtimeSession) Events() <-chan RealtimeSessionEvent {
	return s.eventQueue
}

// History returns a snapshot of current session history.
func (s *RealtimeSession) History() []any {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	return append([]any(nil), s.history...)
}

// SendMessage forwards a user message input to the model.
func (s *RealtimeSession) SendMessage(ctx context.Context, message any) error {
	return s.model.SendEvent(ctx, RealtimeModelSendUserInput{UserInput: message})
}

// SendAudio forwards audio bytes to the model.
func (s *RealtimeSession) SendAudio(ctx context.Context, audio []byte, commit bool) error {
	return s.model.SendEvent(ctx, RealtimeModelSendAudio{Audio: audio, Commit: commit})
}

// Interrupt forwards an interruption request to the model.
func (s *RealtimeSession) Interrupt(ctx context.Context) error {
	return s.model.SendEvent(ctx, RealtimeModelSendInterrupt{})
}

// UpdateAgent switches active agent and pushes a session update to the model.
func (s *RealtimeSession) UpdateAgent(ctx context.Context, agent *RealtimeAgent[any]) error {
	s.currentAgent = agent
	updatedSettings, err := BuildModelSettingsFromAgent(
		s.currentAgent,
		s.contextWrapper,
		s.baseModelSetting,
		nil,
		s.runConfig,
	)
	if err != nil {
		return err
	}
	return s.model.SendEvent(ctx, RealtimeModelSendSessionUpdate{
		SessionSettings: updatedSettings,
	})
}

// ApproveToolCall approves a pending tool call and resumes execution.
func (s *RealtimeSession) ApproveToolCall(ctx context.Context, callID string, always bool) error {
	pending, ok := s.takePendingToolCall(callID)
	if !ok {
		return nil
	}

	s.contextWrapper.ApproveTool(pending.approvalItem, always)
	if s.asyncToolCalls {
		go s.runFunctionToolCall(context.Background(), pending.event, pending.agent, pending.functionTool)
		return nil
	}
	s.runFunctionToolCall(ctx, pending.event, pending.agent, pending.functionTool)
	return nil
}

// RejectToolCall rejects a pending tool call and notifies the model.
func (s *RealtimeSession) RejectToolCall(ctx context.Context, callID string, always bool) error {
	pending, ok := s.takePendingToolCall(callID)
	if !ok {
		return nil
	}

	s.contextWrapper.RejectTool(pending.approvalItem, always)
	return s.sendToolRejection(ctx, pending.event, pending.agent, pending.functionTool)
}

// OnEvent handles model events and emits session-level events.
func (s *RealtimeSession) OnEvent(_ context.Context, event RealtimeModelEvent) error {
	s.putEvent(RealtimeRawModelEvent{Data: event, Info: s.eventInfo})

	switch e := event.(type) {
	case RealtimeModelErrorEvent:
		s.putEvent(RealtimeErrorEvent{Error: e.Error, Info: s.eventInfo})
	case RealtimeModelExceptionEvent:
		message := "realtime model exception"
		if e.Context != nil && strings.TrimSpace(*e.Context) != "" {
			message = *e.Context
		}
		s.putEvent(RealtimeErrorEvent{
			Error: map[string]any{
				"message":   message,
				"exception": e.Exception,
			},
			Info: s.eventInfo,
		})
	case RealtimeModelToolCallEvent:
		agentSnapshot := s.currentAgent
		if s.asyncToolCalls {
			go s.handleToolCall(context.Background(), e, agentSnapshot)
		} else {
			s.handleToolCall(context.Background(), e, agentSnapshot)
		}
	case RealtimeModelTranscriptDeltaEvent:
		transcript, shouldRun := s.handleTranscriptDelta(e)
		if shouldRun {
			go s.runOutputGuardrails(context.Background(), transcript, e.ResponseID)
		}
	case RealtimeModelAudioEvent:
		s.putEvent(RealtimeAudioEvent{
			Audio:        e,
			ItemID:       e.ItemID,
			ContentIndex: e.ContentIndex,
			Info:         s.eventInfo,
		})
	case RealtimeModelAudioInterruptedEvent:
		s.putEvent(RealtimeAudioInterruptedEvent{
			Info:         s.eventInfo,
			ItemID:       e.ItemID,
			ContentIndex: e.ContentIndex,
		})
	case RealtimeModelAudioDoneEvent:
		s.putEvent(RealtimeAudioEndEvent{
			Info:         s.eventInfo,
			ItemID:       e.ItemID,
			ContentIndex: e.ContentIndex,
		})
	case RealtimeModelInputAudioTranscriptionCompletedEvent:
		added, item := s.handleInputAudioTranscriptionCompleted(e)
		if added {
			s.putEvent(RealtimeHistoryAddedEvent{Item: item, Info: s.eventInfo})
		} else {
			s.putEvent(RealtimeHistoryUpdatedEvent{
				History: s.History(),
				Info:    s.eventInfo,
			})
		}
	case RealtimeModelItemUpdatedEvent:
		s.handleHistoryUpdate(e.Item)
	case RealtimeModelItemDeletedEvent:
		s.handleHistoryDelete(e.ItemID)
	case RealtimeModelTurnStartedEvent:
		s.putEvent(RealtimeAgentStartEvent{Agent: s.currentAgent, Info: s.eventInfo})
	case RealtimeModelTurnEndedEvent:
		s.clearGuardrailDebounceState()
		s.putEvent(RealtimeAgentEndEvent{Agent: s.currentAgent, Info: s.eventInfo})
	case RealtimeModelInputAudioTimeoutTriggeredEvent:
		s.putEvent(RealtimeInputAudioTimeoutTriggeredEvent{Info: s.eventInfo})
	}

	return nil
}

func (s *RealtimeSession) handleToolCall(
	ctx context.Context,
	event RealtimeModelToolCallEvent,
	agentSnapshot *RealtimeAgent[any],
) {
	defer func() {
		if recovered := recover(); recovered != nil {
			s.putEvent(RealtimeErrorEvent{
				Error: map[string]any{
					"message": fmt.Sprintf(
						"panic while handling tool call %s: %v",
						event.Name,
						recovered,
					),
				},
				Info: s.eventInfo,
			})
		}
	}()

	if agentSnapshot == nil {
		s.putEvent(RealtimeErrorEvent{
			Error: map[string]any{"message": "tool call received without an active agent"},
			Info:  s.eventInfo,
		})
		return
	}

	tools, err := agentSnapshot.GetAllTools(s.contextWrapper)
	if err != nil {
		s.putEvent(RealtimeErrorEvent{
			Error: map[string]any{"message": fmt.Sprintf("failed to load tools: %v", err)},
			Info:  s.eventInfo,
		})
		return
	}

	handoffs, err := CollectEnabledHandoffs(agentSnapshot, s.contextWrapper)
	if err != nil {
		s.putEvent(RealtimeErrorEvent{
			Error: map[string]any{"message": fmt.Sprintf("failed to load handoffs: %v", err)},
			Info:  s.eventInfo,
		})
		return
	}

	functionMap := make(map[string]agents.FunctionTool)
	for _, tool := range tools {
		functionTool, ok := tool.(agents.FunctionTool)
		if !ok {
			continue
		}
		functionMap[functionTool.Name] = functionTool
	}

	handoffMap := make(map[string]agents.Handoff, len(handoffs))
	for _, handoff := range handoffs {
		handoffMap[handoff.ToolName] = handoff
	}
	realtimeHandoffTargets := make(map[string]*RealtimeAgent[any])
	for _, item := range agentSnapshot.Handoffs {
		switch v := item.(type) {
		case *RealtimeAgent[any]:
			if v != nil {
				realtimeHandoffTargets[RealtimeHandoff(v).ToolName] = v
			}
		case RealtimeAgent[any]:
			clone := v
			realtimeHandoffTargets[RealtimeHandoff(&clone).ToolName] = &clone
		}
	}

	if functionTool, ok := functionMap[event.Name]; ok {
		approvalDecision, err := s.maybeRequestToolApproval(event, agentSnapshot, functionTool)
		if err != nil {
			s.putEvent(RealtimeErrorEvent{
				Error: map[string]any{
					"message": fmt.Sprintf("failed evaluating tool approval for %s: %v", functionTool.Name, err),
				},
				Info: s.eventInfo,
			})
			return
		}
		switch approvalDecision {
		case toolApprovalDecisionApproved:
			s.runFunctionToolCall(ctx, event, agentSnapshot, functionTool)
		case toolApprovalDecisionRejected:
			_ = s.sendToolRejection(ctx, event, agentSnapshot, functionTool)
		case toolApprovalDecisionPending:
			return
		}
		return
	}

	if handoff, ok := handoffMap[event.Name]; ok {
		s.handleHandoffCall(ctx, event, agentSnapshot, handoff, realtimeHandoffTargets[event.Name])
		return
	}

	s.putEvent(RealtimeErrorEvent{
		Error: map[string]any{"message": fmt.Sprintf("tool %s not found", event.Name)},
		Info:  s.eventInfo,
	})
}

type toolApprovalDecision int

const (
	toolApprovalDecisionApproved toolApprovalDecision = iota
	toolApprovalDecisionRejected
	toolApprovalDecisionPending
)

func (s *RealtimeSession) runFunctionToolCall(
	ctx context.Context,
	event RealtimeModelToolCallEvent,
	agentSnapshot *RealtimeAgent[any],
	functionTool agents.FunctionTool,
) {
	defer func() {
		if recovered := recover(); recovered != nil {
			s.putEvent(RealtimeErrorEvent{
				Error: map[string]any{
					"message": fmt.Sprintf(
						"panic while running tool %s: %v",
						functionTool.Name,
						recovered,
					),
				},
				Info: s.eventInfo,
			})
		}
	}()

	s.putEvent(RealtimeToolStartEvent{
		Agent:     agentSnapshot,
		Tool:      functionTool,
		Arguments: event.Arguments,
		Info:      s.eventInfo,
	})

	result, err := functionTool.OnInvokeTool(ctx, event.Arguments)
	if err != nil {
		errorFn := agents.DefaultToolErrorFunction
		if functionTool.FailureErrorFunction != nil {
			errorFn = *functionTool.FailureErrorFunction
		}
		if errorFn == nil {
			s.putEvent(RealtimeErrorEvent{
				Error: map[string]any{
					"message": fmt.Sprintf("error running tool %s: %v", functionTool.Name, err),
				},
				Info: s.eventInfo,
			})
			return
		}

		result, err = errorFn(ctx, err)
		if err != nil {
			s.putEvent(RealtimeErrorEvent{
				Error: map[string]any{
					"message": fmt.Sprintf("error running tool %s: %v", functionTool.Name, err),
				},
				Info: s.eventInfo,
			})
			return
		}
	}

	output := fmt.Sprint(result)
	if err := s.model.SendEvent(ctx, RealtimeModelSendToolOutput{
		ToolCall:      event,
		Output:        output,
		StartResponse: true,
	}); err != nil {
		s.putEvent(RealtimeErrorEvent{
			Error: map[string]any{"message": fmt.Sprintf("failed sending tool output: %v", err)},
			Info:  s.eventInfo,
		})
		return
	}

	s.putEvent(RealtimeToolEndEvent{
		Agent:     agentSnapshot,
		Tool:      functionTool,
		Arguments: event.Arguments,
		Output:    result,
		Info:      s.eventInfo,
	})
}

func (s *RealtimeSession) maybeRequestToolApproval(
	event RealtimeModelToolCallEvent,
	agentSnapshot *RealtimeAgent[any],
	functionTool agents.FunctionTool,
) (toolApprovalDecision, error) {
	needsApproval, err := s.functionNeedsApproval(context.Background(), functionTool, event)
	if err != nil {
		return toolApprovalDecisionRejected, err
	}
	if !needsApproval {
		return toolApprovalDecisionApproved, nil
	}

	approvalItem := s.buildToolApprovalItem(functionTool, event)
	approved, known := s.contextWrapper.GetApprovalStatus(functionTool.Name, event.CallID, &approvalItem)
	if known {
		if approved {
			return toolApprovalDecisionApproved, nil
		}
		return toolApprovalDecisionRejected, nil
	}

	s.storePendingToolCall(event.CallID, pendingRealtimeToolCall{
		event:        event,
		agent:        agentSnapshot,
		functionTool: functionTool,
		approvalItem: approvalItem,
	})

	s.putEvent(RealtimeToolApprovalRequiredEvent{
		Agent:     agentSnapshot,
		Tool:      functionTool,
		CallID:    event.CallID,
		Arguments: event.Arguments,
		Info:      s.eventInfo,
	})
	return toolApprovalDecisionPending, nil
}

func (s *RealtimeSession) functionNeedsApproval(
	ctx context.Context,
	functionTool agents.FunctionTool,
	event RealtimeModelToolCallEvent,
) (bool, error) {
	if functionTool.NeedsApproval == nil {
		return false, nil
	}

	parsedArgs := make(map[string]any)
	if strings.TrimSpace(event.Arguments) != "" {
		if err := json.Unmarshal([]byte(event.Arguments), &parsedArgs); err != nil {
			parsedArgs = map[string]any{}
		}
	}

	return functionTool.NeedsApproval.NeedsApproval(
		ctx,
		s.contextWrapper,
		functionTool,
		parsedArgs,
		event.CallID,
	)
}

func (s *RealtimeSession) buildToolApprovalItem(
	functionTool agents.FunctionTool,
	event RealtimeModelToolCallEvent,
) agents.ToolApprovalItem {
	return agents.ToolApprovalItem{
		ToolName: functionTool.Name,
		RawItem: map[string]any{
			"type":      "function_call",
			"name":      functionTool.Name,
			"call_id":   event.CallID,
			"arguments": event.Arguments,
		},
	}
}

func (s *RealtimeSession) sendToolRejection(
	ctx context.Context,
	event RealtimeModelToolCallEvent,
	agentSnapshot *RealtimeAgent[any],
	functionTool agents.FunctionTool,
) error {
	rejectionMessage := s.resolveApprovalRejectionMessage(functionTool, event.CallID)
	if err := s.model.SendEvent(ctx, RealtimeModelSendToolOutput{
		ToolCall:      event,
		Output:        rejectionMessage,
		StartResponse: true,
	}); err != nil {
		s.putEvent(RealtimeErrorEvent{
			Error: map[string]any{"message": fmt.Sprintf("failed sending tool rejection: %v", err)},
			Info:  s.eventInfo,
		})
		return err
	}

	s.putEvent(RealtimeToolEndEvent{
		Agent:     agentSnapshot,
		Tool:      functionTool,
		Arguments: event.Arguments,
		Output:    rejectionMessage,
		Info:      s.eventInfo,
	})
	return nil
}

func (s *RealtimeSession) resolveApprovalRejectionMessage(
	functionTool agents.FunctionTool,
	callID string,
) string {
	formatterValue := s.runConfig["tool_error_formatter"]
	if formatterValue == nil {
		return defaultRealtimeApprovalRejectionMessage
	}

	args := RealtimeToolErrorFormatterArgs{
		Kind:           RealtimeToolErrorKindApprovalRejected,
		ToolType:       "function",
		ToolName:       functionTool.Name,
		CallID:         callID,
		DefaultMessage: defaultRealtimeApprovalRejectionMessage,
		RunContext:     s.contextWrapper,
	}

	switch formatter := formatterValue.(type) {
	case RealtimeToolErrorFormatter:
		return toolErrorFormatterMessageFromAny(invokeToolErrorFormatterAny(formatter, args))
	case func(RealtimeToolErrorFormatterArgs) any:
		return toolErrorFormatterMessageFromAny(invokeToolErrorFormatterAny(formatter, args))
	case func(RealtimeToolErrorFormatterArgs) string:
		message, err := invokeToolErrorFormatterString(formatter, args)
		return toolErrorFormatterMessageFromAny(message, err)
	case func(RealtimeToolErrorFormatterArgs) (string, error):
		message, err := invokeToolErrorFormatterStringErr(formatter, args)
		return toolErrorFormatterMessageFromAny(message, err)
	default:
		return defaultRealtimeApprovalRejectionMessage
	}
}

func invokeToolErrorFormatterAny(
	formatter func(RealtimeToolErrorFormatterArgs) any,
	args RealtimeToolErrorFormatterArgs,
) (result any, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("tool error formatter panic: %v", recovered)
		}
	}()
	return formatter(args), nil
}

func invokeToolErrorFormatterString(
	formatter func(RealtimeToolErrorFormatterArgs) string,
	args RealtimeToolErrorFormatterArgs,
) (result string, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("tool error formatter panic: %v", recovered)
		}
	}()
	return formatter(args), nil
}

func invokeToolErrorFormatterStringErr(
	formatter func(RealtimeToolErrorFormatterArgs) (string, error),
	args RealtimeToolErrorFormatterArgs,
) (result string, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("tool error formatter panic: %v", recovered)
		}
	}()
	return formatter(args)
}

func toolErrorFormatterMessageFromAny(result any, err error) string {
	if err != nil || result == nil {
		return defaultRealtimeApprovalRejectionMessage
	}
	message, ok := result.(string)
	if !ok {
		return defaultRealtimeApprovalRejectionMessage
	}
	return message
}

func (s *RealtimeSession) handleHandoffCall(
	ctx context.Context,
	event RealtimeModelToolCallEvent,
	agentSnapshot *RealtimeAgent[any],
	handoff agents.Handoff,
	realtimeTarget *RealtimeAgent[any],
) {
	defer func() {
		if recovered := recover(); recovered != nil {
			s.putEvent(RealtimeErrorEvent{
				Error: map[string]any{
					"message": fmt.Sprintf(
						"panic while handling handoff %s: %v",
						handoff.ToolName,
						recovered,
					),
				},
				Info: s.eventInfo,
			})
		}
	}()

	if s.contextWrapper != nil {
		ctx = agents.ContextWithRunContextValue(ctx, s.contextWrapper)
	}

	invokedAgent, err := handoff.OnInvokeHandoff(ctx, event.Arguments)
	if err != nil {
		s.putEvent(RealtimeErrorEvent{
			Error: map[string]any{"message": fmt.Sprintf("handoff failed: %v", err)},
			Info:  s.eventInfo,
		})
		return
	}

	nextAgent := realtimeTarget
	if nextAgent == nil {
		name := strings.TrimSpace(handoff.AgentName)
		if name == "" && invokedAgent != nil {
			name = strings.TrimSpace(invokedAgent.Name)
		}
		if name == "" {
			s.putEvent(RealtimeErrorEvent{
				Error: map[string]any{
					"message": fmt.Sprintf("handoff %s returned no target agent", handoff.ToolName),
				},
				Info: s.eventInfo,
			})
			return
		}
		nextAgent = &RealtimeAgent[any]{Name: name}
	}

	s.currentAgent = nextAgent.Clone()

	updatedSettings, err := BuildModelSettingsFromAgent(
		s.currentAgent,
		s.contextWrapper,
		s.baseModelSetting,
		nil,
		s.runConfig,
	)
	if err != nil {
		s.putEvent(RealtimeErrorEvent{
			Error: map[string]any{"message": fmt.Sprintf("handoff update failed: %v", err)},
			Info:  s.eventInfo,
		})
		return
	}

	transferAgent := invokedAgent
	if transferAgent == nil {
		transferAgent = &agents.Agent{Name: s.currentAgent.Name}
	}

	s.putEvent(RealtimeHandoffEvent{
		FromAgent: agentSnapshot,
		ToAgent:   s.currentAgent,
		Info:      s.eventInfo,
	})

	if err := s.model.SendEvent(ctx, RealtimeModelSendSessionUpdate{
		SessionSettings: updatedSettings,
	}); err != nil {
		s.putEvent(RealtimeErrorEvent{
			Error: map[string]any{"message": fmt.Sprintf("failed sending handoff update: %v", err)},
			Info:  s.eventInfo,
		})
		return
	}

	if err := s.model.SendEvent(ctx, RealtimeModelSendToolOutput{
		ToolCall:      event,
		Output:        handoff.GetTransferMessage(transferAgent),
		StartResponse: true,
	}); err != nil {
		s.putEvent(RealtimeErrorEvent{
			Error: map[string]any{"message": fmt.Sprintf("failed sending handoff tool output: %v", err)},
			Info:  s.eventInfo,
		})
	}
}

func (s *RealtimeSession) putEvent(event RealtimeSessionEvent) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	if s.closed {
		return
	}
	s.eventQueue <- event
}

func (s *RealtimeSession) handleInputAudioTranscriptionCompleted(
	event RealtimeModelInputAudioTranscriptionCompletedEvent,
) (bool, any) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	if s.closed {
		return false, nil
	}

	itemID := strings.TrimSpace(event.ItemID)
	if itemID == "" {
		return false, nil
	}

	for i := range s.history {
		mapping, ok := toStringAnyMap(s.history[i])
		if !ok {
			continue
		}
		if existingID, _ := mapping["item_id"].(string); strings.TrimSpace(existingID) != itemID {
			if existingID, _ = mapping["id"].(string); strings.TrimSpace(existingID) != itemID {
				continue
			}
		}
		if itemType, _ := mapping["type"].(string); itemType != "message" {
			continue
		}
		if role, _ := mapping["role"].(string); role != "user" {
			continue
		}

		newItem := cloneStringAnyMap(mapping)
		newItem["status"] = "completed"
		newContent := make([]any, 0)
		for _, rawPart := range extractContentParts(mapping["content"]) {
			part, ok := toStringAnyMap(rawPart)
			if !ok {
				newContent = append(newContent, rawPart)
				continue
			}
			clonedPart := cloneStringAnyMap(part)
			partType, _ := clonedPart["type"].(string)
			if partType == "input_audio" {
				clonedPart["transcript"] = event.Transcript
			}
			newContent = append(newContent, clonedPart)
		}
		newItem["content"] = newContent
		s.history[i] = newItem
		return false, nil
	}

	newItem := map[string]any{
		"type":    "message",
		"role":    "user",
		"item_id": itemID,
		"status":  "completed",
		"content": []map[string]any{
			{
				"type": "input_text",
				"text": event.Transcript,
			},
		},
	}
	s.history = append(s.history, newItem)
	return true, newItem
}

func (s *RealtimeSession) handleHistoryUpdate(item any) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	if s.closed {
		return
	}

	normalizedItem := normalizeHistoryItem(item)
	itemID := itemIDFromAny(normalizedItem)
	if itemID == "" {
		s.history = append(s.history, normalizedItem)
		s.eventQueue <- RealtimeHistoryAddedEvent{Item: normalizedItem, Info: s.eventInfo}
		return
	}

	for i := range s.history {
		if itemIDFromAny(s.history[i]) == itemID {
			existingItem := normalizeHistoryItem(s.history[i])
			if merged, ok := mergeAssistantTranscriptOnHistoryUpdate(
				existingItem,
				normalizedItem,
				s.itemTranscripts[itemID],
			); ok {
				normalizedItem = merged
			}
			s.history[i] = normalizedItem
			s.eventQueue <- RealtimeHistoryUpdatedEvent{
				History: append([]any(nil), s.history...),
				Info:    s.eventInfo,
			}
			return
		}
	}

	previousItemID := previousItemIDFromAny(normalizedItem)
	if previousItemID != "" {
		for i := range s.history {
			if itemIDFromAny(s.history[i]) != previousItemID {
				continue
			}
			s.history = append(s.history, nil)
			copy(s.history[i+2:], s.history[i+1:])
			s.history[i+1] = normalizedItem
			s.eventQueue <- RealtimeHistoryAddedEvent{Item: normalizedItem, Info: s.eventInfo}
			return
		}
	}

	s.history = append(s.history, normalizedItem)
	s.eventQueue <- RealtimeHistoryAddedEvent{Item: normalizedItem, Info: s.eventInfo}
}

func (s *RealtimeSession) handleHistoryDelete(itemID string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	if s.closed {
		return
	}

	filtered := make([]any, 0, len(s.history))
	for _, item := range s.history {
		if itemIDFromAny(item) != itemID {
			filtered = append(filtered, item)
		}
	}
	s.history = filtered
	s.eventQueue <- RealtimeHistoryUpdatedEvent{
		History: append([]any(nil), s.history...),
		Info:    s.eventInfo,
	}
}

func (s *RealtimeSession) handleTranscriptDelta(event RealtimeModelTranscriptDeltaEvent) (string, bool) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	if s.closed {
		return "", false
	}

	updatedTranscript := s.itemTranscripts[event.ItemID] + event.Delta
	s.itemTranscripts[event.ItemID] = updatedTranscript

	s.updateHistoryForTranscriptLocked(event.ItemID, updatedTranscript)

	nextThreshold := (s.itemGuardrailRunCounts[event.ItemID] + 1) * s.debounceTextLength
	if len(updatedTranscript) >= nextThreshold {
		s.itemGuardrailRunCounts[event.ItemID]++
		return updatedTranscript, true
	}
	return updatedTranscript, false
}

func (s *RealtimeSession) updateHistoryForTranscriptLocked(itemID, transcript string) {
	for index := range s.history {
		itemMap, ok := toStringAnyMap(s.history[index])
		if !ok {
			continue
		}

		existingID, _ := itemMap["item_id"].(string)
		if strings.TrimSpace(existingID) == "" {
			existingID, _ = itemMap["id"].(string)
		}
		if strings.TrimSpace(existingID) != strings.TrimSpace(itemID) {
			continue
		}
		if itemType, _ := itemMap["type"].(string); itemType != "message" {
			continue
		}
		if role, _ := itemMap["role"].(string); role != "assistant" {
			continue
		}

		clonedItem := cloneStringAnyMap(itemMap)
		clonedParts := make([]any, 0)
		hasAudioPart := false
		for _, partRaw := range extractContentParts(itemMap["content"]) {
			partMap, ok := toStringAnyMap(partRaw)
			if !ok {
				clonedParts = append(clonedParts, partRaw)
				continue
			}
			clonedPart := cloneStringAnyMap(partMap)
			partType, _ := clonedPart["type"].(string)
			if partType == "audio" || partType == "output_audio" {
				clonedPart["transcript"] = transcript
				hasAudioPart = true
			}
			clonedParts = append(clonedParts, clonedPart)
		}
		if !hasAudioPart {
			clonedParts = append(clonedParts, map[string]any{
				"type":       "audio",
				"transcript": transcript,
			})
		}
		clonedItem["content"] = clonedParts
		s.history[index] = clonedItem
		return
	}

	s.history = append(s.history, map[string]any{
		"type":    "message",
		"role":    "assistant",
		"item_id": itemID,
		"status":  "in_progress",
		"content": []map[string]any{
			{
				"type":       "audio",
				"transcript": transcript,
			},
		},
	})
}

func (s *RealtimeSession) runOutputGuardrails(
	ctx context.Context,
	message string,
	responseID string,
) {
	defer func() {
		if recovered := recover(); recovered != nil {
			s.putEvent(RealtimeErrorEvent{
				Error: map[string]any{
					"message": fmt.Sprintf("panic while running output guardrails: %v", recovered),
				},
				Info: s.eventInfo,
			})
		}
	}()

	guardrails := s.collectOutputGuardrails()
	if len(guardrails) == 0 {
		return
	}
	if strings.TrimSpace(responseID) != "" && s.isResponseAlreadyInterrupted(responseID) {
		return
	}

	currentAgentName := ""
	if s.currentAgent != nil {
		currentAgentName = s.currentAgent.Name
	}
	agentSnapshot := &agents.Agent{Name: currentAgentName}

	triggered := make([]agents.OutputGuardrailResult, 0)
	for _, guardrail := range guardrails {
		result, err := runOutputGuardrailSafely(ctx, guardrail, agentSnapshot, message)
		if err != nil {
			continue
		}
		if result.Output.TripwireTriggered {
			triggered = append(triggered, result)
		}
	}
	if len(triggered) == 0 {
		return
	}

	if strings.TrimSpace(responseID) != "" && !s.markResponseInterrupted(responseID) {
		return
	}

	s.putEvent(RealtimeGuardrailTrippedEvent{
		GuardrailResults: triggered,
		Message:          message,
		Info:             s.eventInfo,
	})
	_ = s.model.SendEvent(ctx, RealtimeModelSendInterrupt{ForceResponseCancel: true})

	names := make([]string, 0, len(triggered))
	for _, result := range triggered {
		name := strings.TrimSpace(result.Guardrail.Name)
		if name == "" {
			name = "unnamed_guardrail"
		}
		names = append(names, name)
	}

	_ = s.model.SendEvent(ctx, RealtimeModelSendUserInput{
		UserInput: fmt.Sprintf("guardrail triggered: %s", strings.Join(names, ", ")),
	})
}

func runOutputGuardrailSafely(
	ctx context.Context,
	guardrail agents.OutputGuardrail,
	agentSnapshot *agents.Agent,
	message string,
) (result agents.OutputGuardrailResult, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("output guardrail panic: %v", recovered)
		}
	}()
	return guardrail.Run(ctx, agentSnapshot, message)
}

func (s *RealtimeSession) collectOutputGuardrails() []agents.OutputGuardrail {
	guardrails := make([]agents.OutputGuardrail, 0)
	seen := make(map[string]struct{})
	appendIfNew := func(guardrail agents.OutputGuardrail) {
		key := outputGuardrailDedupKey(guardrail)
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		guardrails = append(guardrails, guardrail)
	}

	if s.currentAgent != nil {
		for _, guardrail := range s.currentAgent.OutputGuardrails {
			appendIfNew(guardrail)
		}
	}
	for _, guardrail := range outputGuardrailsFromRunConfig(s.runConfig) {
		appendIfNew(guardrail)
	}
	return guardrails
}

func outputGuardrailDedupKey(guardrail agents.OutputGuardrail) string {
	functionPointer := uintptr(0)
	if guardrail.GuardrailFunction != nil {
		functionPointer = reflect.ValueOf(guardrail.GuardrailFunction).Pointer()
	}
	return fmt.Sprintf("%s:%d", guardrail.Name, functionPointer)
}

func (s *RealtimeSession) markResponseInterrupted(responseID string) bool {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	if s.closed {
		return false
	}
	if _, exists := s.interruptedResponseIDs[responseID]; exists {
		return false
	}
	s.interruptedResponseIDs[responseID] = struct{}{}
	return true
}

func (s *RealtimeSession) isResponseAlreadyInterrupted(responseID string) bool {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	_, exists := s.interruptedResponseIDs[responseID]
	return exists
}

func (s *RealtimeSession) storePendingToolCall(callID string, pending pendingRealtimeToolCall) {
	if strings.TrimSpace(callID) == "" {
		return
	}
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.pendingToolCalls[callID] = pending
}

func (s *RealtimeSession) takePendingToolCall(callID string) (pendingRealtimeToolCall, bool) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	pending, ok := s.pendingToolCalls[callID]
	if !ok {
		return pendingRealtimeToolCall{}, false
	}
	delete(s.pendingToolCalls, callID)
	return pending, true
}

func (s *RealtimeSession) clearGuardrailDebounceState() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.itemTranscripts = make(map[string]string)
	clear(s.itemGuardrailRunCounts)
}

func readDebounceTextLength(runConfig RealtimeRunConfig) int {
	debounce := 100
	settings, ok := toStringAnyMap(runConfig["guardrails_settings"])
	if !ok {
		return debounce
	}
	if value, ok := numericToInt64(settings["debounce_text_length"]); ok && value > 0 {
		return int(value)
	}
	return debounce
}

func outputGuardrailsFromRunConfig(runConfig RealtimeRunConfig) []agents.OutputGuardrail {
	raw := runConfig["output_guardrails"]
	switch v := raw.(type) {
	case []agents.OutputGuardrail:
		return slices.Clone(v)
	case []any:
		out := make([]agents.OutputGuardrail, 0, len(v))
		for _, item := range v {
			if guardrail, ok := item.(agents.OutputGuardrail); ok {
				out = append(out, guardrail)
			}
		}
		return out
	default:
		return nil
	}
}

func itemIDFromAny(item any) string {
	if item == nil {
		return ""
	}
	if mapping, ok := toStringAnyMap(item); ok {
		if id, ok := mapping["item_id"].(string); ok {
			return id
		}
		if id, ok := mapping["id"].(string); ok {
			return id
		}
	}

	value := reflect.ValueOf(item)
	if !value.IsValid() {
		return ""
	}
	if value.Kind() == reflect.Pointer {
		if value.IsNil() {
			return ""
		}
		value = value.Elem()
	}
	if value.Kind() != reflect.Struct {
		return ""
	}

	field := value.FieldByName("ItemID")
	if field.IsValid() && field.Kind() == reflect.String {
		return field.String()
	}
	field = value.FieldByName("ID")
	if field.IsValid() && field.Kind() == reflect.String {
		return field.String()
	}

	return ""
}

func normalizeHistoryItem(item any) any {
	switch typed := item.(type) {
	case RealtimeMessageItem:
		return realtimeMessageItemToMap(typed)
	case *RealtimeMessageItem:
		if typed == nil {
			return nil
		}
		return realtimeMessageItemToMap(*typed)
	case RealtimeToolCallItem:
		return realtimeToolCallItemToMap(typed)
	case *RealtimeToolCallItem:
		if typed == nil {
			return nil
		}
		return realtimeToolCallItemToMap(*typed)
	default:
		if mapping, ok := toStringAnyMap(item); ok {
			return cloneStringAnyMap(mapping)
		}
		return item
	}
}

func previousItemIDFromAny(item any) string {
	if mapping, ok := toStringAnyMap(item); ok {
		if previousID, ok := mapping["previous_item_id"].(string); ok &&
			strings.TrimSpace(previousID) != "" {
			return previousID
		}
	}

	value := reflect.ValueOf(item)
	if !value.IsValid() {
		return ""
	}
	if value.Kind() == reflect.Pointer {
		if value.IsNil() {
			return ""
		}
		value = value.Elem()
	}
	if value.Kind() != reflect.Struct {
		return ""
	}

	field := value.FieldByName("PreviousItemID")
	if !field.IsValid() {
		return ""
	}
	switch field.Kind() {
	case reflect.String:
		return strings.TrimSpace(field.String())
	case reflect.Pointer:
		if field.IsNil() {
			return ""
		}
		if field.Elem().Kind() == reflect.String {
			return strings.TrimSpace(field.Elem().String())
		}
	}
	return ""
}

func realtimeMessageItemToMap(item RealtimeMessageItem) map[string]any {
	mapping := map[string]any{
		"item_id": item.ItemID,
		"type":    item.Type,
		"role":    item.Role,
	}
	if item.PreviousItemID != nil {
		mapping["previous_item_id"] = *item.PreviousItemID
	}
	if item.Status != nil {
		mapping["status"] = *item.Status
	}

	content := make([]any, 0, len(item.Content))
	for _, part := range item.Content {
		partMap := map[string]any{
			"type": part.Type,
		}
		if part.Text != nil {
			partMap["text"] = *part.Text
		}
		if part.Audio != nil {
			partMap["audio"] = *part.Audio
		}
		if part.Transcript != nil {
			partMap["transcript"] = *part.Transcript
		}
		if part.ImageURL != nil {
			partMap["image_url"] = *part.ImageURL
		}
		if part.Detail != nil {
			partMap["detail"] = *part.Detail
		}
		content = append(content, partMap)
	}
	mapping["content"] = content
	return mapping
}

func realtimeToolCallItemToMap(item RealtimeToolCallItem) map[string]any {
	mapping := map[string]any{
		"item_id":   item.ItemID,
		"call_id":   item.CallID,
		"type":      item.Type,
		"status":    item.Status,
		"arguments": item.Arguments,
		"name":      item.Name,
	}
	if item.PreviousItemID != nil {
		mapping["previous_item_id"] = *item.PreviousItemID
	}
	if item.Output != nil {
		mapping["output"] = *item.Output
	}
	return mapping
}

func mergeAssistantTranscriptOnHistoryUpdate(
	existingItem any,
	incomingItem any,
	fallbackTranscript string,
) (any, bool) {
	existingMap, ok := toStringAnyMap(existingItem)
	if !ok {
		return incomingItem, false
	}
	incomingMap, ok := toStringAnyMap(incomingItem)
	if !ok {
		return incomingItem, false
	}
	existingRole, ok := messageRoleFromMessageMap(existingMap)
	if !ok {
		return incomingItem, false
	}
	incomingRole, ok := messageRoleFromMessageMap(incomingMap)
	if !ok || incomingRole != existingRole {
		return incomingItem, false
	}

	merged := cloneStringAnyMap(incomingMap)
	incomingParts := extractContentParts(incomingMap["content"])
	existingParts := extractContentParts(existingMap["content"])

	if incomingRole == "user" {
		incomingImageURLs := make(map[string]struct{})
		for _, incomingRaw := range incomingParts {
			incomingPart, ok := toStringAnyMap(incomingRaw)
			if !ok {
				continue
			}
			partType, _ := incomingPart["type"].(string)
			if partType != "input_image" {
				continue
			}
			imageURL, ok := stringField(incomingPart, "image_url")
			if !ok || strings.TrimSpace(imageURL) == "" {
				continue
			}
			incomingImageURLs[imageURL] = struct{}{}
		}

		missingImages := make([]any, 0)
		for _, existingRaw := range existingParts {
			existingPart, ok := toStringAnyMap(existingRaw)
			if !ok {
				continue
			}
			partType, _ := existingPart["type"].(string)
			if partType != "input_image" {
				continue
			}
			imageURL, ok := stringField(existingPart, "image_url")
			if !ok || strings.TrimSpace(imageURL) == "" {
				continue
			}
			if _, found := incomingImageURLs[imageURL]; found {
				continue
			}
			missingImages = append(missingImages, cloneStringAnyMap(existingPart))
		}
		if len(missingImages) > 0 {
			combined := make([]any, 0, len(missingImages)+len(incomingParts))
			combined = append(combined, missingImages...)
			combined = append(combined, incomingParts...)
			incomingParts = combined
		}
	}

	mergedParts := make([]any, 0, len(incomingParts))

	for idx, incomingRaw := range incomingParts {
		incomingPart, ok := toStringAnyMap(incomingRaw)
		if !ok {
			mergedParts = append(mergedParts, incomingRaw)
			continue
		}
		clonedPart := cloneStringAnyMap(incomingPart)
		partType, _ := clonedPart["type"].(string)
		shouldPreserve := false
		switch incomingRole {
		case "assistant":
			shouldPreserve = partType == "audio" || partType == "output_audio"
		case "user":
			shouldPreserve = partType == "input_audio"
		}
		if shouldPreserve {
			preserved := preserveContentFieldByIndex(
				existingParts,
				idx,
				partType,
				"transcript",
			)
			if incomingRole == "assistant" && preserved == "" && strings.TrimSpace(fallbackTranscript) != "" {
				preserved = fallbackTranscript
			}
			if current, ok := stringField(clonedPart, "transcript"); !ok || strings.TrimSpace(current) == "" {
				if preserved != "" {
					clonedPart["transcript"] = preserved
				}
			}
		}
		switch incomingRole {
		case "assistant":
			if partType == "text" || partType == "output_text" {
				if current, ok := stringField(clonedPart, "text"); !ok || strings.TrimSpace(current) == "" {
					preserved := preserveContentFieldByIndex(existingParts, idx, partType, "text")
					if preserved != "" {
						clonedPart["text"] = preserved
					}
				}
			}
		case "user":
			if partType == "input_text" {
				if current, ok := stringField(clonedPart, "text"); !ok || strings.TrimSpace(current) == "" {
					preserved := preserveContentFieldByIndex(existingParts, idx, partType, "text")
					if preserved != "" {
						clonedPart["text"] = preserved
					}
				}
			}
		case "system":
			if partType == "input_text" || partType == "text" || partType == "output_text" {
				if current, ok := stringField(clonedPart, "text"); !ok || strings.TrimSpace(current) == "" {
					preserved := preserveContentFieldByIndex(existingParts, idx, partType, "text")
					if preserved != "" {
						clonedPart["text"] = preserved
					}
				}
			}
		}
		mergedParts = append(mergedParts, clonedPart)
	}

	merged["content"] = mergedParts
	return merged, true
}

func messageRoleFromMessageMap(item map[string]any) (string, bool) {
	itemType, _ := item["type"].(string)
	role, _ := item["role"].(string)
	if itemType != "message" {
		return "", false
	}
	switch role {
	case "assistant", "user", "system":
		return role, true
	default:
		return "", false
	}
}

func preserveContentFieldByIndex(
	existingParts []any,
	idx int,
	incomingPartType string,
	fieldName string,
) string {
	if idx >= len(existingParts) {
		return ""
	}
	existingPart, ok := toStringAnyMap(existingParts[idx])
	if !ok {
		return ""
	}
	existingPartType, _ := existingPart["type"].(string)
	if !contentPartTypesCompatible(existingPartType, incomingPartType) {
		return ""
	}
	value, ok := stringField(existingPart, fieldName)
	if !ok || strings.TrimSpace(value) == "" {
		return ""
	}
	return value
}

func contentPartTypesCompatible(existingPartType, incomingPartType string) bool {
	if existingPartType == incomingPartType {
		return true
	}
	if (existingPartType == "text" || existingPartType == "output_text") &&
		(incomingPartType == "text" || incomingPartType == "output_text") {
		return true
	}
	if (existingPartType == "audio" || existingPartType == "output_audio") &&
		(incomingPartType == "audio" || incomingPartType == "output_audio") {
		return true
	}
	return false
}
