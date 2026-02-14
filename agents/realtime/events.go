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

import "github.com/nlpodyssey/openai-agents-go/agents"

const (
	realtimeSessionEventTypeAgentStart                = "agent_start"
	realtimeSessionEventTypeAgentEnd                  = "agent_end"
	realtimeSessionEventTypeHandoff                   = "handoff"
	realtimeSessionEventTypeToolStart                 = "tool_start"
	realtimeSessionEventTypeToolEnd                   = "tool_end"
	realtimeSessionEventTypeToolApprovalRequired      = "tool_approval_required"
	realtimeSessionEventTypeRawModelEvent             = "raw_model_event"
	realtimeSessionEventTypeAudioEnd                  = "audio_end"
	realtimeSessionEventTypeAudio                     = "audio"
	realtimeSessionEventTypeAudioInterrupted          = "audio_interrupted"
	realtimeSessionEventTypeError                     = "error"
	realtimeSessionEventTypeHistoryUpdated            = "history_updated"
	realtimeSessionEventTypeHistoryAdded              = "history_added"
	realtimeSessionEventTypeGuardrailTripped          = "guardrail_tripped"
	realtimeSessionEventTypeInputAudioTimeoutDetected = "input_audio_timeout_triggered"
)

// RealtimeEventInfo stores common metadata for session events.
type RealtimeEventInfo struct {
	Context *agents.RunContextWrapper[any]
}

// RealtimeSessionEvent is emitted by realtime sessions for high-level lifecycle updates.
type RealtimeSessionEvent interface {
	Type() string
}

// RealtimeAgentStartEvent indicates a new active agent.
type RealtimeAgentStartEvent struct {
	Agent any
	Info  RealtimeEventInfo
}

func (RealtimeAgentStartEvent) Type() string { return realtimeSessionEventTypeAgentStart }

// RealtimeAgentEndEvent indicates an agent turn end.
type RealtimeAgentEndEvent struct {
	Agent any
	Info  RealtimeEventInfo
}

func (RealtimeAgentEndEvent) Type() string { return realtimeSessionEventTypeAgentEnd }

// RealtimeHandoffEvent indicates handoff between agents.
type RealtimeHandoffEvent struct {
	FromAgent any
	ToAgent   any
	Info      RealtimeEventInfo
}

func (RealtimeHandoffEvent) Type() string { return realtimeSessionEventTypeHandoff }

// RealtimeToolStartEvent indicates tool call start.
type RealtimeToolStartEvent struct {
	Agent     any
	Tool      agents.Tool
	Arguments string
	Info      RealtimeEventInfo
}

func (RealtimeToolStartEvent) Type() string { return realtimeSessionEventTypeToolStart }

// RealtimeToolEndEvent indicates tool call end.
type RealtimeToolEndEvent struct {
	Agent     any
	Tool      agents.Tool
	Arguments string
	Output    any
	Info      RealtimeEventInfo
}

func (RealtimeToolEndEvent) Type() string { return realtimeSessionEventTypeToolEnd }

// RealtimeToolApprovalRequiredEvent indicates human approval requirement.
type RealtimeToolApprovalRequiredEvent struct {
	Agent     any
	Tool      agents.Tool
	CallID    string
	Arguments string
	Info      RealtimeEventInfo
}

func (RealtimeToolApprovalRequiredEvent) Type() string {
	return realtimeSessionEventTypeToolApprovalRequired
}

// RealtimeRawModelEvent forwards raw model events.
type RealtimeRawModelEvent struct {
	Data RealtimeModelEvent
	Info RealtimeEventInfo
}

func (RealtimeRawModelEvent) Type() string { return realtimeSessionEventTypeRawModelEvent }

// RealtimeAudioEndEvent indicates output audio completion.
type RealtimeAudioEndEvent struct {
	Info         RealtimeEventInfo
	ItemID       string
	ContentIndex int
}

func (RealtimeAudioEndEvent) Type() string { return realtimeSessionEventTypeAudioEnd }

// RealtimeAudioEvent wraps audio bytes emitted by model.
type RealtimeAudioEvent struct {
	Audio        RealtimeModelAudioEvent
	ItemID       string
	ContentIndex int
	Info         RealtimeEventInfo
}

func (RealtimeAudioEvent) Type() string { return realtimeSessionEventTypeAudio }

// RealtimeAudioInterruptedEvent indicates barge-in interruption.
type RealtimeAudioInterruptedEvent struct {
	Info         RealtimeEventInfo
	ItemID       string
	ContentIndex int
}

func (RealtimeAudioInterruptedEvent) Type() string {
	return realtimeSessionEventTypeAudioInterrupted
}

// RealtimeErrorEvent indicates a session-level error.
type RealtimeErrorEvent struct {
	Error any
	Info  RealtimeEventInfo
}

func (RealtimeErrorEvent) Type() string { return realtimeSessionEventTypeError }

// RealtimeHistoryUpdatedEvent contains full latest history.
type RealtimeHistoryUpdatedEvent struct {
	History []any
	Info    RealtimeEventInfo
}

func (RealtimeHistoryUpdatedEvent) Type() string { return realtimeSessionEventTypeHistoryUpdated }

// RealtimeHistoryAddedEvent indicates one new history item.
type RealtimeHistoryAddedEvent struct {
	Item any
	Info RealtimeEventInfo
}

func (RealtimeHistoryAddedEvent) Type() string { return realtimeSessionEventTypeHistoryAdded }

// RealtimeGuardrailTrippedEvent indicates output guardrail interruption.
type RealtimeGuardrailTrippedEvent struct {
	GuardrailResults []agents.OutputGuardrailResult
	Message          string
	Info             RealtimeEventInfo
}

func (RealtimeGuardrailTrippedEvent) Type() string { return realtimeSessionEventTypeGuardrailTripped }

// RealtimeInputAudioTimeoutTriggeredEvent indicates user input inactivity timeout.
type RealtimeInputAudioTimeoutTriggeredEvent struct {
	Info RealtimeEventInfo
}

func (RealtimeInputAudioTimeoutTriggeredEvent) Type() string {
	return realtimeSessionEventTypeInputAudioTimeoutDetected
}
