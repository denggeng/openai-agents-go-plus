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

type RealtimeConnectionStatus string

const (
	RealtimeConnectionStatusConnecting   RealtimeConnectionStatus = "connecting"
	RealtimeConnectionStatusConnected    RealtimeConnectionStatus = "connected"
	RealtimeConnectionStatusDisconnected RealtimeConnectionStatus = "disconnected"
)

const (
	realtimeModelEventTypeError                            = "error"
	realtimeModelEventTypeFunctionCall                     = "function_call"
	realtimeModelEventTypeAudio                            = "audio"
	realtimeModelEventTypeAudioInterrupted                 = "audio_interrupted"
	realtimeModelEventTypeAudioDone                        = "audio_done"
	realtimeModelEventTypeInputAudioTranscriptionCompleted = "input_audio_transcription_completed"
	realtimeModelEventTypeInputAudioTimeoutTriggered       = "input_audio_timeout_triggered"
	realtimeModelEventTypeTranscriptDelta                  = "transcript_delta"
	realtimeModelEventTypeItemUpdated                      = "item_updated"
	realtimeModelEventTypeItemDeleted                      = "item_deleted"
	realtimeModelEventTypeConnectionStatus                 = "connection_status"
	realtimeModelEventTypeTurnStarted                      = "turn_started"
	realtimeModelEventTypeTurnEnded                        = "turn_ended"
	realtimeModelEventTypeOther                            = "other"
	realtimeModelEventTypeException                        = "exception"
	realtimeModelEventTypeRawServerEvent                   = "raw_server_event"
)

// RealtimeModelEvent is a transport-level event emitted by realtime models.
type RealtimeModelEvent interface {
	Type() string
}

// RealtimeModelErrorEvent represents a transport-layer error.
type RealtimeModelErrorEvent struct {
	Error any
}

func (RealtimeModelErrorEvent) Type() string { return realtimeModelEventTypeError }

// RealtimeModelToolCallEvent is emitted when the model calls a function/tool.
type RealtimeModelToolCallEvent struct {
	Name           string
	CallID         string
	Arguments      string
	ID             *string
	PreviousItemID *string
}

func (RealtimeModelToolCallEvent) Type() string { return realtimeModelEventTypeFunctionCall }

// RealtimeModelAudioEvent contains raw audio bytes emitted by the model.
type RealtimeModelAudioEvent struct {
	Data         []byte
	ResponseID   string
	ItemID       string
	ContentIndex int
}

func (RealtimeModelAudioEvent) Type() string { return realtimeModelEventTypeAudio }

// RealtimeModelAudioInterruptedEvent indicates audio playback interruption.
type RealtimeModelAudioInterruptedEvent struct {
	ItemID       string
	ContentIndex int
}

func (RealtimeModelAudioInterruptedEvent) Type() string {
	return realtimeModelEventTypeAudioInterrupted
}

// RealtimeModelAudioDoneEvent indicates audio output completion.
type RealtimeModelAudioDoneEvent struct {
	ItemID       string
	ContentIndex int
}

func (RealtimeModelAudioDoneEvent) Type() string { return realtimeModelEventTypeAudioDone }

// RealtimeModelInputAudioTranscriptionCompletedEvent indicates input transcription completion.
type RealtimeModelInputAudioTranscriptionCompletedEvent struct {
	ItemID     string
	Transcript string
}

func (RealtimeModelInputAudioTranscriptionCompletedEvent) Type() string {
	return realtimeModelEventTypeInputAudioTranscriptionCompleted
}

// RealtimeModelInputAudioTimeoutTriggeredEvent indicates VAD idle timeout trigger.
type RealtimeModelInputAudioTimeoutTriggeredEvent struct {
	ItemID       string
	AudioStartMS int
	AudioEndMS   int
}

func (RealtimeModelInputAudioTimeoutTriggeredEvent) Type() string {
	return realtimeModelEventTypeInputAudioTimeoutTriggered
}

// RealtimeModelTranscriptDeltaEvent contains partial transcript output.
type RealtimeModelTranscriptDeltaEvent struct {
	ItemID     string
	Delta      string
	ResponseID string
}

func (RealtimeModelTranscriptDeltaEvent) Type() string {
	return realtimeModelEventTypeTranscriptDelta
}

// RealtimeModelItemUpdatedEvent indicates history item creation/update.
type RealtimeModelItemUpdatedEvent struct {
	Item any
}

func (RealtimeModelItemUpdatedEvent) Type() string { return realtimeModelEventTypeItemUpdated }

// RealtimeModelItemDeletedEvent indicates history item deletion.
type RealtimeModelItemDeletedEvent struct {
	ItemID string
}

func (RealtimeModelItemDeletedEvent) Type() string { return realtimeModelEventTypeItemDeleted }

// RealtimeModelConnectionStatusEvent indicates connection state changes.
type RealtimeModelConnectionStatusEvent struct {
	Status RealtimeConnectionStatus
}

func (RealtimeModelConnectionStatusEvent) Type() string {
	return realtimeModelEventTypeConnectionStatus
}

// RealtimeModelTurnStartedEvent indicates turn generation start.
type RealtimeModelTurnStartedEvent struct{}

func (RealtimeModelTurnStartedEvent) Type() string { return realtimeModelEventTypeTurnStarted }

// RealtimeModelTurnEndedEvent indicates turn generation end.
type RealtimeModelTurnEndedEvent struct{}

func (RealtimeModelTurnEndedEvent) Type() string { return realtimeModelEventTypeTurnEnded }

// RealtimeModelOtherEvent is a catch-all for vendor specific events.
type RealtimeModelOtherEvent struct {
	Data any
}

func (RealtimeModelOtherEvent) Type() string { return realtimeModelEventTypeOther }

// RealtimeModelExceptionEvent indicates a local exception while processing events.
type RealtimeModelExceptionEvent struct {
	Exception error
	Context   *string
}

func (RealtimeModelExceptionEvent) Type() string { return realtimeModelEventTypeException }

// RealtimeModelRawServerEvent wraps unparsed server event payloads.
type RealtimeModelRawServerEvent struct {
	Data any
}

func (RealtimeModelRawServerEvent) Type() string { return realtimeModelEventTypeRawServerEvent }
