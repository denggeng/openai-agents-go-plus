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

const (
	realtimeSendEventTypeRawMessage    = "raw_message"
	realtimeSendEventTypeUserInput     = "user_input"
	realtimeSendEventTypeAudio         = "audio"
	realtimeSendEventTypeToolOutput    = "tool_output"
	realtimeSendEventTypeInterrupt     = "interrupt"
	realtimeSendEventTypeSessionUpdate = "session_update"
)

// RealtimeModelSendEvent is a user/system event that can be sent to a realtime model.
type RealtimeModelSendEvent interface {
	Type() string
}

// RealtimeModelRawClientMessage is a raw message to send to the realtime transport.
type RealtimeModelRawClientMessage struct {
	Type      string
	OtherData map[string]any
}

// RealtimeModelInputTextContent is an input text content part.
type RealtimeModelInputTextContent struct {
	Type string
	Text string
}

// RealtimeModelInputImageContent is an input image content part.
type RealtimeModelInputImageContent struct {
	Type     string
	ImageURL string
	Detail   *string
}

// RealtimeModelUserInputContent is a user message content part.
type RealtimeModelUserInputContent struct {
	Type     string
	Text     *string
	ImageURL *string
	Detail   *string
}

// RealtimeModelUserInputMessage is a structured user message input.
type RealtimeModelUserInputMessage struct {
	Type    string
	Role    string
	Content []RealtimeModelUserInputContent
}

// RealtimeModelSendRawMessage sends a raw transport message.
type RealtimeModelSendRawMessage struct {
	Message RealtimeModelRawClientMessage
}

func (RealtimeModelSendRawMessage) Type() string { return realtimeSendEventTypeRawMessage }

// RealtimeModelSendUserInput sends a user input payload.
//
// Supported input shapes:
// 1. string
// 2. RealtimeModelUserInputMessage
// 3. map[string]any with "type"/"role"/"content" fields
type RealtimeModelSendUserInput struct {
	UserInput any
}

func (RealtimeModelSendUserInput) Type() string { return realtimeSendEventTypeUserInput }

// RealtimeModelSendAudio sends binary audio data.
type RealtimeModelSendAudio struct {
	Audio  []byte
	Commit bool
}

func (RealtimeModelSendAudio) Type() string { return realtimeSendEventTypeAudio }

// RealtimeModelSendToolOutput sends tool output back to the model.
type RealtimeModelSendToolOutput struct {
	ToolCall      RealtimeModelToolCallEvent
	Output        string
	StartResponse bool
}

func (RealtimeModelSendToolOutput) Type() string { return realtimeSendEventTypeToolOutput }

// RealtimeModelSendInterrupt requests response interruption/truncation.
type RealtimeModelSendInterrupt struct {
	ForceResponseCancel bool
}

func (RealtimeModelSendInterrupt) Type() string { return realtimeSendEventTypeInterrupt }

// RealtimeModelSendSessionUpdate sends updated session settings.
type RealtimeModelSendSessionUpdate struct {
	SessionSettings RealtimeSessionModelSettings
}

func (RealtimeModelSendSessionUpdate) Type() string {
	return realtimeSendEventTypeSessionUpdate
}
