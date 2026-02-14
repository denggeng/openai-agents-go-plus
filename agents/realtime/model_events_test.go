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
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAllRealtimeModelEventsHaveType(t *testing.T) {
	context := "context"

	events := []RealtimeModelEvent{
		RealtimeModelErrorEvent{},
		RealtimeModelToolCallEvent{},
		RealtimeModelAudioEvent{},
		RealtimeModelAudioInterruptedEvent{},
		RealtimeModelAudioDoneEvent{},
		RealtimeModelInputAudioTranscriptionCompletedEvent{},
		RealtimeModelInputAudioTimeoutTriggeredEvent{},
		RealtimeModelTranscriptDeltaEvent{},
		RealtimeModelItemUpdatedEvent{},
		RealtimeModelItemDeletedEvent{},
		RealtimeModelConnectionStatusEvent{},
		RealtimeModelTurnStartedEvent{},
		RealtimeModelTurnEndedEvent{},
		RealtimeModelOtherEvent{},
		RealtimeModelExceptionEvent{Exception: errors.New("x"), Context: &context},
		RealtimeModelRawServerEvent{},
	}

	assert.NotEmpty(t, events)
	for _, event := range events {
		assert.NotEmpty(t, event.Type())
	}
}
