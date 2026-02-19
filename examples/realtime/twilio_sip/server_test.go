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

package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type fakeCallTask struct {
	done bool
}

func (t *fakeCallTask) Done() bool { return t.done }

func TestTrackCallTaskIgnoresDuplicateWebhooks(t *testing.T) {
	originalStartObserver := startObserver
	activeCallTasksMu.Lock()
	originalTasks := activeCallTasks
	activeCallTasks = map[string]callTask{}
	activeCallTasksMu.Unlock()

	t.Cleanup(func() {
		startObserver = originalStartObserver
		activeCallTasksMu.Lock()
		activeCallTasks = originalTasks
		activeCallTasksMu.Unlock()
	})

	existing := &fakeCallTask{done: false}
	activeCallTasksMu.Lock()
	activeCallTasks["call-123"] = existing
	activeCallTasksMu.Unlock()

	startCalled := false
	startObserver = func(callID string) callTask {
		startCalled = true
		return &fakeCallTask{done: false}
	}

	trackCallTask("call-123")

	assert.False(t, startCalled)
	activeCallTasksMu.Lock()
	current := activeCallTasks["call-123"]
	activeCallTasksMu.Unlock()
	assert.Same(t, existing, current)
}

func TestTrackCallTaskRestartsAfterCompletion(t *testing.T) {
	originalStartObserver := startObserver
	activeCallTasksMu.Lock()
	originalTasks := activeCallTasks
	activeCallTasks = map[string]callTask{}
	activeCallTasksMu.Unlock()

	t.Cleanup(func() {
		startObserver = originalStartObserver
		activeCallTasksMu.Lock()
		activeCallTasks = originalTasks
		activeCallTasksMu.Unlock()
	})

	existing := &fakeCallTask{done: true}
	activeCallTasksMu.Lock()
	activeCallTasks["call-456"] = existing
	activeCallTasksMu.Unlock()

	newTask := &fakeCallTask{done: false}
	startCalled := false
	startObserver = func(callID string) callTask {
		startCalled = true
		return newTask
	}

	trackCallTask("call-456")

	assert.True(t, startCalled)
	activeCallTasksMu.Lock()
	current := activeCallTasks["call-456"]
	activeCallTasksMu.Unlock()
	assert.Same(t, newTask, current)
}
