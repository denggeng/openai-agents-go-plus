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
	"github.com/stretchr/testify/require"
)

type stubTask struct {
	done bool
}

func (t *stubTask) Done() bool {
	return t.done
}

func setupCallTaskTest(t *testing.T) {
	activeCallTasksMu.Lock()
	oldTasks := activeCallTasks
	activeCallTasks = map[string]callTask{}
	activeCallTasksMu.Unlock()

	oldStartObserver := startObserver

	t.Cleanup(func() {
		activeCallTasksMu.Lock()
		activeCallTasks = oldTasks
		activeCallTasksMu.Unlock()
		startObserver = oldStartObserver
	})
}

func TestTrackCallTaskIgnoresDuplicateWebhooks(t *testing.T) {
	setupCallTaskTest(t)

	existing := &stubTask{done: false}
	activeCallTasksMu.Lock()
	activeCallTasks["call-123"] = existing
	activeCallTasksMu.Unlock()

	called := 0
	startObserver = func(callID string) callTask {
		called++
		return &stubTask{done: false}
	}

	trackCallTask("call-123")

	assert.Equal(t, 0, called)

	activeCallTasksMu.Lock()
	got := activeCallTasks["call-123"]
	activeCallTasksMu.Unlock()

	gotTask, ok := got.(*stubTask)
	require.True(t, ok)
	assert.Same(t, existing, gotTask)
}

func TestTrackCallTaskRestartsAfterCompletion(t *testing.T) {
	setupCallTaskTest(t)

	existing := &stubTask{done: true}
	activeCallTasksMu.Lock()
	activeCallTasks["call-456"] = existing
	activeCallTasksMu.Unlock()

	newTask := &stubTask{done: false}
	called := 0
	startObserver = func(callID string) callTask {
		called++
		return newTask
	}

	trackCallTask("call-456")

	assert.Equal(t, 1, called)

	activeCallTasksMu.Lock()
	got := activeCallTasks["call-456"]
	activeCallTasksMu.Unlock()

	gotTask, ok := got.(*stubTask)
	require.True(t, ok)
	assert.Same(t, newTask, gotTask)
}
