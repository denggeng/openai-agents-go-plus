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

package agents_test

import (
	"runtime"
	"testing"
	"time"

	agents "github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/require"
)

func TestAgentIsReleasedAfterRun(t *testing.T) {
	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Paris"),
		},
	})
	agent := &agents.Agent{
		Name:         "leak-test-agent",
		Instructions: agents.InstructionsStr("Answer questions."),
		Model:        param.NewOpt(agents.NewAgentModel(model)),
	}

	finalized := make(chan struct{})
	runtime.SetFinalizer(agent, func(*agents.Agent) {
		close(finalized)
	})

	_, err := agents.Runner{}.Run(t.Context(), agent, "What is the capital of France?")
	require.NoError(t, err)

	agent = nil
	waitForFinalizer(t, finalized)
}

func waitForFinalizer(t *testing.T, done <-chan struct{}) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		runtime.GC()
		select {
		case <-done:
			return
		default:
		}
		if time.Now().After(deadline) {
			t.Fatal("agent was not collected")
		}
		time.Sleep(10 * time.Millisecond)
	}
}
