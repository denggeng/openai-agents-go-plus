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
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDropAgentToolRunResultHandlesClearedGlobals(t *testing.T) {
	agentToolStateMu.Lock()
	prevResults := agentToolRunResultsByObj
	prevSignature := agentToolRunResultsBySignature
	prevSignatureByObj := agentToolRunResultSignatureByObj
	prevRefs := agentToolCallRefsByObj

	agentToolRunResultsByObj = nil
	agentToolRunResultsBySignature = nil
	agentToolRunResultSignatureByObj = nil
	agentToolCallRefsByObj = nil
	agentToolStateMu.Unlock()

	t.Cleanup(func() {
		agentToolStateMu.Lock()
		agentToolRunResultsByObj = prevResults
		agentToolRunResultsBySignature = prevSignature
		agentToolRunResultSignatureByObj = prevSignatureByObj
		agentToolCallRefsByObj = prevRefs
		agentToolStateMu.Unlock()
	})

	require.NotPanics(t, func() {
		dropAgentToolRunResultByID(123)
	})
}
