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
	"context"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/stretchr/testify/require"
)

func TestInstructionsFromAnyValidAsyncSignature(t *testing.T) {
	valid := func(_ context.Context, _ *agents.Agent) (string, error) {
		return "valid async instructions", nil
	}
	ig, err := agents.InstructionsFromAny(valid)
	require.NoError(t, err)

	value, err := ig.GetInstructions(context.Background(), &agents.Agent{Name: "test"})
	require.NoError(t, err)
	require.Equal(t, "valid async instructions", value)
}

func TestInstructionsFromAnyValidSyncSignature(t *testing.T) {
	valid := func(_ context.Context, _ *agents.Agent) string {
		return "valid sync instructions"
	}
	ig, err := agents.InstructionsFromAny(valid)
	require.NoError(t, err)

	value, err := ig.GetInstructions(context.Background(), &agents.Agent{Name: "test"})
	require.NoError(t, err)
	require.Equal(t, "valid sync instructions", value)
}

func TestInstructionsFromAnyOneParameterRaises(t *testing.T) {
	invalid := func(_ context.Context) string {
		return "should fail"
	}
	_, err := agents.InstructionsFromAny(invalid)
	require.Error(t, err)
	require.Contains(t, err.Error(), "must accept exactly 2 arguments")
	require.Contains(t, err.Error(), "got 1")
}

func TestInstructionsFromAnyThreeParametersRaises(t *testing.T) {
	invalid := func(_ context.Context, _ *agents.Agent, _ string) string {
		return "should fail"
	}
	_, err := agents.InstructionsFromAny(invalid)
	require.Error(t, err)
	require.Contains(t, err.Error(), "must accept exactly 2 arguments")
	require.Contains(t, err.Error(), "got 3")
}

func TestInstructionsFromAnyZeroParametersRaises(t *testing.T) {
	invalid := func() string {
		return "should fail"
	}
	_, err := agents.InstructionsFromAny(invalid)
	require.Error(t, err)
	require.Contains(t, err.Error(), "must accept exactly 2 arguments")
	require.Contains(t, err.Error(), "got 0")
}

func TestInstructionsFromAnyVariadicFails(t *testing.T) {
	invalid := func(_ context.Context, _ *agents.Agent, _ ...string) string {
		return "should fail"
	}
	_, err := agents.InstructionsFromAny(invalid)
	require.Error(t, err)
	require.Contains(t, err.Error(), "must accept exactly 2 arguments")
}

func TestInstructionsFromAnyStringStillWorks(t *testing.T) {
	ig, err := agents.InstructionsFromAny("Static string instructions")
	require.NoError(t, err)
	require.NotNil(t, ig)

	value, err := ig.GetInstructions(context.Background(), &agents.Agent{Name: "test"})
	require.NoError(t, err)
	require.Equal(t, "Static string instructions", value)
}

func TestInstructionsFromAnyNilReturnsNil(t *testing.T) {
	ig, err := agents.InstructionsFromAny(nil)
	require.NoError(t, err)
	require.Nil(t, ig)
}

func TestInstructionsFromAnyNonCallableRaises(t *testing.T) {
	_, err := agents.InstructionsFromAny(123)
	require.Error(t, err)
	require.Contains(t, err.Error(), "Agent instructions must be a string, callable, or nil")
	require.Contains(t, err.Error(), "int")
}
