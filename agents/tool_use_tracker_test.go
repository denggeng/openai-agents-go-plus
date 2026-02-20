package agents_test

import (
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/stretchr/testify/assert"
)

func TestToolUseTrackerAsSerializableUsesNameMapOrRuntimeSnapshot(t *testing.T) {
	tracker := agents.NewAgentToolUseTracker()
	tracker.NameToTools = map[string][]string{
		"agent-a": {"tool-b", "tool-a", "tool-b"},
	}
	assert.Equal(t, map[string][]string{
		"agent-a": {"tool-a", "tool-b"},
	}, tracker.AsSerializable())

	runtimeTracker := agents.NewAgentToolUseTracker()
	agent := &agents.Agent{Name: "runtime-agent"}
	runtimeTracker.AddToolUse(agent, []string{"beta", "alpha"})
	assert.Equal(t, map[string][]string{
		"runtime-agent": {"alpha", "beta"},
	}, runtimeTracker.AsSerializable())
}

func TestToolUseTrackerFromAndSerializeSnapshots(t *testing.T) {
	hydrated := agents.AgentToolUseTrackerFromSerializable(map[string][]string{
		"agent": {"tool-2", "tool-1"},
	})
	assert.Equal(t, map[string][]string{
		"agent": {"tool-1", "tool-2"},
	}, hydrated.AsSerializable())

	runtimeTracker := agents.NewAgentToolUseTracker()
	agent := &agents.Agent{Name: "serialize-agent"}
	runtimeTracker.AddToolUse(agent, []string{"one"})
	runtimeTracker.AddToolUse(agent, []string{"two"})
	assert.Equal(t, map[string][]string{
		"serialize-agent": {"one", "two"},
	}, agents.SerializeToolUseTracker(runtimeTracker))
}

type toolUseTrackerStub struct {
	snapshot map[string][]string
}

func (t toolUseTrackerStub) GetToolUseTrackerSnapshot() map[string][]string {
	return t.snapshot
}

func TestHydrateToolUseTrackerSkipsUnknownAgents(t *testing.T) {
	startingAgent := &agents.Agent{Name: "known-agent"}
	tracker := agents.NewAgentToolUseTracker()
	state := toolUseTrackerStub{
		snapshot: map[string][]string{
			"known-agent":   {"known_tool"},
			"missing-agent": {"missing_tool"},
		},
	}

	agents.HydrateToolUseTracker(tracker, state, startingAgent)

	assert.True(t, tracker.HasUsedTools(startingAgent))
	assert.Equal(t, map[string][]string{
		"known-agent": {"known_tool"},
	}, tracker.AsSerializable())
}
