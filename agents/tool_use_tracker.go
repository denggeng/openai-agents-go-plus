// Copyright 2026 The NLP Odyssey Authors
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

import "sort"

type toolUseTrackerSnapshotProvider interface {
	GetToolUseTrackerSnapshot() map[string][]string
}

// AgentToolUseTrackerFromSerializable restores a tracker from a serialized snapshot.
func AgentToolUseTrackerFromSerializable(data map[string][]string) *AgentToolUseTracker {
	tracker := NewAgentToolUseTracker()
	if len(data) == 0 {
		return tracker
	}
	tracker.NameToTools = make(map[string][]string, len(data))
	for agentName, tools := range data {
		if agentName == "" {
			continue
		}
		tracker.NameToTools[agentName] = uniqueSortedStrings(tools)
	}
	return tracker
}

// SerializeToolUseTracker converts the tracker into a snapshot preserving runtime order.
func SerializeToolUseTracker(toolUseTracker *AgentToolUseTracker) map[string][]string {
	if toolUseTracker == nil {
		return map[string][]string{}
	}
	snapshot := make(map[string][]string)
	for _, item := range toolUseTracker.AgentToTools {
		if item.Agent == nil || item.Agent.Name == "" {
			continue
		}
		snapshot[item.Agent.Name] = append([]string(nil), item.ToolNames...)
	}
	return snapshot
}

// HydrateToolUseTracker seeds the tracker from the serialized snapshot, skipping unknown agents.
func HydrateToolUseTracker(
	toolUseTracker *AgentToolUseTracker,
	runState toolUseTrackerSnapshotProvider,
	startingAgent *Agent,
) {
	if toolUseTracker == nil || runState == nil || startingAgent == nil {
		return
	}
	snapshot := runState.GetToolUseTrackerSnapshot()
	if len(snapshot) == 0 {
		return
	}
	agentMap := buildAgentMap(startingAgent)
	for agentName, tools := range snapshot {
		agent := agentMap[agentName]
		if agent == nil {
			continue
		}
		toolUseTracker.AddToolUse(agent, append([]string(nil), tools...))
	}
}

// AsSerializable returns a deterministic snapshot of tool usage.
func (t *AgentToolUseTracker) AsSerializable() map[string][]string {
	if t == nil {
		return map[string][]string{}
	}
	if len(t.NameToTools) > 0 {
		out := make(map[string][]string, len(t.NameToTools))
		for agentName, tools := range t.NameToTools {
			if agentName == "" {
				continue
			}
			out[agentName] = uniqueSortedStrings(tools)
		}
		return out
	}
	snapshot := make(map[string][]string)
	for _, item := range t.AgentToTools {
		if item.Agent == nil || item.Agent.Name == "" {
			continue
		}
		existing := snapshot[item.Agent.Name]
		snapshot[item.Agent.Name] = append(existing, item.ToolNames...)
	}
	out := make(map[string][]string, len(snapshot))
	for agentName, tools := range snapshot {
		out[agentName] = uniqueSortedStrings(tools)
	}
	return out
}

func uniqueSortedStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, v := range values {
		if v == "" {
			continue
		}
		if _, ok := seen[v]; ok {
			continue
		}
		seen[v] = struct{}{}
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}

func buildAgentMap(startingAgent *Agent) map[string]*Agent {
	if startingAgent == nil {
		return map[string]*Agent{}
	}
	agentMap := make(map[string]*Agent)
	queue := []*Agent{startingAgent}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		if current == nil || current.Name == "" {
			continue
		}
		if _, ok := agentMap[current.Name]; ok {
			continue
		}
		agentMap[current.Name] = current

		for _, tool := range current.Tools {
			var agentTool *Agent
			switch typed := tool.(type) {
			case FunctionTool:
				agentTool = typed.AgentTool
			case *FunctionTool:
				if typed != nil {
					agentTool = typed.AgentTool
				}
			}
			if agentTool == nil || agentTool.Name == "" {
				continue
			}
			if _, ok := agentMap[agentTool.Name]; ok {
				continue
			}
			queue = append(queue, agentTool)
		}
	}
	return agentMap
}
