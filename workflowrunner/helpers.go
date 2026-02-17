package workflowrunner

import "github.com/denggeng/openai-agents-go-plus/agents"

func displayAgentName(agent *agents.Agent) string {
	if agent == nil {
		return ""
	}
	return agent.Name
}
