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

package agents_test

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeMCPPromptServer struct {
	prompts       []*mcp.Prompt
	promptResults map[string]string
	serverName    string
}

func newFakeMCPPromptServer() *fakeMCPPromptServer {
	return &fakeMCPPromptServer{
		promptResults: make(map[string]string),
		serverName:    "fake_prompt_server",
	}
}

func (s *fakeMCPPromptServer) addPrompt(name, description string) {
	s.prompts = append(s.prompts, &mcp.Prompt{
		Name:        name,
		Description: description,
		Arguments:   []*mcp.PromptArgument{},
	})
}

func (s *fakeMCPPromptServer) setPromptResult(name, result string) {
	s.promptResults[name] = result
}

func (s *fakeMCPPromptServer) Connect(context.Context) error { return nil }

func (s *fakeMCPPromptServer) Cleanup(context.Context) error { return nil }

func (s *fakeMCPPromptServer) Name() string { return s.serverName }

func (s *fakeMCPPromptServer) UseStructuredContent() bool { return false }

func (s *fakeMCPPromptServer) ListTools(context.Context, *agents.Agent) ([]*mcp.Tool, error) {
	return nil, nil
}

func (s *fakeMCPPromptServer) CallTool(context.Context, string, map[string]any, map[string]any) (*mcp.CallToolResult, error) {
	return nil, fmt.Errorf("fake server does not support tools")
}

func (s *fakeMCPPromptServer) ListPrompts(context.Context) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{Prompts: s.prompts}, nil
}

func (s *fakeMCPPromptServer) GetPrompt(ctx context.Context, name string, arguments map[string]string) (*mcp.GetPromptResult, error) {
	content, ok := s.promptResults[name]
	if !ok {
		return nil, fmt.Errorf("Prompt '%s' not found", name)
	}
	content = formatPromptContent(content, arguments)

	message := &mcp.PromptMessage{
		Role:    mcp.Role("user"),
		Content: &mcp.TextContent{Text: content},
	}

	return &mcp.GetPromptResult{
		Description: fmt.Sprintf("Generated prompt for %s", name),
		Messages:    []*mcp.PromptMessage{message},
	}, nil
}

var promptTokenPattern = regexp.MustCompile(`\{([a-zA-Z0-9_]+)\}`)

func formatPromptContent(template string, args map[string]string) string {
	if len(args) == 0 || !strings.Contains(template, "{") {
		return template
	}
	matches := promptTokenPattern.FindAllStringSubmatch(template, -1)
	if len(matches) == 0 {
		return template
	}
	for _, match := range matches {
		if _, ok := args[match[1]]; !ok {
			return template
		}
	}
	out := template
	for _, match := range matches {
		key := match[1]
		out = strings.ReplaceAll(out, "{"+key+"}", args[key])
	}
	return out
}

func promptText(t *testing.T, result *mcp.GetPromptResult) string {
	t.Helper()
	require.NotNil(t, result)
	require.Len(t, result.Messages, 1)
	text, ok := result.Messages[0].Content.(*mcp.TextContent)
	require.True(t, ok)
	return text.Text
}

func TestMCPPromptServerListPrompts(t *testing.T) {
	server := newFakeMCPPromptServer()
	server.addPrompt("generate_code_review_instructions", "Generate agent instructions for code review tasks")

	result, err := server.ListPrompts(t.Context())
	require.NoError(t, err)
	require.Len(t, result.Prompts, 1)
	assert.Equal(t, "generate_code_review_instructions", result.Prompts[0].Name)
	assert.Contains(t, result.Prompts[0].Description, "code review")
}

func TestMCPPromptServerGetPromptWithoutArguments(t *testing.T) {
	server := newFakeMCPPromptServer()
	server.addPrompt("simple_prompt", "A simple prompt")
	server.setPromptResult("simple_prompt", "You are a helpful assistant.")

	result, err := server.GetPrompt(t.Context(), "simple_prompt", nil)
	require.NoError(t, err)
	assert.Equal(t, "You are a helpful assistant.", promptText(t, result))
}

func TestMCPPromptServerGetPromptWithArguments(t *testing.T) {
	server := newFakeMCPPromptServer()
	server.addPrompt("generate_code_review_instructions", "Generate agent instructions for code review tasks")
	server.setPromptResult(
		"generate_code_review_instructions",
		"You are a senior {language} code review specialist. Focus on {focus}.",
	)

	result, err := server.GetPrompt(t.Context(), "generate_code_review_instructions", map[string]string{
		"focus":    "security vulnerabilities",
		"language": "python",
	})
	require.NoError(t, err)
	assert.Equal(
		t,
		"You are a senior python code review specialist. Focus on security vulnerabilities.",
		promptText(t, result),
	)
}

func TestMCPPromptServerGetPromptNotFound(t *testing.T) {
	server := newFakeMCPPromptServer()

	_, err := server.GetPrompt(t.Context(), "nonexistent", nil)
	require.Error(t, err)
	assert.ErrorContains(t, err, "Prompt 'nonexistent' not found")
}

func TestMCPPromptServerAgentInstructions(t *testing.T) {
	server := newFakeMCPPromptServer()
	server.addPrompt("generate_code_review_instructions", "Generate agent instructions for code review tasks")
	server.setPromptResult(
		"generate_code_review_instructions",
		"You are a code reviewer. Analyze the provided code for security issues.",
	)

	promptResult, err := server.GetPrompt(t.Context(), "generate_code_review_instructions", nil)
	require.NoError(t, err)
	instructions := promptText(t, promptResult)

	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("prompt_agent").WithInstructions(instructions).WithModelInstance(model)
	agent.MCPServers = []agents.MCPServer{server}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Code analysis complete. Found security vulnerability.")}},
	})

	result, err := agents.Runner{}.Run(t.Context(), agent, "Review this code: def unsafe_exec(cmd): os.system(cmd)")
	require.NoError(t, err)
	assert.Contains(t, result.FinalOutput, "Code analysis complete")
	systemPrompt, err := agent.GetSystemPrompt(t.Context())
	require.NoError(t, err)
	assert.Equal(t, "You are a code reviewer. Analyze the provided code for security issues.", systemPrompt.Or(""))
}

func TestMCPPromptServerAgentInstructionsStreaming(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("streaming=%v", streaming), func(t *testing.T) {
			server := newFakeMCPPromptServer()
			server.addPrompt("generate_code_review_instructions", "Generate agent instructions for code review tasks")
			server.setPromptResult(
				"generate_code_review_instructions",
				"You are a {language} code reviewer focusing on {focus}.",
			)

			promptResult, err := server.GetPrompt(t.Context(), "generate_code_review_instructions", map[string]string{
				"language": "Python",
				"focus":    "security",
			})
			require.NoError(t, err)
			instructions := promptText(t, promptResult)

			model := agentstesting.NewFakeModel(false, nil)
			agent := agents.New("streaming_prompt_agent").WithInstructions(instructions).WithModelInstance(model)
			agent.MCPServers = []agents.MCPServer{server}

			model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
				{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Security analysis complete.")}},
			})

			if streaming {
				streamed, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Review code")
				require.NoError(t, err)
				err = streamed.StreamEvents(func(agents.StreamEvent) error { return nil })
				require.NoError(t, err)
				final, ok := streamed.FinalOutput().(string)
				require.True(t, ok)
				assert.Contains(t, final, "Security analysis complete")
			} else {
				result, err := agents.Runner{}.Run(t.Context(), agent, "Review code")
				require.NoError(t, err)
				assert.Contains(t, result.FinalOutput, "Security analysis complete")
			}

			systemPrompt, err := agent.GetSystemPrompt(t.Context())
			require.NoError(t, err)
			assert.Equal(t, "You are a Python code reviewer focusing on security.", systemPrompt.Or(""))
		})
	}
}

func TestMCPPromptServerMultiplePrompts(t *testing.T) {
	server := newFakeMCPPromptServer()
	server.addPrompt("generate_code_review_instructions", "Generate agent instructions for code review tasks")
	server.addPrompt("generate_testing_instructions", "Generate agent instructions for testing tasks")
	server.setPromptResult("generate_code_review_instructions", "You are a code reviewer.")
	server.setPromptResult("generate_testing_instructions", "You are a test engineer.")

	prompts, err := server.ListPrompts(t.Context())
	require.NoError(t, err)
	require.Len(t, prompts.Prompts, 2)

	var names []string
	for _, prompt := range prompts.Prompts {
		names = append(names, prompt.Name)
	}
	assert.Contains(t, names, "generate_code_review_instructions")
	assert.Contains(t, names, "generate_testing_instructions")

	review, err := server.GetPrompt(t.Context(), "generate_code_review_instructions", nil)
	require.NoError(t, err)
	assert.Equal(t, "You are a code reviewer.", promptText(t, review))

	testingPrompt, err := server.GetPrompt(t.Context(), "generate_testing_instructions", nil)
	require.NoError(t, err)
	assert.Equal(t, "You are a test engineer.", promptText(t, testingPrompt))
}

func TestMCPPromptServerComplexArguments(t *testing.T) {
	server := newFakeMCPPromptServer()
	server.addPrompt("generate_detailed_instructions", "Generate detailed instructions with multiple parameters")
	server.setPromptResult(
		"generate_detailed_instructions",
		"You are a {role} specialist. Your focus is on {focus}. You work with {language} code. Your experience level is {level}.",
	)

	result, err := server.GetPrompt(t.Context(), "generate_detailed_instructions", map[string]string{
		"role":     "security",
		"focus":    "vulnerability detection",
		"language": "Python",
		"level":    "senior",
	})
	require.NoError(t, err)
	assert.Equal(
		t,
		"You are a security specialist. Your focus is on vulnerability detection. You work with Python code. Your experience level is senior.",
		promptText(t, result),
	)
}

func TestMCPPromptServerMissingArguments(t *testing.T) {
	server := newFakeMCPPromptServer()
	server.addPrompt("incomplete_prompt", "Prompt with missing arguments")
	server.setPromptResult("incomplete_prompt", "You are a {role} working on {task}.")

	result, err := server.GetPrompt(t.Context(), "incomplete_prompt", map[string]string{"role": "developer"})
	require.NoError(t, err)
	assert.Equal(t, "You are a {role} working on {task}.", promptText(t, result))
}

func TestMCPPromptServerCleanup(t *testing.T) {
	server := newFakeMCPPromptServer()
	server.addPrompt("test_prompt", "Test prompt")
	server.setPromptResult("test_prompt", "Test result")

	result, err := server.GetPrompt(t.Context(), "test_prompt", nil)
	require.NoError(t, err)
	assert.Equal(t, "Test result", promptText(t, result))

	require.NoError(t, server.Cleanup(t.Context()))

	result, err = server.GetPrompt(t.Context(), "test_prompt", nil)
	require.NoError(t, err)
	assert.Equal(t, "Test result", promptText(t, result))
}
