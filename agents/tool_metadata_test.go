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
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/computer"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type dummyEditor struct{}

func (dummyEditor) CreateFile(agents.ApplyPatchOperation) (any, error) { return nil, nil }
func (dummyEditor) UpdateFile(agents.ApplyPatchOperation) (any, error) { return nil, nil }
func (dummyEditor) DeleteFile(agents.ApplyPatchOperation) (any, error) { return nil, nil }

func TestToolNameProperties(t *testing.T) {
	var dummyComputer computer.Computer

	assert.Equal(t, "file_search", agents.FileSearchTool{}.ToolName())
	assert.Equal(t, "web_search", agents.WebSearchTool{}.ToolName())
	assert.Equal(t, "computer_use_preview", agents.ComputerTool{Computer: dummyComputer}.ToolName())
	assert.Equal(t, "hosted_mcp", agents.HostedMCPTool{}.ToolName())
	assert.Equal(t, "code_interpreter", agents.CodeInterpreterTool{}.ToolName())
	assert.Equal(t, "image_generation", agents.ImageGenerationTool{}.ToolName())
	assert.Equal(t, "local_shell", agents.LocalShellTool{}.ToolName())

	shellTool := agents.ShellTool{
		Executor: func(context.Context, agents.ShellCommandRequest) (any, error) { return "ok", nil },
	}
	require.NoError(t, shellTool.Normalize())
	assert.Equal(t, "shell", shellTool.ToolName())
	assert.Equal(t, map[string]any{"type": "local"}, shellTool.Environment)

	assert.Equal(t, "apply_patch", agents.ApplyPatchTool{Editor: dummyEditor{}}.ToolName())
}

func TestShellCommandOutputStatusProperty(t *testing.T) {
	output := agents.ShellCommandOutput{
		Outcome: agents.ShellCallOutcome{Type: agents.ShellCallOutcomeTimeout},
	}
	assert.Equal(t, "timeout", output.Status())
}

func TestToolDataFromContext(t *testing.T) {
	toolCall := responses.ResponseFunctionToolCall{
		CallID:    "123",
		Name:      "demo",
		Arguments: "{}",
		Type:      constant.ValueOf[constant.FunctionCall](),
	}

	ctx := agents.ContextWithToolData(context.Background(), "123", toolCall)
	toolData := agents.ToolDataFromContext(ctx)
	require.NotNil(t, toolData)
	assert.Equal(t, "demo", toolData.ToolName)
	assert.Equal(t, "123", toolData.ToolCallID)
	assert.Equal(t, "{}", toolData.ToolArguments)
}
