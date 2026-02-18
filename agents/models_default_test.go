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
	"os"
	"reflect"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
)

func unsetEnvForTest(t *testing.T, key string) {
	t.Helper()
	value, ok := os.LookupEnv(key)
	if ok {
		_ = os.Unsetenv(key)
		t.Cleanup(func() {
			_ = os.Setenv(key, value)
		})
		return
	}
	t.Cleanup(func() {
		_ = os.Unsetenv(key)
	})
}

func TestDefaultModelIsGPT41(t *testing.T) {
	unsetEnvForTest(t, openAIDefaultModelEnvVarName)

	assert.Equal(t, "gpt-4.1", GetDefaultModel())
	assert.False(t, IsGPT5Default())
	assert.False(t, GPT5ReasoningSettingsRequired(GetDefaultModel()))

	settings := GetDefaultModelSettings()
	assert.Equal(t, openai.ReasoningParam{}, settings.Reasoning)
	assert.False(t, settings.Verbosity.Valid())
}

func TestDefaultModelEnvGPT5(t *testing.T) {
	t.Setenv(openAIDefaultModelEnvVarName, "gpt-5")

	assert.Equal(t, "gpt-5", GetDefaultModel())
	assert.True(t, IsGPT5Default())
	assert.True(t, GPT5ReasoningSettingsRequired(GetDefaultModel()))

	settings := GetDefaultModelSettings()
	assert.Equal(t, openai.ReasoningEffortLow, settings.Reasoning.Effort)
	assert.Equal(t, param.NewOpt(modelsettings.VerbosityLow), settings.Verbosity)
}

func TestDefaultModelEnvGPT51(t *testing.T) {
	t.Setenv(openAIDefaultModelEnvVarName, "gpt-5.1")

	settings := GetDefaultModelSettings()
	assert.True(t, IsGPT5Default())
	assert.Equal(t, openai.ReasoningEffortNone, settings.Reasoning.Effort)
}

func TestDefaultModelEnvGPT52(t *testing.T) {
	t.Setenv(openAIDefaultModelEnvVarName, "gpt-5.2")

	settings := GetDefaultModelSettings()
	assert.True(t, IsGPT5Default())
	assert.Equal(t, openai.ReasoningEffortNone, settings.Reasoning.Effort)
}

func TestDefaultModelEnvGPT52Codex(t *testing.T) {
	t.Setenv(openAIDefaultModelEnvVarName, "gpt-5.2-codex")

	settings := GetDefaultModelSettings()
	assert.True(t, IsGPT5Default())
	assert.Equal(t, openai.ReasoningEffortLow, settings.Reasoning.Effort)
}

func TestDefaultModelEnvGPT5Mini(t *testing.T) {
	t.Setenv(openAIDefaultModelEnvVarName, "gpt-5-mini")

	settings := GetDefaultModelSettings()
	assert.True(t, IsGPT5Default())
	assert.Equal(t, openai.ReasoningEffortLow, settings.Reasoning.Effort)
}

func TestDefaultModelEnvGPT5Nano(t *testing.T) {
	t.Setenv(openAIDefaultModelEnvVarName, "gpt-5-nano")

	settings := GetDefaultModelSettings()
	assert.True(t, IsGPT5Default())
	assert.Equal(t, openai.ReasoningEffortLow, settings.Reasoning.Effort)
}

func TestDefaultModelEnvGPT5ChatLatest(t *testing.T) {
	t.Setenv(openAIDefaultModelEnvVarName, "gpt-5-chat-latest")

	settings := GetDefaultModelSettings()
	assert.False(t, IsGPT5Default())
	assert.False(t, GPT5ReasoningSettingsRequired(GetDefaultModel()))
	assert.Equal(t, openai.ReasoningParam{}, settings.Reasoning)
}

func TestDefaultModelEnvGPT4O(t *testing.T) {
	t.Setenv(openAIDefaultModelEnvVarName, "gpt-4o")

	settings := GetDefaultModelSettings()
	assert.False(t, IsGPT5Default())
	assert.False(t, GPT5ReasoningSettingsRequired(GetDefaultModel()))
	assert.Equal(t, openai.ReasoningParam{}, settings.Reasoning)
}

func TestAgentUsesGPT5DefaultModelSettings(t *testing.T) {
	t.Setenv(openAIDefaultModelEnvVarName, "gpt-5")

	agent := New("test")
	assert.False(t, agent.Model.Valid())
	assert.Equal(t, openai.ReasoningEffortLow, agent.ModelSettings.Reasoning.Effort)
	assert.Equal(t, param.NewOpt(modelsettings.VerbosityLow), agent.ModelSettings.Verbosity)
}

func TestAgentResetsModelSettingsForNonGPT5Models(t *testing.T) {
	t.Setenv(openAIDefaultModelEnvVarName, "gpt-5")

	agent := New("test").WithModel("gpt-4o")
	assert.True(t, agent.Model.Valid())
	assert.True(t, reflect.DeepEqual(modelsettings.ModelSettings{}, agent.ModelSettings))
}
