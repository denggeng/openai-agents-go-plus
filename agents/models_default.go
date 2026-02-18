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
	"strings"

	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
)

const openAIDefaultModelEnvVarName = "OPENAI_DEFAULT_MODEL"

var gpt5NoneEffortModels = map[string]struct{}{
	"gpt-5.1": {},
	"gpt-5.2": {},
}

// GetDefaultModel returns the default model name.
func GetDefaultModel() string {
	if v, ok := os.LookupEnv(openAIDefaultModelEnvVarName); ok {
		return strings.ToLower(v)
	}
	return "gpt-4.1"
}

// GPT5ReasoningSettingsRequired reports whether the model name is a GPT-5 model
// that requires reasoning settings.
func GPT5ReasoningSettingsRequired(modelName string) bool {
	if strings.HasPrefix(modelName, "gpt-5-chat") {
		return false
	}
	return strings.HasPrefix(modelName, "gpt-5")
}

// IsGPT5Default reports whether the default model is a GPT-5 model.
func IsGPT5Default() bool {
	return GPT5ReasoningSettingsRequired(GetDefaultModel())
}

// GetDefaultModelSettings returns the default model settings for the provided model name.
// If no model name is provided, it uses the current default model.
func GetDefaultModelSettings(modelName ...string) modelsettings.ModelSettings {
	model := ""
	if len(modelName) > 0 {
		model = modelName[0]
	} else {
		model = GetDefaultModel()
	}

	if GPT5ReasoningSettingsRequired(model) {
		if _, ok := gpt5NoneEffortModels[model]; ok {
			return modelsettings.ModelSettings{
				Reasoning: openai.ReasoningParam{Effort: openai.ReasoningEffortNone},
				Verbosity: param.NewOpt(modelsettings.VerbosityLow),
			}
		}
		return modelsettings.ModelSettings{
			Reasoning: openai.ReasoningParam{Effort: openai.ReasoningEffortLow},
			Verbosity: param.NewOpt(modelsettings.VerbosityLow),
		}
	}

	return modelsettings.ModelSettings{}
}

func defaultModelSettingsEqual(settings modelsettings.ModelSettings) bool {
	return reflect.DeepEqual(settings, GetDefaultModelSettings())
}

func maybeResetModelSettingsForNonGPT5(agent *Agent, modelName string, isModelInstance bool) {
	if !IsGPT5Default() {
		return
	}
	if !defaultModelSettingsEqual(agent.ModelSettings) {
		return
	}
	if !isModelInstance && GPT5ReasoningSettingsRequired(modelName) {
		return
	}
	agent.ModelSettings = modelsettings.ModelSettings{}
}
