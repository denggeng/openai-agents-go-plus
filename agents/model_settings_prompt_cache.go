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

import (
	"strings"

	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

func promptCacheRetentionAPIValue(value modelsettings.PromptCacheRetention) string {
	switch value {
	case modelsettings.PromptCacheRetentionInMemory:
		return "in-memory"
	case "":
		return ""
	default:
		return strings.ReplaceAll(string(value), "_", "-")
	}
}

func chatCompletionsPromptCacheRetention(
	value param.Opt[modelsettings.PromptCacheRetention],
) openai.ChatCompletionNewParamsPromptCacheRetention {
	if !value.Valid() {
		return ""
	}
	return openai.ChatCompletionNewParamsPromptCacheRetention(
		promptCacheRetentionAPIValue(value.Value),
	)
}

func responsesPromptCacheRetention(
	value param.Opt[modelsettings.PromptCacheRetention],
) responses.ResponseNewParamsPromptCacheRetention {
	if !value.Valid() {
		return ""
	}
	return responses.ResponseNewParamsPromptCacheRetention(
		promptCacheRetentionAPIValue(value.Value),
	)
}
