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
	"maps"

	"github.com/denggeng/openai-agents-go-plus/modelsettings"
)

func litellmReasoningSummaryValue(settings modelsettings.ModelSettings) string {
	if settings.Reasoning.Summary != "" {
		return string(settings.Reasoning.Summary)
	}
	if settings.Reasoning.GenerateSummary != "" {
		return string(settings.Reasoning.GenerateSummary)
	}
	return ""
}

func litellmExtraJSON(settings modelsettings.ModelSettings) map[string]any {
	extras := mergedModelExtraJSON(settings)
	summary := litellmReasoningSummaryValue(settings)
	if summary == "" {
		return extras
	}

	if extras == nil {
		extras = map[string]any{}
	} else {
		extras = maps.Clone(extras)
	}

	payload := map[string]any{
		"summary": summary,
	}
	if settings.Reasoning.Effort != "" {
		payload["effort"] = string(settings.Reasoning.Effort)
	}
	extras["reasoning_effort"] = payload
	return extras
}
