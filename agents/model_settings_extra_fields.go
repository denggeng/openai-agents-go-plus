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

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
)

func mergedModelExtraJSON(modelSettings modelsettings.ModelSettings) map[string]any {
	if len(modelSettings.ExtraBody) == 0 && len(modelSettings.ExtraArgs) == 0 {
		return nil
	}

	merged := make(map[string]any, len(modelSettings.ExtraBody)+len(modelSettings.ExtraArgs))
	maps.Copy(merged, modelSettings.ExtraBody)

	for k, v := range modelSettings.ExtraArgs {
		if k == "reasoning_effort" {
			if _, exists := merged[k]; exists {
				continue
			}
		}
		merged[k] = v
	}

	if modelSettings.Reasoning.Effort != "" {
		delete(merged, "reasoning_effort")
	}

	if len(merged) == 0 {
		return nil
	}
	return merged
}
