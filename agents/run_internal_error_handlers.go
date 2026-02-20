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
	"context"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

func buildRunErrorData(
	input Input,
	newItems []RunItem,
	rawResponses []ModelResponse,
	lastAgent *Agent,
	reasoningItemIDPolicy ReasoningItemIDPolicy,
) RunErrorData {
	history := ItemHelpers().InputToNewInputList(input)
	output := runItemsToInputItemsWithPolicy(newItems, reasoningItemIDPolicy)
	history = append(history, output...)
	return RunErrorData{
		Input:        input,
		NewItems:     slicesCloneRunItems(newItems),
		History:      slices.Clone(history),
		Output:       slices.Clone(output),
		RawResponses: slices.Clone(rawResponses),
		LastAgent:    lastAgent,
	}
}

func formatFinalOutputText(agent *Agent, finalOutput any) string {
	outputType := agent.OutputType
	if outputType == nil || outputType.IsPlainText() {
		return fmt.Sprint(finalOutput)
	}
	payloadValue := finalOutput
	if isWrappedOutputType(outputType) {
		if mapping, ok := finalOutput.(map[string]any); ok {
			if _, ok := mapping["response"]; !ok {
				payloadValue = map[string]any{"response": finalOutput}
			}
		} else {
			payloadValue = map[string]any{"response": finalOutput}
		}
	}
	payloadBytes, err := json.Marshal(payloadValue)
	if err != nil {
		return fmt.Sprint(finalOutput)
	}
	return string(payloadBytes)
}

func validateHandlerFinalOutput(ctx context.Context, agent *Agent, finalOutput any) (any, error) {
	outputType := agent.OutputType
	if outputType == nil || outputType.IsPlainText() {
		return finalOutput, nil
	}
	payloadValue := finalOutput
	if isWrappedOutputType(outputType) {
		if mapping, ok := finalOutput.(map[string]any); ok {
			if _, ok := mapping["response"]; !ok {
				payloadValue = map[string]any{"response": finalOutput}
			}
		} else {
			payloadValue = map[string]any{"response": finalOutput}
		}
	}
	payloadBytes, err := json.Marshal(payloadValue)
	if err != nil {
		return nil, NewUserError("Invalid run error handler final_output for structured output.")
	}
	validated, err := outputType.ValidateJSON(ctx, string(payloadBytes))
	if err != nil {
		return nil, NewUserError("Invalid run error handler final_output for structured output.")
	}
	return validated, nil
}

func createMessageOutputItem(agent *Agent, outputText string) MessageOutputItem {
	message := responses.ResponseOutputMessage{
		ID:     FakeResponsesID,
		Type:   constant.ValueOf[constant.Message](),
		Role:   constant.ValueOf[constant.Assistant](),
		Status: responses.ResponseOutputMessageStatusCompleted,
		Content: []responses.ResponseOutputMessageContentUnion{
			{
				Type:        "output_text",
				Text:        outputText,
				Annotations: []responses.ResponseOutputTextAnnotationUnion{},
				Logprobs:    []responses.ResponseOutputTextLogprob{},
			},
		},
	}
	return MessageOutputItem{
		Agent:   agent,
		RawItem: message,
		Type:    "message_output_item",
	}
}

func resolveRunErrorHandlerResult(
	ctx context.Context,
	errorHandlers RunErrorHandlers,
	err MaxTurnsExceededError,
	contextWrapper *RunContextWrapper[any],
	runData RunErrorData,
) (*RunErrorHandlerResult, error) {
	handler := errorHandlers.MaxTurns
	if handler == nil {
		return nil, nil
	}
	result, handlerErr := handler(ctx, RunErrorHandlerInput{
		Error:   err,
		Context: contextWrapper,
		RunData: runData,
	})
	if handlerErr != nil {
		return nil, handlerErr
	}
	if result == nil {
		return nil, nil
	}
	switch v := result.(type) {
	case RunErrorHandlerResult:
		return normalizeRunErrorHandlerResult(v), nil
	case *RunErrorHandlerResult:
		if v == nil {
			return nil, nil
		}
		return normalizeRunErrorHandlerResult(*v), nil
	case map[string]any:
		return runErrorHandlerResultFromMap(v)
	default:
		return &RunErrorHandlerResult{
			FinalOutput:      v,
			IncludeInHistory: boolPtr(true),
		}, nil
	}
}

func runErrorHandlerResultFromMap(payload map[string]any) (*RunErrorHandlerResult, error) {
	if _, ok := payload["final_output"]; !ok {
		return &RunErrorHandlerResult{
			FinalOutput:      payload,
			IncludeInHistory: boolPtr(true),
		}, nil
	}
	allowedKeys := map[string]struct{}{
		"final_output":       {},
		"include_in_history": {},
	}
	for key := range payload {
		if _, ok := allowedKeys[key]; !ok {
			return nil, NewUserError("Invalid run error handler result.")
		}
	}
	includeValue := (*bool)(nil)
	if rawInclude, ok := payload["include_in_history"]; ok {
		typed, ok := rawInclude.(bool)
		if !ok {
			return nil, NewUserError("Invalid run error handler result.")
		}
		includeValue = &typed
	}
	return &RunErrorHandlerResult{
		FinalOutput:      payload["final_output"],
		IncludeInHistory: includeValue,
	}, nil
}

func normalizeRunErrorHandlerResult(result RunErrorHandlerResult) *RunErrorHandlerResult {
	if result.IncludeInHistory == nil {
		result.IncludeInHistory = boolPtr(true)
	}
	return &result
}

func isWrappedOutputType(outputType OutputTypeInterface) bool {
	type wrapperInfo interface {
		IsWrapped() bool
	}
	if outputType == nil {
		return false
	}
	if info, ok := outputType.(wrapperInfo); ok {
		return info.IsWrapped()
	}
	return false
}

func boolPtr(value bool) *bool {
	return &value
}
