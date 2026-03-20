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
	"errors"
	"strings"

	"github.com/denggeng/openai-agents-go-plus/retry"
)

type modelReplaySafetyError struct {
	err             error
	replaySafety    string
	responseStarted bool
}

func (e *modelReplaySafetyError) Error() string {
	if e == nil || e.err == nil {
		return ""
	}
	return e.err.Error()
}

func (e *modelReplaySafetyError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.err
}

func wrapModelReplaySafety(err error, replaySafety string, responseStarted bool) error {
	if err == nil {
		return nil
	}

	replaySafety = strings.ToLower(strings.TrimSpace(replaySafety))
	switch replaySafety {
	case "safe", "unsafe":
	default:
		replaySafety = ""
	}

	var existing *modelReplaySafetyError
	if errors.As(err, &existing) {
		baseErr := existing.err
		if baseErr == nil {
			baseErr = err
		}
		if replaySafety == "" {
			replaySafety = existing.replaySafety
		} else if existing.replaySafety == "unsafe" {
			replaySafety = "unsafe"
		}
		return &modelReplaySafetyError{
			err:             baseErr,
			replaySafety:    replaySafety,
			responseStarted: existing.responseStarted || responseStarted,
		}
	}

	return &modelReplaySafetyError{
		err:             err,
		replaySafety:    replaySafety,
		responseStarted: responseStarted,
	}
}

func replaySafetyFromError(err error) string {
	var wrapped *modelReplaySafetyError
	if !errors.As(err, &wrapped) || wrapped == nil {
		return ""
	}
	return wrapped.replaySafety
}

func responseStartedFromError(err error) bool {
	var wrapped *modelReplaySafetyError
	if !errors.As(err, &wrapped) || wrapped == nil {
		return false
	}
	return wrapped.responseStarted
}

func getOpenAIRetryAdvice(request retry.ModelRetryAdviceRequest) *retry.ModelRetryAdvice {
	if request.Error == nil {
		return nil
	}

	err := request.Error
	statefulRequest := isStatefulModelRequest(
		request.PreviousResponseID,
		request.ConversationID,
	)

	if unsafeToReplayOpenAIError(err) {
		return &retry.ModelRetryAdvice{
			Suggested: retry.Bool(false),
			Reason:    err.Error(),
			ReplaySafety: "unsafe",
		}
	}

	switch replaySafety := replaySafetyFromError(err); replaySafety {
	case "unsafe":
		if statefulRequest || responseStartedFromError(err) {
			return &retry.ModelRetryAdvice{
				Suggested:    retry.Bool(false),
				ReplaySafety: "unsafe",
				Reason:       err.Error(),
			}
		}
		return &retry.ModelRetryAdvice{
			Suggested: retry.Bool(true),
			Reason:    err.Error(),
		}
	case "safe":
		return &retry.ModelRetryAdvice{
			Suggested:    retry.Bool(true),
			ReplaySafety: "safe",
			Reason:       err.Error(),
		}
	}

	if timeoutPhase := websocketTimeoutPhaseFromError(err); timeoutPhase != "" {
		normalized := retry.ModelRetryNormalizedError{
			IsTimeout: boolPointer(true),
		}
		switch timeoutPhase {
		case "request lock wait":
			return &retry.ModelRetryAdvice{
				Suggested:    retry.Bool(true),
				ReplaySafety: "safe",
				Reason:       err.Error(),
				Normalized:   &normalized,
			}
		case "connect":
			normalized.IsNetworkError = boolPointer(true)
			return &retry.ModelRetryAdvice{
				Suggested:    retry.Bool(true),
				ReplaySafety: "safe",
				Reason:       err.Error(),
				Normalized:   &normalized,
			}
		default:
			if statefulRequest {
				return &retry.ModelRetryAdvice{
					Suggested:    retry.Bool(false),
					ReplaySafety: "unsafe",
					Reason:       err.Error(),
					Normalized:   &normalized,
				}
			}
			return &retry.ModelRetryAdvice{
				Suggested:  retry.Bool(true),
				Reason:     err.Error(),
				Normalized: &normalized,
			}
		}
	}

	if isNeverSentWebsocketError(err) {
		normalized := retry.ModelRetryNormalizedError{
			IsNetworkError: boolPointer(true),
		}
		return &retry.ModelRetryAdvice{
			Suggested:    retry.Bool(true),
			ReplaySafety: "safe",
			Reason:       err.Error(),
			Normalized:   &normalized,
		}
	}

	normalized := normalizeRetryError(err, nil)
	xShouldRetry := strings.ToLower(strings.TrimSpace(extractHeaderValueFromError(err, "x-should-retry")))
	if xShouldRetry != "" {
		switch xShouldRetry {
		case "true":
			return &retry.ModelRetryAdvice{
				Suggested:    retry.Bool(true),
				RetryAfter:   cloneFloat64Pointer(normalized.RetryAfter),
				ReplaySafety: "safe",
				Reason:       err.Error(),
				Normalized:   &normalized,
			}
		case "false":
			return &retry.ModelRetryAdvice{
				Suggested:  retry.Bool(false),
				RetryAfter: cloneFloat64Pointer(normalized.RetryAfter),
				Reason:     err.Error(),
				Normalized: &normalized,
			}
		}
	}

	if normalized.NetworkError() || normalized.Timeout() {
		return &retry.ModelRetryAdvice{
			Suggested:  retry.Bool(true),
			RetryAfter: cloneFloat64Pointer(normalized.RetryAfter),
			Reason:     err.Error(),
			Normalized: &normalized,
		}
	}

	if normalized.StatusCode != nil {
		statusCode := *normalized.StatusCode
		if statusCode == 408 || statusCode == 409 || statusCode == 429 || statusCode >= 500 {
			advice := &retry.ModelRetryAdvice{
				Suggested:  retry.Bool(true),
				RetryAfter: cloneFloat64Pointer(normalized.RetryAfter),
				Reason:     err.Error(),
				Normalized: &normalized,
			}
			if statefulRequest {
				advice.ReplaySafety = "safe"
			}
			return advice
		}
	}

	if normalized.RetryAfter != nil {
		return &retry.ModelRetryAdvice{
			RetryAfter: cloneFloat64Pointer(normalized.RetryAfter),
			Reason:     err.Error(),
			Normalized: &normalized,
		}
	}

	return nil
}

func unsafeToReplayOpenAIError(err error) bool {
	if err == nil {
		return false
	}

	message := strings.ToLower(err.Error())
	if strings.Contains(
		message,
		"the request may have been accepted, so the sdk will not automatically retry this websocket request",
	) {
		return true
	}

	unsafe := false
	walkErrorChain(err, func(candidate error) bool {
		if value, ok := boolFromField(candidate, "UnsafeToReplay"); ok && value {
			unsafe = true
			return false
		}
		if value, ok := boolFromAnyMap(candidate, "unsafe_to_replay"); ok && value {
			unsafe = true
			return false
		}
		return true
	})
	return unsafe
}

func websocketTimeoutPhaseFromError(err error) string {
	var phase string
	walkErrorChain(err, func(candidate error) bool {
		message := strings.ToLower(strings.TrimSpace(candidate.Error()))
		for _, part := range []string{"request lock wait", "connect", "send", "receive"} {
			if strings.HasPrefix(message, "responses websocket "+part+" timed out") {
				phase = part
				return false
			}
		}
		return true
	})
	return phase
}

func isNeverSentWebsocketError(err error) bool {
	if err == nil {
		return false
	}

	neverSent := false
	walkErrorChain(err, func(candidate error) bool {
		if shouldRetryResponsesWebsocketPreSend(candidate) {
			neverSent = true
			return false
		}
		return true
	})
	return neverSent
}

func extractHeaderValueFromError(err error, key string) string {
	headers := extractHeadersFromError(err)
	if len(headers) == 0 {
		return ""
	}
	return strings.TrimSpace(headers.Get(key))
}

func boolFromField(value any, field string) (bool, bool) {
	raw, ok := anyFromField(value, field)
	if !ok {
		return false, false
	}
	switch typed := raw.(type) {
	case bool:
		return typed, true
	default:
		return false, false
	}
}

func boolFromAnyMap(value any, key string) (bool, bool) {
	raw, ok := anyFromMap(value, key)
	if !ok {
		return false, false
	}
	switch typed := raw.(type) {
	case bool:
		return typed, true
	default:
		return false, false
	}
}
