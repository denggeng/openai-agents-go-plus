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
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/denggeng/openai-agents-go-plus/retry"
)

var sleepForModelRetry = func(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return nil
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func getModelRetryAdvice(model Model, request retry.ModelRetryAdviceRequest) *retry.ModelRetryAdvice {
	advisor, ok := model.(ModelRetryAdvisor)
	if !ok {
		return nil
	}
	return advisor.GetRetryAdvice(request)
}

func getResponseWithRetry(
	ctx context.Context,
	getResponse func(context.Context) (*ModelResponse, error),
	rewind func() error,
	retrySettings *retry.ModelRetrySettings,
	getRetryAdvice func(retry.ModelRetryAdviceRequest) *retry.ModelRetryAdvice,
	previousResponseID string,
	conversationID string,
) (*ModelResponse, error) {
	requestAttempt := 1
	policyAttempt := 1
	failedPolicyAttempts := 0
	compatibilityRetriesTaken := 0
	statefulRequest := isStatefulModelRequest(previousResponseID, conversationID)
	disableWebsocketPreEventRetry := shouldDisableWebsocketPreEventRetry(retrySettings)

	for {
		attemptCtx := withProviderManagedRetriesDisabled(
			ctx,
			shouldDisableProviderManagedRetries(retrySettings, requestAttempt, statefulRequest),
		)
		attemptCtx = withWebsocketPreEventRetriesDisabled(attemptCtx, disableWebsocketPreEventRetry)

		response, err := getResponse(attemptCtx)
		if err == nil {
			if response != nil {
				response.Usage = applyRetryAttemptUsage(
					response.Usage,
					failedPolicyAttempts+compatibilityRetriesTaken,
				)
			}
			return response, nil
		}
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}

		if isConversationLockedError(err) && shouldPreserveConversationLockedCompatibility(retrySettings) {
			if compatibilityRetriesTaken < retry.ConversationLockedRetryBudget {
				compatibilityRetriesTaken++
				if rewind != nil {
					if rewindErr := rewind(); rewindErr != nil {
						return nil, rewindErr
					}
				}
				delaySeconds := math.Pow(2, float64(compatibilityRetriesTaken-1))
				if sleepErr := sleepForModelRetry(ctx, secondsToDuration(delaySeconds)); sleepErr != nil {
					return nil, sleepErr
				}
				requestAttempt++
				continue
			}
		}

		var providerAdvice *retry.ModelRetryAdvice
		if getRetryAdvice != nil {
			providerAdvice = getRetryAdvice(retry.ModelRetryAdviceRequest{
				Error:              err,
				Attempt:            policyAttempt,
				Stream:             false,
				PreviousResponseID: previousResponseID,
				ConversationID:     conversationID,
			})
		}

		decision := evaluateModelRetry(
			err,
			policyAttempt,
			resolvedRetryBudget(retrySettings),
			retrySettings,
			false,
			statefulRequest,
			false,
			providerAdvice,
		)
		if !decision.Retry {
			return nil, err
		}

		if rewind != nil {
			if rewindErr := rewind(); rewindErr != nil {
				return nil, rewindErr
			}
		}
		if sleepErr := sleepForModelRetry(ctx, secondsToDuration(derefFloat64(decision.Delay))); sleepErr != nil {
			return nil, sleepErr
		}
		requestAttempt++
		policyAttempt++
		failedPolicyAttempts++
	}
}

func streamResponseWithRetry(
	ctx context.Context,
	streamResponse func(context.Context, ModelStreamResponseCallback) error,
	rewind func() error,
	retrySettings *retry.ModelRetrySettings,
	getRetryAdvice func(retry.ModelRetryAdviceRequest) *retry.ModelRetryAdvice,
	previousResponseID string,
	conversationID string,
	yield ModelStreamResponseCallback,
) error {
	if yield == nil {
		yield = func(context.Context, TResponseStreamEvent) error { return nil }
	}

	requestAttempt := 1
	policyAttempt := 1
	failedPolicyAttempts := 0
	compatibilityRetriesTaken := 0
	statefulRequest := isStatefulModelRequest(previousResponseID, conversationID)
	disableWebsocketPreEventRetry := shouldDisableWebsocketPreEventRetry(retrySettings)
	outerProviderRetriesDisabled := providerManagedRetriesDisabledFromContext(ctx)
	outerWebsocketPreEventRetriesDisabled := websocketPreEventRetriesDisabledFromContext(ctx)

	for {
		emittedRetryUnsafeEvent := false
		yieldReturnedError := false
		failedAttemptsForDelivery := failedPolicyAttempts + compatibilityRetriesTaken

		attemptCtx := withProviderManagedRetriesDisabled(
			ctx,
			shouldDisableProviderManagedRetries(retrySettings, requestAttempt, statefulRequest),
		)
		attemptCtx = withWebsocketPreEventRetriesDisabled(attemptCtx, disableWebsocketPreEventRetry)

		err := streamResponse(attemptCtx, func(eventCtx context.Context, event TResponseStreamEvent) error {
			if streamEventBlocksRetry(event) {
				emittedRetryUnsafeEvent = true
			}
			consumerCtx := withProviderManagedRetriesDisabled(
				eventCtx,
				outerProviderRetriesDisabled,
			)
			consumerCtx = withWebsocketPreEventRetriesDisabled(
				consumerCtx,
				outerWebsocketPreEventRetriesDisabled,
			)
			consumerCtx = withFailedModelRetryAttempts(consumerCtx, failedAttemptsForDelivery)
			if yieldErr := yield(consumerCtx, event); yieldErr != nil {
				yieldReturnedError = true
				return yieldErr
			}
			return nil
		})
		if err == nil {
			return nil
		}
		if yieldReturnedError {
			return err
		}
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}

		if isConversationLockedError(err) && shouldPreserveConversationLockedCompatibility(retrySettings) {
			if compatibilityRetriesTaken < retry.ConversationLockedRetryBudget {
				compatibilityRetriesTaken++
				if rewind != nil {
					if rewindErr := rewind(); rewindErr != nil {
						return rewindErr
					}
				}
				delaySeconds := math.Pow(2, float64(compatibilityRetriesTaken-1))
				if sleepErr := sleepForModelRetry(ctx, secondsToDuration(delaySeconds)); sleepErr != nil {
					return sleepErr
				}
				requestAttempt++
				continue
			}
		}

		var providerAdvice *retry.ModelRetryAdvice
		if getRetryAdvice != nil {
			providerAdvice = getRetryAdvice(retry.ModelRetryAdviceRequest{
				Error:              err,
				Attempt:            policyAttempt,
				Stream:             true,
				PreviousResponseID: previousResponseID,
				ConversationID:     conversationID,
			})
		}

		decision := evaluateModelRetry(
			err,
			policyAttempt,
			resolvedRetryBudget(retrySettings),
			retrySettings,
			true,
			statefulRequest,
			emittedRetryUnsafeEvent,
			providerAdvice,
		)
		if !decision.Retry {
			return err
		}

		if rewind != nil {
			if rewindErr := rewind(); rewindErr != nil {
				return rewindErr
			}
		}
		if sleepErr := sleepForModelRetry(ctx, secondsToDuration(derefFloat64(decision.Delay))); sleepErr != nil {
			return sleepErr
		}
		requestAttempt++
		policyAttempt++
		failedPolicyAttempts++
	}
}

func evaluateModelRetry(
	err error,
	attempt int,
	maxRetries int,
	retrySettings *retry.ModelRetrySettings,
	stream bool,
	replayUnsafeRequest bool,
	emittedRetryUnsafeEvent bool,
	providerAdvice *retry.ModelRetryAdvice,
) retry.RetryDecision {
	if attempt > maxRetries {
		return retry.RetryDecision{Retry: false}
	}

	normalized := normalizeRetryError(err, providerAdvice)
	if normalized.Abort() || emittedRetryUnsafeEvent || (providerAdvice != nil && providerAdvice.ReplaySafety == "unsafe") {
		return retry.RetryDecision{
			Retry:  false,
			Reason: retryAdviceReason(providerAdvice),
		}
	}
	if retrySettings == nil || retrySettings.Policy == nil {
		return retry.RetryDecision{Retry: false}
	}

	decision := retrySettings.Policy.Evaluate(retry.RetryPolicyContext{
		Error:          err,
		Attempt:        attempt,
		MaxRetries:     maxRetries,
		Stream:         stream,
		Normalized:     normalized,
		ProviderAdvice: providerAdvice,
	})
	if !decision.Retry {
		return decision
	}

	providerMarksReplaySafe := providerAdvice != nil && providerAdvice.ReplaySafety == "safe"
	if replayUnsafeRequest && !decision.ApprovesReplay() && !providerMarksReplaySafe {
		return retry.RetryDecision{
			Retry:  false,
			Reason: firstNonEmpty(decision.Reason, retryAdviceReason(providerAdvice)),
		}
	}

	delay := decision.Delay
	if delay == nil {
		delay = cloneFloat64Pointer(normalized.RetryAfter)
	}
	if delay == nil && retrySettings != nil {
		defaultDelay := retrySettings.DefaultDelaySeconds(attempt)
		delay = &defaultDelay
	}

	return retry.RetryDecision{
		Retry:  true,
		Delay:  delay,
		Reason: firstNonEmpty(decision.Reason, retryAdviceReason(providerAdvice)),
	}
}

func normalizeRetryError(err error, providerAdvice *retry.ModelRetryAdvice) retry.ModelRetryNormalizedError {
	retryAfter := parseRetryAfterFromHeaders(extractHeadersFromError(err))
	normalized := retry.ModelRetryNormalizedError{
		StatusCode:     intPointer(getStatusCodeFromError(err)),
		ErrorCode:      getErrorCodeFromError(err),
		Message:        errorMessage(err),
		RequestID:      getRequestIDFromError(err),
		RetryAfter:     retryAfter,
		RetryAfterSet:  retryAfter != nil,
		IsAbort:        boolPointer(isAbortLikeError(err)),
		IsNetworkError: boolPointer(isNetworkLikeError(err)),
		IsTimeout:      boolPointer(isTimeoutLikeError(err)),
	}

	if providerAdvice == nil {
		return normalized
	}
	if providerAdvice.RetryAfter != nil {
		normalized.RetryAfter = cloneFloat64Pointer(providerAdvice.RetryAfter)
		normalized.RetryAfterSet = true
	}
	if providerAdvice.Normalized == nil {
		return normalized
	}

	override := providerAdvice.Normalized
	if override.StatusCode != nil {
		normalized.StatusCode = intPointer(*override.StatusCode)
	}
	if override.ErrorCode != "" {
		normalized.ErrorCode = override.ErrorCode
	}
	if override.Message != "" {
		normalized.Message = override.Message
	}
	if override.RequestID != "" {
		normalized.RequestID = override.RequestID
	}
	if override.RetryAfter != nil || override.RetryAfterSet {
		normalized.RetryAfter = cloneFloat64Pointer(override.RetryAfter)
		normalized.RetryAfterSet = true
	}
	if override.IsAbort != nil {
		normalized.IsAbort = boolPointer(*override.IsAbort)
	}
	if override.IsNetworkError != nil {
		normalized.IsNetworkError = boolPointer(*override.IsNetworkError)
	}
	if override.IsTimeout != nil {
		normalized.IsTimeout = boolPointer(*override.IsTimeout)
	}
	return normalized
}

func shouldPreserveConversationLockedCompatibility(retrySettings *retry.ModelRetrySettings) bool {
	if retrySettings == nil || retrySettings.MaxRetries == nil {
		return true
	}
	return *retrySettings.MaxRetries > 0
}

func shouldDisableProviderManagedRetries(
	retrySettings *retry.ModelRetrySettings,
	attempt int,
	statefulRequest bool,
) bool {
	if retrySettings != nil && retrySettings.MaxRetries != nil && *retrySettings.MaxRetries <= 0 {
		return true
	}

	if attempt > 1 {
		if statefulRequest {
			return true
		}
		if retrySettings == nil || retrySettings.Policy == nil {
			return false
		}
		return resolvedRetryBudget(retrySettings) > 0
	}

	if retrySettings == nil {
		return false
	}
	if !statefulRequest {
		return false
	}
	return retrySettings.MaxRetries != nil && *retrySettings.MaxRetries > 0 && retrySettings.Policy != nil
}

func shouldDisableWebsocketPreEventRetry(retrySettings *retry.ModelRetrySettings) bool {
	if retrySettings == nil {
		return false
	}
	if retrySettings.MaxRetries != nil && *retrySettings.MaxRetries <= 0 {
		return true
	}
	if retrySettings.Policy == nil {
		return false
	}
	return retrySettings.MaxRetries != nil &&
		*retrySettings.MaxRetries > 0 &&
		retry.PolicyRetriesSafeTransportErrors(retrySettings.Policy)
}

func resolvedRetryBudget(retrySettings *retry.ModelRetrySettings) int {
	if retrySettings == nil || retrySettings.MaxRetries == nil || *retrySettings.MaxRetries < 0 {
		return 0
	}
	return *retrySettings.MaxRetries
}

func streamEventBlocksRetry(event TResponseStreamEvent) bool {
	switch event.Type {
	case "response.created", "response.in_progress":
		return false
	default:
		return true
	}
}

func isStatefulModelRequest(previousResponseID, conversationID string) bool {
	return strings.TrimSpace(previousResponseID) != "" || strings.TrimSpace(conversationID) != ""
}

func isConversationLockedError(err error) bool {
	return strings.EqualFold(getErrorCodeFromError(err), "conversation_locked")
}

func errorMessage(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func retryAdviceReason(advice *retry.ModelRetryAdvice) string {
	if advice == nil {
		return ""
	}
	return advice.Reason
}

func getStatusCodeFromError(err error) int {
	statusCode := 0
	walkErrorChain(err, func(candidate error) bool {
		switch typed := candidate.(type) {
		case interface{ StatusCode() int }:
			statusCode = typed.StatusCode()
			return false
		case interface{ Status() int }:
			statusCode = typed.Status()
			return false
		}
		if response, ok := extractHTTPResponse(candidate); ok && response != nil {
			statusCode = response.StatusCode
			return false
		}
		if reflected, ok := intFromField(candidate, "StatusCode"); ok {
			statusCode = reflected
			return false
		}
		if reflected, ok := intFromField(candidate, "Status"); ok {
			statusCode = reflected
			return false
		}
		return true
	})
	return statusCode
}

func getErrorCodeFromError(err error) string {
	var code string
	walkErrorChain(err, func(candidate error) bool {
		if reflected, ok := stringFromField(candidate, "Code"); ok && reflected != "" {
			code = reflected
			return false
		}
		if reflected, ok := stringFromMap(candidate, "code"); ok && reflected != "" {
			code = reflected
			return false
		}
		return true
	})
	return code
}

func getRequestIDFromError(err error) string {
	var requestID string
	walkErrorChain(err, func(candidate error) bool {
		if reflected, ok := stringFromField(candidate, "RequestID"); ok && reflected != "" {
			requestID = reflected
			return false
		}
		if response, ok := extractHTTPResponse(candidate); ok && response != nil {
			requestID = strings.TrimSpace(response.Header.Get("x-request-id"))
			return requestID == ""
		}
		return true
	})
	return requestID
}

func isAbortLikeError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) {
		return true
	}
	abortLike := false
	walkErrorChain(err, func(candidate error) bool {
		if errors.Is(candidate, context.Canceled) {
			abortLike = true
			return false
		}
		name := candidateName(candidate)
		if name == "AbortError" || name == "CancelledError" {
			abortLike = true
			return false
		}
		return true
	})
	return abortLike
}

func isTimeoutLikeError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	timeoutLike := false
	walkErrorChain(err, func(candidate error) bool {
		if errors.Is(candidate, context.DeadlineExceeded) {
			timeoutLike = true
			return false
		}
		if netErr, ok := candidate.(net.Error); ok && netErr.Timeout() {
			timeoutLike = true
			return false
		}
		return true
	})
	return timeoutLike
}

func isNetworkLikeError(err error) bool {
	if err == nil {
		return false
	}
	networkLike := false
	walkErrorChain(err, func(candidate error) bool {
		switch candidate.(type) {
		case net.Error, *url.Error, *net.OpError:
			networkLike = true
			return false
		}
		if errors.Is(candidate, net.ErrClosed) {
			networkLike = true
			return false
		}
		name := candidateName(candidate)
		if strings.HasPrefix(name, "ConnectionClosed") {
			networkLike = true
			return false
		}
		return true
	})
	if networkLike {
		return true
	}
	message := strings.ToLower(errorMessage(err))
	return strings.Contains(message, "connection error") ||
		strings.Contains(message, "network error") ||
		strings.Contains(message, "socket hang up") ||
		strings.Contains(message, "connection closed")
}

func extractHeadersFromError(err error) http.Header {
	var headers http.Header
	walkErrorChain(err, func(candidate error) bool {
		if response, ok := extractHTTPResponse(candidate); ok && response != nil {
			headers = response.Header.Clone()
			return false
		}
		if header, ok := headerFromField(candidate, "Headers"); ok && len(header) > 0 {
			headers = header.Clone()
			return false
		}
		if header, ok := headerFromField(candidate, "ResponseHeaders"); ok && len(header) > 0 {
			headers = header.Clone()
			return false
		}
		return true
	})
	return headers
}

func parseRetryAfterFromHeaders(headers http.Header) *float64 {
	if len(headers) == 0 {
		return nil
	}
	if retryAfterMS := strings.TrimSpace(headers.Get("retry-after-ms")); retryAfterMS != "" {
		if parsed, err := strconvParseFloat(retryAfterMS); err == nil && parsed >= 0 {
			seconds := parsed / 1000
			return &seconds
		}
	}
	retryAfter := strings.TrimSpace(headers.Get("retry-after"))
	if retryAfter == "" {
		return nil
	}
	if parsed, err := strconvParseFloat(retryAfter); err == nil {
		if parsed >= 0 {
			return &parsed
		}
		return nil
	}
	if parsedTime, err := http.ParseTime(retryAfter); err == nil {
		seconds := maxFloat64(parsedTime.Sub(time.Now()).Seconds(), 0)
		return &seconds
	}
	return nil
}

func walkErrorChain(err error, visit func(error) bool) {
	if err == nil || visit == nil {
		return
	}
	queue := []error{err}
	seen := make(map[uintptr]struct{}, 8)
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		if current == nil {
			continue
		}
		if id := itemIdentity(current); id != 0 {
			if _, ok := seen[id]; ok {
				continue
			}
			seen[id] = struct{}{}
		}
		if !visit(current) {
			return
		}
		switch typed := any(current).(type) {
		case interface{ Unwrap() []error }:
			queue = append(queue, typed.Unwrap()...)
		case interface{ Unwrap() error }:
			queue = append(queue, typed.Unwrap())
		}
	}
}

func extractHTTPResponse(err error) (*http.Response, bool) {
	response, ok := anyFromField(err, "Response")
	if !ok {
		return nil, false
	}
	httpResponse, ok := response.(*http.Response)
	return httpResponse, ok
}

func headerFromField(err error, field string) (http.Header, bool) {
	value, ok := anyFromField(err, field)
	if !ok {
		return nil, false
	}
	switch typed := value.(type) {
	case http.Header:
		return typed, true
	case map[string][]string:
		return http.Header(typed), true
	case map[string]string:
		header := make(http.Header, len(typed))
		for key, val := range typed {
			header.Set(key, val)
		}
		return header, true
	default:
		return nil, false
	}
}

func intFromField(value any, field string) (int, bool) {
	raw, ok := anyFromField(value, field)
	if !ok {
		return 0, false
	}
	switch typed := raw.(type) {
	case int:
		return typed, true
	case int64:
		return int(typed), true
	case int32:
		return int(typed), true
	case uint:
		return int(typed), true
	case uint64:
		return int(typed), true
	case uint32:
		return int(typed), true
	default:
		return 0, false
	}
}

func candidateName(err error) string {
	if err == nil {
		return ""
	}
	parts := strings.Split(fmt.Sprintf("%T", err), ".")
	return parts[len(parts)-1]
}

func boolPointer(v bool) *bool { return &v }

func intPointer(v int) *int {
	if v == 0 {
		return nil
	}
	return &v
}

func secondsToDuration(seconds float64) time.Duration {
	if seconds <= 0 {
		return 0
	}
	return time.Duration(seconds * float64(time.Second))
}

func derefFloat64(v *float64) float64 {
	if v == nil {
		return 0
	}
	return *v
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func cloneFloat64Pointer(v *float64) *float64 {
	if v == nil {
		return nil
	}
	cloned := *v
	return &cloned
}

func maxFloat64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func strconvParseFloat(value string) (float64, error) {
	return strconv.ParseFloat(value, 64)
}
