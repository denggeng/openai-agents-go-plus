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

package retry

import (
	"math/rand"
)

const (
	DefaultInitialDelaySeconds    = 0.25
	DefaultMaxDelaySeconds        = 2.0
	DefaultBackoffMultiplier      = 2.0
	DefaultBackoffUsesJitter      = true
	ConversationLockedRetryBudget = 3
)

type ModelRetryBackoffSettings struct {
	InitialDelay *float64 `json:"initial_delay,omitempty"`
	MaxDelay     *float64 `json:"max_delay,omitempty"`
	Multiplier   *float64 `json:"multiplier,omitempty"`
	Jitter       *bool    `json:"jitter,omitempty"`
}

func (s *ModelRetryBackoffSettings) ResolveInitialDelay() float64 {
	if s != nil && s.InitialDelay != nil {
		return maxFloat64(*s.InitialDelay, 0)
	}
	return DefaultInitialDelaySeconds
}

func (s *ModelRetryBackoffSettings) ResolveMaxDelay() float64 {
	if s != nil && s.MaxDelay != nil {
		return maxFloat64(*s.MaxDelay, 0)
	}
	return DefaultMaxDelaySeconds
}

func (s *ModelRetryBackoffSettings) ResolveMultiplier() float64 {
	if s != nil && s.Multiplier != nil {
		return maxFloat64(*s.Multiplier, 0)
	}
	return DefaultBackoffMultiplier
}

func (s *ModelRetryBackoffSettings) ResolveJitter() bool {
	if s != nil && s.Jitter != nil {
		return *s.Jitter
	}
	return DefaultBackoffUsesJitter
}

type ModelRetryNormalizedError struct {
	StatusCode *int     `json:"status_code,omitempty"`
	ErrorCode  string   `json:"error_code,omitempty"`
	Message    string   `json:"message,omitempty"`
	RequestID  string   `json:"request_id,omitempty"`
	RetryAfter *float64 `json:"retry_after,omitempty"`
	// RetryAfterSet records whether RetryAfter was explicitly provided, including an
	// intentional nil override that should suppress header-derived retry delays.
	RetryAfterSet  bool  `json:"-"`
	IsAbort        *bool `json:"is_abort,omitempty"`
	IsNetworkError *bool `json:"is_network_error,omitempty"`
	IsTimeout      *bool `json:"is_timeout,omitempty"`
}

func (e ModelRetryNormalizedError) Abort() bool {
	return e.IsAbort != nil && *e.IsAbort
}

func (e ModelRetryNormalizedError) NetworkError() bool {
	return e.IsNetworkError != nil && *e.IsNetworkError
}

func (e ModelRetryNormalizedError) Timeout() bool {
	return e.IsTimeout != nil && *e.IsTimeout
}

type ModelRetryAdvice struct {
	Suggested    *bool                      `json:"suggested,omitempty"`
	RetryAfter   *float64                   `json:"retry_after,omitempty"`
	ReplaySafety string                     `json:"replay_safety,omitempty"`
	Reason       string                     `json:"reason,omitempty"`
	Normalized   *ModelRetryNormalizedError `json:"normalized,omitempty"`
}

type ModelRetryAdviceRequest struct {
	Error              error
	Attempt            int
	Stream             bool
	PreviousResponseID string
	ConversationID     string
}

type RetryDecision struct {
	Retry  bool
	Delay  *float64
	Reason string

	hardVeto       bool
	approvesReplay bool
}

func (d RetryDecision) WithHardVeto() RetryDecision {
	d.hardVeto = true
	return d
}

func (d RetryDecision) WithReplaySafeApproval() RetryDecision {
	d.approvesReplay = true
	return d
}

func (d RetryDecision) HardVeto() bool {
	return d.hardVeto
}

func (d RetryDecision) ApprovesReplay() bool {
	return d.approvesReplay
}

type RetryPolicyContext struct {
	Error          error
	Attempt        int
	MaxRetries     int
	Stream         bool
	Normalized     ModelRetryNormalizedError
	ProviderAdvice *ModelRetryAdvice
}

type RetryPolicy interface {
	Evaluate(RetryPolicyContext) RetryDecision
}

type RetryPolicyFunc func(RetryPolicyContext) RetryDecision

func (f RetryPolicyFunc) Evaluate(ctx RetryPolicyContext) RetryDecision {
	return f(ctx)
}

type retryPolicyCapabilities interface {
	retriesSafeTransportErrors() bool
	retriesAllTransientErrors() bool
}

type retryPolicySpec struct {
	fn                              RetryPolicyFunc
	retriesSafeTransportErrorsValue bool
	retriesAllTransientErrorsValue  bool
}

func (p retryPolicySpec) Evaluate(ctx RetryPolicyContext) RetryDecision {
	return p.fn(ctx)
}

func (p retryPolicySpec) retriesSafeTransportErrors() bool {
	return p.retriesSafeTransportErrorsValue
}

func (p retryPolicySpec) retriesAllTransientErrors() bool {
	return p.retriesAllTransientErrorsValue
}

type Policies struct{}

var RetryPolicies Policies

func PolicyRetriesSafeTransportErrors(policy RetryPolicy) bool {
	if policy == nil {
		return false
	}
	provider, ok := policy.(retryPolicyCapabilities)
	return ok && provider.retriesSafeTransportErrors()
}

func PolicyRetriesAllTransientErrors(policy RetryPolicy) bool {
	if policy == nil {
		return false
	}
	provider, ok := policy.(retryPolicyCapabilities)
	return ok && provider.retriesAllTransientErrors()
}

func (Policies) Never() RetryPolicy {
	return retryPolicySpec{
		fn: func(RetryPolicyContext) RetryDecision {
			return RetryDecision{Retry: false}
		},
	}
}

func (Policies) ProviderSuggested() RetryPolicy {
	return retryPolicySpec{
		fn: func(ctx RetryPolicyContext) RetryDecision {
			advice := ctx.ProviderAdvice
			if advice == nil || advice.Suggested == nil {
				return RetryDecision{Retry: false}
			}
			if !*advice.Suggested {
				return RetryDecision{
					Retry:  false,
					Reason: advice.Reason,
				}.WithHardVeto()
			}
			decision := RetryDecision{
				Retry:  true,
				Delay:  cloneFloat64Pointer(advice.RetryAfter),
				Reason: advice.Reason,
			}
			if advice.ReplaySafety == "safe" {
				decision = decision.WithReplaySafeApproval()
			}
			return decision
		},
		retriesSafeTransportErrorsValue: true,
	}
}

func (Policies) NetworkError() RetryPolicy {
	return retryPolicySpec{
		fn: func(ctx RetryPolicyContext) RetryDecision {
			return RetryDecision{
				Retry: ctx.Normalized.NetworkError() || ctx.Normalized.Timeout(),
			}
		},
		retriesSafeTransportErrorsValue: true,
	}
}

func (Policies) RetryAfter() RetryPolicy {
	return retryPolicySpec{
		fn: func(ctx RetryPolicyContext) RetryDecision {
			delay := cloneFloat64Pointer(ctx.Normalized.RetryAfter)
			if delay == nil && ctx.ProviderAdvice != nil {
				delay = cloneFloat64Pointer(ctx.ProviderAdvice.RetryAfter)
			}
			if delay == nil {
				return RetryDecision{Retry: false}
			}
			return RetryDecision{Retry: true, Delay: delay}
		},
	}
}

func (Policies) HTTPStatus(statuses ...int) RetryPolicy {
	allowed := make(map[int]struct{}, len(statuses))
	for _, status := range statuses {
		allowed[status] = struct{}{}
	}
	return retryPolicySpec{
		fn: func(ctx RetryPolicyContext) RetryDecision {
			if ctx.Normalized.StatusCode == nil {
				return RetryDecision{Retry: false}
			}
			_, ok := allowed[*ctx.Normalized.StatusCode]
			return RetryDecision{Retry: ok}
		},
	}
}

func (p Policies) All(policies ...RetryPolicy) RetryPolicy {
	filtered := compactPolicies(policies)
	if len(filtered) == 0 {
		return p.Never()
	}
	return retryPolicySpec{
		fn: func(ctx RetryPolicyContext) RetryDecision {
			merged := RetryDecision{Retry: true}
			for _, policy := range filtered {
				decision := policy.Evaluate(ctx)
				if decision.HardVeto() {
					return decision
				}
				if !decision.Retry {
					return decision
				}
				if decision.Delay != nil {
					merged.Delay = cloneFloat64Pointer(decision.Delay)
				}
				if decision.Reason != "" {
					merged.Reason = decision.Reason
				}
				if decision.ApprovesReplay() {
					merged = merged.WithReplaySafeApproval()
				}
			}
			return merged
		},
		retriesSafeTransportErrorsValue: allPolicies(filtered, PolicyRetriesSafeTransportErrors),
		retriesAllTransientErrorsValue:  allPolicies(filtered, PolicyRetriesAllTransientErrors),
	}
}

func (p Policies) Any(policies ...RetryPolicy) RetryPolicy {
	filtered := compactPolicies(policies)
	if len(filtered) == 0 {
		return p.Never()
	}
	return retryPolicySpec{
		fn: func(ctx RetryPolicyContext) RetryDecision {
			var firstPositive *RetryDecision
			var lastNegative *RetryDecision
			for _, policy := range filtered {
				decision := policy.Evaluate(ctx)
				if decision.HardVeto() {
					return decision
				}
				if decision.Retry {
					if firstPositive == nil {
						copy := decision
						firstPositive = &copy
					} else {
						merged := mergePositiveRetryDecisions(*firstPositive, decision)
						firstPositive = &merged
					}
					continue
				}
				copy := decision
				lastNegative = &copy
			}
			if firstPositive != nil {
				return *firstPositive
			}
			if lastNegative != nil {
				return *lastNegative
			}
			return RetryDecision{Retry: false}
		},
		retriesSafeTransportErrorsValue: anyPolicy(filtered, PolicyRetriesSafeTransportErrors),
		retriesAllTransientErrorsValue:  anyPolicy(filtered, PolicyRetriesAllTransientErrors),
	}
}

type ModelRetrySettings struct {
	MaxRetries *int                       `json:"max_retries,omitempty"`
	Backoff    *ModelRetryBackoffSettings `json:"backoff,omitempty"`
	Policy     RetryPolicy                `json:"-"`
}

func (s *ModelRetrySettings) ResolvedMaxRetries() (int, bool) {
	if s == nil || s.MaxRetries == nil {
		return 0, false
	}
	return maxInt(*s.MaxRetries, 0), true
}

func (s *ModelRetrySettings) DefaultDelaySeconds(attempt int) float64 {
	backoff := s.Backoff
	initialDelay := backoff.ResolveInitialDelay()
	maxDelay := backoff.ResolveMaxDelay()
	multiplier := backoff.ResolveMultiplier()
	useJitter := backoff.ResolveJitter()

	delay := initialDelay
	for i := 1; i < attempt; i++ {
		delay *= multiplier
	}
	if delay > maxDelay {
		delay = maxDelay
	}
	if !useJitter {
		return delay
	}
	return minFloat64(maxFloat64(delay*(0.875+rand.Float64()*0.25), 0), maxDelay)
}

func Bool(v bool) *bool { return &v }

func Float64(v float64) *float64 { return &v }

func Int(v int) *int { return &v }

func cloneFloat64Pointer(v *float64) *float64 {
	if v == nil {
		return nil
	}
	cloned := *v
	return &cloned
}

func mergePositiveRetryDecisions(existing, incoming RetryDecision) RetryDecision {
	merged := RetryDecision{
		Retry:  true,
		Delay:  cloneFloat64Pointer(existing.Delay),
		Reason: existing.Reason,
	}
	if existing.ApprovesReplay() {
		merged = merged.WithReplaySafeApproval()
	}
	if incoming.Delay != nil {
		merged.Delay = cloneFloat64Pointer(incoming.Delay)
	}
	if incoming.Reason != "" {
		merged.Reason = incoming.Reason
	}
	if incoming.ApprovesReplay() {
		merged = merged.WithReplaySafeApproval()
	}
	return merged
}

func compactPolicies(policies []RetryPolicy) []RetryPolicy {
	out := make([]RetryPolicy, 0, len(policies))
	for _, policy := range policies {
		if policy != nil {
			out = append(out, policy)
		}
	}
	return out
}

func allPolicies(policies []RetryPolicy, pred func(RetryPolicy) bool) bool {
	if len(policies) == 0 {
		return false
	}
	for _, policy := range policies {
		if !pred(policy) {
			return false
		}
	}
	return true
}

func anyPolicy(policies []RetryPolicy, pred func(RetryPolicy) bool) bool {
	for _, policy := range policies {
		if pred(policy) {
			return true
		}
	}
	return false
}

func minFloat64(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func maxFloat64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
