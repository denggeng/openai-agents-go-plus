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

	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3/option"
)

type providerManagedRetriesDisabledContextKey struct{}
type websocketPreEventRetriesDisabledContextKey struct{}
type failedModelRetryAttemptsContextKey struct{}

func withProviderManagedRetriesDisabled(ctx context.Context, disabled bool) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, providerManagedRetriesDisabledContextKey{}, disabled)
}

func providerManagedRetriesDisabledFromContext(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	disabled, _ := ctx.Value(providerManagedRetriesDisabledContextKey{}).(bool)
	return disabled
}

func withWebsocketPreEventRetriesDisabled(ctx context.Context, disabled bool) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, websocketPreEventRetriesDisabledContextKey{}, disabled)
}

func websocketPreEventRetriesDisabledFromContext(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	disabled, _ := ctx.Value(websocketPreEventRetriesDisabledContextKey{}).(bool)
	return disabled
}

func withFailedModelRetryAttempts(ctx context.Context, attempts int) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, failedModelRetryAttemptsContextKey{}, attempts)
}

func modelRetryAttemptsFromContext(ctx context.Context) int {
	if ctx == nil {
		return 0
	}
	attempts, _ := ctx.Value(failedModelRetryAttemptsContextKey{}).(int)
	if attempts < 0 {
		return 0
	}
	return attempts
}

func appendProviderRetryDisableOption(ctx context.Context, opts []option.RequestOption) []option.RequestOption {
	if !providerManagedRetriesDisabledFromContext(ctx) {
		return opts
	}
	return append(opts, option.WithMaxRetries(0))
}

func applyRetryAttemptUsage(u *usage.Usage, failedAttempts int) *usage.Usage {
	if u == nil {
		u = usage.NewUsage()
	}
	if u.Requests == 0 {
		u.Requests = 1
	}
	if failedAttempts > 0 {
		u.Requests += uint64(failedAttempts)
	}
	return u
}
