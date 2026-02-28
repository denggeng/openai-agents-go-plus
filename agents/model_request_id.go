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
	"net/http"
	"strings"
)

type modelRequestIDContextKey struct{}

func contextWithModelRequestID(ctx context.Context, requestID string) context.Context {
	if requestID == "" {
		return ctx
	}
	return context.WithValue(ctx, modelRequestIDContextKey{}, requestID)
}

func modelRequestIDFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	requestID, _ := ctx.Value(modelRequestIDContextKey{}).(string)
	return requestID
}

func requestIDFromHeaders(resp *http.Response) string {
	if resp == nil || resp.Header == nil {
		return ""
	}
	for _, key := range []string{"x-request-id", "openai-request-id", "request-id"} {
		value := strings.TrimSpace(resp.Header.Get(key))
		if value != "" {
			return value
		}
	}
	return ""
}
