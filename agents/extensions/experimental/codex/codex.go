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

package codex

// Codex is a lightweight entrypoint for experimental Codex extension workflows.
type Codex struct {
	execClient CodexExecClient
	options    CodexOptions
}

// NewCodex creates a Codex client from nil, struct, or map-based options.
func NewCodex(options any) (*Codex, error) {
	resolvedOptions, err := CoerceCodexOptions(options)
	if err != nil {
		return nil, err
	}
	if resolvedOptions == nil {
		resolvedOptions = &CodexOptions{}
	}

	execClient, err := NewCodexExec(
		resolvedOptions.CodexPathOverride,
		normalizeEnv(*resolvedOptions),
		resolvedOptions.CodexSubprocessStreamLimitBytes,
	)
	if err != nil {
		return nil, err
	}
	return &Codex{
		execClient: execClient,
		options:    *cloneCodexOptions(resolvedOptions),
	}, nil
}

func newCodexWithExec(execClient CodexExecClient, options CodexOptions) *Codex {
	return &Codex{
		execClient: execClient,
		options:    *cloneCodexOptions(&options),
	}
}

// StartThread creates a new thread with optional per-thread options.
func (c *Codex) StartThread(options any) (*Thread, error) {
	resolvedOptions, err := CoerceThreadOptions(options)
	if err != nil {
		return nil, err
	}
	if resolvedOptions == nil {
		resolvedOptions = &ThreadOptions{}
	}
	return newThread(c.execClient, c.options, *resolvedOptions, nil), nil
}

// ResumeThread creates a thread handle bound to an existing thread id.
func (c *Codex) ResumeThread(threadID string, options any) (*Thread, error) {
	resolvedOptions, err := CoerceThreadOptions(options)
	if err != nil {
		return nil, err
	}
	if resolvedOptions == nil {
		resolvedOptions = &ThreadOptions{}
	}
	id := threadID
	return newThread(c.execClient, c.options, *resolvedOptions, &id), nil
}

func normalizeEnv(options CodexOptions) map[string]string {
	if options.Env == nil {
		return nil
	}
	out := make(map[string]string, len(options.Env))
	for key, value := range options.Env {
		out[anyToString(key)] = anyToString(value)
	}
	return out
}
