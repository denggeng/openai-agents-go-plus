Compared against `.upstream/openai-agents-python` HEAD `05dc0685` (`0.12.4`).
Previous report baseline was `.upstream/openai-agents-python` `e00f377a` (`0.11.1`).
Scope is parity-relevant runtime behavior, serialization, and test intent, not just exported API names.

Method:
- reviewed upstream `src/agents/` and changed tests from `e00f377a..05dc0685`
- re-audited the previously reported gaps against the latest Go implementation, including internal runner/realtime/MCP call wiring
- ran `go test ./...` on 2026-03-19 after the latest follow-up changes; the current Go test suite passed

## Aligned in the audited 0.12.x delta
- `ModelSettings.extra_args` request forwarding and `reasoning_effort` precedence are aligned at request-build time.
- `ModelSettings.Resolve()` now matches Python for both `ExtraArgs` merge semantics and nested `Retry` deep-merge semantics.
- `ModelSettings.prompt_cache_retention` is now first-class in Go, serialized, resolved, and forwarded to both Chat Completions and Responses requests, including the Python-to-OpenAI-Go value translation for `in_memory`.
- `RunResult.ToInputList()` and `RunResultStreaming.ToInputList()` now support Python’s `mode="normalized"` continuation helper, using `ModelInputItems` when continuation history diverges from preserved session history.
- Runner-managed model retry remains aligned, including provider-managed retry suppression, OpenAI retry classification, websocket replay-safety handling, and retry-aware usage aggregation.
- MCP Streamable HTTP shared-session semantics remain aligned, including serialized shared-session access and isolated-session fallback for transient shared-session failures.
- MCP approval rejection-message propagation remains aligned through run context, run-state persistence, HITL resume, and realtime explicit rejections.
- MCP SSE and Streamable HTTP convenience configuration is now aligned for the audited Python delta: headers, auth, HTTP client factories, request timeout, and SSE read-timeout behavior are all surfaced in Go. Because the Go MCP SDK does not expose first-class timeout knobs, Go implements the timeout split at the HTTP wrapper layer instead of via transport fields.
- Existing Go implementations already covered the audited Python deltas for Advanced SQLite table-name plumbing, run-state `tool_input` persistence, and optional trace API-key serialization.

## P0 gaps
- none

## P1 gaps
- none

## P2 notes
- none identified in the audited 0.12.x delta
