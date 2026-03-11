Compared against `.upstream/openai-agents-python` HEAD `e00f377a` (`0.11.1`). Focus was runtime behavior and function logic, not just exported API shape.

## Aligned
- HTTP Responses request/stream behavior is mostly aligned for non-websocket transport. Go now captures `request_id` in `agents/models_openai_responses.go`, persists `ModelResponse.RequestID` in `agents/items.go`, and finalizes streamed runs on `response.completed`, `response.failed`, and `response.incomplete` in `agents/run.go`, matching current Python `models/openai_responses.py` / `run.py`.
- Tracing export sanitation is aligned. `tracing/processors.go` now sanitizes JSON-incompatible payloads and truncates oversized `span_data.input` / `span_data.output` before OpenAI trace ingest, matching Python `tracing/processors.py`.
- Core runtime lifecycle hooks are present. `OpenAIProvider.Aclose`, `MultiProvider.Aclose`, `ResolveComputer` / `DisposeResolvedComputers`, MCP approval/failure/meta hooks in `agents/mcp_util.go`, and basic trace persistence/reattach in `agents/trace_resume.go` now exist. The remaining trace semantic mismatch is narrower and called out below.
- Responses tool-search parity is now aligned. Go has `ToolSearchTool`, `tool_namespace(...)`, `FunctionTool.DeferLoading`, collision-safe lookup keys, namespace-aware approvals/tool context, tool-search call/output run items, deferred hosted MCP payloads, prompt-managed opaque tool-search allowance, and OpenAI Responses conversion/validation in `agents/tool_search.go`, `agents/tool_identity.go`, `agents/run_context.go`, `agents/tool_context.go`, `agents/run_impl.go`, and `agents/models_openai_responses.go`.
- RunState schema compatibility is aligned to current Python. Go now emits and accepts schema `1.5`, preserves namespaced function calls and tool-search run items through serialization/deserialization, and keeps resumed approval/dispatch data compatible with current Python snapshots.
- Responses computer-tool wire-shape selection is aligned. `agents/models_openai_responses.go` now picks GA `computer` vs preview `computer_use_preview` from the effective request model/tool choice while keeping the runtime `ToolName()` compatibility alias.

## P0 gaps
- Responses websocket transport is scaffolded but still non-functional in Go.
  - `agents/responses_websocket_session.go` and `agents/models_openai_provider.go` can construct websocket-backed sessions/models.
  - But `agents/models_openai_responses_ws.go` returns explicit `not implemented` errors from both `GetResponse()` and `StreamResponse()`.
  - Python `src/agents/models/openai_responses.py` implements persistent websocket connections, request/connect/send/recv timeouts, streamed event parsing, connection reuse, and `get_response()` by consuming streamed terminal events.
  - Impact: `UseResponsesWebsocket` and `NewResponsesWebSocketSession()` are effectively unusable in Go.

## P1 gaps
- Function-tool timeout semantics are missing.
  - Python `FunctionTool` supports `timeout_seconds`, `timeout_behavior`, and `timeout_error_function`, enforced in `src/agents/tool.py::invoke_function_tool`.
  - Go now exposes matching fields/types in `agents/tool_function.go` / `agents/errors.go`, but `agents/run_impl.go::runSingleTool` still invokes handlers without timeout enforcement or timeout-as-result handling.
- MCP display metadata fallback/persistence lags.
  - Python `src/agents/_mcp_tool_metadata.py` falls back model-facing description to title when description is empty, stores `_mcp_title`, and persists `ToolCallItem.title` / `description` through run-state serialization.
  - Go `agents/mcp_util.go` uses only `tool.Description`, ignores title fallback, and `agents/run_state_serialization.go` has no `title` field for run items or interruptions.
- Trace reattach matching is looser than Python.
  - Python only reattaches when trace id, workflow name, group id, metadata, and tracing API key match the effective resumed settings (`src/agents/tracing/context.py`).
  - Go `agents/trace_resume.go` reattaches whenever the trace id was seen and the API-key hash is compatible, then merges metadata; resumed runs with overridden trace settings can still reuse the old trace instead of starting a new one.

## Suggested implementation order
1. Replace the websocket stubs in `agents/models_openai_responses_ws.go` with a real transport; then harden `ResponsesWebSocketSession` around the live provider/model lifecycle.
2. Add function-tool timeout semantics in `agents/run_impl.go`, including timeout-as-result vs raise-exception behavior and tests.
3. Finish parity cleanup: MCP title/description fallback + RunState persistence, then stricter trace-reattach matching.
