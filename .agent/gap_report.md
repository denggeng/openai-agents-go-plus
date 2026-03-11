Compared against `.upstream/openai-agents-python` HEAD `e00f377a` (`0.11.1`). Focus was runtime behavior and function logic, not just exported API shape.

## Aligned
- HTTP Responses request/stream behavior is mostly aligned for non-websocket transport. Go now captures `request_id` in `agents/models_openai_responses.go`, persists `ModelResponse.RequestID` in `agents/items.go`, and finalizes streamed runs on `response.completed`, `response.failed`, and `response.incomplete` in `agents/run.go`, matching current Python `models/openai_responses.py` / `run.py`.
- Tracing export sanitation is aligned. `tracing/processors.go` now sanitizes JSON-incompatible payloads and truncates oversized `span_data.input` / `span_data.output` before OpenAI trace ingest, matching Python `tracing/processors.py`.
- Core runtime lifecycle hooks are present. `OpenAIProvider.Aclose`, `MultiProvider.Aclose`, `ResolveComputer` / `DisposeResolvedComputers`, MCP approval/failure/meta hooks in `agents/mcp_util.go`, and trace persistence/reattach in `agents/trace_resume.go` now exist.
- Responses tool-search parity is now aligned. Go has `ToolSearchTool`, `tool_namespace(...)`, `FunctionTool.DeferLoading`, collision-safe lookup keys, namespace-aware approvals/tool context, tool-search call/output run items, deferred hosted MCP payloads, prompt-managed opaque tool-search allowance, and OpenAI Responses conversion/validation in `agents/tool_search.go`, `agents/tool_identity.go`, `agents/run_context.go`, `agents/tool_context.go`, `agents/run_impl.go`, and `agents/models_openai_responses.go`.
- RunState schema compatibility is aligned to current Python. Go now emits and accepts schema `1.5`, preserves namespaced function calls and tool-search run items through serialization/deserialization, and keeps resumed approval/dispatch data compatible with current Python snapshots.
- Responses computer-tool wire-shape selection is aligned. `agents/models_openai_responses.go` now picks GA `computer` vs preview `computer_use_preview` from the effective request model/tool choice while keeping the runtime `ToolName()` compatibility alias.
- Function-tool timeout semantics are aligned. `agents/tool_function_invoke.go`, `agents/run_impl.go`, and `agents/realtime/session.go` now enforce `timeout_seconds`, `timeout_behavior`, and `timeout_error_function`, including timeout-as-result vs raise-exception behavior in runner and realtime flows.
- MCP display metadata fallback/persistence is aligned. Local MCP tools now carry title/description fallback through `agents/mcp_util.go` and `agents/run_impl.go`, hosted MCP calls recover display metadata from prior `mcp_list_tools` history, and RunState now round-trips `ToolCallItem` titles/descriptions.
- Trace reattach semantics are aligned. `agents/trace_resume.go` now reattaches only when trace id, workflow name, group id, metadata, and tracing API key/hash all match the effective resumed settings; disabled runs no longer reattach, reattached traces preserve the live tracing key/hash, and the started-trace cache is now bounded like upstream Python.
- Responses websocket transport is aligned. `agents/models_openai_responses_ws.go` now uses a real websocket transport with persistent connection reuse, request serialization, explicit websocket base URL support, typed event parsing, request timeout handling, terminal-response consumption for `GetResponse()`, and provider/session lifecycle cleanup, matching current Python `models/openai_responses.py` behavior for the supported Go surface.

## P0 gaps
- none

## P1 gaps
- none

## Suggested implementation order
1. Rerun a full Python-vs-Go audit before merge/release to confirm no new upstream drift.
2. If the audit stays clean, this branch is ready to merge.
