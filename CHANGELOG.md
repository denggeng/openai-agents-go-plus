# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## 0.12.1 - 2026-03-24

### Changed
- Removed private `gopkg.inshopline.com/...` transitive packages from the module graph by switching Redis dependencies back to the public `github.com/redis/go-redis/v9` release line.

### Fixed
- Open-source builds no longer depend on private inshopline module sources or checksums.

## 0.12.0 - 2026-03-20

### Added
- Runner-managed model retry with provider-specific retry advice, OpenAI retry normalization, websocket replay-safety handling, and retry-aware usage accounting.
- `ModelSettings.prompt_cache_retention` support for both Chat Completions and Responses requests.
- `RunResult.ToInputList()` and `RunResultStreaming.ToInputList()` `normalized` mode for continuation flows.
- MCP SSE and Streamable HTTP convenience configuration for headers, auth, HTTP client factories, request timeout, and SSE read-timeout behavior.

### Changed
- Python parity was refreshed through upstream `openai-agents-python` `0.12.4`.
- `ModelSettings.Resolve()` now deep-merges nested `Retry` settings and `ExtraArgs`.
- MCP Streamable HTTP shared-session behavior now matches upstream more closely, including serialized shared-session access and isolated-session fallback.
- Approval rejection messages now persist consistently through run context, run-state resume, HITL, and realtime flows.

### Fixed
- Stream retry callbacks no longer leak internal retry-disable flags to stream consumers.
- Retry normalization can explicitly clear header-derived `Retry-After` delays when adapters request it.

## 0.11.0 - 2026-03-13

### Added
- OpenAI Responses websocket transport with persistent connection reuse, typed event handling, timeout handling, and terminal response finalization.
- Namespaced function tools, tool search support, and RunState schema compatibility updates for newer Python SDK behavior.
- Function-tool timeouts with configurable timeout behavior and timeout error handlers.
- Multi-provider prefix routing with `openai_prefix_mode` and `unknown_prefix_mode` parity behavior.

### Changed
- Python parity was refreshed through upstream `openai-agents-python` `0.11.1`.
- MCP tool metadata fallback and persistence were aligned for both local and hosted MCP tools.
- Trace resume and reattach semantics were tightened to match workflow, metadata, and tracing-key compatibility checks.

### Fixed
- Responses websocket reuse and tool-timeout validation were hardened.

## 0.9.2+patch1 - 2026-02-20

### Fixed
- MCP stdio tests: avoid stdout noise from transitive deps after dependency bumps.

## 0.9.2 - 2026-02-20

### Added
- Core agent loop with tools, handoffs, guardrails, and streaming events.
- Streaming cancellation with immediate and after-turn modes.
- Session memory backends: SQLite, Redis, Dapr, Postgres, encrypted, and advanced branching sessions.
- Realtime and voice pipelines (STT/TTS) with examples.
- MCP integrations (local and hosted) with approval workflows.
- Tracing with processors/exporters and OpenAI backend export support.
- Custom model providers and LiteLLM proxy integration.
- HITL approval flows with resumable run state.
