# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Fixed
- MCP stdio tests: avoid stdout noise from transitive deps after dependency bumps.

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
