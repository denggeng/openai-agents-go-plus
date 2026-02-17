# Repository Guidelines

## Project Structure & Module Organization
- `agents/` is the core SDK (agent loop, tools, guardrails, handoffs, realtime, and extensions).
- `agentstesting/` contains fake models and helpers used across tests.
- `openaitypes/`, `modelsettings/`, `memory/`, `tracing/`, `usage/`, `util/` provide shared types and utilities.
- `examples/` hosts runnable samples grouped by scenario (e.g., `examples/basic`, `examples/agent_patterns`).
- `.upstaream/` tracks the upstream Python SDK snapshot and migration notes (see `gap_report_20260212.md`).

## Build, Test, and Development Commands
Use standard Go tooling:
```
go test ./...
```
Runs the full test suite.
```
go test ./agents -run ApplyPatchTool
```
Runs a focused subset by name.
```
gofmt -w ./agents ./openaitypes ./memory
```
Formats Go sources. Keep files gofmt-clean.

## Coding Style & Naming Conventions
- Go formatting: tabs for indentation, gofmt for line wrapping and alignment.
- Names: exported identifiers in `CamelCase`, unexported in `camelCase`, package names lower-case.
- File naming: use `*_test.go` for tests and keep package-private helpers unexported unless reused.

## Testing Guidelines
- Tests use Go’s `testing` package plus `stretchr/testify` (`assert`/`require`).
- Add unit tests alongside the package under test, e.g., `agents/*_test.go`.
- When porting features from Python, copy/translate tests first, then implement Go code until tests pass.

## Porting Workflow (Python → Go)
- Reference upstream sources under `.upstaream/openai-agents-python/`.
- Start from the closest Python tests and mirror behavior.
- Update `gap_report_20260212.md` if you close a migration gap.

## Commit & Pull Request Guidelines
- Existing history mixes simple imperative messages (`Add ...`, `Fix ...`) and Conventional-style prefixes (`fix:`, `feat:`, `fix(memory):`).
- Follow that pattern: short summary, optional scope for clarity.
- PRs should include: purpose, test coverage (commands run), and any behavior changes.
- Link related issues when applicable and call out breaking changes explicitly.

## Security & Configuration Tips
- Examples require `OPENAI_API_KEY` in the environment.
- Avoid committing secrets; prefer env vars and local config only.
