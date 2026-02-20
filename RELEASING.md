# Releasing

This document describes the standard release process for this repository.

## 1) Pre-flight checks

- Ensure you are on `main` and the working tree is clean.
- Run tests:

```bash
go test ./...
```

- Format code when needed:

```bash
gofmt -w ./agents ./openaitypes ./memory
```

- Confirm `README.md` and `CHANGELOG.md` are up to date.

## 2) Versioning

We use SemVer tags (e.g. `v0.1.0`, `v0.2.0`, `v1.0.0`).

Suggested first public release: `v0.1.0`.

## 3) Tag and push

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

## 4) Create GitHub Release

- Title: `vX.Y.Z`
- Notes: copy from `CHANGELOG.md` (latest section)
- Attach any relevant migration notes if needed.

## 5) Verify module availability

```bash
go list -m -versions github.com/denggeng/openai-agents-go-plus
go get github.com/denggeng/openai-agents-go-plus@vX.Y.Z
```

## 6) Post-release

- Bump `CHANGELOG.md` with a new "Unreleased" section if it was finalized.
- Announce changes to users if applicable.
