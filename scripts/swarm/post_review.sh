#!/usr/bin/env bash
set -euo pipefail

# Post local reviewer notes to a GitHub PR.
# Usage: post_review.sh <prNumberOrBranch> [--file .agent/review_notes.md]

TARGET="${1:-}"; shift || true
FILE=".agent/review_notes.md"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file) FILE="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "$TARGET" ]]; then
  echo "Usage: post_review.sh <prNumberOrBranch> [--file .agent/review_notes.md]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$FILE" ]]; then
  echo "[swarm] file not found: $FILE" >&2
  exit 1
fi

body=$(cat "$FILE")

# Avoid huge comments
max=6000
if (( ${#body} > max )); then
  body="${body:0:max}\n\n[swarm] (truncated)"
fi

gh pr comment "$TARGET" --body "$body"

echo "[swarm] posted review notes to PR: $TARGET"
