#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/review-pr-local.sh [PR_NUMBER_OR_URL] [--output FILE] [--dry-run] [--prompt-file FILE]

Examples:
  scripts/review-pr-local.sh 123
  scripts/review-pr-local.sh https://github.com/owner/repo/pull/123 --dry-run
  scripts/review-pr-local.sh 123 --output /tmp/pr-123-review.md
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

PR_REF=""
OUTPUT_FILE=""
DRY_RUN=0
PROMPT_FILE=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --output)
      [ "$#" -ge 2 ] || { echo "--output requires a file path" >&2; exit 1; }
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --prompt-file)
      [ "$#" -ge 2 ] || { echo "--prompt-file requires a file path" >&2; exit 1; }
      PROMPT_FILE="$2"
      shift 2
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [ -z "$PR_REF" ]; then
        PR_REF="$1"
      else
        echo "Unexpected argument: $1" >&2
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

require_cmd git
require_cmd gh
require_cmd codex

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "Run this script inside a git repository checkout." >&2
  exit 1
}

gh auth status >/dev/null 2>&1 || {
  echo "GitHub CLI is not authenticated. Run: gh auth login" >&2
  exit 1
}

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set in your local environment." >&2
  exit 1
fi

if [ -n "$PROMPT_FILE" ] && [ ! -f "$PROMPT_FILE" ]; then
  echo "Prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

if [ -n "$PROMPT_FILE" ]; then
  echo "Warning: --prompt-file is ignored because this codex version cannot combine --base with a custom prompt." >&2
fi

if [ -z "$PR_REF" ]; then
  PR_REF="$(gh pr view --json number -q .number 2>/dev/null || true)"
  if [ -z "$PR_REF" ]; then
    echo "Cannot infer PR from current branch. Pass a PR number or PR URL." >&2
    exit 1
  fi
fi

PR_NUMBER="$(gh pr view "$PR_REF" --json number -q .number)"
BASE_REF="$(gh pr view "$PR_REF" --json baseRefName -q .baseRefName)"
BASE_SHA="$(gh pr view "$PR_REF" --json baseRefOid -q .baseRefOid)"
HEAD_SHA="$(gh pr view "$PR_REF" --json headRefOid -q .headRefOid)"
PR_TITLE="$(gh pr view "$PR_REF" --json title -q .title)"
REPO_FULL="$(gh repo view --json nameWithOwner -q .nameWithOwner)"

git fetch --no-tags origin "${BASE_REF}:${BASE_REF}" >/dev/null 2>&1 || git fetch --no-tags origin "${BASE_REF}" >/dev/null

if [ -z "$OUTPUT_FILE" ]; then
  OUTPUT_FILE="$(mktemp -t codex-pr-${PR_NUMBER}.XXXXXX.md)"
fi

LOG_FILE="$(mktemp -t codex-pr-${PR_NUMBER}.XXXXXX.log)"
trap 'rm -f "$LOG_FILE"' EXIT

set +e
codex exec review --base "$BASE_REF" --output-last-message "$OUTPUT_FILE" >"$LOG_FILE" 2>&1
review_exit_code=$?
set -e

if [ "$review_exit_code" -ne 0 ]; then
  original="${LOG_FILE}.raw"
  mv "$LOG_FILE" "$original"
  {
    echo "## Codex Review Execution Error"
    echo
    echo "codex exited with code ${review_exit_code}."
    echo
    echo "Raw output (from codex stdout/stderr):"
    echo
    echo '```text'
    cat "$original"
    echo '```'
  } >"$OUTPUT_FILE"
fi

if [ ! -s "$OUTPUT_FILE" ]; then
  {
    echo "## Codex Review Execution Error"
    echo
    echo "Codex completed but did not write a final review message."
    echo
    echo "Raw output (from codex stdout/stderr):"
    echo
    echo '```text'
    cat "$LOG_FILE"
    echo '```'
  } >"$OUTPUT_FILE"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry-run complete. Review body file: $OUTPUT_FILE"
  exit 0
fi

gh pr review "$PR_REF" --comment --body-file "$OUTPUT_FILE"
echo "Posted review comment to PR ${PR_REF} from: $OUTPUT_FILE"
