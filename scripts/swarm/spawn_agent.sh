#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   spawn_agent.sh <role> <prompt> --base <baseBranch> --branch <newBranch> [--session <tmuxSessionName>]

ROLE="${1:-}"; shift || true
PROMPT="${1:-}"; shift || true

BASE="main"
BRANCH=""
SESSION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2;;
    --branch) BRANCH="$2"; shift 2;;
    --session) SESSION="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "$ROLE" || -z "$PROMPT" || -z "$BRANCH" ]]; then
  echo "Usage: spawn_agent.sh <role> <prompt> --base <baseBranch> --branch <newBranch> [--session <tmuxSessionName>]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_bins git tmux jq gh codex script
ensure_openclaw_dirs
registry_migrate_if_needed

TASK_ID="${ROLE}-$(date +%Y%m%d-%H%M%S)"
WORKTREE="$WORKTREES_DIR/$TASK_ID"
LOG_PATH="$LOG_DIR/$TASK_ID.log"

# create worktree + branch
(
  cd "$REPO_ROOT"
  git fetch origin "$BASE" >/dev/null 2>&1 || true
  git worktree add -b "$BRANCH" "$WORKTREE" "origin/$BASE" >/dev/null
)

if [[ -z "$SESSION" ]]; then
  SESSION="codex-${ROLE}-${TASK_ID}"
fi

# Start a tmux session running Codex with full logging.
(
  cd "$WORKTREE"
  tmux new-session -d -s "$SESSION" -c "$WORKTREE" "script -q '$LOG_PATH' codex"
)

# Record the task.
task_add "$TASK_ID" "$ROLE" "$BRANCH" "$BASE" "$WORKTREE" "$SESSION" "$PROMPT" "$LOG_PATH"

# Send the prompt + Enter into Codex.
# (single-line + C-m is the most reliable across multi-attach tmux)
tmux send-keys -t "$SESSION" -l -- "$PROMPT"
tmux send-keys -t "$SESSION" C-m

cat <<EOF
[swarm] spawned:
  id: $TASK_ID
  role: $ROLE
  branch: $BRANCH
  base: $BASE
  worktree: $WORKTREE
  tmux: $SESSION
  log: $LOG_PATH
EOF
