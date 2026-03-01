#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_bins tmux jq git
registry_migrate_if_needed

# Clean up tasks marked merged=true:
# - kill tmux session if exists
# - remove git worktree
# - prune from registry

kept=0
removed=0

new="$(mktemp)"
echo '{"schema":1,"tasks":[]}' > "$new"

while IFS= read -r task; do
  id=$(jq -r '.id' <<<"$task")
  merged=$(jq -r '.merged // false' <<<"$task")
  session=$(jq -r '.tmux_session' <<<"$task")
  worktree=$(jq -r '.worktree' <<<"$task")

  if [ "$merged" = "true" ]; then
    echo "[swarm] cleanup merged task $id"
    tmux kill-session -t "$session" 2>/dev/null || true
    ( cd "$REPO_ROOT" && git worktree remove --force "$worktree" 2>/dev/null ) || true
    removed=$((removed+1))
  else
    tmp2="$(mktemp)"
    jq --argjson t "$task" '.tasks += [$t]' "$new" > "$tmp2"
    mv "$tmp2" "$new"
    kept=$((kept+1))
  fi

done < <(list_tasks)

mv "$new" "$REGISTRY"

echo "[swarm] cleanup done kept=$kept removed=$removed"
