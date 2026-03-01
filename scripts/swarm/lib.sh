#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENCLAW_DIR="$REPO_ROOT/.openclaw"
REGISTRY="$OPENCLAW_DIR/active-tasks.json"
LOG_DIR="$OPENCLAW_DIR/logs"
WORKTREES_DIR="$OPENCLAW_DIR/worktrees"
CONFIG_FILE="$OPENCLAW_DIR/config.json"

now_iso() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

require_bins() {
  local missing=0
  for b in "$@"; do
    if ! command -v "$b" >/dev/null 2>&1; then
      echo "[swarm] missing required binary: $b" >&2
      missing=1
    fi
  done
  if [ "$missing" -eq 1 ]; then
    exit 2
  fi
}

ensure_openclaw_dirs() {
  mkdir -p "$LOG_DIR" "$WORKTREES_DIR"
  if [ ! -f "$REGISTRY" ]; then
    echo '{"schema":1,"tasks":[]}' > "$REGISTRY"
  fi
  if [ ! -f "$CONFIG_FILE" ]; then
    cat > "$CONFIG_FILE" <<'JSON'
{
  "auto_merge": false,
  "merge_method": "squash",
  "delete_branch": true,
  "require_ci_success": true,
  "require_codex_reviewer_pass": true,
  "codex_reviewer_pass_regex": "REVIEW PASS",
  "require_gemini_no_blocking": true,
  "gemini_blocking_regex": "(?i)(blocking|critical|major)",
  "stall_minutes": 15,
  "max_respawn_attempts": 3
}
JSON
  fi
}

config_get() {
  local jq_expr="$1"
  ensure_openclaw_dirs
  jq -r "$jq_expr" "$CONFIG_FILE"
}

registry_migrate_if_needed() {
  ensure_openclaw_dirs
  # If registry is legacy {tasks:[]}, wrap with schema.
  if jq -e 'has("schema")' "$REGISTRY" >/dev/null 2>&1; then
    return 0
  fi
  local tmp
  tmp="$(mktemp)"
  jq '{schema:1,tasks:(.tasks // [])}' "$REGISTRY" > "$tmp"
  mv "$tmp" "$REGISTRY"
}

task_add() {
  local id="$1" role="$2" branch="$3" base="$4" worktree="$5" tmux_session="$6" prompt="$7" log_path="$8"
  registry_migrate_if_needed
  local tmp created_at
  created_at="$(now_iso)"
  tmp="$(mktemp)"
  jq --arg id "$id" \
     --arg role "$role" \
     --arg branch "$branch" \
     --arg base "$base" \
     --arg worktree "$worktree" \
     --arg tmux_session "$tmux_session" \
     --arg prompt "$prompt" \
     --arg log_path "$log_path" \
     --arg created_at "$created_at" \
     '.tasks += [{
        id:$id,
        role:$role,
        branch:$branch,
        base:$base,
        worktree:$worktree,
        tmux_session:$tmux_session,
        log_path:$log_path,
        status:"spawned",
        attempts:0,
        pr_number:null,
        pr_state:null,
        pr_url:null,
        tmux_alive:null,
        ci:null,
        gemini_ok:null,
        codex_review_ok:null,
        ready_to_merge:false,
        merged:false,
        created_at:$created_at,
        updated_at:$created_at,
        prompt:$prompt
     }]' \
     "$REGISTRY" > "$tmp"
  mv "$tmp" "$REGISTRY"
}

# Update task by id with a jq filter that mutates the root object.
# Example: task_update "$id" '(.tasks[]|select(.id==$id)|.status)="done"'

task_update() {
  local id="$1" jq_filter="$2"
  registry_migrate_if_needed
  local tmp
  tmp="$(mktemp)"
  jq --arg id "$id" --arg updated_at "$(now_iso)" \
     "(.tasks[] | select(.id==\$id) | .updated_at)=\$updated_at | $jq_filter" \
     "$REGISTRY" > "$tmp" || { rm -f "$tmp"; return 1; }
  mv "$tmp" "$REGISTRY"
}

get_task() {
  local id="$1"
  registry_migrate_if_needed
  jq -c --arg id "$id" '.tasks[] | select(.id==$id)' "$REGISTRY"
}

list_tasks() {
  registry_migrate_if_needed
  jq -c '.tasks[]' "$REGISTRY"
}

log_mtime_epoch() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo 0
    return 0
  fi
  # macOS stat
  stat -f %m "$path" 2>/dev/null || stat -c %Y "$path" 2>/dev/null || echo 0
}

minutes_since_epoch() {
  local epoch="$1"
  if [ "$epoch" -le 0 ]; then
    echo 999999
    return 0
  fi
  local now
  now=$(date +%s)
  echo $(( (now - epoch) / 60 ))
}
