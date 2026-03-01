#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_bins tmux jq gh git
registry_migrate_if_needed

stall_minutes=$(config_get '.stall_minutes')
max_respawn=$(config_get '.max_respawn_attempts')
auto_merge=$(config_get '.auto_merge')
merge_method=$(config_get '.merge_method')
delete_branch=$(config_get '.delete_branch')
require_ci=$(config_get '.require_ci_success')
require_codex=$(config_get '.require_codex_reviewer_pass')
require_gemini=$(config_get '.require_gemini_no_blocking')

codex_pass_re=$(config_get '.codex_reviewer_pass_regex')
gemini_block_re=$(config_get '.gemini_blocking_regex')

# Helpers
pr_ci_ok() {
  local pr="$1"
  local rollup
  rollup=$(gh pr view "$pr" --json statusCheckRollup 2>/dev/null || echo '{}')
  # If no checks, treat as ok? We'll treat empty as ok.
  local total
  total=$(jq '.statusCheckRollup | length' <<<"$rollup" 2>/dev/null || echo 0)
  if [ "$total" -eq 0 ]; then
    echo "ok"
    return 0
  fi
  local failed
  failed=$(jq '[.statusCheckRollup[] | select((.conclusion // "") != "SUCCESS")] | length' <<<"$rollup")
  if [ "$failed" -eq 0 ]; then
    echo "ok"
  else
    echo "fail"
  fi
}

gemini_ok() {
  local pr="$1"
  local comments
  comments=$(gh pr view "$pr" --json comments 2>/dev/null || echo '{}')
  local body
  body=$(jq -r '.comments | map(select(.author.login=="gemini-code-assist")) | .[-1].bodyText // ""' <<<"$comments")
  if [ -z "$body" ]; then
    # no gemini output yet
    echo "unknown"
    return 0
  fi
  if echo "$body" | perl -ne "exit(\$_ =~ /$gemini_block_re/ ? 0 : 1)"; then
    echo "fail"
  else
    echo "ok"
  fi
}

codex_review_ok() {
  # local reviewer writes REVIEW PASS in .agent/review_notes.md in repo root.
  local file="$REPO_ROOT/.agent/review_notes.md"
  if [ ! -f "$file" ]; then
    echo "unknown"
    return 0
  fi
  if perl -ne "exit(\$_ =~ /$codex_pass_re/ ? 0 : 1)" "$file"; then
    echo "ok"
  else
    echo "fail"
  fi
}

try_respawn() {
  local id="$1" worktree="$2" role="$3" prompt="$4" attempts="$5"
  if [ "$attempts" -ge "$max_respawn" ]; then
    return 1
  fi
  local new_attempt=$((attempts + 1))
  local session="codex-${role}-${id}-retry${new_attempt}"
  local log_path="$LOG_DIR/${id}-retry${new_attempt}.log"

  echo "[swarm] respawn $id attempt=$new_attempt session=$session"

  # start new session
  ( cd "$worktree" && tmux new-session -d -s "$session" -c "$worktree" "script -q '$log_path' codex" )
  tmux send-keys -t "$session" -l -- "RETRY #$new_attempt: $prompt"
  tmux send-keys -t "$session" C-m

  task_update "$id" "(.tasks[]|select(.id==\$id)|.attempts)=$new_attempt | (.tasks[]|select(.id==\$id)|.tmux_session)=\"$session\" | (.tasks[]|select(.id==\$id)|.log_path)=\"$log_path\" | (.tasks[]|select(.id==\$id)|.status)=\"respawned\""
  return 0
}

# Iterate tasks
while IFS= read -r task; do
  id=$(jq -r '.id' <<<"$task")
  role=$(jq -r '.role' <<<"$task")
  branch=$(jq -r '.branch' <<<"$task")
  session=$(jq -r '.tmux_session' <<<"$task")
  worktree=$(jq -r '.worktree' <<<"$task")
  log_path=$(jq -r '.log_path' <<<"$task")
  attempts=$(jq -r '.attempts' <<<"$task")
  prompt=$(jq -r '.prompt' <<<"$task")

  # tmux alive
  alive="no"
  if tmux has-session -t "$session" 2>/dev/null; then
    alive="yes"
  fi

  # stall detection by log mtime
  mtime=$(log_mtime_epoch "$log_path")
  mins=$(minutes_since_epoch "$mtime")
  stalled="no"
  if [ "$alive" = "yes" ] && [ "$mins" -ge "$stall_minutes" ]; then
    stalled="yes"
  fi

  pr_num=""; pr_state=""; pr_url=""
  if pr_json=$(cd "$REPO_ROOT" && gh pr view "$branch" --json number,state,url 2>/dev/null); then
    pr_num=$(jq -r '.number' <<<"$pr_json")
    pr_state=$(jq -r '.state' <<<"$pr_json")
    pr_url=$(jq -r '.url' <<<"$pr_json")
  fi

  # Gate checks
  ci="unknown"; gem="unknown"; codexr="unknown"; ready="false"

  if [ -n "$pr_num" ]; then
    ci=$(pr_ci_ok "$pr_num")
    gem=$(gemini_ok "$pr_num")
    codexr=$(codex_review_ok)

    ready="true"
    if [ "$require_ci" = "true" ] && [ "$ci" != "ok" ]; then ready="false"; fi
    if [ "$require_gemini" = "true" ] && [ "$gem" != "ok" ]; then ready="false"; fi
    if [ "$require_codex" = "true" ] && [ "$codexr" != "ok" ]; then ready="false"; fi

    # Only merge if PR is open
    if [ "$pr_state" != "OPEN" ]; then
      ready="false"
    fi
  else
    ready="false"
  fi

  # Update registry snapshot
  tmp="$(mktemp)"
  jq --arg id "$id" \
     --arg alive "$alive" \
     --arg pr_num "$pr_num" \
     --arg pr_state "$pr_state" \
     --arg pr_url "$pr_url" \
     --arg ci "$ci" \
     --arg gem "$gem" \
     --arg codexr "$codexr" \
     --argjson ready_to_merge $( [ "$ready" = "true" ] && echo true || echo false ) \
     --argjson stalled $( [ "$stalled" = "yes" ] && echo true || echo false ) \
     --arg updated_at "$(now_iso)" \
     '(.tasks[] | select(.id==$id) | .updated_at)=$updated_at
      | (.tasks[] | select(.id==$id) | .tmux_alive)=$alive
      | (.tasks[] | select(.id==$id) | .stalled)=$stalled
      | (.tasks[] | select(.id==$id) | .pr_number)=(if $pr_num=="" then null else ($pr_num|tonumber) end)
      | (.tasks[] | select(.id==$id) | .pr_state)=(if $pr_state=="" then null else $pr_state end)
      | (.tasks[] | select(.id==$id) | .pr_url)=(if $pr_url=="" then null else $pr_url end)
      | (.tasks[] | select(.id==$id) | .ci)=$ci
      | (.tasks[] | select(.id==$id) | .gemini_ok)=(if $gem=="unknown" then null else ($gem=="ok") end)
      | (.tasks[] | select(.id==$id) | .codex_review_ok)=(if $codexr=="unknown" then null else ($codexr=="ok") end)
      | (.tasks[] | select(.id==$id) | .ready_to_merge)=$ready_to_merge
     ' "$REGISTRY" > "$tmp"
  mv "$tmp" "$REGISTRY"

  # Respawn on dead/stalled
  if [ "$alive" = "no" ] || [ "$stalled" = "yes" ]; then
    try_respawn "$id" "$worktree" "$role" "$prompt" "$attempts" || true
  fi

  # Auto-merge
  if [ "$auto_merge" = "true" ] && [ "$ready" = "true" ]; then
    echo "[swarm] auto-merge PR #$pr_num ($pr_url)"
    args=("$pr_num")
    if [ "$merge_method" = "squash" ]; then args+=(--squash); fi
    if [ "$merge_method" = "merge" ]; then args+=(--merge); fi
    if [ "$merge_method" = "rebase" ]; then args+=(--rebase); fi
    if [ "$delete_branch" = "true" ]; then args+=(--delete-branch); fi

    # Use --auto? We already gated on checks being green, so merge now.
    if cd "$REPO_ROOT" && gh pr merge "${args[@]}" --yes; then
      task_update "$id" "(.tasks[]|select(.id==\$id)|.merged)=true | (.tasks[]|select(.id==\$id)|.status)=\"merged\""
    else
      task_update "$id" "(.tasks[]|select(.id==\$id)|.status)=\"merge_failed\""
    fi
  fi

done < <(list_tasks)

# summary
if jq -e '.tasks|length>0' "$REGISTRY" >/dev/null 2>&1; then
  echo "[swarm] monitor ok $(now_iso)"
  jq -r '.tasks[] | "- \(.id) [\(.role)] tmux=\(.tmux_alive // "?") stalled=\(.stalled // false) pr=\(.pr_number // "-") ci=\(.ci // "-") gemini_ok=\(.gemini_ok // "-") codex_ok=\(.codex_review_ok // "-") ready=\(.ready_to_merge) merged=\(.merged)"' "$REGISTRY" || true
fi
