# Agent Swarm (local tmux + Codex + GitHub PR) + Auto-merge

This folder implements a lightweight version of the workflow described in the article:

- each agent gets its own git worktree + tmux session
- tasks are tracked in a deterministic JSON registry
- a monitor script checks: tmux alive, PR exists, CI/checks status, Gemini status, local codex-review status
- optional auto-respawn (dead/stalled sessions)
- optional auto-merge when gates pass

## Prereqs

- macOS/Linux
- `tmux`, `gh`, `jq`, `git`
- `codex` CLI installed and logged in

## State directory

Everything local lives under `.openclaw/` (ignored by git):

- `.openclaw/active-tasks.json` — task registry
- `.openclaw/logs/<taskId>.log` — full terminal logs (via `script`)
- `.openclaw/worktrees/<taskId>/` — git worktrees
- `.openclaw/config.json` — gates + auto-merge settings

## Quick start

```bash
mkdir -p .openclaw/logs
[ -f .openclaw/active-tasks.json ] || echo '{"schema":1,"tasks":[]}' > .openclaw/active-tasks.json

# spawn 3 agents (diff/dev/review) on separate worktrees
scripts/swarm/spawn_agent.sh diff   "scan python↔go gaps; output task list" \
  --base main --branch swarm/diff-$(date +%Y%m%d-%H%M)

scripts/swarm/spawn_agent.sh dev    "implement tasks; push; open/update PR" \
  --base main --branch swarm/dev-$(date +%Y%m%d-%H%M)

scripts/swarm/spawn_agent.sh review "review latest PR; write .agent/review_notes.md; post to PR" \
  --base main --branch swarm/review-$(date +%Y%m%d-%H%M)

# monitor loop (run manually or via cron)
scripts/swarm/monitor.sh
```

## Gates (definition of done)

Monitor determines `ready_to_merge` using `.openclaw/config.json`:

- CI checks success (via `gh pr view --json statusCheckRollup`)
- local codex-reviewer PASS (regex match in `.agent/review_notes.md`)
- Gemini no-blocking (Gemini latest comment must NOT contain blocking keywords)

When `auto_merge=true`, `monitor.sh` will run `gh pr merge` automatically **only when gates are met**.

## Cron

Every 10 minutes:

```cron
*/10 * * * * cd /Users/denggeng/work-dg/go/openai-agents-go-plus && scripts/swarm/monitor.sh >> .openclaw/monitor.log 2>&1
```

Cleanup daily:

```cron
0 3 * * * cd /Users/denggeng/work-dg/go/openai-agents-go-plus && scripts/swarm/cleanup.sh >> .openclaw/cleanup.log 2>&1
```

## Notes

- This repo already uses `.agent/` for local review/dev status files.
- Auto-respawn is conservative: it restarts a new tmux session if a session is dead or its log hasn't updated for `stall_minutes`.
