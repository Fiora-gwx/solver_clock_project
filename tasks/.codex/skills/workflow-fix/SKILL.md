---
name: workflow-fix
description: Apply only explicitly approved review fixes in the repository's main working tree, record each applied fix with impact details, and preserve unresolved issues clearly. Use when the task is in fixing status after review approval.
---

# Workflow Fix

## Overview

Fix only approved review items and keep repair scope tightly controlled.

## Source of Truth

Always follow:
- `tasks/_templates/tasklifecycle.md`
- `tasks/_templates/task.template.md`
- `tasks/_templates/progress.template.md`

Do not edit these templates unless the user explicitly asks to change the workflow standard.

## Input Contract

- `task_file`: task path such as `tasks/NNN-slug.md`
- `current_status`: current task status
- `approved_scope`: explicit approved issue IDs or fix scope
- `user_request`: latest user instruction

## Output Contract

- `status_transition`: `from -> to` or `no_change`
- `artifacts_changed`: changed files list or `none`
- `next_action`: exact next step for remaining fixes or done handoff
- `checkpoint_hint`: checkpoint reminder or `N/A`

## Required Fix Workflow

1. Apply only approved issues.
2. If user asks questions about issues, pause fixes and answer first.
3. For each approved fix, report:
   what changed, affected files, expected impact.
4. Update `Fix Log` and `Progress Log` with timestamp and issue linkage.
5. Summarize fixed and unresolved issues after each batch.
6. Remind rollback checkpoint after each safe batch.

## Guardrails

- Never fix unapproved items.
- Never do opportunistic refactor or new feature work.
- Never create per-task branch or worktree for this repository.
- Never mark done without explicit user confirmation.
