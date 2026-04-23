---
name: workflow-execute
description: Execute only approved task scope in this repository's `main` working tree, keep progress logs timestamped, report checkpoints frequently, and stop immediately on blockers. Use when a task is in in_progress status after planning approval.
---

# Workflow Execute

## Overview

Implement approved scope with strict progress tracking and no hidden scope expansion.

## Source of Truth

Always follow:
- `tasks/_templates/tasklifecycle.md`
- `tasks/_templates/task.template.md`
- `tasks/_templates/progress.template.md`

Do not edit these templates unless the user explicitly asks to change the workflow standard.

## Input Contract

- `task_file`: task path such as `tasks/NNN-slug.md`
- `current_status`: current task status
- `approved_scope`: approved implementation scope
- `user_request`: latest user instruction

## Output Contract

- `status_transition`: `from -> to` or `no_change`
- `artifacts_changed`: changed files list or `none`
- `next_action`: exact next step for execution/review handoff
- `checkpoint_hint`: checkpoint reminder or `N/A`

## Required Execute Workflow

1. Ensure scope is approved before implementation.
2. Set task status to `in_progress` when execution starts.
3. Implement directly in the current repository working tree on `main`.
4. Keep unrelated local changes out of the task scope, or document them explicitly if they affect execution.
5. Update `Progress Log` with timestamp after each meaningful subtask:
   completed work, files changed, result, new risk/discovery.
6. Remind checkpoint at rollback-safe milestones.
7. If blocked, set status `blocked`, record blocker/tried/unblock condition, then stop.

## Guardrails

- Never create per-task branch or worktree for this repository.
- Never do review fixes in this stage.
- Never archive or mark done in this stage.
- Never implement items outside approved scope.
