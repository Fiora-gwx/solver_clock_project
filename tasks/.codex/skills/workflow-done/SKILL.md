---
name: workflow-done
description: Finalize a completed main-only task by updating completion metadata, archiving task files, and updating `tasks/PROGRESS.md` plus `tasks/LESSONS.md`. Use when implementation and approved fixes are complete and final handoff is confirmed.
---

# Workflow Done

## Overview

Close the task lifecycle cleanly with explicit completion bookkeeping.

## Source of Truth

Always follow:
- `tasks/_templates/tasklifecycle.md`
- `tasks/_templates/task.template.md`
- `tasks/_templates/lessons.template.md`

Do not edit these templates unless the user explicitly asks to change the workflow standard.

## Input Contract

- `task_file`: task path such as `tasks/NNN-slug.md`
- `current_status`: current task status
- `approved_scope`: finalization approval scope
- `user_request`: latest user instruction

## Output Contract

- `status_transition`: `from -> to` or `no_change`
- `artifacts_changed`: changed files list or `none`
- `next_action`: exact final handoff or commit step
- `checkpoint_hint`: checkpoint reminder or `N/A`

## Required Done Workflow

1. Verify preconditions:
   implementation complete, review complete, approved fixes complete, user confirms finalization.
2. Update completion sections and set status `done`.
3. Archive task file to `tasks/_archive/`.
4. Update `tasks/PROGRESS.md`.
5. Update `tasks/LESSONS.md` when reusable lessons exist.
6. Provide final completion summary and suggested final commit message.
7. Provide the final completion checkpoint reminder.

## Guardrails

- Never introduce new features in done stage.
- Never skip archive/progress bookkeeping.
- Never finalize without explicit confirmation.
