---
name: workflow-plan
description: Build a complete main-only task plan before any implementation, including scope boundaries, execution steps, test strategy, risk review, and approval-ready summary. Use when a task under `tasks/` is in pending status or when planning quality must be improved before any code change.
---

# Workflow Plan

## Overview

Produce a decision-ready plan and keep the task status at `pending` until explicit approval.

## Source of Truth

Always follow:
- `tasks/_templates/tasklifecycle.md`
- `tasks/_templates/task.template.md`
- `tasks/_templates/progress.template.md`

Do not edit these templates unless the user explicitly asks to change the workflow standard.

## Input Contract

- `task_file`: task path such as `tasks/NNN-slug.md`
- `current_status`: current task status
- `approved_scope`: approved scope or `none`
- `user_request`: latest user instruction

## Output Contract

- `status_transition`: `from -> to` or `no_change`
- `artifacts_changed`: changed files list or `none`
- `next_action`: exact approval or execution next step
- `checkpoint_hint`: checkpoint reminder or `N/A`

## Required Plan Workflow

1. Read relevant codebase areas before writing the plan.
2. Confirm planning stage behavior from lifecycle rules.
3. Fill required sections in `task_file`:
   Goal, Background/Context, Scope, Non-goals, Approach, Execution Plan, Test Plan, Risks, Open Questions.
4. Set repository-specific metadata:
   `Execution Branch: main`, `Workspace: current repository working tree`.
5. Run plan self-review:
   Gaps, Feasibility, Risks Review, Split Decision.
6. Ensure task status is `pending`.
7. Provide an approval-ready summary and pause.

## Guardrails

- Never implement product code in this stage.
- Never create per-task branch or worktree for this repository.
- Never silently expand scope.
- Never skip test planning.
- End with a clear wait-for-approval handoff.
