---
name: workflow-orchestrator
description: Coordinate this repository's main-only task lifecycle by routing requests to the correct stage agent (plan/execute/review/fix/done) using task status, user intent, and repo-wide single-writer safety. Use when you need stage selection, transition validation, or next-step decisions for task files under `tasks/`.
---

# Workflow Orchestrator

## Overview

Route each task request to the correct lifecycle agent while keeping stage transitions safe, explicit, and auditable.

## Source of Truth

Always follow these repository-local files:
- `tasks/_templates/tasklifecycle.md`
- `tasks/_templates/task.template.md`
- `tasks/_templates/progress.template.md`

## Input Contract

All requests must provide:
- `task_file`: task path such as `tasks/NNN-slug.md`
- `current_status`: current task status
- `approved_scope`: user-approved fix/implementation scope, or `none`
- `user_request`: latest user instruction

## Output Contract

Always return:
- `status_transition`: `from -> to` or `no_change`
- `artifacts_changed`: changed files list or `none`
- `next_action`: exact next step and target agent
- `checkpoint_hint`: checkpoint reminder or `N/A`

## Routing Rules

Status mapping:
- `pending -> workflow-plan`
- `in_progress -> workflow-execute`
- `review -> workflow-review`
- `fixing -> workflow-fix`
- `done -> workflow-done`

Special handling:
- `blocked`: do not auto-advance; return unblock condition and wait for confirmation
- `cancelled`: do not auto-advance; return confirmation instruction only

Intent override policy:
- If `user_request` explicitly asks for another stage, validate safety first.
- Reject unsafe jumps, for example `pending -> execute` without approved plan.

## Repository Concurrency Policy

- Read-only analysis may happen in parallel when it does not write files.
- Any write-capable lifecycle stage must run in single-writer mode across the whole repository because all work happens on `main`.
- Never assign concurrent write-capable stage agents in this repository.

## Guardrails

- Keep `_templates` aligned with the current repository workflow.
- Do not implement task code in this agent.
- Return deterministic routing with minimal ambiguity.
