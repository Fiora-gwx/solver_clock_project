---
name: workflow-review
description: Review implementation quality on a main-only workflow without applying fixes, identify risks and test gaps, and output a structured issue table for user approval. Use when implementation is complete and the task enters review status.
---

# Workflow Review

## Overview

Review only. No fixes. Produce structured findings for approval.

## Source of Truth

Always follow:
- `tasks/_templates/tasklifecycle.md`
- `tasks/_templates/task.template.md`

Do not edit these templates unless the user explicitly asks to change the workflow standard.

## Input Contract

- `task_file`: task path such as `tasks/NNN-slug.md`
- `current_status`: current task status
- `approved_scope`: approved scope snapshot
- `user_request`: latest user instruction

## Output Contract

- `status_transition`: `from -> to` or `no_change`
- `artifacts_changed`: changed files list or `none`
- `next_action`: exact discussion/fix-approval next step
- `checkpoint_hint`: checkpoint reminder or `N/A`

## Required Review Workflow

1. Review the full task diff against the task-start baseline on `main`.
   State the baseline explicitly: pre-task commit, checkpoint commit, or explicit changed-file list.
2. Evaluate bugs, risk points, test coverage gaps, maintainability concerns.
3. Fill review issue table in exact format:

| # | 问题描述 | 风险等级 | 影响范围（涉及哪些文件/模块） | 推荐修复方式 |
|---|---|---|---|---|
| 1 | ... | low/medium/high | ... | ... |

4. Summarize completed, validated, uncertain parts.
5. Pause and wait for explicit fix approval.

## Guardrails

- Never apply fixes in review stage.
- Never skip issue table formatting.
- Always provide a review checkpoint reminder before fixes.
- Exclude unrelated local changes from findings and call them out if they affect the review baseline.
