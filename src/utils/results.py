from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import dump_json, resolve_repo_path


SUMMARY_FIELDNAMES: tuple[str, ...] = (
    "backend",
    "dataset",
    "model_asset",
    "solver",
    "schedule",
    "nfe",
    "num_samples",
    "fid",
)

RESULT_ID_FIELDS: tuple[str, ...] = (
    "backend",
    "dataset",
    "model_asset",
    "solver",
    "schedule",
    "nfe",
)


def write_run_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    return dump_json(payload, path)


def compact_result_row(row: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for field in SUMMARY_FIELDNAMES:
        value = row.get(field, "")
        compacted[field] = "" if value is None else value
    return compacted


def result_row_identity(row: dict[str, Any]) -> tuple[str, ...]:
    compacted = compact_result_row(row)
    return tuple(str(compacted.get(field, "")) for field in RESULT_ID_FIELDS)


def write_result_rows(csv_path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    resolved = resolve_repo_path(csv_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = [compact_result_row(dict(row)) for row in rows]
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_FIELDNAMES))
        writer.writeheader()
        writer.writerows(normalized_rows)
    return resolved


def compact_result_csv(
    csv_path: str | Path,
    *,
    keep_row: Callable[[dict[str, Any]], bool] | None = None,
) -> Path:
    resolved = resolve_repo_path(csv_path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if keep_row is not None:
        rows = [row for row in rows if keep_row(row)]
    return write_result_rows(resolved, rows)


def append_result_row(csv_path: str | Path, row: dict[str, Any]) -> Path:
    resolved = resolve_repo_path(csv_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    if resolved.exists():
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))

    normalized_row = compact_result_row(dict(row))
    row_identity = result_row_identity(normalized_row)
    replaced = False
    for index, existing_row in enumerate(existing_rows):
        if result_row_identity(existing_row) == row_identity:
            existing_rows[index] = normalized_row
            replaced = True
            break
    if not replaced:
        existing_rows.append(normalized_row)
    return write_result_rows(resolved, existing_rows)
