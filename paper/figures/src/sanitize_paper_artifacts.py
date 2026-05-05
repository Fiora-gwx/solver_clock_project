#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PAPER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PAPER_ROOT.parent

TEXT_TARGETS = [
    PAPER_ROOT / "project_context.md",
    PAPER_ROOT / "appendix/reproducibility.tex",
]

TEXT_GLOBS = [
    "results/**/*.md",
]


def sanitize_string(value: str) -> str:
    repo = str(REPO_ROOT)
    return value.replace(repo + "/", "").replace(repo, ".")


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_string(value)
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    return value


def sanitize_csv(path: Path) -> bool:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    sanitized = [[sanitize_string(cell) for cell in row] for row in rows]
    if sanitized == rows:
        return False
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(sanitized)
    return True


def sanitize_json(path: Path) -> bool:
    original_text = path.read_text()
    data = json.loads(original_text)
    sanitized = sanitize_json_value(data)
    if sanitized == data:
        return False
    path.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n")
    return True


def sanitize_text(path: Path) -> bool:
    original = path.read_text()
    sanitized = sanitize_string(original)
    if sanitized == original:
        return False
    path.write_text(sanitized)
    return True


def main() -> None:
    changed: list[Path] = []
    for path in PAPER_ROOT.glob("results/**/*.csv"):
        if sanitize_csv(path):
            changed.append(path)
    for path in PAPER_ROOT.glob("results/**/*.json"):
        if sanitize_json(path):
            changed.append(path)
    for path in TEXT_TARGETS:
        if path.exists() and sanitize_text(path):
            changed.append(path)
    for pattern in TEXT_GLOBS:
        for path in PAPER_ROOT.glob(pattern):
            if sanitize_text(path):
                changed.append(path)

    for path in changed:
        print(path.relative_to(REPO_ROOT))
    print(f"sanitized_files={len(changed)}")


if __name__ == "__main__":
    main()
