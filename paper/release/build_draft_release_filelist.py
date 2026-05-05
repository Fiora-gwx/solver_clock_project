#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PAPER_ROOT / "release"
FILELIST = RELEASE_ROOT / "draft_release_filelist.txt"
EXCLUSIONS = RELEASE_ROOT / "draft_release_exclusions.txt"

INCLUDE_PREFIXES = (
    "appendix/",
    "figures/",
    "release/",
    "results/",
    "sections/",
    "tables/",
)
INCLUDE_FILES = {
    "checklist.tex",
    "neurips_2026.pdf",
    "neurips_2026.sty",
    "neurips_2026.tex",
    "references.bib",
}
EXCLUDE_SUFFIXES = (
    ".aux",
    ".bbl",
    ".blg",
    ".log",
    ".out",
    ".pyc",
)
EXCLUDE_PARTS = {
    "__pycache__",
}
EXCLUDE_PREFIXES = (
    "internal_reviews/",
    "release/dist/",
)
EXCLUDE_FILES = {
    "project_context.md",
    "results/t2i/sd15_euler_nfe10_cfg7p5_schedule_profile_seed0.csv",
}
EXCLUDE_CONTAINS = (
    "offline_authorized_schedule",
    "authorized_offline_schedule",
)


def relative_files() -> list[str]:
    return sorted(
        str(path.relative_to(PAPER_ROOT))
        for path in PAPER_ROOT.rglob("*")
        if path.is_file()
    )


def excluded_reason(path: str) -> str | None:
    parts = set(Path(path).parts)
    if parts & EXCLUDE_PARTS:
        return "python bytecode cache"
    if path.endswith(EXCLUDE_SUFFIXES):
        return "latex/python build artifact"
    if path.startswith("release/dist/"):
        return "generated release archive output"
    if any(path.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return "internal review note, not part of anonymous artifact release"
    if any(token in path for token in EXCLUDE_CONTAINS):
        return "authorized-offline numeric schedule bundles are excluded from the anonymous artifact package"
    if path in EXCLUDE_FILES:
        if path.endswith("sd15_euler_nfe10_cfg7p5_schedule_profile_seed0.csv"):
            return "raw AYS numeric schedule values are excluded from the anonymous artifact package"
        return "working project context, not part of anonymous artifact release"
    if not (path in INCLUDE_FILES or any(path.startswith(prefix) for prefix in INCLUDE_PREFIXES)):
        return "not in draft release allowlist"
    return None


def main() -> None:
    included: list[str] = []
    excluded: list[tuple[str, str]] = []
    for path in relative_files():
        reason = excluded_reason(path)
        if reason is None:
            included.append(path)
        else:
            excluded.append((path, reason))
    FILELIST.write_text("\n".join(included) + "\n")
    EXCLUSIONS.write_text("\n".join(f"{path}\t{reason}" for path, reason in excluded) + "\n")
    print(f"[release-filelist] included={len(included)} filelist={FILELIST}")
    print(f"[release-filelist] excluded={len(excluded)} exclusions={EXCLUSIONS}")


if __name__ == "__main__":
    main()
