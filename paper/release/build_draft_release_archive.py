#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import build_draft_release_filelist


PAPER_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PAPER_ROOT / "release"
DIST_ROOT = RELEASE_ROOT / "dist"
FILELIST = RELEASE_ROOT / "draft_release_filelist.txt"
ARCHIVE_NAME = "dgpde_neurips2026_draft_artifacts.zip"
ARCHIVE_ROOT = "dgpde_neurips2026_draft_artifacts"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def main() -> None:
    build_draft_release_filelist.main()
    paths = [line.strip() for line in FILELIST.read_text().splitlines() if line.strip()]
    missing = [path for path in paths if not (PAPER_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"release file list contains missing files: {missing}")

    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = DIST_ROOT / ARCHIVE_NAME
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in paths:
            source = PAPER_ROOT / path
            archive_name = f"{ARCHIVE_ROOT}/{path}"
            archive.writestr(zip_info(archive_name), source.read_bytes())

    checksum = digest(archive_path)
    checksum_path = archive_path.with_suffix(".sha256")
    checksum_path.write_text(f"{checksum}  {ARCHIVE_NAME}\n")
    print(f"[release-archive] files={len(paths)} archive={archive_path}")
    print(f"[release-archive] sha256={checksum} checksum={checksum_path}")


if __name__ == "__main__":
    main()
