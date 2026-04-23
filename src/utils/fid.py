from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from .config import resolve_repo_path


def compute_fid(samples_dir: str | Path, reference_stats: str | Path) -> float:
    samples_path = resolve_repo_path(samples_dir)
    reference_path = resolve_repo_path(reference_stats)
    command = [sys.executable, "-m", "pytorch_fid", str(reference_path), str(samples_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    match = re.search(r"FID:\s*([0-9.]+)", result.stdout)
    if match is None:
        raise RuntimeError(f"Unable to parse FID output:\n{result.stdout}\n{result.stderr}")
    return float(match.group(1))
