#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.fid import compute_fid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID between generated samples and a precomputed stats file.")
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--reference-stats", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fid_value = compute_fid(args.samples_dir, args.reference_stats)
    print(f"{fid_value:.6f}")


if __name__ == "__main__":
    main()
