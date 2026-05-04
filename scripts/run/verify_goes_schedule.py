#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goes.verify import verify_goes_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a GOES schedule.json and optional ScheduleBundle directory.")
    parser.add_argument("--schedule-json", "--schedule", dest="schedule_json", required=True)
    parser.add_argument("--bundle-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_goes_schedule(args.schedule_json, bundle_dir=args.bundle_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
