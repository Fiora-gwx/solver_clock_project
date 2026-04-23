#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.runtime_env import load_runtime_envs, run_in_runtime_env


PROBE_CODE = r"""
import importlib.metadata as metadata
import json
import sys
import traceback
from pathlib import Path

backend = sys.argv[1]
repo_root = Path(sys.argv[2])
result = {
    "backend": backend,
    "python": sys.executable,
}

for package_name in ("torch", "torchvision", "transformers", "diffusers"):
    try:
        result[package_name] = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        result[package_name] = None

try:
    import torch

    result["cuda_available"] = bool(torch.cuda.is_available())
    result["torch_cuda"] = torch.version.cuda
except Exception as exc:
    result["cuda_available"] = False
    result["torch_cuda"] = None
    result["cuda_error"] = repr(exc)

sys.path.insert(0, str(repo_root / "third_party" / "diffusers" / "src"))

if backend == "pndm":
    sys.path.insert(0, str(repo_root / "third_party" / "STORK" / "external" / "PNDM"))
    try:
        from diffusers import EulerDiscreteScheduler
        from model.ddim import Model

        result["backend_import"] = "ok"
        result["backend_detail"] = {
            "scheduler": EulerDiscreteScheduler.__name__,
            "model": Model.__name__,
        }
    except Exception:
        result["backend_import"] = "error"
        result["backend_error"] = traceback.format_exc(limit=8)
elif backend == "diffusers":
    try:
        from diffusers import DiffusionPipeline

        result["backend_import"] = "ok"
        result["backend_detail"] = {
            "pipeline": DiffusionPipeline.__name__,
        }
    except Exception:
        result["backend_import"] = "error"
        result["backend_error"] = traceback.format_exc(limit=12)
elif backend == "sana":
    try:
        from diffusers import EulerDiscreteScheduler

        result["backend_import"] = "ok"
        result["backend_detail"] = {
            "scheduler": EulerDiscreteScheduler.__name__,
        }
    except Exception:
        result["backend_import"] = "error"
        result["backend_error"] = traceback.format_exc(limit=8)

if not result.get("cuda_available"):
    result["status"] = "cuda_unavailable"
elif result.get("backend_import") == "error":
    result["status"] = "backend_import_error"
else:
    result["status"] = "ok"

print(json.dumps(result))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the configured backend conda environments used by the experiment launcher.")
    parser.add_argument("--runtime-config", default="configs/runtime_envs.yaml")
    parser.add_argument("--json", action="store_true", default=False)
    return parser.parse_args()


def probe_runtime_envs(runtime_config: str) -> list[dict[str, object]]:
    runtime_envs = load_runtime_envs(runtime_config)
    reports: list[dict[str, object]] = []
    for backend, runtime_env in runtime_envs.items():
        completed = run_in_runtime_env(
            runtime_env,
            ["-c", PROBE_CODE, backend, str(REPO_ROOT)],
            capture_output=True,
        )
        payload = json.loads(completed.stdout.strip())
        reports.append(payload)
    return reports


def print_human_report(reports: list[dict[str, object]]) -> None:
    for report in reports:
        print(f"[{report['backend']}] status={report['status']}")
        print(f"  python: {report['python']}")
        print(
            "  packages: "
            f"torch={report.get('torch')} "
            f"torchvision={report.get('torchvision')} "
            f"transformers={report.get('transformers')} "
            f"diffusers={report.get('diffusers')}"
        )
        print(f"  cuda: available={report.get('cuda_available')} torch_cuda={report.get('torch_cuda')}")
        if report.get("backend_import") == "error":
            print("  backend_import: error")
            error = str(report.get("backend_error", "")).strip().splitlines()
            if error:
                print(f"  backend_error: {error[-1]}")
        else:
            print("  backend_import: ok")


def main() -> None:
    args = parse_args()
    reports = probe_runtime_envs(args.runtime_config)
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        print_human_report(reports)


if __name__ == "__main__":
    main()
