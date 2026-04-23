from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .config import load_yaml, repo_root, resolve_repo_path


@dataclass(frozen=True)
class RuntimeEnv:
    backend: str
    name: str
    python: Path
    purpose: str = ""


def load_runtime_envs(config_path: str | Path = "configs/runtime_envs.yaml") -> dict[str, RuntimeEnv]:
    payload = load_yaml(config_path)
    runtime_envs: dict[str, RuntimeEnv] = {}
    for backend, config in payload.get("envs", {}).items():
        runtime_envs[backend] = RuntimeEnv(
            backend=backend,
            name=config.get("name", backend),
            python=resolve_repo_path(config["python"]),
            purpose=config.get("purpose", ""),
        )
    return runtime_envs


def get_runtime_env(backend: str, config_path: str | Path = "configs/runtime_envs.yaml") -> RuntimeEnv:
    runtime_envs = load_runtime_envs(config_path)
    if backend not in runtime_envs:
        raise KeyError(f"Unknown runtime backend: {backend}")
    return runtime_envs[backend]


def build_repo_pythonpath(extra_paths: Iterable[str | Path] | None = None) -> str:
    ordered_paths = [str(repo_root())]
    if extra_paths:
        ordered_paths.extend(str(resolve_repo_path(path)) for path in extra_paths)
    existing = os.environ.get("PYTHONPATH")
    if existing:
        ordered_paths.append(existing)
    return os.pathsep.join(ordered_paths)


def build_subprocess_env(
    extra_paths: Iterable[str | Path] | None = None,
    *,
    env_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = build_repo_pythonpath(extra_paths=extra_paths)
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    return env


def command_preview(runtime_env: RuntimeEnv, arguments: Sequence[str | Path]) -> str:
    command = [str(runtime_env.python), *[str(argument) for argument in arguments]]
    return shlex.join(command)


def run_in_runtime_env(
    runtime_env: RuntimeEnv,
    arguments: Sequence[str | Path],
    *,
    cwd: str | Path | None = None,
    extra_paths: Iterable[str | Path] | None = None,
    env_overrides: Mapping[str, str] | None = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [str(runtime_env.python), *[str(argument) for argument in arguments]]
    return subprocess.run(
        command,
        cwd=str(resolve_repo_path(cwd or repo_root())),
        env=build_subprocess_env(extra_paths=extra_paths, env_overrides=env_overrides),
        check=check,
        text=True,
        capture_output=capture_output,
    )
