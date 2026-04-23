from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import load_yaml, repo_root, resolve_repo_path


@dataclass(frozen=True)
class AssetRecord:
    key: str
    path: Path
    asset_type: str


class AssetManifest:
    def __init__(self, manifest_path: str | Path = "configs/assets_manifest.yaml") -> None:
        payload = load_yaml(manifest_path)
        self._project_root = resolve_repo_path(payload.get("project_root", repo_root()))
        self._assets = payload.get("assets", {})

    @property
    def project_root(self) -> Path:
        return self._project_root

    def has(self, key: str) -> bool:
        return key in self._assets

    def record(self, key: str) -> AssetRecord:
        if key not in self._assets:
            raise KeyError(f"Unknown asset key: {key}")
        payload = self._assets[key]
        return AssetRecord(
            key=key,
            path=self._project_root / payload["path"],
            asset_type=payload.get("type", "unknown"),
        )

    def path(self, key: str) -> Path:
        return self.record(key).path
