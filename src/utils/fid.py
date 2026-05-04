from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import resolve_repo_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def compute_fid(samples_dir: str | Path, reference_stats: str | Path) -> float:
    samples_path = resolve_repo_path(samples_dir)
    reference_path = resolve_repo_path(reference_stats)
    command = [sys.executable, "-m", "pytorch_fid", str(reference_path), str(samples_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "pytorch_fid failed with exit code "
            f"{result.returncode}:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    match = re.search(r"FID:\s*([0-9.]+)", result.stdout)
    if match is None:
        raise RuntimeError(f"Unable to parse FID output:\n{result.stdout}\n{result.stderr}")
    return float(match.group(1))


def image_files(samples_dir: str | Path) -> list[str]:
    samples_path = resolve_repo_path(samples_dir)
    files = sorted(
        str(path)
        for path in samples_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No image files found in {samples_path}.")
    return files


def reference_features_from_stats(reference_stats: str | Path) -> np.ndarray:
    reference_path = resolve_repo_path(reference_stats)
    with np.load(reference_path) as payload:
        for key in ("features", "activations", "pool_3", "real_features"):
            if key in payload:
                features = np.asarray(payload[key], dtype=np.float64)
                break
        else:
            available = ", ".join(sorted(payload.files))
            raise ValueError(
                "KID requires reference feature activations in the stats npz. "
                "Expected one of features, activations, pool_3, or real_features; "
                f"found keys: {available}."
            )
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("Reference KID features must have shape [num_images, feature_dim] with at least two rows.")
    return features


def polynomial_mmd_unbiased(
    generated_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> float:
    generated = np.asarray(generated_features, dtype=np.float64)
    reference = np.asarray(reference_features, dtype=np.float64)
    if generated.ndim != 2 or reference.ndim != 2:
        raise ValueError("KID features must be two-dimensional.")
    if generated.shape[1] != reference.shape[1]:
        raise ValueError("Generated and reference KID features must have the same feature dimension.")
    if generated.shape[0] < 2 or reference.shape[0] < 2:
        raise ValueError("KID requires at least two generated and two reference features.")
    scale = (1.0 / generated.shape[1]) if gamma is None else float(gamma)
    kernel_xx = (scale * generated.dot(generated.T) + float(coef0)) ** int(degree)
    kernel_yy = (scale * reference.dot(reference.T) + float(coef0)) ** int(degree)
    kernel_xy = (scale * generated.dot(reference.T) + float(coef0)) ** int(degree)
    m = generated.shape[0]
    n = reference.shape[0]
    xx = (kernel_xx.sum() - np.trace(kernel_xx)) / (m * (m - 1))
    yy = (kernel_yy.sum() - np.trace(kernel_yy)) / (n * (n - 1))
    xy = kernel_xy.mean()
    return float(xx + yy - 2.0 * xy)


def kid_from_features(
    generated_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    subsets: int = 100,
    subset_size: int = 1000,
    seed: int = 0,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> float:
    generated = np.asarray(generated_features, dtype=np.float64)
    reference = np.asarray(reference_features, dtype=np.float64)
    size = min(int(subset_size), generated.shape[0], reference.shape[0])
    if int(subsets) < 1:
        raise ValueError("KID subsets must be positive.")
    if size < 2:
        raise ValueError("KID subset size must be at least two after clipping to available features.")
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(subsets)):
        generated_indices = rng.choice(generated.shape[0], size=size, replace=False)
        reference_indices = rng.choice(reference.shape[0], size=size, replace=False)
        values.append(
            polynomial_mmd_unbiased(
                generated[generated_indices],
                reference[reference_indices],
                degree=degree,
                gamma=gamma,
                coef0=coef0,
            )
        )
    return float(np.mean(values))


def compute_kid(
    samples_dir: str | Path,
    reference_stats: str | Path,
    *,
    batch_size: int = 50,
    dims: int = 2048,
    device: str = "cuda",
    num_workers: int = 1,
    subsets: int = 100,
    subset_size: int = 1000,
    seed: int = 0,
) -> float:
    import torch
    from pytorch_fid.fid_score import get_activations
    from pytorch_fid.inception import InceptionV3

    files = image_files(samples_dir)
    reference_features = reference_features_from_stats(reference_stats)
    block_index = InceptionV3.BLOCK_INDEX_BY_DIM[int(dims)]
    resolved_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = InceptionV3([block_index]).to(resolved_device)
    generated_features = get_activations(
        files,
        model,
        batch_size=int(batch_size),
        dims=int(dims),
        device=resolved_device,
        num_workers=int(num_workers),
    )
    return kid_from_features(
        generated_features,
        reference_features,
        subsets=int(subsets),
        subset_size=int(subset_size),
        seed=int(seed),
    )
