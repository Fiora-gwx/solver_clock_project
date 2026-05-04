import numpy as np
import pytest

from src.utils.fid import kid_from_features, polynomial_mmd_unbiased, reference_features_from_stats


def test_polynomial_mmd_unbiased_is_symmetric() -> None:
    generated = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    reference = np.asarray([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]], dtype=np.float64)

    left = polynomial_mmd_unbiased(generated, reference, degree=2, gamma=0.5, coef0=1.0)
    right = polynomial_mmd_unbiased(reference, generated, degree=2, gamma=0.5, coef0=1.0)

    assert np.isclose(left, right)


def test_kid_from_features_is_deterministic_for_fixed_seed() -> None:
    generated = np.arange(20, dtype=np.float64).reshape(10, 2)
    reference = generated + 0.5

    first = kid_from_features(generated, reference, subsets=5, subset_size=4, seed=123)
    second = kid_from_features(generated, reference, subsets=5, subset_size=4, seed=123)

    assert first == second


def test_reference_features_from_stats_requires_feature_arrays(tmp_path) -> None:
    feature_path = tmp_path / "features.npz"
    np.savez(feature_path, features=np.ones((3, 2), dtype=np.float64))

    loaded = reference_features_from_stats(feature_path)

    assert loaded.shape == (3, 2)

    stats_only_path = tmp_path / "fid_stats_only.npz"
    np.savez(stats_only_path, mu=np.zeros(2), sigma=np.eye(2))
    with pytest.raises(ValueError, match="KID requires reference feature activations"):
        reference_features_from_stats(stats_only_path)
