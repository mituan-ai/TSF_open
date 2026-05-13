from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SequenceDataset:
    windows: np.ndarray
    targets: np.ndarray
    feature_names: tuple[str, ...]
    target_name: str

    def as_dict(self) -> dict[str, object]:
        return {
            "n_samples": int(self.windows.shape[0]),
            "window_size": int(self.windows.shape[1]),
            "n_features": int(self.windows.shape[2]),
            "feature_names": list(self.feature_names),
            "target_name": self.target_name,
        }


@dataclass(frozen=True)
class StandardizationStats:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / np.maximum(self.std, 1e-12)


def fit_standardization(values: np.ndarray) -> StandardizationStats:
    array = np.asarray(values, dtype=np.float64)
    return StandardizationStats(mean=array.mean(axis=0), std=array.std(axis=0))


def build_sliding_windows(
    features: np.ndarray,
    target: np.ndarray,
    *,
    window_size: int,
    horizon: int,
    feature_names: tuple[str, ...],
    target_name: str,
) -> SequenceDataset:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must have shape (time, n_features)")
    if y.ndim != 1:
        raise ValueError("target must have shape (time,)")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and target must have the same time length")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match features")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    n_samples = x.shape[0] - window_size - horizon + 1
    if n_samples <= 0:
        raise ValueError("not enough time steps for the requested window_size and horizon")

    windows = np.empty((n_samples, window_size, x.shape[1]), dtype=np.float64)
    targets = np.empty(n_samples, dtype=np.float64)
    for sample_index in range(n_samples):
        start = sample_index
        end = start + window_size
        windows[sample_index] = x[start:end]
        targets[sample_index] = y[end + horizon - 1]
    return SequenceDataset(
        windows=windows,
        targets=targets,
        feature_names=tuple(feature_names),
        target_name=target_name,
    )
