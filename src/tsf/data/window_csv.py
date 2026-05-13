from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class WindowDatasetMetadata:
    dataset_name: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    window_size_steps: int
    base_step_minutes: int | None
    dataset_dir: Path
    data_path: Path
    raw: dict[str, object]

    @classmethod
    def from_raw(
        cls,
        *,
        dataset_dir: Path,
        data_path: Path,
        raw: dict[str, object],
    ) -> "WindowDatasetMetadata":
        config = raw.get("config", {})
        if not isinstance(config, dict):
            config = {}
        feature_names = tuple(str(name) for name in raw["feature_names"])
        if not feature_names:
            raise ValueError("metadata feature_names must not be empty")
        return cls(
            dataset_name=str(raw.get("dataset_name", raw.get("dataset", dataset_dir.name))),
            feature_names=feature_names,
            target_names=_target_names_from_metadata(raw),
            window_size_steps=_window_size_from_metadata(raw, config),
            base_step_minutes=(
                int(config["base_step_minutes"])
                if config.get("base_step_minutes") is not None
                else None
            ),
            dataset_dir=dataset_dir,
            data_path=data_path,
            raw=raw,
        )

    @classmethod
    def load_csv(cls, dataset_dir: Path) -> "WindowDatasetMetadata":
        csv_dir = dataset_dir / "csv"
        metadata_path = csv_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls.from_raw(dataset_dir=dataset_dir, data_path=csv_dir, raw=raw)

    @classmethod
    def load_npz(cls, npz_path: Path, raw: dict[str, object]) -> "WindowDatasetMetadata":
        return cls.from_raw(dataset_dir=npz_path.parent, data_path=npz_path, raw=raw)

    @property
    def target_name(self) -> str:
        if len(self.target_names) != 1:
            raise ValueError("target_name is only available for single-target datasets")
        return self.target_names[0]

    def infer_window_columns(self, fieldnames: Iterable[str]) -> tuple[str, ...]:
        available = set(fieldnames)
        matched: dict[tuple[str, int], str] = {}
        offsets_by_feature: list[set[int]] = []
        for feature_name in self.feature_names:
            pattern = re.compile(rf"^{re.escape(feature_name)}_t_minus_(\d+)$")
            offsets: set[int] = set()
            for column in available:
                match = pattern.match(column)
                if match is None:
                    continue
                offset = int(match.group(1))
                offsets.add(offset)
                matched[(feature_name, offset)] = column
            if not offsets:
                raise ValueError(f"No window columns found for feature: {feature_name}")
            offsets_by_feature.append(offsets)

        common_offsets = set.intersection(*offsets_by_feature)
        if len(common_offsets) != self.window_size_steps:
            raise ValueError(
                f"Expected {self.window_size_steps} common window offsets, "
                f"found {len(common_offsets)}"
            )
        columns: list[str] = []
        for offset in sorted(common_offsets, reverse=True):
            columns.extend(matched[(feature, offset)] for feature in self.feature_names)
        return tuple(columns)


@dataclass(frozen=True)
class WindowDataset:
    metadata: WindowDatasetMetadata
    windows: np.ndarray
    targets: np.ndarray
    sample_metadata: tuple[dict[str, object], ...]
    source_path: Path

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.metadata.feature_names

    @property
    def target_name(self) -> str:
        return self.metadata.target_name

    @property
    def target_names(self) -> tuple[str, ...]:
        return self.metadata.target_names

    @property
    def source_csv(self) -> Path:
        return self.source_path

    def select_indices(self, indices: np.ndarray) -> "WindowDataset":
        normalized_indices = np.asarray(indices, dtype=np.int64)
        return WindowDataset(
            metadata=self.metadata,
            windows=self.windows[normalized_indices],
            targets=self.targets[normalized_indices],
            sample_metadata=tuple(self.sample_metadata[int(index)] for index in normalized_indices),
            source_path=self.source_path,
        )

    def indices_matching(self, filters: dict[str, object]) -> np.ndarray:
        selected: list[int] = []
        for index, meta in enumerate(self.sample_metadata):
            if all(meta.get(column) == value for column, value in filters.items()):
                selected.append(index)
        return np.asarray(selected, dtype=np.int64)

    def indices_where(self, *, domain_bucket: str | None = None, scenarios: Iterable[str] | None = None) -> np.ndarray:
        scenario_set = set(scenarios) if scenarios is not None else None
        selected: list[int] = []
        for index, meta in enumerate(self.sample_metadata):
            if domain_bucket is not None and meta.get("domain_bucket") != domain_bucket:
                continue
            if scenario_set is not None and meta.get("scenario_name") not in scenario_set:
                continue
            selected.append(index)
        return np.asarray(selected, dtype=np.int64)

    def domain_buckets(self) -> tuple[str, ...]:
        return tuple(sorted({str(meta["domain_bucket"]) for meta in self.sample_metadata}))

    def scenario_names(self) -> tuple[str, ...]:
        return tuple(sorted({str(meta["scenario_name"]) for meta in self.sample_metadata}))


@dataclass(frozen=True)
class FixedScenarioSplit:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    test_group_indices: dict[str, np.ndarray]
    scenario_column: str = "scenario_name"

    def as_dict(self, dataset: WindowDataset) -> dict[str, object]:
        return {
            "train_samples": int(self.train_indices.size),
            "val_samples": int(self.val_indices.size),
            "test_samples": int(self.test_indices.size),
            "scenario_column": self.scenario_column,
            "train_scenarios": _scenario_list(dataset, self.train_indices, self.scenario_column),
            "val_scenarios": _scenario_list(dataset, self.val_indices, self.scenario_column),
            "test_groups": {
                name: {
                    "samples": int(indices.size),
                    "scenarios": _scenario_list(dataset, indices, self.scenario_column),
                }
                for name, indices in sorted(self.test_group_indices.items())
            },
        }


@dataclass(frozen=True)
class WindowNormalization:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    def transform_windows(self, windows: np.ndarray) -> np.ndarray:
        values = np.asarray(windows, dtype=np.float32)
        return ((values - self.input_mean) / np.maximum(self.input_std, 1e-12)).astype(np.float32)

    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        values = np.asarray(targets, dtype=np.float32)
        return ((values - self.target_mean) / np.maximum(self.target_std, 1e-12)).astype(np.float32)

    def inverse_transform_targets(self, targets: np.ndarray) -> np.ndarray:
        values = np.asarray(targets, dtype=np.float64)
        return values * np.maximum(self.target_std, 1e-12) + self.target_mean

    def as_dict(self, feature_names: Iterable[str], target_names: Iterable[str]) -> dict[str, object]:
        names = tuple(feature_names)
        targets = tuple(target_names)
        return {
            "feature_names": list(names),
            "target_names": list(targets),
            "input_mean": {name: float(self.input_mean[index]) for index, name in enumerate(names)},
            "input_std": {name: float(self.input_std[index]) for index, name in enumerate(names)},
            "target_mean": {
                name: float(self.target_mean[index])
                for index, name in enumerate(targets)
            },
            "target_std": {
                name: float(self.target_std[index])
                for index, name in enumerate(targets)
            },
        }


def load_window_dataset(dataset_dir: Path, *, data_name: str = "windows.npz") -> WindowDataset:
    npz_path = dataset_dir / data_name
    if npz_path.exists():
        return load_window_npz_dataset(npz_path)
    if data_name.endswith(".npz"):
        raise FileNotFoundError(f"Window NPZ not found: {npz_path}")
    return load_window_csv_dataset(dataset_dir, csv_name=data_name)


def load_window_npz_dataset(npz_path: Path) -> WindowDataset:
    if not npz_path.exists():
        raise FileNotFoundError(f"Window NPZ not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as payload:
        raw_metadata_text = str(payload["metadata_json"].item())
        raw_metadata = json.loads(raw_metadata_text)
        metadata = WindowDatasetMetadata.load_npz(npz_path, raw_metadata)
        windows = np.asarray(payload["windows"], dtype=np.float32)
        targets = np.asarray(payload["targets"], dtype=np.float32)
        sample_metadata_json = payload["sample_metadata_json"]
        sample_metadata = tuple(
            json.loads(str(item))
            for item in np.asarray(sample_metadata_json).reshape(-1)
        )
    _validate_loaded_dataset(
        windows=windows,
        targets=targets,
        sample_metadata=sample_metadata,
        metadata=metadata,
        source_path=npz_path,
    )
    return WindowDataset(
        metadata=metadata,
        windows=windows,
        targets=targets,
        sample_metadata=sample_metadata,
        source_path=npz_path,
    )


def load_window_csv_dataset(dataset_dir: Path, *, csv_name: str = "all_windows.csv") -> WindowDataset:
    metadata = WindowDatasetMetadata.load_csv(dataset_dir)
    csv_path = metadata.data_path / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Window CSV not found: {csv_path}")

    target_columns = metadata.target_names
    rows: list[list[float]] = []
    targets: list[list[float]] = []
    sample_metadata: list[dict[str, object]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Window CSV has no header: {csv_path}")
        window_columns = metadata.infer_window_columns(reader.fieldnames)
        expected_columns = set((*target_columns, *window_columns))
        missing_columns = sorted(expected_columns.difference(reader.fieldnames))
        if missing_columns:
            raise ValueError(f"Window CSV missing columns: {missing_columns[:10]}")
        metadata_columns = tuple(
            column
            for column in reader.fieldnames
            if column not in expected_columns
        )
        for row in reader:
            rows.append([float(row[column]) for column in window_columns])
            targets.append([float(row[column]) for column in target_columns])
            sample_metadata.append({column: _parse_meta_value(row[column]) for column in metadata_columns})

    if not rows:
        raise ValueError(f"Window CSV contains no samples: {csv_path}")
    flat_windows = np.asarray(rows, dtype=np.float32)
    windows = flat_windows.reshape(len(rows), metadata.window_size_steps, len(metadata.feature_names))
    target_array = np.asarray(targets, dtype=np.float32)
    if target_array.shape[1] == 1:
        target_array = target_array[:, 0]
    return WindowDataset(
        metadata=metadata,
        windows=windows,
        targets=target_array,
        sample_metadata=tuple(sample_metadata),
        source_path=csv_path,
    )


def build_fixed_group_split(
    dataset: WindowDataset,
    *,
    validation_column: str,
    validation_groups: Iterable[object],
    train_column: str,
    train_value: object,
    test_group_column: str,
    test_groups: Iterable[object],
    scenario_column: str = "scenario_name",
) -> FixedScenarioSplit:
    validation_set = set(validation_groups)
    if not validation_set:
        raise ValueError("validation_groups must not be empty")
    train_pool = dataset.indices_matching({train_column: train_value})
    val_indices = np.asarray(
        [
            int(index)
            for index in train_pool
            if dataset.sample_metadata[int(index)].get(validation_column) in validation_set
        ],
        dtype=np.int64,
    )
    if val_indices.size == 0:
        raise ValueError(f"No validation samples found for groups: {sorted(validation_set)}")
    train_indices = np.asarray(
        [
            int(index)
            for index in train_pool
            if dataset.sample_metadata[int(index)].get(validation_column) not in validation_set
        ],
        dtype=np.int64,
    )
    if train_indices.size == 0:
        raise ValueError("No training samples remain after validation split")

    test_group_indices: dict[str, np.ndarray] = {}
    for group in test_groups:
        indices = dataset.indices_matching({test_group_column: group})
        if indices.size == 0:
            raise ValueError(f"No test samples found for group: {group}")
        test_group_indices[str(group)] = indices
    test_indices = np.concatenate(tuple(test_group_indices.values()))
    _validate_no_overlap(train_indices, val_indices, test_indices)
    return FixedScenarioSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        test_group_indices=test_group_indices,
        scenario_column=scenario_column,
    )


def build_fixed_scenario_split(
    dataset: WindowDataset,
    *,
    validation_scenarios: Iterable[str],
    test_domain_buckets: Iterable[str] = ("same_family", "near", "far"),
    scenario_column: str = "scenario_name",
) -> FixedScenarioSplit:
    return build_fixed_group_split(
        dataset,
        validation_column=scenario_column,
        validation_groups=validation_scenarios,
        train_column="domain_bucket",
        train_value="train",
        test_group_column="domain_bucket",
        test_groups=test_domain_buckets,
        scenario_column=scenario_column,
    )


def fit_window_normalization(windows: np.ndarray, targets: np.ndarray) -> WindowNormalization:
    x = np.asarray(windows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("windows must have shape (samples, time, features)")
    if y.ndim not in {1, 2}:
        raise ValueError("targets must have shape (samples,) or (samples, targets)")
    input_mean = x.reshape(-1, x.shape[-1]).mean(axis=0).astype(np.float32)
    input_std = x.reshape(-1, x.shape[-1]).std(axis=0).astype(np.float32)
    target_values = y.reshape(-1, 1) if y.ndim == 1 else y
    target_mean = target_values.mean(axis=0).astype(np.float32)
    target_std = target_values.std(axis=0).astype(np.float32)
    return WindowNormalization(
        input_mean=input_mean,
        input_std=np.maximum(input_std, 1e-12).astype(np.float32),
        target_mean=target_mean,
        target_std=np.maximum(target_std, 1e-12).astype(np.float32),
    )


def _parse_meta_value(value: str) -> object:
    if value == "":
        return ""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _target_names_from_metadata(raw: dict[str, object]) -> tuple[str, ...]:
    if "target_columns" in raw:
        target_columns = raw["target_columns"]
        if not isinstance(target_columns, list) or not target_columns:
            raise ValueError("metadata target_columns must be a non-empty list")
        return tuple(str(column) for column in target_columns)
    if "target_name" in raw:
        return (str(raw["target_name"]),)
    raise ValueError("metadata must define target_name or target_columns")


def _window_size_from_metadata(raw: dict[str, object], config: dict[str, object]) -> int:
    for value in (
        raw.get("input_window_steps"),
        config.get("window_size_steps"),
        config.get("window_size_minutes"),
    ):
        if value is not None:
            window_size = int(value)
            if window_size <= 0:
                raise ValueError("window size must be positive")
            return window_size
    raise ValueError("metadata must define input_window_steps, config.window_size_steps, or config.window_size_minutes")


def _scenario_list(dataset: WindowDataset, indices: np.ndarray, scenario_column: str) -> list[str]:
    return sorted({str(dataset.sample_metadata[int(index)][scenario_column]) for index in indices})


def _validate_no_overlap(*groups: np.ndarray) -> None:
    seen: set[int] = set()
    for group in groups:
        current = {int(index) for index in group}
        if seen.intersection(current):
            raise ValueError("Split indices overlap")
        seen.update(current)


def _validate_loaded_dataset(
    *,
    windows: np.ndarray,
    targets: np.ndarray,
    sample_metadata: tuple[dict[str, object], ...],
    metadata: WindowDatasetMetadata,
    source_path: Path,
) -> None:
    if windows.ndim != 3:
        raise ValueError(f"windows in {source_path} must have shape (samples, time, features)")
    if windows.shape[1] != metadata.window_size_steps:
        raise ValueError(
            f"windows time dimension {windows.shape[1]} does not match metadata "
            f"window_size_steps={metadata.window_size_steps}"
        )
    if windows.shape[2] != len(metadata.feature_names):
        raise ValueError(
            f"windows feature dimension {windows.shape[2]} does not match "
            f"{len(metadata.feature_names)} feature names"
        )
    if targets.ndim not in {1, 2}:
        raise ValueError(f"targets in {source_path} must have shape (samples,) or (samples, targets)")
    if targets.shape[0] != windows.shape[0]:
        raise ValueError("targets sample count must match windows")
    if len(sample_metadata) != windows.shape[0]:
        raise ValueError("sample_metadata length must match windows")


WindowCsvMetadata = WindowDatasetMetadata
WindowCsvDataset = WindowDataset
