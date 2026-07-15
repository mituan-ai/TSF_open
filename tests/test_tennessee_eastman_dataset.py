from __future__ import annotations

from collections import Counter
from pathlib import Path

from tsf.data import load_window_dataset
from tsf.semantic_field import SemanticFieldSpec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPOSITORY_ROOT / "resources" / "datasets" / "tennessee_eastman"
ARTIFACT_DIR = REPOSITORY_ROOT / "resources" / "semantic_artifacts" / "tennessee_eastman"


def test_tennessee_eastman_public_bundle_protocol() -> None:
    dataset = load_window_dataset(DATASET_DIR)

    assert dataset.windows.shape == (47_775, 20, 33)
    assert dataset.targets.shape == (47_775,)
    assert dataset.feature_names == tuple(
        [f"xmeas_{index}" for index in range(1, 23)]
        + [f"xmv_{index}" for index in range(1, 12)]
    )
    assert Counter(
        str(row["domain_bucket"]) for row in dataset.sample_metadata
    ) == {
        "train": 19_110,
        "same_family": 1_365,
        "near": 6_825,
        "far": 20_475,
    }
    assert sum(
        str(row["domain_bucket"]) == "train"
        and int(row["simulation_run"]) in {31, 32, 33, 34, 35}
        for row in dataset.sample_metadata
    ) == 2_730


def test_tennessee_eastman_semantic_directions_match_features() -> None:
    dataset = load_window_dataset(DATASET_DIR)
    artifact = SemanticFieldSpec.load_artifact(ARTIFACT_DIR)

    assert artifact.variable_names == dataset.feature_names
    assert artifact.directions.shape == (33, 128)
