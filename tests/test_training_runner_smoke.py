from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch", reason="training smoke tests require the train extra")

from tsf.training.config import load_experiment_config
from tsf.training.runner import run_experiment


def test_baseline_training_smoke_outputs_required_files() -> None:
    config = load_experiment_config(
        Path("configs/experiments/indpensim_gru.yaml"),
        root=Path.cwd(),
    )

    run_dir = run_experiment(
        config,
        overrides={
            "training.max_epochs": 1,
            "training.device": "cpu",
            "output.run_name_prefix": "test_",
        },
    )

    assert run_dir.name.startswith("test_indpensim_gru_")
    assert (run_dir / "config_snapshot.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "metrics_by_group.csv").exists()
    assert (run_dir / "metrics_by_scenario.csv").exists()
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "best_checkpoint.pt").exists()
    assert (run_dir / "normalization.json").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "environment.json").exists()


def test_tsf_training_smoke_outputs_semantic_snapshot() -> None:
    config = load_experiment_config(
        Path("configs/experiments/indpensim_gru_tsf.yaml"),
        root=Path.cwd(),
    )

    run_dir = run_experiment(
        config,
        overrides={
            "training.max_epochs": 1,
            "training.device": "cpu",
            "output.run_name_prefix": "test_",
        },
    )

    assert run_dir.name.startswith("test_indpensim_gru_tsf_")
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "semantic_field_metadata_snapshot.json").exists()


def test_ladle_multi_target_training_smoke_outputs_predictions() -> None:
    config = load_experiment_config(
        Path("configs/experiments/ladle_preheating_gru.yaml"),
        root=Path.cwd(),
    )

    run_dir = run_experiment(
        config,
        overrides={
            "training.max_epochs": 1,
            "training.device": "cpu",
            "output.run_name_prefix": "test_",
        },
    )

    assert run_dir.name.startswith("test_ladle_preheating_gru_")
    predictions_header = (run_dir / "predictions.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "target_01" in predictions_header
    assert "prediction_05" in predictions_header
    assert (run_dir / "metrics_by_scenario.csv").exists()
