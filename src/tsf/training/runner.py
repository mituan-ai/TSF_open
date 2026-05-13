from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

from tsf.data.window_csv import (
    FixedScenarioSplit,
    WindowDataset,
    WindowNormalization,
    build_fixed_group_split,
    build_fixed_scenario_split,
    fit_window_normalization,
    load_window_dataset,
)
from tsf.experiment_io import build_unique_output_dir, write_json
from tsf.models.forecaster import build_forecast_model
from tsf.training.config import ExperimentConfig
from tsf.training.metrics import regression_metrics

try:  # pragma: no cover - covered by training smoke tests when torch exists.
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PreparedData:
    dataset: WindowDataset
    split: FixedScenarioSplit
    normalization: WindowNormalization
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def run_experiment(
    config: ExperimentConfig,
    *,
    overrides: dict[str, object] | None = None,
) -> Path:
    if torch is None or DataLoader is None or TensorDataset is None:
        raise ImportError("PyTorch is required to run training experiments")
    resolved_config = _apply_overrides(config, overrides or {})
    set_global_seed(resolved_config.seed)
    device = torch.device(str(resolved_config.training.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    run_dir = _build_run_dir(resolved_config)
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_run_header(run_dir, resolved_config, device)

    prepared = prepare_data(resolved_config, device=device)
    output_dim = int(np.asarray(prepared.dataset.targets).shape[1]) if prepared.dataset.targets.ndim == 2 else 1
    configured_output_dim = int(resolved_config.model.get("output_dim", output_dim))
    if configured_output_dim != output_dim:
        raise ValueError(
            f"model.output_dim={configured_output_dim} does not match "
            f"dataset target dimension {output_dim}"
        )
    write_json(run_dir / "normalization.json", prepared.normalization.as_dict(
        prepared.dataset.feature_names,
        prepared.dataset.target_names,
    ))
    write_json(run_dir / "split.json", prepared.split.as_dict(prepared.dataset))
    write_json(run_dir / "dataset_summary.json", _dataset_summary(prepared.dataset))

    if resolved_config.method_name.lower() == "tsf":
        artifact_dir = Path(str(resolved_config.method["semantic_artifact"]))
        shutil.copy2(artifact_dir / "semantic_field_metadata.json", run_dir / "semantic_field_metadata_snapshot.json")

    backbone_config = _backbone_config(resolved_config)
    model = build_forecast_model(
        feature_names=prepared.dataset.feature_names,
        method_config=resolved_config.method,
        backbone_config=backbone_config,
        output_dim=configured_output_dim,
    ).to(device)
    total_parameters, trainable_parameters = count_parameters(model)
    history, best_epoch = train_model(
        model=model,
        prepared=prepared,
        config=resolved_config,
        run_dir=run_dir,
        device=device,
    )
    best_path = run_dir / "best_checkpoint.pt"
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    predictions = predict(model, prepared.test_loader, prepared.normalization, device=device)
    write_predictions_csv(run_dir / "predictions.csv", predictions)

    metrics_payload = build_metrics(
        predictions=predictions,
        dataset=prepared.dataset,
        split=prepared.split,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        inference_time_ms_per_step=measure_inference_time(
            model=model,
            loader=prepared.test_loader,
            device=device,
            warmup_loops=int(
                resolved_config.evaluation.get(
                    "timing_warmup_loops",
                    resolved_config.evaluation.get("timing_warmup_batches", 2),
                )
            ),
            repeat_loops=int(
                resolved_config.evaluation.get(
                    "timing_repeat_loops",
                    resolved_config.evaluation.get("timing_repeat_batches", 20),
                )
            ),
        ),
        peak_rss_mb=peak_rss_mb(),
        best_epoch=best_epoch,
    )
    write_json(run_dir / "metrics.json", metrics_payload["metrics"])
    write_group_metrics_csv(run_dir / "metrics_by_group.csv", metrics_payload["by_group"])
    write_group_metrics_csv(run_dir / "metrics_by_scenario.csv", metrics_payload["by_scenario"])
    write_history_csv(run_dir / "history.csv", history)
    write_json(run_dir / "environment.json", environment_payload(resolved_config, run_dir, device))
    return run_dir


def prepare_data(config: ExperimentConfig, *, device: torch.device) -> PreparedData:
    dataset = load_window_dataset(
        Path(str(config.dataset["path"])),
        data_name=str(config.dataset.get("data_name", config.dataset.get("csv_name", "windows.npz"))),
    )
    split = _build_split_from_config(dataset, config.dataset)
    normalization = fit_window_normalization(
        dataset.windows[split.train_indices],
        dataset.targets[split.train_indices],
    )
    batch_size = int(config.training.get("batch_size", 64))
    num_workers = int(config.training.get("num_workers", 0))
    train_loader = _build_loader(
        dataset,
        split.train_indices,
        normalization,
        batch_size=batch_size,
        shuffle=True,
        seed=config.seed,
        device=device,
        num_workers=num_workers,
    )
    val_loader = _build_loader(
        dataset,
        split.val_indices,
        normalization,
        batch_size=batch_size,
        shuffle=False,
        seed=config.seed,
        device=device,
        num_workers=num_workers,
    )
    test_loader = _build_loader(
        dataset,
        split.test_indices,
        normalization,
        batch_size=batch_size,
        shuffle=False,
        seed=config.seed,
        device=device,
        num_workers=num_workers,
    )
    return PreparedData(
        dataset=dataset,
        split=split,
        normalization=normalization,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def train_model(
    *,
    model: torch.nn.Module,
    prepared: PreparedData,
    config: ExperimentConfig,
    run_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, object]], int]:
    optimizer_name = str(config.training.get("optimizer", "AdamW")).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.get("learning_rate", 1e-3)),
        weight_decay=float(config.training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.training.get("scheduler_factor", 0.5)),
        patience=int(config.training.get("scheduler_patience", 15)),
        min_lr=float(config.training.get("min_lr", 1e-5)),
    )
    criterion = torch.nn.MSELoss()
    max_epochs = int(config.training.get("max_epochs", 300))
    early_patience = int(config.training.get("early_stopping_patience", 40))
    min_delta = float(config.training.get("early_stopping_min_delta", 1e-4))
    grad_clip = float(config.training.get("gradient_clip_norm", 1.0))

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    log_path = run_dir / "run.log"

    for epoch in range(1, max_epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            loader=prepared.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            grad_clip=grad_clip,
            device=device,
        )
        val_predictions = predict(model, prepared.val_loader, prepared.normalization, device=device)
        val_metrics = regression_metrics(val_predictions["target"], val_predictions["prediction"])
        val_rmse = float(val_metrics["RMSE"])
        scheduler.step(val_rmse)
        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_MAE": val_metrics["MAE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_R^2": val_metrics["R^2"],
            "learning_rate": lr,
        }
        history.append(row)
        _append_log(log_path, json.dumps(row, ensure_ascii=False))

        if val_rmse < best_val - min_delta:
            best_val = val_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_rmse": val_rmse,
                },
                run_dir / "best_checkpoint.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_patience:
                _append_log(log_path, f"early_stopping epoch={epoch} best_epoch={best_epoch}")
                break

    if not (run_dir / "best_checkpoint.pt").exists():
        raise RuntimeError("Training did not produce a best checkpoint")
    return history, best_epoch


def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    normalization: WindowNormalization,
    *,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x, y, index = batch
            output = _as_2d(model(x.to(device)).detach().cpu().numpy())
            predictions.append(output)
            targets.append(_as_2d(y.detach().cpu().numpy()))
            sample_indices.append(index.detach().cpu().numpy().reshape(-1))
    pred_norm = np.concatenate(predictions, axis=0)
    target_norm = np.concatenate(targets, axis=0)
    return {
        "sample_index": np.concatenate(sample_indices).astype(np.int64),
        "prediction": normalization.inverse_transform_targets(pred_norm),
        "target": normalization.inverse_transform_targets(target_norm),
        "prediction_normalized": pred_norm,
        "target_normalized": target_norm,
    }


def measure_inference_time(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_loops: int,
    repeat_loops: int,
) -> float:
    model.eval()
    batches = [batch[0].to(device) for batch in loader]
    if not batches:
        return float("nan")
    warmup_count = max(int(warmup_loops), 0)
    repeat_count = max(int(repeat_loops), 1)
    with torch.no_grad():
        for _ in range(warmup_count):
            for x in batches:
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        measured_samples = 0
        for _ in range(repeat_count):
            for x in batches:
                _ = model(x)
                measured_samples += int(x.shape[0])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return float((elapsed / measured_samples) * 1000.0)


def build_metrics(
    *,
    predictions: dict[str, np.ndarray],
    dataset: WindowDataset,
    split: FixedScenarioSplit,
    total_parameters: int,
    trainable_parameters: int,
    inference_time_ms_per_step: float,
    peak_rss_mb: float,
    best_epoch: int,
) -> dict[str, object]:
    by_group: list[dict[str, object]] = []
    by_scenario: list[dict[str, object]] = []
    index_to_prediction_row = {
        int(sample_index): row
        for row, sample_index in enumerate(predictions["sample_index"])
    }

    overall_metrics = regression_metrics(predictions["target"], predictions["prediction"])
    overall = _with_efficiency(
        overall_metrics,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        inference_time_ms_per_step=inference_time_ms_per_step,
        peak_rss_mb=peak_rss_mb,
    )
    overall["best_epoch"] = int(best_epoch)

    for group_name, indices in sorted(split.test_group_indices.items()):
        row_indices = np.asarray([index_to_prediction_row[int(index)] for index in indices], dtype=np.int64)
        metrics = regression_metrics(predictions["target"][row_indices], predictions["prediction"][row_indices])
        by_group.append({"group": group_name, "n_samples": int(row_indices.size), **metrics})

    for scenario in _unique_scenarios(dataset, split.test_indices, split.scenario_column):
        sample_indices = [
            int(index)
            for index in split.test_indices
            if dataset.sample_metadata[int(index)][split.scenario_column] == scenario
        ]
        row_indices = np.asarray([index_to_prediction_row[index] for index in sample_indices], dtype=np.int64)
        metrics = regression_metrics(predictions["target"][row_indices], predictions["prediction"][row_indices])
        by_scenario.append({"scenario_name": scenario, "n_samples": int(row_indices.size), **metrics})

    return {
        "metrics": overall,
        "by_group": by_group,
        "by_scenario": by_scenario,
    }


def write_predictions_csv(output_path: Path, predictions: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = _as_2d(predictions["target"])
    prediction = _as_2d(predictions["prediction"])
    target_normalized = _as_2d(predictions["target_normalized"])
    prediction_normalized = _as_2d(predictions["prediction_normalized"])
    output_dim = target.shape[1]
    fieldnames = ["sample_index"]
    if output_dim == 1:
        fieldnames.extend([
            "target",
            "prediction",
            "error",
            "target_normalized",
            "prediction_normalized",
        ])
    else:
        for target_index in range(output_dim):
            suffix = f"target_{target_index + 1:02d}"
            fieldnames.extend([
                suffix,
                f"prediction_{target_index + 1:02d}",
                f"error_{target_index + 1:02d}",
                f"{suffix}_normalized",
                f"prediction_{target_index + 1:02d}_normalized",
            ])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(predictions["sample_index"])):
            row: dict[str, object] = {"sample_index": int(predictions["sample_index"][index])}
            if output_dim == 1:
                row.update(
                    {
                        "target": float(target[index, 0]),
                        "prediction": float(prediction[index, 0]),
                        "error": float(prediction[index, 0] - target[index, 0]),
                        "target_normalized": float(target_normalized[index, 0]),
                        "prediction_normalized": float(prediction_normalized[index, 0]),
                    }
                )
            else:
                for target_index in range(output_dim):
                    suffix = f"target_{target_index + 1:02d}"
                    row.update(
                        {
                            suffix: float(target[index, target_index]),
                            f"prediction_{target_index + 1:02d}": float(prediction[index, target_index]),
                            f"error_{target_index + 1:02d}": float(
                                prediction[index, target_index] - target[index, target_index]
                            ),
                            f"{suffix}_normalized": float(target_normalized[index, target_index]),
                            f"prediction_{target_index + 1:02d}_normalized": float(
                                prediction_normalized[index, target_index]
                            ),
                        }
                    )
            writer.writerow(row)


def write_group_metrics_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [name for name in ("group", "scenario_name", "n_samples", "MAE", "RMSE", "R^2") if name in fieldnames]
    ordered = preferred + [name for name in fieldnames if name not in preferred]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def write_history_csv(output_path: Path, history: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return float(usage / (1024 * 1024))
    return float(usage / 1024)


def environment_payload(config: ExperimentConfig, run_dir: Path, device: torch.device) -> dict[str, object]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "command": sys.argv,
        "seed": config.seed,
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__ if torch is not None else None,
        "git": _git_payload(config.root),
    }


def _build_loader(
    dataset: WindowDataset,
    indices: np.ndarray,
    normalization: WindowNormalization,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    x = normalization.transform_windows(dataset.windows[indices])
    y = _as_2d(normalization.transform_targets(dataset.targets[indices]))
    sample_indices = indices.astype(np.int64)
    tensor_dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
        torch.as_tensor(sample_indices, dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = device.type == "cuda"
    return DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for x, y, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x.to(device))
        loss = criterion(prediction, y.to(device))
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def _build_run_dir(config: ExperimentConfig) -> Path:
    method = config.method_name.lower()
    dataset = config.dataset_name
    backbone = config.backbone_name.lower()
    prefix = str(config.output.get("run_name_prefix", ""))
    if method == "baseline":
        stem = f"{prefix}{dataset}_{backbone}"
    else:
        stem = f"{prefix}{dataset}_{backbone}_{method}"
    return build_unique_output_dir(Path(str(config.output["root"])), stem)


def _write_run_header(run_dir: Path, config: ExperimentConfig, device: torch.device) -> None:
    write_json(run_dir / "config_snapshot.json", config.as_dict())
    _append_log(run_dir / "run.log", f"created_at_utc={datetime.now(timezone.utc).isoformat()}")
    _append_log(run_dir / "run.log", f"config={config.config_path}")
    _append_log(run_dir / "run.log", f"device={device}")


def _dataset_summary(dataset: WindowDataset) -> dict[str, object]:
    domain_buckets = dataset.domain_buckets() if "domain_bucket" in dataset.sample_metadata[0] else ()
    scenario_names = dataset.scenario_names() if "scenario_name" in dataset.sample_metadata[0] else ()
    return {
        "dataset_name": dataset.metadata.dataset_name,
        "source_path": str(dataset.source_path),
        "n_samples": int(dataset.windows.shape[0]),
        "window_size": int(dataset.windows.shape[1]),
        "n_features": int(dataset.windows.shape[2]),
        "feature_names": list(dataset.feature_names),
        "target_names": list(dataset.target_names),
        "domain_buckets": domain_buckets,
        "scenario_names": scenario_names,
    }


def _backbone_config(config: ExperimentConfig) -> dict[str, Any]:
    backbone = config.model.get("backbone", {})
    if not isinstance(backbone, dict):
        raise ValueError("model.backbone must be a mapping")
    return backbone


def _build_split_from_config(dataset: WindowDataset, dataset_config: dict[str, Any]) -> FixedScenarioSplit:
    if "split" in dataset_config:
        split_config = dataset_config["split"]
        if not isinstance(split_config, dict):
            raise ValueError("dataset.split must be a mapping")
        return build_fixed_group_split(
            dataset,
            validation_column=str(split_config["validation_column"]),
            validation_groups=tuple(split_config["validation_groups"]),
            train_column=str(split_config["train_column"]),
            train_value=split_config["train_value"],
            test_group_column=str(split_config["test_group_column"]),
            test_groups=tuple(split_config["test_groups"]),
            scenario_column=str(split_config.get("scenario_column", split_config["validation_column"])),
        )
    val_scenarios = tuple(dataset_config.get("validation_scenarios", ()))
    return build_fixed_scenario_split(
        dataset,
        validation_scenarios=val_scenarios,
        test_domain_buckets=tuple(dataset_config.get("test_domain_buckets", ("same_family", "near", "far"))),
        scenario_column=str(dataset_config.get("scenario_column", "scenario_name")),
    )


def _with_efficiency(
    metrics: dict[str, float],
    *,
    total_parameters: int,
    trainable_parameters: int,
    inference_time_ms_per_step: float,
    peak_rss_mb: float,
) -> dict[str, object]:
    return {
        **metrics,
        "Inference Time (ms/step)": float(inference_time_ms_per_step),
        "Parameters": int(total_parameters),
        "Trainable Parameters": int(trainable_parameters),
        "Peak RSS (MB)": float(peak_rss_mb),
    }


def _unique_scenarios(dataset: WindowDataset, indices: np.ndarray, scenario_column: str) -> tuple[str, ...]:
    return tuple(sorted({str(dataset.sample_metadata[int(index)][scenario_column]) for index in indices}))


def _as_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected 1D or 2D target array, got shape {array.shape}")


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _apply_overrides(config: ExperimentConfig, overrides: dict[str, object]) -> ExperimentConfig:
    if not overrides:
        return config
    training = dict(config.training)
    output = dict(config.output)
    evaluation = dict(config.evaluation)
    for key, value in overrides.items():
        if key.startswith("training."):
            training[key.split(".", 1)[1]] = value
        elif key.startswith("evaluation."):
            evaluation[key.split(".", 1)[1]] = value
        elif key.startswith("output."):
            output[key.split(".", 1)[1]] = value
        else:
            raise ValueError(f"Unsupported override: {key}")
    return ExperimentConfig(
        config_path=config.config_path,
        root=config.root,
        dataset=config.dataset,
        model=config.model,
        method=config.method,
        training=training,
        evaluation=evaluation,
        output=output,
        raw={**config.raw, "training": training, "evaluation": evaluation, "output": output},
    )


def _git_payload(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit": commit, "status_short": status}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "status_short": []}
