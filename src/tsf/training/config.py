from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tsf.experiment_io import load_simple_yaml_mapping, resolve_reference_path


@dataclass(frozen=True)
class ExperimentConfig:
    config_path: Path
    root: Path
    dataset: dict[str, Any]
    model: dict[str, Any]
    method: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    output: dict[str, Any]
    raw: dict[str, Any]

    @property
    def dataset_name(self) -> str:
        return str(self.dataset["name"])

    @property
    def method_name(self) -> str:
        return str(self.method.get("name", "baseline"))

    @property
    def backbone_name(self) -> str:
        backbone = self.model.get("backbone", {})
        if not isinstance(backbone, dict):
            raise ValueError("model.backbone must be a mapping")
        return str(backbone.get("type", "gru"))

    @property
    def seed(self) -> int:
        return int(self.training.get("seed", 42))

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "method": self.method,
            "training": self.training,
            "evaluation": self.evaluation,
            "output": self.output,
        }


def load_experiment_config(config_path: Path, *, root: Path) -> ExperimentConfig:
    payload = load_simple_yaml_mapping(config_path)
    required_sections = ("dataset", "model", "method", "training", "evaluation", "output")
    for section in required_sections:
        if section not in payload or not isinstance(payload[section], dict):
            raise ValueError(f"Experiment config must contain mapping section: {section}")
    resolved = _resolve_paths(payload, config_path=config_path, root=root)
    return ExperimentConfig(
        config_path=config_path,
        root=root,
        dataset=resolved["dataset"],
        model=resolved["model"],
        method=resolved["method"],
        training=resolved["training"],
        evaluation=resolved["evaluation"],
        output=resolved["output"],
        raw=resolved,
    )


def _resolve_paths(payload: dict[str, object], *, config_path: Path, root: Path) -> dict[str, Any]:
    resolved = dict(payload)
    dataset = dict(resolved["dataset"])  # type: ignore[arg-type]
    if "path" in dataset:
        dataset["path"] = str(resolve_reference_path(str(dataset["path"]), config_path=config_path, root=root))
    else:
        dataset["path"] = str(root / "resources/datasets" / str(dataset["name"]))
    resolved["dataset"] = dataset

    method = dict(resolved["method"])  # type: ignore[arg-type]
    if method.get("semantic_artifact"):
        method["semantic_artifact"] = str(
            resolve_reference_path(str(method["semantic_artifact"]), config_path=config_path, root=root)
        )
    resolved["method"] = method

    output = dict(resolved["output"])  # type: ignore[arg-type]
    output_root = Path(str(output.get("root", "outputs/runs")))
    output["root"] = str(output_root if output_root.is_absolute() else root / output_root)
    resolved["output"] = output
    return resolved  # type: ignore[return-value]
