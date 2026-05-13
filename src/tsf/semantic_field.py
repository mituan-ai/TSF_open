from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def normalize_rows(matrix: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def construct_semantic_field(
    normalized_inputs: np.ndarray,
    semantic_directions: np.ndarray,
) -> np.ndarray:
    x = np.asarray(normalized_inputs, dtype=np.float64)
    directions = np.asarray(semantic_directions, dtype=np.float64)
    if x.shape[-1] != directions.shape[0]:
        raise ValueError(
            f"input feature dimension {x.shape[-1]} does not match "
            f"{directions.shape[0]} semantic directions"
        )
    return np.einsum("...d,dk->...k", x, directions, dtype=np.float64)


def validate_feature_order(
    *,
    feature_names: Iterable[str],
    semantic_variable_names: Iterable[str],
) -> tuple[str, ...]:
    features = tuple(feature_names)
    variables = tuple(semantic_variable_names)
    if features != variables:
        raise ValueError(
            "feature_names must match semantic variable order exactly: "
            f"features={list(features)}, semantic_variables={list(variables)}"
        )
    return features


@dataclass(frozen=True)
class SemanticFieldSpec:
    variable_names: tuple[str, ...]
    descriptions: tuple[str, ...]
    embeddings: np.ndarray
    directions: np.ndarray
    embedding_model: str

    @property
    def semantic_matrix(self) -> np.ndarray:
        return self.directions.T.copy()

    def transform(self, normalized_inputs: np.ndarray) -> np.ndarray:
        return construct_semantic_field(normalized_inputs, self.directions)

    def as_dict(self) -> dict[str, object]:
        return {
            "variable_names": list(self.variable_names),
            "descriptions": list(self.descriptions),
            "embedding_model": self.embedding_model,
            "embedding_dim": int(self.embeddings.shape[1]),
            "semantic_dim": int(self.directions.shape[1]),
            "embeddings": self.embeddings,
            "directions": self.directions,
        }

    def validate_feature_order(self, feature_names: Iterable[str]) -> tuple[str, ...]:
        return validate_feature_order(
            feature_names=feature_names,
            semantic_variable_names=self.variable_names,
        )

    def save_artifact(self, artifact_dir: Path) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        np.save(artifact_dir / "embeddings.npy", self.embeddings)
        np.save(artifact_dir / "directions.npy", self.directions)
        metadata = {
            "artifact_type": "semantic_field",
            "variable_names": list(self.variable_names),
            "descriptions": list(self.descriptions),
            "embedding_model": self.embedding_model,
            "embedding_dim": int(self.embeddings.shape[1]),
            "semantic_dim": int(self.directions.shape[1]),
            "direction_method": "direct_embedding",
            "arrays": {
                "embeddings": "embeddings.npy",
                "directions": "directions.npy",
            },
        }
        (artifact_dir / "semantic_field_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_artifact(
        cls,
        artifact_dir: Path,
        *,
        feature_names: Iterable[str] | None = None,
    ) -> "SemanticFieldSpec":
        metadata_path = artifact_dir / "semantic_field_metadata.json"
        if not metadata_path.exists():
            metadata_path = artifact_dir / "artifact_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Semantic field metadata not found in {artifact_dir}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays = metadata.get("arrays", {})
        if isinstance(arrays, dict):
            embeddings_name = arrays.get("embeddings")
            directions_name = str(arrays.get("directions", "directions.npy"))
        else:
            embeddings_name = "embeddings.npy"
            directions_name = "directions.npy"

        variable_names = tuple(str(name) for name in metadata["variable_names"])
        descriptions = tuple(
            str(description)
            for description in metadata.get("descriptions", ("",) * len(variable_names))
        )
        if len(descriptions) != len(variable_names):
            raise ValueError("semantic artifact descriptions must match variable_names")

        directions = np.load(artifact_dir / directions_name).astype(np.float64)
        embeddings_path = artifact_dir / str(embeddings_name) if embeddings_name else None
        embeddings = (
            np.load(embeddings_path).astype(np.float64)
            if embeddings_path is not None and embeddings_path.exists()
            else directions.copy()
        )
        spec = cls(
            variable_names=variable_names,
            descriptions=descriptions,
            embeddings=embeddings,
            directions=directions,
            embedding_model=str(metadata.get("embedding_model", "unknown_offline_embedding")),
        )
        spec.validate_shapes()
        if feature_names is not None:
            spec.validate_feature_order(feature_names)
        return spec

    def validate_shapes(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (n_variables, embedding_dim)")
        if self.directions.ndim != 2:
            raise ValueError("directions must have shape (n_variables, semantic_dim)")
        if self.embeddings.shape[0] != len(self.variable_names):
            raise ValueError("embeddings row count must match variable_names")
        if self.directions.shape[0] != len(self.variable_names):
            raise ValueError("directions row count must match variable_names")

    @classmethod
    def from_embeddings(
        cls,
        *,
        variable_names: Iterable[str],
        descriptions: Iterable[str],
        embeddings: np.ndarray,
        embedding_model: str,
        normalize: bool = False,
    ) -> "SemanticFieldSpec":
        names = tuple(variable_names)
        descs = tuple(" ".join(description.split()) for description in descriptions)
        if len(names) != len(descs):
            raise ValueError("variable_names and descriptions must have the same length")
        if not names:
            raise ValueError("at least one variable is required")
        embedding_values = np.asarray(embeddings, dtype=np.float64)
        directions = normalize_rows(embedding_values) if normalize else embedding_values.copy()
        spec = cls(
            variable_names=names,
            descriptions=descs,
            embeddings=embedding_values,
            directions=directions,
            embedding_model=embedding_model,
        )
        spec.validate_shapes()
        return spec


@dataclass(frozen=True)
class RestrictedSemanticInputLayer:
    numeric_scale: np.ndarray
    semantic_weight: np.ndarray
    bias: np.ndarray

    @classmethod
    def initialize(
        cls,
        *,
        n_variables: int,
        semantic_dim: int,
        seed: int = 42,
        numeric_scale: float = 1.0,
    ) -> "RestrictedSemanticInputLayer":
        if n_variables <= 0:
            raise ValueError("n_variables must be positive")
        if semantic_dim <= 0:
            raise ValueError("semantic_dim must be positive")
        rng = np.random.default_rng(seed)
        semantic_weight = rng.standard_normal((semantic_dim, n_variables), dtype=np.float64)
        semantic_weight = semantic_weight / np.sqrt(semantic_dim)
        return cls(
            numeric_scale=np.full(n_variables, numeric_scale, dtype=np.float64),
            semantic_weight=semantic_weight.astype(np.float64),
            bias=np.zeros(n_variables, dtype=np.float64),
        )

    def transform(self, normalized_inputs: np.ndarray, semantic_directions: np.ndarray) -> np.ndarray:
        x = np.asarray(normalized_inputs, dtype=np.float64)
        directions = np.asarray(semantic_directions, dtype=np.float64)
        if x.shape[-1] != self.numeric_scale.shape[0]:
            raise ValueError("normalized_inputs does not match numeric_scale")
        semantic_field = construct_semantic_field(x, directions)
        if semantic_field.shape[-1] != self.semantic_weight.shape[0]:
            raise ValueError("semantic_field does not match semantic_weight")
        numeric_part = x * self.numeric_scale
        semantic_part = semantic_field @ self.semantic_weight
        return numeric_part + semantic_part + self.bias
