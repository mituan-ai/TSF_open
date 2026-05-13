from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from tsf.semantic_field import SemanticFieldSpec

try:  # pragma: no cover - exercised by optional torch tests when installed.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SemanticFieldForwardOutput:
    transformed: Tensor
    semantic_field: Tensor
    numeric_residual: Tensor
    semantic_projection: Tensor


def require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for tsf.methods.semantic_field_layer")


if nn is not None:

    class SemanticFieldLayer(nn.Module):
        """Task-semantic field input factorization layer.

        The layer keeps semantic directions fixed as a buffer and learns only
        the semantic projection B, diagonal numeric residual D, and bias.
        """

        def __init__(
            self,
            *,
            directions: np.ndarray,
            output_dim: int | None = None,
            numeric_path: str = "diagonal",
            dtype: torch.dtype = torch.float32,
        ) -> None:
            super().__init__()
            if numeric_path != "diagonal":
                raise ValueError("Only diagonal numeric_path is supported in the first TSF implementation")
            direction_values = np.asarray(directions, dtype=np.float64)
            if direction_values.ndim != 2:
                raise ValueError("directions must have shape (n_variables, semantic_dim)")
            if direction_values.shape[0] <= 0 or direction_values.shape[1] <= 0:
                raise ValueError("directions must have non-empty dimensions")

            n_variables = int(direction_values.shape[0])
            semantic_dim = int(direction_values.shape[1])
            resolved_output_dim = n_variables if output_dim is None else int(output_dim)
            if resolved_output_dim <= 0:
                raise ValueError("output_dim must be positive")

            self.n_variables = n_variables
            self.semantic_dim = semantic_dim
            self.output_dim = resolved_output_dim
            self.numeric_path = numeric_path
            self.residual_dim = min(n_variables, resolved_output_dim)
            self.register_buffer("directions", torch.as_tensor(direction_values, dtype=dtype))
            self.semantic_projection = nn.Linear(semantic_dim, resolved_output_dim, bias=False, dtype=dtype)
            self.numeric_scale = nn.Parameter(torch.ones(self.residual_dim, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(resolved_output_dim, dtype=dtype))

        @property
        def semantic_matrix(self) -> Tensor:
            return self.directions.transpose(0, 1)

        @classmethod
        def from_spec(
            cls,
            spec: SemanticFieldSpec,
            *,
            feature_names: Iterable[str] | None = None,
            output_dim: int | None = None,
            numeric_path: str = "diagonal",
            dtype: torch.dtype = torch.float32,
        ) -> "SemanticFieldLayer":
            if feature_names is not None:
                spec.validate_feature_order(feature_names)
            return cls(
                directions=spec.directions,
                output_dim=output_dim,
                numeric_path=numeric_path,
                dtype=dtype,
            )

        @classmethod
        def from_artifact(
            cls,
            artifact_dir: Path,
            *,
            feature_names: Iterable[str],
            output_dim: int | None = None,
            numeric_path: str = "diagonal",
            dtype: torch.dtype = torch.float32,
        ) -> "SemanticFieldLayer":
            spec = SemanticFieldSpec.load_artifact(artifact_dir, feature_names=feature_names)
            return cls.from_spec(
                spec,
                feature_names=feature_names,
                output_dim=output_dim,
                numeric_path=numeric_path,
                dtype=dtype,
            )

        def compute_semantic_field(self, normalized_inputs: Tensor) -> Tensor:
            if normalized_inputs.shape[-1] != self.n_variables:
                raise ValueError(
                    f"input feature dimension {normalized_inputs.shape[-1]} does not match "
                    f"{self.n_variables} semantic directions"
                )
            return torch.matmul(normalized_inputs, self.directions)

        def forward(
            self,
            normalized_inputs: Tensor,
            *,
            return_components: bool = False,
        ) -> Tensor | SemanticFieldForwardOutput:
            semantic_field = self.compute_semantic_field(normalized_inputs)
            semantic_part = self.semantic_projection(semantic_field)
            scaled_inputs = normalized_inputs[..., : self.residual_dim] * self.numeric_scale
            if self.output_dim == self.residual_dim:
                numeric_part = scaled_inputs
            else:
                pad_shape = (*scaled_inputs.shape[:-1], self.output_dim - self.residual_dim)
                numeric_padding = scaled_inputs.new_zeros(pad_shape)
                numeric_part = torch.cat([scaled_inputs, numeric_padding], dim=-1)
            transformed = numeric_part + semantic_part + self.bias
            if return_components:
                return SemanticFieldForwardOutput(
                    transformed=transformed,
                    semantic_field=semantic_field,
                    numeric_residual=numeric_part,
                    semantic_projection=semantic_part,
                )
            return transformed


    class SemanticFieldBackboneInput(nn.Module):
        """Small adapter that can feed TSF-transformed inputs to any backbone."""

        def __init__(self, field_layer: SemanticFieldLayer, backbone: nn.Module | None = None) -> None:
            super().__init__()
            self.field_layer = field_layer
            self.backbone = backbone

        def forward(self, normalized_inputs: Tensor) -> Tensor | tuple[Tensor, Tensor]:
            output = self.field_layer(normalized_inputs, return_components=True)
            if not isinstance(output, SemanticFieldForwardOutput):
                raise TypeError("field_layer must return SemanticFieldForwardOutput")
            if self.backbone is None:
                return output.transformed, output.semantic_field
            return self.backbone(output.transformed)

else:

    class SemanticFieldLayer:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()


    class SemanticFieldBackboneInput:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()


__all__ = [
    "SemanticFieldBackboneInput",
    "SemanticFieldForwardOutput",
    "SemanticFieldLayer",
    "require_torch",
]
