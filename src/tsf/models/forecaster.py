from __future__ import annotations

from pathlib import Path
from typing import Any

from tsf.methods.semantic_field_layer import SemanticFieldLayer
from tsf.models.backbones import build_backbone, require_torch

try:  # pragma: no cover - exercised by torch tests.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment]


if nn is not None:

    class ForecastModel(nn.Module):
        """Window-level forecaster with an optional TSF input adapter."""

        def __init__(
            self,
            *,
            input_adapter: nn.Module,
            backbone: nn.Module,
            head: nn.Module,
        ) -> None:
            super().__init__()
            self.input_adapter = input_adapter
            self.backbone = backbone
            self.head = head

        def forward(self, normalized_inputs: Tensor) -> Tensor:
            adapted_inputs = self.input_adapter(normalized_inputs)
            encoded = self.backbone(adapted_inputs)
            return self.head(encoded)


else:

    class ForecastModel:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()


def build_forecast_model(
    *,
    feature_names: tuple[str, ...],
    method_config: dict[str, Any],
    backbone_config: dict[str, Any],
    output_dim: int = 1,
) -> ForecastModel:
    require_torch()
    method_name = str(method_config.get("name", "baseline")).lower()
    input_dim = len(feature_names)
    if method_name == "baseline":
        input_adapter = nn.Identity()
    elif method_name == "tsf":
        artifact_dir = method_config.get("semantic_artifact")
        if not artifact_dir:
            raise ValueError("method.semantic_artifact is required for TSF mode")
        input_adapter = SemanticFieldLayer.from_artifact(
            Path(str(artifact_dir)),
            feature_names=feature_names,
            output_dim=input_dim,
            numeric_path=str(method_config.get("numeric_path", "diagonal")),
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unsupported method: {method_name}")

    kind = str(backbone_config.get("type", "gru"))
    backbone = build_backbone(kind, input_dim=input_dim, config=backbone_config)
    head = nn.Linear(int(backbone.output_dim), output_dim)
    return ForecastModel(input_adapter=input_adapter, backbone=backbone, head=head)
