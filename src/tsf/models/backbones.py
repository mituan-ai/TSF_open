from __future__ import annotations

import math
from typing import Any

try:  # pragma: no cover - import path exercised by torch tests.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment]


SUPPORTED_BACKBONES = (
    "gru",
    "lstm",
    "transformer",
    "informer",
    "mamba",
    "itransformer",
    "patchtst",
    "moderntcn",
)

PUBLIC_EXAMPLE_BACKBONES = ("gru",)


def require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for tsf.models")


if nn is not None:

    class GRUBackbone(nn.Module):
        """GRU encoder for window-level forecasting."""

        def __init__(
            self,
            *,
            input_dim: int,
            hidden_size: int,
            num_layers: int,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            _validate_positive("input_dim", input_dim)
            _validate_positive("hidden_size", hidden_size)
            _validate_positive("num_layers", num_layers)
            effective_dropout = float(dropout) if num_layers > 1 else 0.0
            self.input_dim = int(input_dim)
            self.output_dim = int(hidden_size)
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=effective_dropout,
            )

        def forward(self, inputs: Tensor) -> Tensor:
            _require_sequence_inputs(inputs, "GRUBackbone")
            _, hidden = self.gru(inputs)
            return hidden[-1]


    class LSTMBackbone(nn.Module):
        """LSTM encoder with the same public contract as GRUBackbone."""

        def __init__(
            self,
            *,
            input_dim: int,
            hidden_size: int,
            num_layers: int,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            _validate_positive("input_dim", input_dim)
            _validate_positive("hidden_size", hidden_size)
            _validate_positive("num_layers", num_layers)
            effective_dropout = float(dropout) if num_layers > 1 else 0.0
            self.input_dim = int(input_dim)
            self.output_dim = int(hidden_size)
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=effective_dropout,
            )

        def forward(self, inputs: Tensor) -> Tensor:
            _require_sequence_inputs(inputs, "LSTMBackbone")
            _, (hidden, _) = self.lstm(inputs)
            return hidden[-1]


    class TransformerBackbone(nn.Module):
        """Encoder-only temporal Transformer for sequence windows."""

        def __init__(
            self,
            *,
            input_dim: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            max_window_size: int = 512,
        ) -> None:
            super().__init__()
            _validate_transformer_args(input_dim, d_model, num_layers, num_heads, d_ff, max_window_size)
            self.input_dim = int(input_dim)
            self.output_dim = int(d_model)
            self.input_projection = nn.Linear(input_dim, d_model)
            self.position_embedding = nn.Parameter(torch.zeros(1, max_window_size, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=float(dropout),
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            _init_parameter(self.position_embedding, std=0.02)

        def forward(self, inputs: Tensor) -> Tensor:
            _require_sequence_inputs(inputs, "TransformerBackbone")
            if inputs.shape[1] > self.position_embedding.shape[1]:
                raise ValueError(
                    "TransformerBackbone input window exceeds configured max_window_size"
                )
            projected = self.input_projection(inputs)
            encoded = self.encoder(projected + self.position_embedding[:, : inputs.shape[1], :])
            return self.norm(encoded[:, -1, :])


    class InformerBackbone(nn.Module):
        """Lightweight HuggingFace Informer encoder wrapper."""

        def __init__(
            self,
            *,
            input_dim: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            distil: bool = True,
            sampling_factor: int = 5,
            max_window_size: int = 512,
        ) -> None:
            super().__init__()
            _validate_transformer_args(input_dim, d_model, num_layers, num_heads, d_ff, max_window_size)
            _validate_positive("sampling_factor", sampling_factor)
            try:
                from transformers import InformerConfig, InformerModel
            except ImportError as exc:  # pragma: no cover - depends on optional extra.
                raise ImportError(
                    "Informer backbone requires transformers. "
                    "Install the train extra or add transformers before running Informer experiments."
                ) from exc
            self.input_dim = int(input_dim)
            self.output_dim = int(d_model)
            self.max_window_size = int(max_window_size)
            self.context_length = int(max_window_size) - 1
            self.required_past_length = self.context_length + 1
            self.time_feature_size = 1
            config = InformerConfig(
                input_size=input_dim,
                prediction_length=1,
                context_length=self.context_length,
                lags_sequence=[1],
                d_model=d_model,
                encoder_layers=num_layers,
                decoder_layers=1,
                encoder_attention_heads=num_heads,
                decoder_attention_heads=num_heads,
                encoder_ffn_dim=d_ff,
                decoder_ffn_dim=d_ff,
                dropout=float(dropout),
                attention_dropout=float(dropout),
                activation_dropout=float(dropout),
                num_time_features=self.time_feature_size,
                num_dynamic_real_features=0,
                num_static_real_features=0,
                num_static_categorical_features=0,
                distil=bool(distil),
                sampling_factor=int(sampling_factor),
            )
            self.model = InformerModel(config)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, inputs: Tensor) -> Tensor:
            _require_sequence_inputs(inputs, "InformerBackbone")
            if inputs.shape[1] > self.max_window_size:
                raise ValueError("InformerBackbone input window exceeds configured max_window_size")
            if inputs.shape[1] < self.required_past_length:
                inputs = torch.nn.functional.pad(inputs, (0, 0, self.required_past_length - inputs.shape[1], 0))
            batch_size, sequence_length, _ = inputs.shape
            past_time_features = torch.zeros(
                batch_size,
                sequence_length,
                self.time_feature_size,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            future_time_features = torch.zeros(
                batch_size,
                1,
                self.time_feature_size,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            outputs = self.model(
                past_values=inputs,
                past_time_features=past_time_features,
                past_observed_mask=torch.ones_like(inputs),
                future_time_features=future_time_features,
            )
            hidden_state = getattr(outputs, "last_hidden_state", None)
            if hidden_state is None:
                hidden_state = outputs[0]
            return self.norm(hidden_state[:, -1, :])


    class MambaBackbone(nn.Module):
        """Lightweight HuggingFace Mamba sequence wrapper."""

        def __init__(
            self,
            *,
            input_dim: int,
            d_model: int,
            num_layers: int,
            state_size: int,
            conv_kernel: int,
            expand: int,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            _validate_positive("input_dim", input_dim)
            _validate_positive("d_model", d_model)
            _validate_positive("num_layers", num_layers)
            _validate_positive("state_size", state_size)
            _validate_positive("conv_kernel", conv_kernel)
            _validate_positive("expand", expand)
            try:
                from transformers import MambaConfig, MambaModel
            except ImportError as exc:  # pragma: no cover - depends on optional extra.
                raise ImportError(
                    "Mamba backbone requires transformers with MambaModel support. "
                    "Install the train extra or add transformers before running Mamba experiments."
                ) from exc
            from tsf.models.mamba_kernels import enable_hf_mamba_kernels

            enable_hf_mamba_kernels()
            self.input_dim = int(input_dim)
            self.output_dim = int(d_model)
            self.input_projection = nn.Linear(input_dim, d_model)
            self.model = MambaModel(
                MambaConfig(
                    vocab_size=1,
                    hidden_size=d_model,
                    state_size=state_size,
                    num_hidden_layers=num_layers,
                    conv_kernel=conv_kernel,
                    expand=expand,
                    intermediate_size=d_model * expand,
                )
            )
            self.dropout = nn.Dropout(float(dropout))
            self.norm = nn.LayerNorm(d_model)

        def forward(self, inputs: Tensor) -> Tensor:
            _require_sequence_inputs(inputs, "MambaBackbone")
            projected = self.dropout(self.input_projection(inputs))
            outputs = self.model(inputs_embeds=projected)
            hidden_state = getattr(outputs, "last_hidden_state", None)
            if hidden_state is None:
                hidden_state = outputs[0]
            return self.norm(hidden_state[:, -1, :])


    class ITransformerBackbone(nn.Module):
        """Inverted Transformer that treats variables as tokens."""

        def __init__(
            self,
            *,
            input_dim: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            max_window_size: int = 512,
        ) -> None:
            super().__init__()
            _validate_transformer_args(input_dim, d_model, num_layers, num_heads, d_ff, max_window_size)
            self.input_dim = int(input_dim)
            self.output_dim = int(d_model)
            self.max_window_size = int(max_window_size)
            self.temporal_projection = nn.Linear(max_window_size, d_model)
            self.variable_embedding = nn.Parameter(torch.zeros(1, input_dim, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=float(dropout),
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            _init_parameter(self.variable_embedding, std=0.02)

        def forward(self, inputs: Tensor) -> Tensor:
            _require_sequence_inputs(inputs, "ITransformerBackbone")
            if inputs.shape[1] > self.max_window_size:
                raise ValueError("ITransformerBackbone input window exceeds configured max_window_size")
            transposed = inputs.transpose(1, 2)
            if transposed.shape[-1] < self.max_window_size:
                transposed = torch.nn.functional.pad(
                    transposed,
                    (self.max_window_size - transposed.shape[-1], 0),
                )
            tokens = self.temporal_projection(transposed)
            encoded = self.encoder(tokens + self.variable_embedding)
            return self.norm(encoded.mean(dim=1))


    class PatchTSTBackbone(nn.Module):
        """Patch-based channel-independent Transformer encoder."""

        def __init__(
            self,
            *,
            input_dim: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            patch_len: int,
            stride: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            _validate_transformer_args(input_dim, d_model, num_layers, num_heads, d_ff, patch_len)
            _validate_positive("patch_len", patch_len)
            _validate_positive("stride", stride)
            self.input_dim = int(input_dim)
            self.output_dim = int(d_model)
            self.patch_len = int(patch_len)
            self.stride = int(stride)
            self.patch_projection = nn.Linear(patch_len, d_model)
            self.feature_embedding = nn.Parameter(torch.zeros(1, input_dim, 1, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=float(dropout),
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            _init_parameter(self.feature_embedding, std=0.02)

        def forward(self, inputs: Tensor) -> Tensor:
            _require_sequence_inputs(inputs, "PatchTSTBackbone")
            transposed = inputs.transpose(1, 2)
            if transposed.shape[-1] < self.patch_len:
                transposed = torch.nn.functional.pad(transposed, (self.patch_len - transposed.shape[-1], 0))
            patches = transposed.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            projected = self.patch_projection(patches) + self.feature_embedding
            batch_size, feature_dim, patch_count, model_dim = projected.shape
            tokens = projected.reshape(batch_size, feature_dim * patch_count, model_dim)
            encoded = self.encoder(tokens)
            return self.norm(encoded.mean(dim=1))


    class ModernTCNBackbone(nn.Module):
        """Modern pure-convolution sequence encoder for window-level forecasting."""

        def __init__(
            self,
            *,
            input_dim: int,
            d_model: int,
            num_layers: int,
            kernel_size: int,
            expansion: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            _validate_positive("input_dim", input_dim)
            _validate_positive("d_model", d_model)
            _validate_positive("num_layers", num_layers)
            _validate_positive("kernel_size", kernel_size)
            _validate_positive("expansion", expansion)
            if kernel_size % 2 == 0:
                raise ValueError("kernel_size must be odd for ModernTCNBackbone")
            self.input_dim = int(input_dim)
            self.output_dim = int(d_model)
            self.input_projection = nn.Linear(input_dim, d_model)
            self.blocks = nn.ModuleList(
                [
                    _ModernTCNBlock(
                        d_model=d_model,
                        kernel_size=kernel_size,
                        expansion=expansion,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.norm = nn.LayerNorm(d_model)

        def forward(self, inputs: Tensor) -> Tensor:
            _require_sequence_inputs(inputs, "ModernTCNBackbone")
            hidden = self.input_projection(inputs).transpose(1, 2)
            for block in self.blocks:
                hidden = block(hidden)
            pooled = hidden.mean(dim=-1)
            return self.norm(pooled)


    class _ModernTCNBlock(nn.Module):
        def __init__(self, *, d_model: int, kernel_size: int, expansion: int, dropout: float) -> None:
            super().__init__()
            hidden_dim = d_model * expansion
            self.depthwise = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=d_model,
            )
            self.norm = nn.BatchNorm1d(d_model)
            self.pointwise = nn.Sequential(
                nn.Conv1d(d_model, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Conv1d(hidden_dim, d_model, kernel_size=1),
                nn.Dropout(float(dropout)),
            )

        def forward(self, inputs: Tensor) -> Tensor:
            residual = inputs
            hidden = self.depthwise(inputs)
            hidden = self.norm(hidden)
            hidden = self.pointwise(hidden)
            return residual + hidden


else:

    class GRUBackbone:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()

    class LSTMBackbone:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()

    class TransformerBackbone:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()

    class InformerBackbone:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()

    class MambaBackbone:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()

    class ITransformerBackbone:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()

    class PatchTSTBackbone:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()

    class ModernTCNBackbone:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()


def build_backbone(kind: str, *, input_dim: int, config: dict[str, Any]) -> nn.Module:
    require_torch()
    normalized_kind = kind.lower()
    if normalized_kind == "gru":
        return GRUBackbone(
            input_dim=input_dim,
            hidden_size=int(config.get("hidden_size", 96)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.2)),
        )
    if normalized_kind == "lstm":
        return LSTMBackbone(
            input_dim=input_dim,
            hidden_size=int(config.get("hidden_size", 96)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.2)),
        )
    if normalized_kind == "transformer":
        return TransformerBackbone(
            input_dim=input_dim,
            d_model=_d_model(config),
            num_layers=int(config.get("num_layers", 2)),
            num_heads=_num_heads(config),
            d_ff=_d_ff(config),
            dropout=float(config.get("dropout", 0.1)),
            max_window_size=int(config.get("max_window_size", 512)),
        )
    if normalized_kind == "informer":
        return InformerBackbone(
            input_dim=input_dim,
            d_model=_d_model(config),
            num_layers=int(config.get("num_layers", 2)),
            num_heads=_num_heads(config),
            d_ff=_d_ff(config),
            dropout=float(config.get("dropout", 0.1)),
            distil=bool(config.get("distil", True)),
            sampling_factor=int(config.get("sampling_factor", 5)),
            max_window_size=int(config.get("max_window_size", 512)),
        )
    if normalized_kind == "mamba":
        return MambaBackbone(
            input_dim=input_dim,
            d_model=_d_model(config),
            num_layers=int(config.get("num_layers", 2)),
            state_size=int(config.get("state_size", config.get("state_dim", 16))),
            conv_kernel=int(config.get("conv_kernel", config.get("conv_kernel_size", 4))),
            expand=int(config.get("expand", config.get("expand_factor", 2))),
            dropout=float(config.get("dropout", 0.1)),
        )
    if normalized_kind == "itransformer":
        return ITransformerBackbone(
            input_dim=input_dim,
            d_model=_d_model(config),
            num_layers=int(config.get("num_layers", 2)),
            num_heads=_num_heads(config),
            d_ff=_d_ff(config),
            dropout=float(config.get("dropout", 0.1)),
            max_window_size=int(config.get("max_window_size", 512)),
        )
    if normalized_kind == "patchtst":
        return PatchTSTBackbone(
            input_dim=input_dim,
            d_model=_d_model(config),
            num_layers=int(config.get("num_layers", 2)),
            num_heads=_num_heads(config),
            d_ff=_d_ff(config),
            patch_len=int(config.get("patch_len", 8)),
            stride=int(config.get("stride", 4)),
            dropout=float(config.get("dropout", 0.1)),
        )
    if normalized_kind == "moderntcn":
        return ModernTCNBackbone(
            input_dim=input_dim,
            d_model=_d_model(config),
            num_layers=int(config.get("num_layers", 2)),
            kernel_size=int(config.get("kernel_size", 7)),
            expansion=int(config.get("expansion", 2)),
            dropout=float(config.get("dropout", 0.1)),
        )
    raise ValueError(f"Unsupported backbone: {kind}. Expected one of {SUPPORTED_BACKBONES}")


def _require_sequence_inputs(inputs: Tensor, module_name: str) -> None:
    if inputs.ndim != 3:
        raise ValueError(f"{module_name} inputs must have shape (batch, time, features)")


def _validate_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_transformer_args(
    input_dim: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    max_window_size: int,
) -> None:
    _validate_positive("input_dim", input_dim)
    _validate_positive("d_model", d_model)
    _validate_positive("num_layers", num_layers)
    _validate_positive("num_heads", num_heads)
    _validate_positive("d_ff", d_ff)
    _validate_positive("max_window_size", max_window_size)
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")


def _d_model(config: dict[str, Any]) -> int:
    return int(config.get("d_model", config.get("hidden_size", 96)))


def _num_heads(config: dict[str, Any]) -> int:
    return int(config.get("num_heads", config.get("num_attention_heads", 4)))


def _d_ff(config: dict[str, Any]) -> int:
    d_model = _d_model(config)
    return int(config.get("d_ff", config.get("feedforward_size", d_model * 4)))


def _init_parameter(parameter: nn.Parameter, *, std: float) -> None:
    bound = std * math.sqrt(3.0)
    nn.init.uniform_(parameter, -bound, bound)
