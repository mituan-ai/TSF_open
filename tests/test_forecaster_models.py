from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="forecast model tests require the train extra")

from tsf.models.backbones import SUPPORTED_BACKBONES, build_backbone
from tsf.models.forecaster import build_forecast_model
from tsf.semantic_field import SemanticFieldSpec


FEATURE_NAMES = (
    "time_h",
    "aeration",
    "agitator_rpm",
    "sugar_feed",
    "acid",
    "base",
    "cooling_water",
    "heating_water",
    "wfi",
    "air_head_pressure",
    "dumped_broth",
    "do2",
    "volume",
    "weight",
    "ph",
    "temp",
    "co2_out",
    "paa_flow",
    "oil_flow",
    "our",
    "o2_out",
    "cer",
    "ammonia_shots",
)


BACKBONE_CONFIGS = {
    "gru": {"type": "gru", "hidden_size": 16, "num_layers": 1, "dropout": 0.0},
    "lstm": {"type": "lstm", "hidden_size": 16, "num_layers": 1, "dropout": 0.0},
    "transformer": {
        "type": "transformer",
        "d_model": 16,
        "num_layers": 1,
        "num_heads": 4,
        "d_ff": 32,
        "dropout": 0.0,
        "max_window_size": 128,
    },
    "informer": {
        "type": "informer",
        "d_model": 16,
        "num_layers": 1,
        "num_heads": 4,
        "d_ff": 32,
        "dropout": 0.0,
        "sampling_factor": 5,
        "distil": True,
        "max_window_size": 128,
    },
    "mamba": {
        "type": "mamba",
        "d_model": 16,
        "num_layers": 1,
        "state_size": 8,
        "conv_kernel": 4,
        "expand": 2,
        "dropout": 0.0,
    },
    "itransformer": {
        "type": "itransformer",
        "d_model": 16,
        "num_layers": 1,
        "num_heads": 4,
        "d_ff": 32,
        "dropout": 0.0,
        "max_window_size": 128,
    },
    "patchtst": {
        "type": "patchtst",
        "d_model": 16,
        "num_layers": 1,
        "num_heads": 4,
        "d_ff": 32,
        "patch_len": 8,
        "stride": 4,
        "dropout": 0.0,
    },
    "moderntcn": {
        "type": "moderntcn",
        "d_model": 16,
        "num_layers": 1,
        "kernel_size": 7,
        "expansion": 2,
        "dropout": 0.0,
    },
}


def test_baseline_gru_forecaster_forward_shape() -> None:
    model = build_forecast_model(
        feature_names=FEATURE_NAMES,
        method_config={"name": "baseline"},
        backbone_config={"type": "gru", "hidden_size": 16, "num_layers": 1, "dropout": 0.0},
    )

    output = model(torch.zeros((4, 120, 23), dtype=torch.float32))

    assert output.shape == (4, 1)


@pytest.mark.parametrize("backbone_name", SUPPORTED_BACKBONES)
def test_baseline_forecaster_backbones_forward_shape(backbone_name: str) -> None:
    try:
        model = build_forecast_model(
            feature_names=FEATURE_NAMES,
            method_config={"name": "baseline"},
            backbone_config=BACKBONE_CONFIGS[backbone_name],
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    output = model(torch.zeros((4, 120, 23), dtype=torch.float32))

    assert output.shape == (4, 1)


@pytest.mark.parametrize("backbone_name", SUPPORTED_BACKBONES)
def test_tsf_forecaster_backbones_forward_shape(backbone_name: str) -> None:
    artifact = "resources/semantic_artifacts/indpensim"
    try:
        model = build_forecast_model(
            feature_names=FEATURE_NAMES,
            method_config={"name": "tsf", "semantic_artifact": artifact, "numeric_path": "diagonal"},
            backbone_config=BACKBONE_CONFIGS[backbone_name],
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    output = model(torch.zeros((4, 120, 23), dtype=torch.float32))

    assert output.shape == (4, 1)


def test_build_backbone_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported backbone"):
        build_backbone("unknown", input_dim=23, config={"type": "unknown"})


def test_baseline_gru_forecaster_supports_multi_target_output() -> None:
    feature_names = tuple(f"feature_{index}" for index in range(14))
    model = build_forecast_model(
        feature_names=feature_names,
        method_config={"name": "baseline"},
        backbone_config={"type": "gru", "hidden_size": 16, "num_layers": 1, "dropout": 0.0},
        output_dim=5,
    )

    output = model(torch.zeros((4, 60, 14), dtype=torch.float32))

    assert output.shape == (4, 5)


def test_tsf_gru_forecaster_loads_artifact_and_forward_shape() -> None:
    artifact = "resources/semantic_artifacts/indpensim"
    spec = SemanticFieldSpec.load_artifact(artifact_dir=Path(artifact), feature_names=FEATURE_NAMES)
    model = build_forecast_model(
        feature_names=FEATURE_NAMES,
        method_config={"name": "tsf", "semantic_artifact": artifact, "numeric_path": "diagonal"},
        backbone_config={"type": "gru", "hidden_size": 16, "num_layers": 1, "dropout": 0.0},
    )

    output = model(torch.zeros((4, 120, 23), dtype=torch.float32))

    assert spec.directions.shape == (23, 128)
    assert output.shape == (4, 1)


def test_ladle_tsf_gru_forecaster_supports_multi_target_output() -> None:
    artifact = "resources/semantic_artifacts/ladle_preheating"
    feature_names = (
        "时间间隔",
        "煤气阀门开度",
        "CO2流量",
        "O2流量",
        "CO流量",
        "N2流量",
        "煤气压力",
        "空气阀门开度",
        "空气O2流量",
        "空气N2流量",
        "空气CO2流量",
        "空气压力",
        "燃烧效率指标",
        "温度",
    )
    spec = SemanticFieldSpec.load_artifact(artifact_dir=Path(artifact), feature_names=feature_names)
    model = build_forecast_model(
        feature_names=feature_names,
        method_config={"name": "tsf", "semantic_artifact": artifact, "numeric_path": "diagonal"},
        backbone_config={"type": "gru", "hidden_size": 16, "num_layers": 1, "dropout": 0.0},
        output_dim=5,
    )

    output = model(torch.zeros((4, 60, 14), dtype=torch.float32))

    assert spec.directions.shape == (14, 128)
    assert output.shape == (4, 5)
