from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tsf.experiment_io import load_simple_yaml_mapping
from tsf.llm_semantics import (
    EmbeddingGenerationConfig,
    build_semantic_field_artifact_from_cards,
    load_semantic_card_payload,
    validate_semantic_card_payload,
)


def _semantic_card(variable_name: str, embedding_text: str = "") -> dict[str, object]:
    card: dict[str, object] = {
        "variable_name": variable_name,
        "canonical_name": f"{variable_name} name",
        "unit": "engineering unit",
        "physical_meaning": "measured process input",
        "process_role": "industrial process variable",
        "information_role": "observation",
        "temporal_relation_to_target": "lagged",
        "relations_to_other_variables": [],
    }
    if embedding_text:
        card["embedding_text"] = embedding_text
    return card


def _legacy_semantic_card(variable_name: str, embedding_text: str) -> dict[str, object]:
    return {
        "variable_name": variable_name,
        "canonical_name": f"{variable_name} name",
        "unit": "engineering unit",
        "physical_meaning": "measured process input",
        "process_role": "industrial process variable",
        "information_role": "observation",
        "online_availability": "online at prediction time",
        "temporal_relation_to_target": "lagged",
        "relations_to_other_variables": [],
        "embedding_text": embedding_text,
        "confidence": 0.8,
        "review_notes": "test card",
    }


def test_api_embedding_config_uses_env_references(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSF_EMBEDDING_BASE_URL", "https://example.test/v1")

    payload = load_simple_yaml_mapping(Path("configs/embedding.yaml"))
    config = EmbeddingGenerationConfig.from_mapping(payload)

    assert config.provider == "openai_compatible"
    assert config.model == "text-embedding-v4"
    assert config.base_url == "https://example.test/v1"
    assert config.api_key_env == "TSF_EMBEDDING_API_KEY"
    assert config.batch_size == 10
    assert config.timeout_seconds == 120
    assert config.truncate_dim == 128


def test_local_env_files_are_ignored_by_git() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "configs/env/api.local.env"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "configs/env/api.local.env" in result.stdout


def test_embedding_text_rejects_order_dependent_wording() -> None:
    payload = {
        "cards": [
            _legacy_semantic_card(
                "gas_pressure",
                "This variable is the first column and measures fuel-side pressure.",
            )
        ]
    }

    with pytest.raises(ValueError, match="column-order-dependent"):
        validate_semantic_card_payload(payload, feature_names=("gas_pressure",))


def test_direct_embedding_artifact_with_monkeypatched_embedder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card_path = tmp_path / "semantic_cards.json"
    card_path.write_text(
        json.dumps(
            {
                "artifact_type": "tsf_semantic_cards",
                "review_status": "generated",
                "cards_sha256": "test",
                "cards": [
                    _semantic_card("gas_pressure"),
                    _semantic_card("temperature"),
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fake_embed_texts(texts, config):
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        return vectors, {
            "provider": config.provider,
            "model": config.model,
            "truncate_dim": config.truncate_dim,
        }

    monkeypatch.setattr("tsf.llm_semantics.embed_texts", fake_embed_texts)
    config = EmbeddingGenerationConfig(
        provider="openai_compatible",
        model="text-embedding-v4",
        truncate_dim=3,
        batch_size=10,
    )
    output_dir = tmp_path / "artifact"

    build_semantic_field_artifact_from_cards(
        card_path=card_path,
        output_dir=output_dir,
        embedding_config=config,
        semantic_dim=3,
    )

    directions = np.load(output_dir / "directions.npy")
    metadata = json.loads((output_dir / "semantic_field_metadata.json").read_text(encoding="utf-8"))

    assert directions.shape == (2, 3)
    assert not (output_dir / "projection.npy").exists()
    assert metadata["direction_method"] == "direct_embedding"
    assert metadata["embedding_generation"]["truncate_dim"] == 3


def test_semantic_card_payload_drops_legacy_audit_fields(tmp_path: Path) -> None:
    card_path = tmp_path / "semantic_cards.json"
    card_path.write_text(
        json.dumps(
            {
                "cards": [
                    _legacy_semantic_card(
                        "gas_pressure",
                        "Gas pressure is a process state.",
                    )
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = load_semantic_card_payload(card_path)
    card = payload["cards"][0]

    assert "online_availability" not in card
    assert "confidence" not in card
    assert "review_notes" not in card
    assert "embedding_text" in card
    assert "variable=" not in card["embedding_text"].lower()
    assert "unit=" not in card["embedding_text"].lower()
    assert "工程" not in card["embedding_text"]


def test_generated_semantic_cards_can_build_artifact_without_required_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card_path = tmp_path / "semantic_cards.json"
    card_path.write_text(
        json.dumps(
            {
                "artifact_type": "tsf_semantic_cards",
                "review_status": "generated",
                "cards": [_semantic_card("gas_pressure")],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fake_embed_texts(texts, config):
        return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64), {
            "provider": config.provider,
            "model": config.model,
            "truncate_dim": config.truncate_dim,
        }

    monkeypatch.setattr("tsf.llm_semantics.embed_texts", fake_embed_texts)
    config = EmbeddingGenerationConfig(
        provider="openai_compatible",
        model="text-embedding-v4",
        truncate_dim=3,
        batch_size=10,
    )

    build_semantic_field_artifact_from_cards(
        card_path=card_path,
        output_dir=tmp_path / "artifact",
        embedding_config=config,
        semantic_dim=3,
    )

    directions = np.load(tmp_path / "artifact/directions.npy")
    assert directions.shape == (1, 3)
