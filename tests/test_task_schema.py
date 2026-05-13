from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsf.llm_semantics import (
    LLMGenerationConfig,
    EMBEDDING_TEXT_RENDERER_VERSION,
    load_semantic_cards,
    load_semantic_card_payload,
    render_semantic_embedding_text,
    save_semantic_cards,
    validate_semantic_card_payload,
)
from tsf.experiment_io import load_env_file, load_simple_yaml_mapping
from tsf.task_schema import SEMANTIC_PROMPT_VERSION, build_template_task_spec, load_task_spec


def test_semantic_prompt_contains_information_boundary() -> None:
    prompt = build_template_task_spec().build_semantic_prompt()

    assert "Do not predict numeric values" in prompt
    assert "future windows" in prompt
    assert SEMANTIC_PROMPT_VERSION in prompt
    assert "target_variable" in prompt
    assert "feed_flow" in prompt


def test_task_spec_round_trip_from_json(tmp_path: Path) -> None:
    spec = build_template_task_spec()
    path = tmp_path / "task_spec.json"
    path.write_text(json.dumps(spec.as_dict(), ensure_ascii=False), encoding="utf-8")

    loaded = load_task_spec(path)

    assert loaded.dataset == spec.dataset
    assert loaded.feature_names == spec.feature_names
    assert loaded.as_dict() == spec.as_dict()


def test_load_semantic_cards_requires_generated_json() -> None:
    with pytest.raises(ValueError, match="generated JSON"):
        load_semantic_cards(Path("resources/datasets/ladle_preheating/VARIABLES.md"))


def test_semantic_card_payload_requires_fixed_order() -> None:
    payload = {
        "cards": [
            {
                "variable_name": "air",
                "canonical_name": "Air",
                "unit": "m3/h",
                "physical_meaning": "Oxidizer-side flow.",
                "process_role": "Supports combustion.",
                "information_role": "observation",
                "temporal_relation_to_target": "lagged",
                "relations_to_other_variables": ["gas"],
            }
        ]
    }

    with pytest.raises(ValueError, match="variable order mismatch"):
        validate_semantic_card_payload(payload, feature_names=("gas",))


def test_save_and_load_json_semantic_cards_uses_rendered_embedding_text(tmp_path: Path) -> None:
    task_spec = build_template_task_spec()
    payload = {"cards": []}
    for variable in task_spec.variables:
        payload["cards"].append(
            {
                "variable_name": variable.name,
                "canonical_name": variable.name,
                "unit": variable.unit,
                "physical_meaning": "measured input",
                "process_role": "process variable",
                "information_role": "observation",
                "temporal_relation_to_target": "lagged",
                "relations_to_other_variables": [],
            }
        )
    card_path = tmp_path / "semantic_cards.json"

    save_semantic_cards(payload=payload, task_spec=task_spec, output_path=card_path)
    cards = load_semantic_cards(card_path)

    assert [card.variable_name for card in cards] == list(task_spec.feature_names)
    assert cards[0].description == "feed_flow, m3/h. measured input; process variable; observation. lagged."
    stored = json.loads(card_path.read_text(encoding="utf-8"))
    assert stored["review_status"] == "generated"
    assert stored["prompt_version"] == SEMANTIC_PROMPT_VERSION
    assert stored["embedding_text_renderer"] == EMBEDDING_TEXT_RENDERER_VERSION
    assert set(stored["cards"][0]) == {
        "variable_name",
        "canonical_name",
        "unit",
        "physical_meaning",
        "process_role",
        "information_role",
        "temporal_relation_to_target",
        "relations_to_other_variables",
        "embedding_text",
    }


def test_render_semantic_embedding_text_is_natural_and_excludes_audit_fields() -> None:
    text = render_semantic_embedding_text(
        {
            "variable_name": "temperature",
            "canonical_name": "ladle temperature",
            "unit": "°C",
            "physical_meaning": "internal thermal state of the ladle",
            "process_role": "historical target state input",
            "information_role": "history_target",
            "temporal_relation_to_target": "influences future temperature through thermal inertia",
            "relations_to_other_variables": ["combustion efficiency index", "CO flow"],
            "confidence": 1.0,
        }
    )

    assert text == (
        "ladle temperature, °C. internal thermal state of the ladle; "
        "historical target state input; historical target state. influences future "
        "temperature through thermal inertia; combustion efficiency index, CO flow."
    )
    assert "combustion efficiency index" in text
    assert "变量=" not in text
    assert "单位=" not in text
    assert "unit=" not in text
    assert "leak" not in text
    assert "only" not in text


def test_render_semantic_embedding_text_requires_english_fields() -> None:
    with pytest.raises(ValueError, match="must be written in English"):
        render_semantic_embedding_text(
            {
                "variable_name": "temperature",
                "canonical_name": "钢包温度",
                "unit": "°C",
                "physical_meaning": "钢包内部热状态",
                "process_role": "historical target state input",
                "information_role": "history_target",
                "temporal_relation_to_target": "influences future temperature",
                "relations_to_other_variables": ["combustion efficiency index"],
            }
        )


def test_loading_semantic_card_payload_rewrites_embedding_text_deterministically(tmp_path: Path) -> None:
    card_path = tmp_path / "semantic_cards.json"
    card_path.write_text(
        json.dumps(
            {
                "cards": [
                    {
                        "variable_name": "air",
                        "canonical_name": "Air flow",
                        "unit": "m3/h",
                        "physical_meaning": "oxidizer-side flow",
                        "process_role": "supports combustion",
                        "information_role": "control",
                        "temporal_relation_to_target": "lagged thermal effect",
                        "relations_to_other_variables": ["fuel flow"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = load_semantic_card_payload(card_path)

    assert payload["cards"][0]["embedding_text"] == (
        "Air flow, m3/h. oxidizer-side flow; supports combustion; control. "
        "lagged thermal effect; fuel flow."
    )


def test_yaml_env_reference_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_path = tmp_path / "config.env"
    yaml_path = tmp_path / "config.yaml"
    env_path.write_text("TSF_TEST_MODEL=example-model\n", encoding="utf-8")
    yaml_path.write_text("model: ${TSF_TEST_MODEL}\nmissing: ${TSF_TEST_MISSING}\n", encoding="utf-8")
    monkeypatch.delenv("TSF_TEST_MODEL", raising=False)

    load_env_file(env_path)
    payload = load_simple_yaml_mapping(yaml_path)

    assert payload["model"] == "example-model"
    assert payload["missing"] == ""


def test_llm_config_omits_temperature_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSF_LLM_MODEL", "example-chat-model")
    monkeypatch.setenv("TSF_LLM_BASE_URL", "https://api.example.test/v1")
    payload = load_simple_yaml_mapping(Path("configs/llm.yaml"))
    config = LLMGenerationConfig.from_mapping(payload)

    assert config.temperature is None
    assert config.reasoning_effort == "max"
    assert config.max_tokens == 24000
