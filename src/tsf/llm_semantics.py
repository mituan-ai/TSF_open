from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from tsf.experiment_io import write_json
from tsf.semantic_field import (
    SemanticFieldSpec,
    validate_feature_order,
)
from tsf.task_schema import (
    SEMANTIC_CARD_FIELDS,
    SEMANTIC_CARD_INPUT_FIELDS,
    SEMANTIC_PROMPT_VERSION,
    ForecastTaskSpec,
    SemanticCard,
)


EMBEDDING_TEXT_RENDERER_VERSION = "tsf-compact-variable-semantics-v3"


@dataclass(frozen=True)
class LLMGenerationConfig:
    provider: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float | None = None
    reasoning_effort: str | None = None
    max_tokens: int = 12000
    timeout_seconds: float = 120.0

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "LLMGenerationConfig":
        model = str(payload["model"]).strip()
        base_url = str(payload["base_url"]).strip()
        if not model:
            raise ValueError("LLM config model is empty; fill configs/env/api.local.env")
        if not base_url:
            raise ValueError("LLM config base_url is empty; fill configs/env/api.local.env")
        return cls(
            provider=str(payload.get("provider", "openai_compatible")),
            model=model,
            base_url=base_url,
            api_key_env=str(payload.get("api_key_env", "OPENAI_API_KEY")),
            temperature=(
                None
                if payload.get("temperature") in {None, "", "null", "none"}
                else float(payload["temperature"])
            ),
            reasoning_effort=(
                None
                if payload.get("reasoning_effort") in {None, "", "null", "none"}
                else str(payload["reasoning_effort"])
            ),
            max_tokens=int(payload.get("max_tokens", 12000)),
            timeout_seconds=float(payload.get("timeout_seconds", 120.0)),
        )

    def public_metadata(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class EmbeddingGenerationConfig:
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    batch_size: int = 64
    timeout_seconds: float = 120.0
    normalize_vectors: bool = True
    truncate_dim: int | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "EmbeddingGenerationConfig":
        model = str(payload["model"]).strip()
        if not model:
            raise ValueError("Embedding config model is empty; fill configs/env/api.local.env")
        base_url_value = None if payload.get("base_url") is None else str(payload.get("base_url")).strip()
        return cls(
            provider=str(payload.get("provider", "openai_compatible")),
            model=model,
            base_url=None if not base_url_value else base_url_value,
            api_key_env=str(payload.get("api_key_env", "OPENAI_API_KEY")),
            batch_size=int(payload.get("batch_size", 64)),
            timeout_seconds=float(payload.get("timeout_seconds", 120.0)),
            normalize_vectors=bool(payload.get("normalize_vectors", True)),
            truncate_dim=(
                None
                if payload.get("truncate_dim") in {None, "", "null", "none"}
                else int(payload["truncate_dim"])
            ),
        )

    def public_metadata(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "batch_size": self.batch_size,
            "timeout_seconds": self.timeout_seconds,
            "normalize_vectors": self.normalize_vectors,
            "truncate_dim": self.truncate_dim,
        }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(text)


ORDER_DEPENDENCE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"第\s*\d+\s*列",
        r"第\s*[一二三四五六七八九十百千万]+\s*列",
        r"上一列",
        r"下一列",
        r"前一列",
        r"后一列",
        r"\bcolumn\s+\d+\b",
        r"\b(first|second|third|fourth|fifth|last)\s+column\b",
        r"\bprevious\s+column\b",
        r"\bnext\s+column\b",
    )
)


def validate_embedding_text_not_order_dependent(text: str, *, variable_name: str) -> None:
    for pattern in ORDER_DEPENDENCE_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                f"semantic card {variable_name} contains column-order-dependent wording: "
                f"{pattern.pattern}"
            )


def _clean_text(value: object) -> str:
    text = " ".join(str(value).strip().split())
    return text.strip("。.;； ")


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _display_name(card: dict[str, Any]) -> str:
    variable_name = _clean_text(card.get("variable_name", ""))
    canonical_name = _clean_text(card.get("canonical_name", ""))
    if canonical_name and canonical_name != variable_name:
        if _contains_cjk(canonical_name + variable_name):
            return f"{canonical_name}（{variable_name}）"
        return f"{canonical_name} ({variable_name})"
    return variable_name


def _relation_text(card: dict[str, Any], *, chinese: bool) -> str:
    relations = card.get("relations_to_other_variables", [])
    if not isinstance(relations, list):
        return ""
    cleaned = [_clean_text(item) for item in relations if _clean_text(item)]
    if not cleaned:
        return ""
    selected = []
    for item in cleaned[:3]:
        item = item.strip("。.;； ")
        if chinese:
            for prefix in ("与", "和", "同"):
                if item.startswith(prefix) and len(item) > 1:
                    item = item[len(prefix) :]
                    break
        selected.append(item)
    if chinese:
        return "、".join(selected)
    return ", ".join(selected)


def _role_phrase(information_role: str, *, chinese: bool) -> str:
    normalized = information_role.strip().lower()
    if chinese:
        mapping = {
            "control": "控制量",
            "state": "状态量",
            "observation": "观测量",
            "derived": "派生量",
            "time": "时间特征",
            "history_target": "历史目标状态",
            "other": "其他输入",
        }
        return mapping.get(normalized, information_role)
    mapping = {
        "control": "control",
        "state": "state",
        "observation": "observation",
        "derived": "derived",
        "time": "time feature",
        "history_target": "historical target state",
        "other": "other input",
    }
    return mapping.get(normalized, normalized.replace("_", " "))


def _ensure_english_text(text: str, *, field_name: str, variable_name: str) -> str:
    normalized = _clean_text(text)
    if normalized and _contains_cjk(normalized):
        raise ValueError(
            f"semantic card {variable_name} field {field_name} must be written in English"
        )
    return normalized


def _normalize_information_role(role: str, *, variable_name: str) -> str:
    normalized = _clean_text(role).strip().lower()
    allowed = {
        "control": "control",
        "state": "state",
        "observation": "observation",
        "derived": "derived",
        "time": "time feature",
        "history_target": "historical target state",
        "other": "other input",
    }
    if not normalized:
        raise ValueError(f"semantic card {variable_name} field information_role is empty")
    if _contains_cjk(normalized):
        raise ValueError(
            f"semantic card {variable_name} field information_role must be written in English"
        )
    if normalized not in allowed:
        raise ValueError(
            f"semantic card {variable_name} field information_role must be one of {sorted(allowed)}"
        )
    return allowed[normalized]


def _normalize_relation_clause(text: str) -> str:
    normalized = _clean_text(text)
    if not normalized:
        return ""
    normalized = normalized.replace("，", ",")
    normalized = normalized.replace("；", ";")
    normalized = normalized.replace("、", ", ")
    normalized = normalized.replace(" and ", ", ")
    normalized = normalized.replace(" or ", ", ")
    normalized = " ".join(normalized.split())
    return normalized[0].upper() + normalized[1:] if normalized else ""


def render_semantic_embedding_text(card: dict[str, Any]) -> str:
    """Render stable compact text for embedding from structured card fields."""

    variable_name = _clean_text(card.get("variable_name", ""))
    name = _ensure_english_text(
        card.get("canonical_name") or variable_name,
        field_name="canonical_name",
        variable_name=variable_name,
    )
    unit = _ensure_english_text(card.get("unit", ""), field_name="unit", variable_name=variable_name)
    physical_meaning = _ensure_english_text(
        card.get("physical_meaning", ""),
        field_name="physical_meaning",
        variable_name=variable_name,
    )
    process_role = _ensure_english_text(
        card.get("process_role", ""),
        field_name="process_role",
        variable_name=variable_name,
    )
    information_role = _normalize_information_role(
        card.get("information_role", ""),
        variable_name=variable_name,
    )
    temporal_relation = _ensure_english_text(
        card.get("temporal_relation_to_target", ""),
        field_name="temporal_relation_to_target",
        variable_name=variable_name,
    )
    relation_text = _relation_text(card, chinese=False)
    if relation_text and _contains_cjk(relation_text):
        raise ValueError(
            f"semantic card {variable_name} field relations_to_other_variables must be written in English"
        )

    parts: list[str] = []
    identity = ", ".join([item for item in (name, unit) if item])
    if identity:
        parts.append(f"{identity}.")
    meaning_parts = [item for item in (physical_meaning, process_role, information_role) if item]
    if meaning_parts:
        parts.append("; ".join(meaning_parts) + ".")
    dynamics_parts = [item for item in (temporal_relation, relation_text) if item]
    if dynamics_parts:
        parts.append("; ".join(dynamics_parts) + ".")
    rendered = " ".join(parts).strip()
    validate_embedding_text_not_order_dependent(rendered, variable_name=variable_name or name)
    return rendered


def _task_spec_units(task_spec: ForecastTaskSpec) -> dict[str, str]:
    return {variable.name: variable.unit for variable in task_spec.variables if variable.unit}


def attach_task_units_to_cards(
    cards: list[dict[str, Any]],
    task_spec: ForecastTaskSpec | None,
) -> list[dict[str, Any]]:
    if task_spec is None:
        return cards
    units = _task_spec_units(task_spec)
    enriched_cards: list[dict[str, Any]] = []
    for card in cards:
        enriched = dict(card)
        variable_name = _clean_text(enriched.get("variable_name", ""))
        if variable_name in units and not _clean_text(enriched.get("unit", "")):
            enriched["unit"] = units[variable_name]
        enriched_cards.append(enriched)
    return enriched_cards


def export_semantic_prompt(task_spec: ForecastTaskSpec, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(task_spec.build_semantic_prompt() + "\n", encoding="utf-8")


def write_prompt_package(task_spec: ForecastTaskSpec, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = task_spec.build_semantic_prompt()
    (output_dir / "semantic_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    write_json(
        output_dir / "task_spec.json",
        {
            **task_spec.as_dict(),
            "prompt_version": SEMANTIC_PROMPT_VERSION,
            "prompt_sha256": sha256_text(prompt),
        },
    )


def call_llm_for_semantic_cards(
    task_spec: ForecastTaskSpec,
    config: LLMGenerationConfig,
) -> dict[str, Any]:
    if config.provider != "openai_compatible":
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install the llm extra to call external LLM APIs: uv sync --extra llm") from exc

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {config.api_key_env}")

    prompt = task_spec.build_semantic_prompt()
    client = OpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout_seconds)
    request_kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You produce strict JSON for scientific preprocessing. "
                    "Return JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": config.max_tokens,
        "response_format": {"type": "json_object"},
    }
    if config.temperature is not None:
        request_kwargs["temperature"] = config.temperature
    if config.reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = config.reasoning_effort
    response = client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned an empty semantic-card response")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise RuntimeError("LLM semantic-card response must be a JSON object")
    return payload


def validate_semantic_card_payload(
    payload: dict[str, Any],
    *,
    feature_names: tuple[str, ...],
    task_spec: ForecastTaskSpec | None = None,
) -> dict[str, Any]:
    cards = payload.get("cards")
    if not isinstance(cards, list):
        raise ValueError("semantic card payload must contain a cards list")
    if len(cards) != len(feature_names):
        raise ValueError(
            f"semantic card count {len(cards)} does not match feature count {len(feature_names)}"
        )

    normalized_cards: list[dict[str, Any]] = []
    cards_with_units = attach_task_units_to_cards(cards, task_spec)
    for index, item in enumerate(cards_with_units):
        if not isinstance(item, dict):
            raise ValueError("each semantic card must be a JSON object")
        required_fields = (
            SEMANTIC_CARD_INPUT_FIELDS
            if "embedding_text" not in item
            else SEMANTIC_CARD_FIELDS
        )
        missing = [field for field in required_fields if field not in item]
        if missing:
            raise ValueError(f"semantic card for {feature_names[index]} is missing fields: {missing}")
        variable_name = str(item["variable_name"])
        if variable_name != feature_names[index]:
            raise ValueError(
                "semantic card variable order mismatch: "
                f"expected {feature_names[index]!r} at index {index}, got {variable_name!r}"
            )
        llm_embedding_text = str(item.get("embedding_text", "")).strip()
        if llm_embedding_text:
            validate_embedding_text_not_order_dependent(
                llm_embedding_text,
                variable_name=variable_name,
            )
        normalized_item = {field: item[field] for field in SEMANTIC_CARD_INPUT_FIELDS}
        normalized_item["variable_name"] = variable_name
        if not isinstance(normalized_item["relations_to_other_variables"], list):
            raise ValueError(f"semantic card {variable_name} relations_to_other_variables must be a list")
        normalized_item["embedding_text"] = render_semantic_embedding_text(normalized_item)
        normalized_cards.append(normalized_item)

    return {**payload, "cards": normalized_cards}


def save_semantic_cards(
    *,
    payload: dict[str, Any],
    task_spec: ForecastTaskSpec,
    output_path: Path,
    llm_config: LLMGenerationConfig | None = None,
    review_status: str = "generated",
) -> None:
    prompt = task_spec.build_semantic_prompt()
    validated = validate_semantic_card_payload(
        payload,
        feature_names=task_spec.feature_names,
        task_spec=task_spec,
    )
    output = {
        "artifact_type": "tsf_semantic_cards",
        "prompt_version": SEMANTIC_PROMPT_VERSION,
        "prompt_sha256": sha256_text(prompt),
        "task_spec": task_spec.as_dict(),
        "llm_generation": None if llm_config is None else llm_config.public_metadata(),
        "review_status": review_status,
        "embedding_text_renderer": EMBEDDING_TEXT_RENDERER_VERSION,
        "cards_sha256": sha256_json(validated["cards"]),
        "cards": validated["cards"],
    }
    write_json(output_path, output)


def generate_semantic_cards_with_llm(
    *,
    task_spec: ForecastTaskSpec,
    llm_config: LLMGenerationConfig,
    output_path: Path,
) -> dict[str, Any]:
    payload = call_llm_for_semantic_cards(task_spec, llm_config)
    save_semantic_cards(
        payload=payload,
        task_spec=task_spec,
        output_path=output_path,
        llm_config=llm_config,
    )
    return payload


def load_semantic_cards(card_path: Path) -> tuple[SemanticCard, ...]:
    if card_path.suffix.lower() != ".json":
        raise ValueError("semantic cards must be generated JSON")
    payload = load_semantic_card_payload(card_path)
    items = payload["cards"]
    cards: list[SemanticCard] = []
    for item in items:
        cards.append(
            SemanticCard(
                variable_name=str(item["variable_name"]),
                description=str(item["embedding_text"]),
                source=str(item.get("source", "offline_card")),
                confidence=(
                    None
                    if item.get("confidence") is None
                    else float(item["confidence"])
                ),
            )
        )
    return tuple(cards)


def load_semantic_card_payload(card_path: Path) -> dict[str, Any]:
    payload = json.loads(card_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("semantic card JSON must contain an object")
    cards = payload.get("cards")
    if not isinstance(cards, list):
        raise ValueError("semantic card JSON must contain a cards list")
    feature_names = tuple(str(item["variable_name"]) for item in cards if isinstance(item, dict))
    if len(feature_names) != len(cards):
        raise ValueError("each semantic card item must contain variable_name")
    task_spec_payload = payload.get("task_spec")
    task_spec = (
        ForecastTaskSpec.from_mapping(task_spec_payload)
        if isinstance(task_spec_payload, dict)
        else None
    )
    return validate_semantic_card_payload(
        payload,
        feature_names=feature_names,
        task_spec=task_spec,
    )


def embedding_texts_from_card_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    cards = payload.get("cards")
    if not isinstance(cards, list):
        raise ValueError("semantic card payload must contain a cards list")
    return tuple(render_semantic_embedding_text(card) for card in cards)


def rewrite_semantic_card_file_with_rendered_embeddings(card_path: Path) -> None:
    payload = json.loads(card_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("semantic card JSON must contain an object")
    validated = load_semantic_card_payload(card_path)
    task_spec_payload = payload.get("task_spec")
    prompt_metadata: dict[str, object] = {}
    if isinstance(task_spec_payload, dict):
        task_spec = ForecastTaskSpec.from_mapping(task_spec_payload)
        prompt = task_spec.build_semantic_prompt()
        prompt_metadata = {
            "prompt_version": SEMANTIC_PROMPT_VERSION,
            "prompt_sha256": sha256_text(prompt),
        }
    payload.update(
        {
            **prompt_metadata,
            "review_status": "generated",
            "embedding_text_renderer": EMBEDDING_TEXT_RENDERER_VERSION,
            "cards_sha256": sha256_json(validated["cards"]),
            "cards": validated["cards"],
        }
    )
    write_json(card_path, payload)


def embed_texts_openai_compatible(
    texts: tuple[str, ...],
    config: EmbeddingGenerationConfig,
) -> np.ndarray:
    if config.provider != "openai_compatible":
        raise ValueError(f"Unsupported embedding provider: {config.provider}")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install the llm extra to call embedding APIs: uv sync --extra llm") from exc

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {config.api_key_env}")

    client = OpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout_seconds)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), config.batch_size):
        batch = texts[start : start + config.batch_size]
        request_kwargs: dict[str, Any] = {
            "model": config.model,
            "input": list(batch),
        }
        if config.truncate_dim is not None:
            request_kwargs["dimensions"] = config.truncate_dim
        response = client.embeddings.create(**request_kwargs)
        vectors.extend(item.embedding for item in response.data)

    embeddings = np.asarray(vectors, dtype=np.float64)
    if config.truncate_dim is not None and embeddings.shape[1] != config.truncate_dim:
        raise ValueError(
            "embedding API returned an unexpected dimension: "
            f"expected {config.truncate_dim}, got {embeddings.shape[1]}"
        )
    if config.normalize_vectors:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
    return embeddings.astype(np.float64)


def embed_texts(
    texts: tuple[str, ...],
    config: EmbeddingGenerationConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    if config.provider == "openai_compatible":
        embeddings = embed_texts_openai_compatible(texts, config)
        metadata = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            **config.public_metadata(),
        }
        return embeddings, metadata
    raise ValueError(f"Unsupported embedding provider: {config.provider}")


def build_semantic_field_artifact_from_cards(
    *,
    card_path: Path,
    output_dir: Path,
    embedding_config: EmbeddingGenerationConfig,
    semantic_dim: int | None = None,
    feature_names: tuple[str, ...] | None = None,
) -> SemanticFieldSpec:
    card_payload = load_semantic_card_payload(card_path)
    cards = card_payload["cards"]
    card_feature_names = tuple(str(card["variable_name"]) for card in cards)
    if feature_names is not None:
        validate_feature_order(
            feature_names=feature_names,
            semantic_variable_names=card_feature_names,
        )
    descriptions = embedding_texts_from_card_payload(card_payload)
    embeddings, embedding_metadata = embed_texts(descriptions, embedding_config)
    if embeddings.ndim != 2:
        raise ValueError("embedding provider must return a 2D array")
    if embeddings.shape[0] != len(card_feature_names):
        raise ValueError(
            "embedding provider returned a row count that does not match semantic cards: "
            f"cards={len(card_feature_names)}, embeddings={embeddings.shape[0]}"
        )
    resolved_dim = int(embeddings.shape[1])
    if embedding_config.truncate_dim is not None and resolved_dim != embedding_config.truncate_dim:
        raise ValueError(
            "embedding output dimension does not match truncate_dim: "
            f"truncate_dim={embedding_config.truncate_dim}, got={resolved_dim}"
        )
    if semantic_dim is not None and resolved_dim != semantic_dim:
        raise ValueError(
            "semantic_dim must match the embedding output dimension in direct_embedding mode: "
            f"semantic_dim={semantic_dim}, got={resolved_dim}"
        )
    spec = SemanticFieldSpec(
        variable_names=card_feature_names,
        descriptions=descriptions,
        embeddings=embeddings,
        directions=embeddings.astype(np.float64),
        embedding_model=embedding_config.model,
    )
    spec.save_artifact(output_dir)
    metadata_path = output_dir / "semantic_field_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "artifact_type": "tsf_semantic_field",
            "semantic_cards_path": str(card_path),
            "semantic_cards_sha256": card_payload.get("cards_sha256", sha256_json(cards)),
            "prompt_version": card_payload.get("prompt_version", SEMANTIC_PROMPT_VERSION),
            "review_status": card_payload.get("review_status", "unknown"),
            "embedding_generation": embedding_metadata,
            "direction_method": "direct_embedding",
            "feature_names": list(card_feature_names),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return spec
