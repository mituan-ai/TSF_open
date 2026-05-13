from __future__ import annotations

import ast
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any


ENV_REFERENCE_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
SAFE_OUTPUT_PART_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def parse_yaml_scalar(raw_value: str) -> object:
    env_match = ENV_REFERENCE_PATTERN.match(raw_value)
    if env_match:
        return os.environ.get(env_match.group(1), "")
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if raw_value.startswith("[") or raw_value.startswith("{") or raw_value.startswith(("'", '"')):
        try:
            return ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            if raw_value.startswith("[") and raw_value.endswith("]"):
                inner = raw_value[1:-1].strip()
                if not inner:
                    return []
                return [segment.strip().strip("'\"") for segment in inner.split(",")]
            raise
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def load_simple_yaml_mapping(config_path: Path) -> dict[str, object]:
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, root)]

    for line_number, raw_line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped_line = raw_line.split("#", 1)[0].rstrip()
        if not stripped_line:
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"Unsupported indentation in {config_path}:{line_number}")

        content = stripped_line.lstrip(" ")
        key, separator, value_text = content.partition(":")
        if not separator:
            raise ValueError(f"Invalid YAML entry in {config_path}:{line_number}")

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current_mapping = stack[-1][1]

        normalized_key = key.strip()
        normalized_value = value_text.strip()
        if not normalized_value:
            next_mapping: dict[str, object] = {}
            current_mapping[normalized_key] = next_mapping
            stack.append((indent, next_mapping))
            continue

        current_mapping[normalized_key] = parse_yaml_scalar(normalized_value)

    return root


def load_env_file(env_path: Path, *, override: bool = True) -> None:
    if not env_path.exists():
        return
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"Invalid env entry in {env_path}:{line_number}")
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError(f"Invalid empty env key in {env_path}:{line_number}")
        normalized_value = value.strip().strip("'\"")
        if override or normalized_key not in os.environ:
            os.environ[normalized_key] = normalized_value


def load_api_env_files(root: Path) -> None:
    env_dir = root / "configs/env"
    load_env_file(env_dir / "api.env", override=False)
    load_env_file(env_dir / "api.local.env", override=True)


def resolve_reference_path(reference: str, *, config_path: Path, root: Path) -> Path:
    raw_path = Path(reference)
    if raw_path.is_absolute():
        return raw_path
    local_candidate = (config_path.parent / raw_path).resolve()
    if local_candidate.exists():
        return local_candidate
    root_candidate = (root / raw_path).resolve()
    if root_candidate.exists():
        return root_candidate
    return local_candidate


def normalize_output_name(raw_name: str) -> str:
    normalized = Path(raw_name).name
    if normalized in {"", ".", ".."}:
        raise ValueError(f"Invalid output name: {raw_name}")
    return normalized


def slugify_output_part(raw_name: object, *, fallback: str = "unnamed") -> str:
    normalized = SAFE_OUTPUT_PART_PATTERN.sub("_", str(raw_name).strip())
    normalized = normalized.strip("._-")
    if not normalized:
        normalized = fallback
    return normalize_output_name(normalized)


def current_timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_unique_output_dir(parent: Path, stem: str, *, timestamp_label: str | None = None) -> Path:
    normalized_stem = slugify_output_part(stem)
    resolved_timestamp = current_timestamp_label() if timestamp_label is None else timestamp_label
    candidate = parent / f"{normalized_stem}_{resolved_timestamp}"
    collision_index = 1
    while candidate.exists():
        candidate = parent / f"{normalized_stem}_{resolved_timestamp}_{collision_index:02d}"
        collision_index += 1
    return candidate


def to_jsonable(value: object) -> object:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
