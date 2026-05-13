from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import prepare_script_runtime

ROOT = prepare_script_runtime(__file__)

from tsf.experiment_io import (
    build_unique_output_dir,
    load_api_env_files,
    load_simple_yaml_mapping,
    resolve_reference_path,
    slugify_output_part,
)
from tsf.llm_semantics import load_semantic_card_payload
from tsf.llm_semantics import EmbeddingGenerationConfig, build_semantic_field_artifact_from_cards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorize TSF semantic cards and save a frozen semantic-field artifact."
    )
    parser.add_argument("--semantic-cards", required=True, help="Semantic cards JSON path.")
    parser.add_argument("--embedding-config", default="configs/embedding.yaml", help="Embedding YAML config path.")
    parser.add_argument(
        "--output-dir",
        help=(
            "Output artifact directory. If omitted, a readable directory is created "
            "under outputs/semantic_artifacts/<dataset>/."
        ),
    )
    parser.add_argument(
        "--semantic-dim",
        type=int,
        help="Expected semantic dimension k. If omitted, it is inferred from embedding output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    card_path = resolve_reference_path(
        args.semantic_cards,
        config_path=ROOT / args.semantic_cards,
        root=ROOT,
    )
    embedding_config_path = resolve_reference_path(
        args.embedding_config,
        config_path=ROOT / args.embedding_config,
        root=ROOT,
    )
    load_api_env_files(ROOT)
    embedding_config = EmbeddingGenerationConfig.from_mapping(
        load_simple_yaml_mapping(embedding_config_path)
    )
    semantic_dim = args.semantic_dim
    output_dim_label = semantic_dim or embedding_config.truncate_dim or "auto"
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (ROOT / output_dir).resolve()
    else:
        card_payload = load_semantic_card_payload(card_path)
        task_spec = card_payload.get("task_spec", {})
        dataset_name = (
            task_spec.get("dataset", "semantic_cards")
            if isinstance(task_spec, dict)
            else "semantic_cards"
        )
        model_label = slugify_output_part(embedding_config.model, fallback="embedding")
        dataset_label = slugify_output_part(dataset_name, fallback="dataset")
        output_dir = build_unique_output_dir(
            ROOT
            / "outputs"
            / "semantic_artifacts"
            / dataset_label,
            f"{dataset_label}_{model_label}_k{output_dim_label}",
        )
    build_semantic_field_artifact_from_cards(
        card_path=card_path,
        output_dir=output_dir,
        embedding_config=embedding_config,
        semantic_dim=semantic_dim,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
