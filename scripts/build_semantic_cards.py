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
from tsf.llm_semantics import (
    LLMGenerationConfig,
    generate_semantic_cards_with_llm,
    write_prompt_package,
)
from tsf.task_schema import load_task_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an offline TSF semantic-card prompt package and optionally call an LLM API."
    )
    parser.add_argument("--task-spec", required=True, help="Dataset task-spec JSON path.")
    parser.add_argument("--llm-config", default="configs/llm.yaml", help="LLM YAML config path.")
    parser.add_argument(
        "--output-dir",
        help=(
            "Output directory for prompt and semantic cards. If omitted, a readable "
            "directory is created under outputs/semantics/<dataset>/."
        ),
    )
    parser.add_argument(
        "--call-api",
        action="store_true",
        help="Actually call the configured LLM API. Without this flag, only prompt files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_spec_path = resolve_reference_path(
        args.task_spec,
        config_path=ROOT / args.task_spec,
        root=ROOT,
    )
    llm_config_path = resolve_reference_path(
        args.llm_config,
        config_path=ROOT / args.llm_config,
        root=ROOT,
    )
    task_spec = load_task_spec(task_spec_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (ROOT / output_dir).resolve()
    else:
        llm_label = "prompt_only"
        if args.call_api:
            load_api_env_files(ROOT)
            llm_payload = load_simple_yaml_mapping(llm_config_path)
            llm_label = slugify_output_part(llm_payload.get("model", "llm"), fallback="llm")
        output_dir = build_unique_output_dir(
            ROOT / "outputs" / "semantics" / slugify_output_part(task_spec.dataset, fallback="dataset"),
            llm_label,
        )
    write_prompt_package(task_spec, output_dir)
    if args.call_api:
        load_api_env_files(ROOT)
        llm_config = LLMGenerationConfig.from_mapping(load_simple_yaml_mapping(llm_config_path))
        generate_semantic_cards_with_llm(
            task_spec=task_spec,
            llm_config=llm_config,
            output_path=output_dir / "semantic_cards.json",
        )
        print("wrote semantic_cards.json")
    print(output_dir)


if __name__ == "__main__":
    main()
