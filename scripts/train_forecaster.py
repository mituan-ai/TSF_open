from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import prepare_script_runtime


ROOT = prepare_script_runtime(__file__)

from tsf.training.config import load_experiment_config  # noqa: E402
from tsf.training.runner import run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TSF forecasting experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an experiment YAML config.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional short-run override for training.max_epochs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional override for training.device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for training.seed.",
    )
    parser.add_argument(
        "--run-name-prefix",
        type=str,
        default=None,
        help="Optional prefix for the generated run directory name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_experiment_config(config_path, root=ROOT)
    overrides: dict[str, object] = {}
    if args.max_epochs is not None:
        overrides["training.max_epochs"] = args.max_epochs
    if args.device is not None:
        overrides["training.device"] = args.device
    if args.seed is not None:
        overrides["training.seed"] = args.seed
    if args.run_name_prefix is not None:
        overrides["output.run_name_prefix"] = args.run_name_prefix
    run_dir = run_experiment(config, overrides=overrides)
    print(run_dir)


if __name__ == "__main__":
    main()
