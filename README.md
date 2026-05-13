# TSF

[English](README.md) | [中文](README_CN.md)

**An open-source method implementation for industrial time-series forecasting and soft sensing.**

TSF provides runnable code, processed NumPy data bundles, precomputed semantic resources, and example GRU / TSF-GRU configurations for industrial forecasting and soft-sensing workflows. The repository is designed for users who want to run, inspect, and adapt the method without rebuilding private data pipelines.

## What's Included

- Public processed data bundles for three industrial forecasting or soft-sensing tasks
- Minimal runnable configs for baseline GRU and TSF-GRU experiments
- Training, validation, testing, and metric export utilities
- Precomputed semantic resources, so the released examples do not require external API calls
- Tests for data loading, model construction, semantic-resource loading, and training smoke runs

## Quick Start

```bash
git clone https://github.com/mituan-ai/TSF_open.git
cd TSF_open
uv sync --extra train --extra dev
uv run --extra train python scripts/train_forecaster.py \
  --config configs/experiments/indpensim_gru_tsf.yaml \
  --max-epochs 1 \
  --device cpu
```

The run writes outputs to `outputs/runs/`. This directory is ignored by git and is intended for local experiment artifacts.

## Requirements

| Component | Requirement |
| --- | --- |
| Python | `>=3.14` |
| Package manager | `uv` recommended |
| Core dependency | `numpy` |
| Training dependencies | `torch`, `scikit-learn`, `transformers`, and `kernels` via `--extra train` |
| Test dependencies | `pytest` and `matplotlib` via `--extra dev` |
| GPU | Optional; use `--device cuda` when CUDA is available |

## Datasets

Each dataset directory contains a processed `windows.npz` bundle, a `task_spec.json` task protocol, and reader-facing dataset notes.

| Dataset | Task | Input shape | Target shape | Evaluation split |
| --- | --- | --- | --- | --- |
| `indpensim` | Offline penicillin soft sensing | `(818, 120, 23)` | `(818,)` | `train/same_family/near/far` batches |
| `thickener_dewatering` | Underflow concentration soft sensing | `(5149, 30, 5)` | `(5149,)` | `train/near/far` scenarios |
| `ladle_preheating` | 5-step temperature forecasting | `(19840, 60, 14)` | `(19840, 5)` | process-level holdout |

`windows.npz` stores:

| Array | Description |
| --- | --- |
| `windows` | float32 input windows with shape `(samples, time, features)` |
| `targets` | float32 prediction targets |
| `sample_metadata_json` | one JSON metadata row per sample |
| `metadata_json` | feature names, target names, and public bundle metadata |

Detailed data boundaries and split notes are available in:

- `resources/datasets/indpensim/README.md`
- `resources/datasets/thickener_dewatering/README.md`
- `resources/datasets/ladle_preheating/README.md`

## Training Examples

Run a GRU baseline:

```bash
uv run --extra train python scripts/train_forecaster.py \
  --config configs/experiments/indpensim_gru.yaml \
  --device cuda
```

Run TSF-GRU:

```bash
uv run --extra train python scripts/train_forecaster.py \
  --config configs/experiments/indpensim_gru_tsf.yaml \
  --device cuda
```

Available configs:

```text
configs/experiments/
├── indpensim_gru.yaml
├── indpensim_gru_tsf.yaml
├── ladle_preheating_gru.yaml
├── ladle_preheating_gru_tsf.yaml
├── thickener_dewatering_gru.yaml
└── thickener_dewatering_gru_tsf.yaml
```

Training outputs include config snapshots, normalization statistics, split summaries, metrics, predictions, logs, and checkpoints.

## Semantic Resources

The released example configs use precomputed semantic resources stored in the repository:

```text
resources/semantic_artifacts/<dataset>/
├── directions.npy
└── semantic_field_metadata.json
```

This means the public examples can be reproduced without an API key. You only need local API credentials if you want to generate semantic cards for a new task or rebuild semantic resources:

```bash
cp configs/env/api.env configs/env/api.local.env
```

Fill credentials in `configs/env/api.local.env`. The local file is ignored by git.

Generate a semantic-card prompt package for a task:

```bash
uv run python scripts/build_semantic_cards.py \
  --task-spec resources/datasets/indpensim/task_spec.json
```

## Project Structure

```text
src/tsf/
├── data/                 # NPZ data loading, splits, and normalization
├── methods/              # TSF input adapter
├── models/               # Time-series backbones and forecast-model wrapper
├── training/             # Config parsing, training loop, and metrics
├── llm_semantics.py      # Semantic-card validation and resource building
├── semantic_field.py     # Semantic-resource loading and NumPy utilities
└── task_schema.py        # task_spec.json protocol and prompt-package construction
```

## Development

```bash
uv sync --extra dev --extra train
uv run pytest tests/test_window_npz_data.py tests/test_semantic_field.py tests/test_embedding_config.py -q
uv run --extra train pytest tests/test_forecaster_models.py tests/test_training_runner_smoke.py -q
```

## Citation

If this repository is useful in your work, please cite the accompanying paper. The final publication metadata will be updated in `CITATION.cff`.

```bibtex
@article{tsf2026,
  title = {LLM-Guided Task-Semantic Field Factorization for Industrial Time-Series Forecasting},
  author = {TSF Authors},
  year = {2026},
  note = {Manuscript citation metadata to be updated after publication}
}
```

## License

This repository is released under the MIT License. See `LICENSE`.
