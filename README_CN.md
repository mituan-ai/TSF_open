# TSF

[English](README.md) | [中文](README_CN.md)

**面向工业时间序列预测与软测量的开源方法实现。**

TSF 提供可运行代码、处理后的 NumPy 数据包、预计算语义资源，以及工业预测和软测量流程中的 GRU / TSF-GRU 示例配置。本仓库面向希望运行、检查和改造该方法的用户，不要求重新构建私有数据处理流程。

## 主要内容

- 三个工业预测或软测量任务的公开处理数据包
- 基线 GRU 与 TSF-GRU 的最小可运行配置
- 训练、验证、测试流程和指标导出脚本
- 已预计算的语义资源，运行公开示例不需要调用外部 API
- 数据读取、模型构建、语义资源加载和训练 smoke test

## 快速开始

```bash
git clone https://github.com/mituan-ai/TSF_open.git
cd TSF_open
uv sync --extra train --extra dev
uv run --extra train python scripts/train_forecaster.py \
  --config configs/experiments/indpensim_gru_tsf.yaml \
  --max-epochs 1 \
  --device cpu
```

运行结果会写入 `outputs/runs/`。该目录由 git 忽略，适合保存本地实验输出。

## 环境要求

| 组件 | 要求 |
| --- | --- |
| Python | `>=3.14` |
| 包管理器 | 推荐 `uv` |
| 基础依赖 | `numpy` |
| 训练依赖 | 通过 `--extra train` 安装 `torch`、`scikit-learn`、`transformers`、`kernels` |
| 测试依赖 | 通过 `--extra dev` 安装 `pytest`、`matplotlib` |
| GPU | 可选；有 CUDA 环境时使用 `--device cuda` |

## 数据集

每个数据集目录都包含处理后的 `windows.npz`、任务协议 `task_spec.json` 和面向读者的数据说明。

| 数据集 | 任务 | 输入形状 | 目标形状 | 评估划分 |
| --- | --- | --- | --- | --- |
| `indpensim` | 青霉素离线浓度软测量 | `(818, 120, 23)` | `(818,)` | `train/same_family/near/far` batches |
| `thickener_dewatering` | 底流浓度软测量 | `(5149, 30, 5)` | `(5149,)` | `train/near/far` scenarios |
| `ladle_preheating` | 5 步温度预测 | `(19840, 60, 14)` | `(19840, 5)` | process-level holdout |

`windows.npz` 中的主要数组：

| 数组 | 说明 |
| --- | --- |
| `windows` | float32 输入窗口，形状为 `(samples, time, features)` |
| `targets` | float32 预测目标 |
| `sample_metadata_json` | 每个样本对应的一行 JSON 元数据 |
| `metadata_json` | 特征名、目标名和公开数据包元数据 |

更详细的数据边界和划分说明见：

- `resources/datasets/indpensim/README.md`
- `resources/datasets/thickener_dewatering/README.md`
- `resources/datasets/ladle_preheating/README.md`

## 训练示例

运行基线 GRU：

```bash
uv run --extra train python scripts/train_forecaster.py \
  --config configs/experiments/indpensim_gru.yaml \
  --device cuda
```

运行 TSF-GRU：

```bash
uv run --extra train python scripts/train_forecaster.py \
  --config configs/experiments/indpensim_gru_tsf.yaml \
  --device cuda
```

可用配置：

```text
configs/experiments/
├── indpensim_gru.yaml
├── indpensim_gru_tsf.yaml
├── ladle_preheating_gru.yaml
├── ladle_preheating_gru_tsf.yaml
├── thickener_dewatering_gru.yaml
└── thickener_dewatering_gru_tsf.yaml
```

训练输出包括配置快照、归一化统计、数据划分摘要、指标、预测结果、日志和 checkpoint。

## 语义资源

公开示例配置默认使用仓库内已经生成好的语义资源：

```text
resources/semantic_artifacts/<dataset>/
├── directions.npy
└── semantic_field_metadata.json
```

因此，直接运行公开示例不需要 API key。只有在你要为新任务重新生成语义卡片或重建语义资源时，才需要配置本地 API：

```bash
cp configs/env/api.env configs/env/api.local.env
```

然后在 `configs/env/api.local.env` 中填写本地凭据。该文件已被 git 忽略。

生成新任务的语义卡片提示包：

```bash
uv run python scripts/build_semantic_cards.py \
  --task-spec resources/datasets/indpensim/task_spec.json
```

## 项目结构

```text
src/tsf/
├── data/                 # NPZ 数据读取、划分与归一化
├── methods/              # TSF 输入适配层
├── models/               # 时间序列 backbone 与预测模型封装
├── training/             # 配置解析、训练循环与指标
├── llm_semantics.py      # 语义卡片校验与资源构建
├── semantic_field.py     # 语义资源加载与 NumPy 工具
└── task_schema.py        # task_spec.json 协议与提示包构建
```

## 开发与测试

```bash
uv sync --extra dev --extra train
uv run pytest tests/test_window_npz_data.py tests/test_semantic_field.py tests/test_embedding_config.py -q
uv run --extra train pytest tests/test_forecaster_models.py tests/test_training_runner_smoke.py -q
```

## 引用

如果本仓库对你的研究有帮助，请引用配套论文。最终出版信息会在 `CITATION.cff` 中更新。

```bibtex
@article{tsf2026,
  title = {LLM-Guided Task-Semantic Field Factorization for Industrial Time-Series Forecasting},
  author = {TSF Authors},
  year = {2026},
  note = {Manuscript citation metadata to be updated after publication}
}
```

## 许可

本仓库使用 MIT License。详见 `LICENSE`。
