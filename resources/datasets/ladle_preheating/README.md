# Ladle Preheating Multi-Step Temperature Forecasting

This directory contains the public, processed NumPy bundle for the ladle preheating forecasting task. It includes window-level arrays only, not private plant records, spreadsheets, or intermediate CSV exports.

## Task

- Dataset key: `ladle_preheating`
- Input: `60` historical process steps
- Target: future 5-step temperature sequence
- Samples: `19840`
- Features: `14`
- Target dimension: `5`
- Public data file: `windows.npz`

Historical temperature is a legal input feature inside the observed window. Future temperatures are labels only.

## Public Bundle

`windows.npz` stores:

| Array | Shape | Description |
| --- | --- | --- |
| `windows` | `(19840, 60, 14)` | Window-level float32 input tensor |
| `targets` | `(19840, 5)` | Float32 5-step temperature target |
| `sample_metadata_json` | `(19840,)` | JSON rows with process-level split metadata |
| `metadata_json` | scalar | Feature names, target names, window size, and export metadata |

The canonical feature order is:

```text
时间间隔, 煤气阀门开度, CO2流量, O2流量, CO流量, N2流量,
煤气压力, 空气阀门开度, 空气O2流量, 空气N2流量,
空气CO2流量, 空气压力, 燃烧效率指标, 温度
```

## Evaluation Protocol

The split is process-level: entire processes are held out rather than mixing windows from the same process across train and test.

| Group | Processes | Samples |
| --- | --- | ---: |
| Train pool | all valid processes except `13`, `23`, and `33` | `17736` |
| Test | `process_13`, `process_23`, `process_33` | `2104` |

The default validation processes are:

```text
process_02, process_06, process_11, process_26, process_29
```

## Notes

- Training scripts read `windows.npz` directly through `tsf.data.load_window_dataset`.
- `task_spec.json` is the task protocol used to generate variable semantic cards.
- `VARIABLES.md` provides the public variable table used for method and data documentation.
