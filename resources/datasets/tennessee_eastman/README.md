# Tennessee Eastman Process Product-Composition Soft Sensing

This directory contains the processed NumPy bundle used by the public TSF example for the Tennessee Eastman Process (TEP). The task estimates the delayed product-stream component G concentration from online process measurements and manipulated variables.

## Task

- Dataset key: `tennessee_eastman`
- Input: `20` online samples at 3-minute resolution (`60` minutes total)
- Features: `22` continuous process measurements and `11` manipulated variables
- Target: product-stream component G concentration, `XMEAS(40)`, in mole percent
- Analyzer delay: `15` minutes
- Samples: `47,775`
- Public data file: `windows.npz`

Composition analyzers `XMEAS(23:41)`, disturbance labels, future measurements, and test statistics are excluded from the model input.

## Public Source

The bundle is derived from the extended TEP simulations released by Rieth et al.:

- Dataset: [Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation](https://doi.org/10.7910/DVN/6C3JR1)
- Repository: Harvard Dataverse
- Base sampling interval: 3 minutes

The original files are not redistributed here. `windows.npz` contains only task-specific, window-level arrays and protocol metadata.

## Public Bundle

| Array | Shape | Description |
| --- | --- | --- |
| `windows` | `(47775, 20, 33)` | Float32 online input windows |
| `targets` | `(47775,)` | Float32 component G targets |
| `sample_metadata_json` | `(47775,)` | Split, condition, run, and timing metadata |
| `metadata_json` | scalar | Feature order, task definition, source, and split policy |

## Evaluation Protocol

Independent simulation trajectories are separated before training. Condition identifiers are used only for protocol grouping and evaluation.

| Group | Conditions | Runs | Samples |
| --- | --- | ---: | ---: |
| `train` | normal operation and random variations `0, 8–12` | `1–30` | `16,380` |
| `validation` | normal operation and random variations `0, 8–12` | `31–35` | `2,730` |
| `same_family` | normal operation `0` | `36–50` | `1,365` |
| `near` | random variations `8–12` | `36–50` | `6,825` |
| `far` | unseen conditions `1–7, 13–20` | `36–50` | `20,475` |

See [SCENARIOS.md](SCENARIOS.md) for the operating-condition groups and [VARIABLES.md](VARIABLES.md) for the input definition.

## Notes

- Training scripts read `windows.npz` through `tsf.data.load_window_dataset`.
- `task_spec.json` defines the legal information boundary used to generate variable semantic cards.
- The target report occurs 15 minutes after the online input window ends.
