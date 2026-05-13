# Thickener Dewatering Soft Sensing

This directory contains the public, processed NumPy bundle for the thickener/dewatering soft-sensing task. It includes window-level arrays only, not simulation code or intermediate CSV exports.

## Task

- Dataset key: `thickener_dewatering`
- Input: `30` minute historical window ending at the target minute
- Target: current `underflow_concentration`
- Samples: `5149`
- Features: `5`
- Target dimension: `1`
- Public data file: `windows.npz`

The input boundary excludes current or historical `underflow_concentration` from the input window.

## Public Bundle

`windows.npz` stores:

| Array | Shape | Description |
| --- | --- | --- |
| `windows` | `(5149, 30, 5)` | Window-level float32 input tensor |
| `targets` | `(5149,)` | Float32 regression target |
| `sample_metadata_json` | `(5149,)` | JSON rows with split and scenario metadata |
| `metadata_json` | scalar | Feature names, target names, window size, and export metadata |

The canonical feature order is:

```text
q_in, p2, p3, phase_pressurizing, phase_discharging
```

## Evaluation Protocol

| Group | Scenarios | Samples |
| --- | --- | ---: |
| `train` | `train_jitter_1` to `train_jitter_6` | `1626` |
| `near` | `near_train_like_jitter` | `271` |
| `far` | 12 load, sensor, actuator, delay, and restart-shift scenarios | `3252` |

The default validation scenario is `train_jitter_6`; near and far scenarios are held out for evaluation.

## Notes

- Training scripts read `windows.npz` directly through `tsf.data.load_window_dataset`.
- `task_spec.json` is the task protocol used to generate variable semantic cards.
- `SCENARIOS.md` describes the scenario protocol and the held-out domain buckets.
