# IndPenSim Offline Penicillin Soft Sensing

This directory contains the public, processed NumPy bundle used by the TSF open-source method implementation. It does not include raw simulator tables or intermediate CSV exports.

## Task

- Dataset key: `indpensim`
- Input: `120` historical online process steps at 12-minute resolution
- Target: current offline penicillin concentration, `penicillin_offline`
- Samples: `818`
- Features: `23`
- Target dimension: `1`
- Public data file: `windows.npz`

The input boundary excludes Raman spectra, online penicillin estimates, offline assay side variables, future labels, and hidden control annotations.

## Public Bundle

`windows.npz` stores:

| Array | Shape | Description |
| --- | --- | --- |
| `windows` | `(818, 120, 23)` | Window-level float32 input tensor |
| `targets` | `(818,)` | Float32 regression target |
| `sample_metadata_json` | `(818,)` | JSON rows with split and scenario metadata |
| `metadata_json` | scalar | Feature names, target names, window size, and export metadata |

The canonical feature order is:

```text
time_h, aeration, agitator_rpm, sugar_feed, acid, base,
cooling_water, heating_water, wfi, air_head_pressure,
dumped_broth, do2, volume, weight, ph, temp, co2_out,
paa_flow, oil_flow, our, o2_out, cer, ammonia_shots
```

## Evaluation Protocol

| Group | Scenarios | Samples |
| --- | --- | ---: |
| `train` | `train_recipe_001` to `train_recipe_029` | `541` |
| `same_family` | `same_family_recipe_030` | `19` |
| `near` | `near_operator_040`, `near_operator_032` | `38` |
| `far` | `far_apc_090`, `far_apc_067`, `far_fault_091` to `far_fault_100` | `220` |

The default validation scenarios are:

```text
train_recipe_005, train_recipe_011, train_recipe_017,
train_recipe_023, train_recipe_029
```

## Notes

- Training scripts read `windows.npz` directly through `tsf.data.load_window_dataset`.
- `task_spec.json` is the task protocol used to generate variable semantic cards.
- `SCENARIOS.md` records the public scenario protocol for reader inspection.
