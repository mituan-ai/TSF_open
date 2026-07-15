# Tennessee Eastman Process Evaluation Protocol

TSF separates both simulation trajectories and operating conditions. Windows from the same trajectory never appear in both development and test groups.

| Group | Fault / condition IDs | Simulation runs | Samples | Interpretation |
| --- | --- | ---: | ---: | --- |
| `train` | `0, 8–12` | `1–30` | `16,380` | Normal operation and random variations used for fitting |
| `validation` | `0, 8–12` | `31–35` | `2,730` | Independent model-selection trajectories |
| `same_family` | `0` | `36–50` | `1,365` | Unseen normal-operation trajectories |
| `near` | `8–12` | `36–50` | `6,825` | Unseen trajectories from known variation families |
| `far` | `1–7, 13–20` | `36–50` | `20,475` | Operating conditions absent from training |

## Condition Families

- `0`: normal operation
- `1–7`: step changes in feed ratios, compositions, temperatures, or supply conditions
- `8–12`: random variations in feed composition and cooling temperatures
- `13`: slow reaction-kinetics drift
- `14–15`: cooling-valve sticking
- `16–20`: disturbances without a public mechanistic description

Condition IDs, names, and domain groups are evaluation metadata. They are not model inputs and do not enter the variable semantic cards.
