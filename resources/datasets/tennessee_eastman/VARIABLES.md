# Tennessee Eastman Process Variables

Each input window contains 33 online variables sampled every 3 minutes. The model receives 20 samples, covering the 60 minutes before the product sample. The product analyzer reports `XMEAS(40)` 15 minutes later.

## Continuous Measurements

| Range | Count | Process role |
| --- | ---: | --- |
| `XMEAS(1:6)` | 6 | Feed and recycle flow measurements |
| `XMEAS(7:9)` | 3 | Reactor pressure, level, and temperature |
| `XMEAS(10:14)` | 5 | Purge and separator measurements |
| `XMEAS(15:19)` | 5 | Stripper state, product flow, and steam flow |
| `XMEAS(20:22)` | 3 | Compressor and cooling-water measurements |

## Manipulated Variables

`XMV(1:11)` are valve positions for feed, compressor recycle, purge, separator underflow, product flow, steam, and cooling-water control.

## Target and Exclusions

- Target: `XMEAS(40)`, product-stream component G concentration in mole percent.
- Excluded: all composition analyzers `XMEAS(23:41)` from the input.
- Excluded: condition labels, future values, analyzer outputs, and test statistics.

The canonical machine-readable descriptions and units are stored in `task_spec.json`.
