from __future__ import annotations

import math

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    error = pred - true
    mae = float(np.mean(np.abs(error)))
    rmse = float(math.sqrt(np.mean(np.square(error))))
    ss_res = float(np.sum(np.square(error)))
    ss_tot = float(np.sum(np.square(true - true.mean())))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R^2": r2,
    }
