"""Evaluation metrics for the CS549 traffic forecasting final project.

The leaderboard ranks submissions by MAE only. RMSE and MAPE are provided
alongside it for your own diagnostics and for the final report.
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error. This is the official leaderboard metric."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Mean Absolute Percentage Error, masking near-zero ground truth.

    MAPE is undefined when y_true == 0 and unstable when y_true is close to
    zero. Rather than adding a flat epsilon to the denominator (which
    silently distorts the metric for every row), rows with |y_true| <
    epsilon are excluded from the MAPE computation entirely. `epsilon`
    defaults to 1.0 mph.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= epsilon
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, mape_epsilon: float = 1.0) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred, epsilon=mape_epsilon),
    }
