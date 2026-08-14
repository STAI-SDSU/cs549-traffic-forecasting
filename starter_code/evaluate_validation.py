#!/usr/bin/env python3
"""Score predictions against the released validation labels.

This script never touches hidden test labels -- validation.npz is the only
file with labels you have access to, and it is what you should use for
model development and hyperparameter tuning.

Run directly for a working demo (scores a naive "predict the last observed
value" baseline on validation):

    python starter_code/evaluate_validation.py

Or import `evaluate_predictions` from your own training code:

    from starter_code.evaluate_validation import evaluate_predictions
    metrics = evaluate_predictions(my_val_predictions)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from starter_code.load_data import load_split  # noqa: E402
from evaluation.metrics import compute_all_metrics  # noqa: E402

MAPE_EPSILON = 1.0  # mph; ground-truth speeds below this are excluded from MAPE


def evaluate_predictions(y_pred: np.ndarray, mape_epsilon: float = MAPE_EPSILON) -> dict:
    """Scores `y_pred` (shape [num_val_samples, 12, num_sensors]) against validation.npz.

    Only (sample, sensor) pairs where valid_mask is True are scored --
    entries where the target overlapped a long missing-data gap were never
    observed (NaN in Y) and must not be compared against a prediction.
    """
    val = load_split("validation")
    y_true = val["Y"]
    valid_mask = val["valid_mask"]

    if y_pred.shape != y_true.shape:
        raise ValueError(f"y_pred shape {y_pred.shape} does not match validation Y shape {y_true.shape}")

    forecast_horizon = y_true.shape[1]
    expanded_mask = np.repeat(valid_mask[:, None, :], forecast_horizon, axis=1)

    metrics = compute_all_metrics(y_true[expanded_mask], y_pred[expanded_mask], mape_epsilon=mape_epsilon)
    print(f"Scored on {int(expanded_mask.sum())}/{expanded_mask.size} valid entries:")
    print(f"    MAE : {metrics['MAE']:.4f}")
    print(f"    RMSE: {metrics['RMSE']:.4f}")
    print(f"    MAPE: {metrics['MAPE']:.4f}%")
    return metrics


def _persistence_demo() -> np.ndarray:
    """Naive baseline used only to demonstrate this script end-to-end."""
    val = load_split("validation")
    last_observed = val["X"][:, -1:, :]
    forecast_horizon = val["Y"].shape[1]
    return np.repeat(last_observed, forecast_horizon, axis=1)


def main() -> None:
    print("No predictions supplied -- demonstrating with a naive persistence baseline "
          "(predict the last observed value for every future step).\n")
    y_pred = _persistence_demo()
    evaluate_predictions(y_pred)


if __name__ == "__main__":
    main()
