#!/usr/bin/env python3
"""Baseline 1: per-sensor linear regression.

For each of the 207 sensors independently, fits a linear regression that
maps the 12 history steps to the 12 future steps for that sensor:

    Y[:, :, sensor] = LinearRegression().fit(X[:, :, sensor], Y[:, :, sensor])

This is a simple traditional ML baseline: it actually learns from data, but
leaves plenty of room to do better with models that share information across
sensors (random forests, gradient boosting, MLPs, RNNs, GNNs, ...).

Run:
    python baselines/baseline_1_linear_regression.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from starter_code.load_data import load_split  # noqa: E402
from starter_code.evaluate_validation import evaluate_predictions  # noqa: E402
from starter_code.create_submission import build_submission  # noqa: E402


def fit_per_sensor_models(X_train: np.ndarray, Y_train: np.ndarray, valid_mask: np.ndarray) -> list[LinearRegression]:
    """X_train, Y_train: [num_samples, 12, num_sensors]. Returns one fitted model per sensor.

    Rows where `valid_mask[:, s]` is False overlapped a long missing-data gap
    for that sensor (X/Y contain NaN there) and are excluded from that
    sensor's training set.
    """
    num_sensors = X_train.shape[2]
    models = []
    for s in range(num_sensors):
        rows = valid_mask[:, s]
        model = LinearRegression()
        model.fit(X_train[rows, :, s], Y_train[rows, :, s])
        models.append(model)
    return models


def predict_per_sensor(models: list[LinearRegression], X: np.ndarray) -> np.ndarray:
    """Predicts for every row.

    Rows whose input overlapped a long gap are filled with 0.0 purely so
    `.predict()` (which rejects NaN) doesn't raise; those specific
    predictions are never used, since sample_submission.csv never asks for
    an excluded (sample, sensor) pair.
    """
    num_samples, _, num_sensors = X.shape
    forecast_horizon = models[0].coef_.shape[0]
    preds = np.empty((num_samples, forecast_horizon, num_sensors), dtype=np.float32)
    X_safe = np.nan_to_num(X, nan=0.0)
    for s, model in enumerate(models):
        preds[:, :, s] = model.predict(X_safe[:, :, s])
    return preds


def main() -> None:
    print("Baseline 1: Per-sensor Linear Regression\n")

    train = load_split("train")
    models = fit_per_sensor_models(train["X"], train["Y"], train["valid_mask"])
    print(f"Fitted {len(models)} independent linear regression models (one per sensor).")

    val = load_split("validation")
    val_pred = predict_per_sensor(models, val["X"])
    print("\nValidation performance:")
    evaluate_predictions(val_pred)

    test = load_split("test_features")
    test_pred = predict_per_sensor(models, test["X"])
    build_submission(test_pred, out_path="baseline_1_submission.csv")


if __name__ == "__main__":
    main()
