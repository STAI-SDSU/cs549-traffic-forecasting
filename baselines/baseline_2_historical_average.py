#!/usr/bin/env python3
"""Baseline 2: historical (time-of-day / weekend-vs-weekday) average.

For each sensor, predicts the average observed speed at that same
5-minute-of-day, weekday-vs-weekend combination, using ONLY the training
split. This captures the strong daily commute pattern without any sequence
modeling at all -- a useful reference point below which a "real" model
should not fall.

Run:
    python baselines/baseline_2_historical_average.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from starter_code.load_data import load_split  # noqa: E402
from starter_code.evaluate_validation import evaluate_predictions  # noqa: E402
from starter_code.create_submission import build_submission  # noqa: E402

INTERVAL = np.timedelta64(5, "m")
HISTORY_LENGTH = 12
FORECAST_HORIZON = 12


def _step_offsets(n_steps: int, sign: int) -> np.ndarray:
    """sign=-1 for history steps ending at 0, sign=+1 for future steps starting at 1."""
    if sign < 0:
        return np.arange(-(n_steps - 1), 1) * INTERVAL
    return np.arange(1, n_steps + 1) * INTERVAL


def build_lookup_table(train: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Reconstruct the (deduplicated) train-period time series and average by bucket."""
    X, Y, origins, sensor_ids = train["X"], train["Y"], train["timestamps"], train["sensor_ids"]
    n, H, S = X.shape
    x_ts = origins[:, None] + _step_offsets(H, sign=-1)[None, :]
    y_ts = origins[:, None] + _step_offsets(FORECAST_HORIZON, sign=+1)[None, :]

    all_ts = np.concatenate([x_ts.reshape(-1), y_ts.reshape(-1)])
    all_vals = np.concatenate([X.reshape(-1, S), Y.reshape(-1, S)], axis=0)

    series = pd.DataFrame(all_vals, index=pd.to_datetime(all_ts), columns=sensor_ids)
    series = series[~series.index.duplicated(keep="first")].sort_index()

    minute_of_day = series.index.hour * 60 + series.index.minute
    is_weekend = series.index.dayofweek >= 5
    lookup_table = series.groupby([is_weekend, minute_of_day]).mean()  # pandas .mean() skips NaN by default
    lookup_table.index.names = ["is_weekend", "minute_of_day"]

    fallback_mean = series.mean(axis=0)
    return lookup_table, fallback_mean


def predict_from_table(origins: np.ndarray, lookup_table: pd.DataFrame, fallback_mean: pd.Series) -> np.ndarray:
    n = len(origins)
    S = lookup_table.shape[1]
    preds = np.empty((n, FORECAST_HORIZON, S), dtype=np.float32)
    y_offsets = _step_offsets(FORECAST_HORIZON, sign=+1)
    for h in range(FORECAST_HORIZON):
        target_ts = pd.to_datetime(origins + y_offsets[h])
        keys = list(zip(target_ts.dayofweek >= 5, target_ts.hour * 60 + target_ts.minute))
        rows = lookup_table.reindex(keys)
        rows = rows.fillna(fallback_mean)
        preds[:, h, :] = rows.to_numpy()
    return preds


def main() -> None:
    print("Baseline 2: Historical Average (time-of-day / weekend)\n")

    train = load_split("train")
    lookup_table, fallback_mean = build_lookup_table(train)
    print(f"Lookup table computed from the training split only: {lookup_table.shape[0]} (weekend, minute-of-day) buckets")

    val = load_split("validation")
    val_pred = predict_from_table(val["timestamps"], lookup_table, fallback_mean)
    print("\nValidation performance:")
    evaluate_predictions(val_pred)

    test = load_split("test_features")
    test_pred = predict_from_table(test["timestamps"], lookup_table, fallback_mean)
    build_submission(test_pred, out_path="baseline_2_submission.csv")


if __name__ == "__main__":
    main()
