#!/usr/bin/env python3
"""Load the released train / validation / test data.

Run directly to print the shape and dtype of every array in each file:

    python starter_code/load_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_split(name: str) -> dict:
    """Load one of "train", "validation", "test_features" as a dict of arrays.

    train.npz / validation.npz contain: X, Y, valid_mask, timestamps,
    sensor_ids, sample_id.

    test_features.npz contains the same fields EXCEPT Y -- the test targets
    are never released. That is exactly what your model must predict.
    """
    path = DATA_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- did you download the released data into data/?")
    return dict(np.load(path, allow_pickle=False))


def main() -> None:
    for name in ["train", "validation", "test_features"]:
        data = load_split(name)
        print(f"\n{name}.npz:")
        for key, value in data.items():
            print(f"    {key:12s} shape={value.shape} dtype={value.dtype}")

    train = load_split("train")
    print("\nExample: X[sample, timestep, sensor] holds the previous 60 minutes of speed (mph).")
    print(f"  train['X'].shape  = {train['X'].shape}  ->  "
          f"({train['X'].shape[0]} samples, {train['X'].shape[1]} history steps, {train['X'].shape[2]} sensors)")
    print(f"  train['Y'].shape  = {train['Y'].shape}  ->  next {train['Y'].shape[1]} steps to predict")
    print("  valid_mask[i, s] == False means sensor s's input or target for sample i overlapped a long")
    print("  missing-data gap; X/Y contain NaN at exactly those (sample, sensor) entries. Exclude them")
    print("  from training and from your own validation scoring (see starter_code/evaluate_validation.py).")


if __name__ == "__main__":
    main()
