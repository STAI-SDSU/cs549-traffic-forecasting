#!/usr/bin/env python3
"""Turn a prediction tensor into a valid Kaggle submission.csv.

`data/sample_submission.csv` is the source of truth for which Ids are
required -- some (sample, sensor) pairs were excluded from scoring because
of long missing-data intervals in that sensor's history, so it does NOT ask
for a prediction for every sensor on every test sample. You do not need to
reconstruct this yourself: this script reads the required Ids straight from
sample_submission.csv and pulls the matching value out of your prediction
tensor.

Run directly for a working demo (builds a submission from a naive
persistence prediction on the test set):

    python starter_code/create_submission.py

Or import `build_submission` from your own code once you have a real
prediction tensor of shape [num_test_samples, 12, num_sensors]:

    from starter_code.create_submission import build_submission
    build_submission(my_test_predictions)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from starter_code.load_data import DATA_DIR, load_split  # noqa: E402

ID_PATTERN = re.compile(r"^(sample_\d+)_sensor_(\d+)_h(\d+)$")
PLAUSIBLE_SPEED_RANGE = (0.0, 100.0)  # mph; generous sanity-check bounds


class InvalidSubmissionError(Exception):
    pass


def build_submission(predictions: np.ndarray, out_path: str | Path = "submission.csv") -> pd.DataFrame:
    """predictions: [num_test_samples, 12, num_sensors], same order as test_features.npz's sample_id.

    Looks up every Id required by data/sample_submission.csv, reads the
    corresponding value out of `predictions`, validates the result, and
    writes `out_path`.
    """
    test = load_split("test_features")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

    forecast_horizon = predictions.shape[1]
    num_sensors = predictions.shape[2]
    if predictions.shape[0] != len(test["sample_id"]):
        raise InvalidSubmissionError(
            f"predictions has {predictions.shape[0]} samples, expected {len(test['sample_id'])}"
        )

    sample_row = {sid: i for i, sid in enumerate(test["sample_id"])}

    parsed = sample_submission["Id"].str.extract(ID_PATTERN)
    if parsed.isna().any().any():
        bad = sample_submission["Id"][parsed.isna().any(axis=1)].iloc[0]
        raise InvalidSubmissionError(f"unrecognized Id format in sample_submission.csv, e.g. {bad!r}")

    sample_ids = parsed[0].to_numpy()
    sensor_idx = parsed[1].to_numpy(dtype=int)
    horizon = parsed[2].to_numpy(dtype=int)

    try:
        row_idx = np.array([sample_row[s] for s in sample_ids])
    except KeyError as exc:
        raise InvalidSubmissionError(f"sample_submission.csv references unknown sample id {exc}") from exc

    if (sensor_idx >= num_sensors).any() or (horizon < 1).any() or (horizon > forecast_horizon).any():
        raise InvalidSubmissionError("sample_submission.csv references a sensor/horizon out of range for `predictions`")

    values = predictions[row_idx, horizon - 1, sensor_idx]

    submission = pd.DataFrame({"Id": sample_submission["Id"], "Prediction": values})
    _validate(submission, sample_submission)

    out_path = Path(out_path)
    submission.to_csv(out_path, index=False)
    print(f"Wrote {len(submission)}-row submission to {out_path}")
    return submission


def _validate(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["Id", "Prediction"]:
        raise InvalidSubmissionError(f"submission must have exactly the columns ['Id', 'Prediction'], got {list(submission.columns)}")

    if len(submission) != len(sample_submission):
        raise InvalidSubmissionError(f"submission has {len(submission)} rows, expected exactly {len(sample_submission)}")

    if not (submission["Id"].to_numpy() == sample_submission["Id"].to_numpy()).all():
        raise InvalidSubmissionError("submission Ids do not exactly match sample_submission.csv, in order")

    values = submission["Prediction"].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise InvalidSubmissionError(f"submission contains {int(np.isnan(values).sum())} NaN prediction(s)")
    if np.isinf(values).any():
        raise InvalidSubmissionError(f"submission contains {int(np.isinf(values).sum())} infinite prediction(s)")

    lo, hi = PLAUSIBLE_SPEED_RANGE
    out_of_range = int(((values < lo) | (values > hi)).sum())
    if out_of_range > 0:
        print(f"WARNING: {out_of_range} prediction(s) fall outside the plausible speed range [{lo}, {hi}] mph.")

    print("Submission is valid: correct row count, correct/ordered Ids, no NaN/Inf values.")


def _persistence_demo() -> np.ndarray:
    """Naive baseline used only to demonstrate this script end-to-end."""
    test = load_split("test_features")
    last_observed = test["X"][:, -1:, :]
    forecast_horizon = 12
    return np.repeat(last_observed, forecast_horizon, axis=1)


def main() -> None:
    print("No predictions supplied -- demonstrating with a naive persistence baseline "
          "(predict the last observed value for every future step).\n")
    predictions = _persistence_demo()
    build_submission(predictions, out_path="submission.csv")


if __name__ == "__main__":
    main()
