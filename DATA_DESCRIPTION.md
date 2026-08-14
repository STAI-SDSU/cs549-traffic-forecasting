# Data Description

All released data lives in `data/`. This document describes every file.
Loading examples are in [`starter_code/load_data.py`](starter_code/load_data.py).

## Task recap

* History window: 12 time steps (60 minutes, sampled every 5 minutes).
* Forecast horizon: 12 time steps (60 minutes ahead).
* 207 sensors, each with its own speed history/forecast.
* Target variable: traffic speed in mph.

## `train.npz` / `validation.npz`

```python
import numpy as np
data = np.load("data/train.npz", allow_pickle=False)

X          = data["X"]           # [num_samples, 12, num_sensors] previous 60 min of speed (mph)
Y          = data["Y"]           # [num_samples, 12, num_sensors] next 60 min of speed (mph)
valid_mask = data["valid_mask"]  # [num_samples, num_sensors] bool
timestamps = data["timestamps"]  # forecast-origin timestamp for each sample
sensor_ids = data["sensor_ids"]  # sensor id string for each of the num_sensors columns
sample_id  = data["sample_id"]   # e.g. "sample_000001"
```

`X.shape == (num_samples, 12, num_sensors)`, `Y.shape == (num_samples, 12, num_sensors)`.

**`validation.npz` has the exact same fields, including `Y`.** You are
expected to use validation labels freely for model development, model
comparison, and hyperparameter tuning — see
[`starter_code/evaluate_validation.py`](starter_code/evaluate_validation.py).

### Missing data: `valid_mask`

A small fraction of sensor readings are missing for long stretches (multiple
hours to multiple days). Short gaps were filled in before this data was
released; long gaps (over about an hour) were **not** filled in with
fabricated values.

`valid_mask[i, s]` is `False` when sample `i`'s input or target window for
sensor `s` overlapped one of these long gaps. At exactly those
`(sample, sensor)` positions, `X` and/or `Y` contain `NaN`.

**You must exclude `valid_mask == False` entries from training and from your
own metrics.** For example, to compute a masked MAE by hand:

```python
forecast_horizon = Y.shape[1]
expanded_mask = np.repeat(valid_mask[:, None, :], forecast_horizon, axis=1)
mae = np.mean(np.abs(Y[expanded_mask] - predictions[expanded_mask]))
```

`starter_code/evaluate_validation.py` and both example baselines already do
this correctly — use them as a reference.

## `test_features.npz`

Same fields as above **except there is no `Y`** — the test targets are
never released; this is exactly what you are asked to predict.

```python
test = np.load("data/test_features.npz", allow_pickle=False)
list(test.keys())  # ['X', 'valid_mask', 'timestamps', 'sensor_ids', 'sample_id']
```

## `sample_submission.csv`

This file defines the **exact required prediction Ids and their order**.

> Generate predictions only for the Ids listed in `sample_submission.csv`.

Some sensor / forecast-origin pairs were excluded from the test set because
of long missing-data intervals in that sensor's own history (the same
`valid_mask` concept as above, applied to the test split). **You do not need
to reconstruct the validity mask yourself when creating a submission** —
`sample_submission.csv` already only lists the Ids you need to predict, and
[`starter_code/create_submission.py`](starter_code/create_submission.py)
reads directly from it.

```csv
Id,Prediction
sample_000001_sensor_001_h01,0.0
sample_000001_sensor_001_h02,0.0
...
```

Id format: `{sample_id}_sensor_{sensor_index:03d}_h{horizon:02d}`, where
`sensor_index` is the positional index into the `sensor_ids` array (0-based,
**not** the raw METR-LA sensor id number), and `horizon` is 1-indexed
(`h01`..`h12`).

## `sensor_metadata.csv`

```csv
sensor_index,sensor_id,index,latitude,longitude
sensor_000,773869,0,34.15497,-118.31829
...
```

`sensor_index` matches the `sensor_XXX` label used in submission Ids and the
`sensor_ids` array. Latitude/longitude may be useful if you want to build a
spatially-aware model (e.g. a graph neural network over sensor proximity).

## Building a submission

Use [`starter_code/create_submission.py`](starter_code/create_submission.py):

```python
from starter_code.create_submission import build_submission

# predictions: [num_test_samples, 12, num_sensors], in the same sample order
# as test_features.npz's sample_id array
build_submission(predictions, out_path="submission.csv")
```

It validates row count, Id match/order, and absence of NaN/Inf before
writing `submission.csv` with columns `Id,Prediction`.
