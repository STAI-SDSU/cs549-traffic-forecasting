# Project Description

## Prediction Task

Given the previous 12 five-minute traffic speed observations (60 minutes of
history) for a sensor, predict the next 12 five-minute observations (60
minutes ahead) for that same sensor. This is repeated across 207 sensors on
the LA highway network (METR-LA).

```text
X: [12 history steps, num_sensors]  -->  model  -->  Y: [12 future steps, num_sensors]
```

See [`DATA_DESCRIPTION.md`](DATA_DESCRIPTION.md) for exact array shapes and
file formats.

## Expected Experimental Workflow

We recommend working through the project roughly in this order:

```text
Data inspection
  -> baseline models (>= 2)
  -> deep learning model (>= 1)
  -> hyperparameter tuning
  -> validation evaluation
  -> improved final model
  -> test prediction
  -> leaderboard submission
  -> error analysis
```

Start with [`starter_code/load_data.py`](starter_code/load_data.py) and
[`examples/getting_started.ipynb`](examples/getting_started.ipynb) to get
oriented, then read through both baseline scripts in
[`baselines/`](baselines/) before writing your own model — they show the
correct pattern for handling `valid_mask`, scoring on validation, and
producing a submission.

## Final Report Expectations

Your final report should include:

* **Problem description** — restate the forecasting task in your own words.
* **Data processing** — how you loaded, cleaned, normalized, and/or
  engineered features from the released data.
* **Model descriptions**, covering:
  * at least two baseline models;
  * at least one deep learning model;
  * your final improved model.
* **Experimental setup** — training details, hyperparameters, hardware,
  validation strategy.
* **Quantitative model comparison** — a table of MAE (and optionally RMSE /
  MAPE) for every model you built, on the validation set.
* **Validation metrics** — MAE, and any additional metrics you find useful.
* **Forecasting-horizon analysis** — how does error change from the 5-minute
  prediction (h01) out to the 60-minute prediction (h12)? Does it grow
  linearly, saturate, or behave differently for different models?
* **Error analysis** — where does your model struggle? Consider specific
  sensors, times of day, congestion vs. free-flow conditions, or specific
  horizons.
* **Discussion of limitations** — what would you try next with more time or
  data?
* **External-code / AI-use disclosure** — list any external code, pretrained
  models, or AI tools you used, and how (see [`RULES.md`](RULES.md)).
* **Conclusions.**

Explain **why** your final model performs better (or worse) than your
baselines — a leaderboard rank alone, without analysis, is not sufficient.
