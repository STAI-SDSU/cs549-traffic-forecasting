# CS549 Machine Learning
## Final Project: Traffic Forecasting Challenge
### Fall 2026

Welcome! This repository is the main entry point for the CS549 final
project: a traffic speed forecasting competition built on the METR-LA
sensor dataset.

## Overview

Working in groups of four or five, you will build and compare machine
learning models that forecast highway traffic speed. The task is:

> Given the previous 60 minutes of traffic speed observations, predict
> traffic speeds for the following 60 minutes.

METR-LA contains readings from 207 traffic sensors on the Los Angeles
highway network, sampled every 5 minutes. Concretely:

```text
Input:  previous 12 time steps  (60 minutes of history)
Output: next 12 time steps      (60 minutes ahead)
Target: traffic speed (mph)
```

Your model sees, for each of 207 sensors, a window of 12 past speed
readings, and must output 12 future speed readings for that sensor.

See [`DATA_DESCRIPTION.md`](DATA_DESCRIPTION.md) for the exact file formats,
[`PROJECT_DESCRIPTION.md`](PROJECT_DESCRIPTION.md) for the detailed
assignment and report expectations, and [`RULES.md`](RULES.md) for
collaboration and academic-integrity rules.

## Project Objectives

Over the course of the project you should:

* build and compare multiple machine learning models;
* evaluate models using appropriate metrics;
* analyze forecasting errors;
* improve upon baseline models;
* produce predictions for the hidden test set and submit to the leaderboard.

## Required Models

Each group must implement and evaluate:

1. **At least two baseline models** (starter examples are provided in
   [`baselines/`](baselines/) — see below).
2. **At least one deep learning model.**
3. **One final, improved model** used for your leaderboard submission.

Possible methods for your deep learning and final models include (not an
exhaustive list, and you are not required to use all of them):

```text
Linear Regression        MLP                       LSTM
Random Forest            CNN                       GRU
Gradient Boosting        RNN                       Transformer
XGBoost                  Graph Neural Networks      Ensemble methods
SVR
```

There is no requirement to implement a persistence (naive "last observed
value") model, though you may find it a useful sanity check during
development.

## Evaluation

The primary competition metric is:

```text
Mean Absolute Error (MAE)
```

Lower is better. You may also report RMSE and MAPE in your final report;
`evaluation/metrics.py` computes all three.

## Getting Started

```bash
git clone <this-repository>
cd cs549-traffic-forecasting
pip install -r requirements.txt
```

1. Inspect the data: `python starter_code/load_data.py`
2. Run a baseline end-to-end: `python baselines/baseline_1_linear_regression.py`
3. Walk through [`examples/getting_started.ipynb`](examples/getting_started.ipynb)
   for a guided introduction (loading data, plotting, a first prediction,
   validation scoring, and a submission file).
4. Read [`DATA_DESCRIPTION.md`](DATA_DESCRIPTION.md) for the full data
   format before you start modifying the pipeline.

You do not need to understand or reproduce how the data was generated —
everything you need is already in `data/`.

## Data

Released in `data/`:

```text
train.npz              -- training inputs AND labels
validation.npz         -- validation inputs AND labels
test_features.npz      -- test inputs only (no labels -- this is what's hidden)
sample_submission.csv  -- required submission Ids, in order
sensor_metadata.csv    -- sensor latitude/longitude
```

Full details, array shapes, and the missing-data `valid_mask` are documented
in [`DATA_DESCRIPTION.md`](DATA_DESCRIPTION.md).

## Deliverables

```text
Project Proposal
Code and Model Submission
Leaderboard Prediction Submission
Final Report
```

See [`PROJECT_DESCRIPTION.md`](PROJECT_DESCRIPTION.md) for what each of
these should contain.

## Grading

```text
Project Proposal:             5%
Code and Model Submission:    5%
Final Report:                15%
Leaderboard Performance:      5%
Total Final Project:         30%
```

Leaderboard performance credit is assigned by rank:

```text
Top 10% of teams:                          full credit
>10%–30%:                                  90% credit
>30%–60%:                                  80% credit
Remaining teams with a valid submission:   50% credit
No valid submission:                       0%
```

> **Final leaderboard grades are determined using the private leaderboard,
> not the public leaderboard.** The public leaderboard (visible during the
> competition) is provided only so you can sanity-check your submission
> format and get a rough sense of standing during development.

## Repository Layout

```text
cs549-traffic-forecasting/
├── README.md                  -- this file
├── PROJECT_DESCRIPTION.md     -- detailed assignment instructions
├── DATA_DESCRIPTION.md        -- data file formats
├── RULES.md                   -- collaboration and integrity rules
├── requirements.txt
├── data/                      -- released train/validation/test data
├── starter_code/              -- load data, score validation, build a submission
├── baselines/                 -- two worked baseline examples
├── evaluation/                -- metrics used by the leaderboard
└── examples/getting_started.ipynb
```

## Questions

Post questions to the course discussion forum / Canvas so the whole class
benefits from the answer. Do not share code, trained models, or predictions
with other groups (see [`RULES.md`](RULES.md)).
