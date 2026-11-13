# OULAD assessment score estimator

A Flask regression demo using the tuned XGBoost selected in
[`notebook/Model Training.ipynb`](notebook/Model%20Training.ipynb). Contextual inputs
are retained; previous assessment results are optional. The application calculates
the history features and loads the saved preprocessing + model pipeline.

## Hyperparameter tuning

The experiment uses the existing student-disjoint 60/20/20 train/validation/test
partitions. All five algorithms are tuned separately for context and 28-day history
features. Each search includes the original recipe. Selection minimizes validation
RMSE, with preprocessing fitted only on training students. Test scores never select
algorithms or hyperparameters. This is a bounded validation search, not exhaustive
optimization or cross-validation. The test holdout was inspected previously.

| Algorithm | Search beyond the original recipe, per feature set |
| --- | --- |
| Linear Regression | Intercept on/off |
| K-Nearest Neighbors | 15 or 50 neighbors; uniform or distance weighting |
| Decision Tree | Depth 8, 16, or unlimited; leaf minimum 10 or 40 |
| Random Forest | Six seeded random trials: 150/250 trees, depth 12/20/unlimited, leaf minimum 3/10/20, feature fraction 0.6/1.0 |
| XGBoost | Six seeded random trials: 300/600 trees, depth 3/5/7, rate 0.03/0.07, child weight 1/10, lambda 1/10, row/column fractions 0.8/1.0 |

Seed is 42. Exact spaces and execution order are in [`src/tuning.py`](src/tuning.py).
The run evaluated **58 configurations across both feature sets**.
The complete parameters and metrics are in [`artifacts/tuning_trials.csv`](artifacts/tuning_trials.csv)
and [`artifacts/tuning_results.json`](artifacts/tuning_results.json).

### Validation comparison after tuning

Each row shows the best configuration for that algorithm on each feature set.
RMSE and MAE are measured in score points; lower is better.

| Algorithm | Context RMSE | Context MAE | History 28d RMSE | History 28d MAE |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 17.727 | 13.641 | 16.287 | 12.213 |
| K-Neighbors Regressor | 17.502 | 13.401 | 15.316 | 11.289 |
| Decision Tree | 17.334 | 13.354 | 15.485 | 11.405 |
| Random Forest | 17.162 | 13.201 | 14.978 | 10.998 |
| XGBoost | 17.051 | 13.130 | 14.897 | 10.955 |

The best original recipe and best tuned recipe on each validation feature set:

| Feature set | Original best RMSE | Tuned best RMSE |
| --- | ---: | ---: |
| Context | 17.05060 | 17.05060 |
| History 28d | 14.98640 | 14.89736 |

### Frozen selected recipes

**Context: XGBoost**, validation RMSE 17.05060.

```json
{
  "n_estimators": 300,
  "max_depth": 4,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "objective": "reg:squarederror",
  "tree_method": "hist",
  "random_state": 42,
  "n_jobs": 4
}
```

**History: XGBoost**, validation RMSE 14.89736.

```json
{
  "n_estimators": 300,
  "max_depth": 7,
  "learning_rate": 0.03,
  "subsample": 0.8,
  "colsample_bytree": 1.0,
  "objective": "reg:squarederror",
  "tree_method": "hist",
  "random_state": 42,
  "n_jobs": 4,
  "reg_lambda": 1.0,
  "min_child_weight": 10
}
```

The machine-readable recipes are in [`src/model_selection.json`](src/model_selection.json).
Final models are refitted on train + validation. The 56-day sensitivity model uses
the selected 28-day recipe without a second search. Simple mean/median and prior-score
baselines remain in the executed notebook and results JSON.

## Major findings after tuning

The test set contains 33,774 assessment records from 4,659 held-out students.
R² is a regression measure, not classification accuracy.

| Model / assumed grade-release delay | Test R² | Test RMSE | Test MAE |
| --- | ---: | ---: | ---: |
| Context (XGBoost) | 0.160 | 17.128 | 13.084 |
| History (XGBoost, 28 days) | 0.356 | 14.991 | 10.999 |
| History (XGBoost, 56 days) | 0.305 | 15.573 | 11.550 |

With a 28-day assumed release delay, history reduces test RMSE by **12.5%** and MAE by **15.9%** versus the tuned context reference.

With a 56-day assumed release delay, history reduces test RMSE by **9.1%** and MAE by **11.7%** versus the tuned context reference.

The retained notable-improvement gate **did not pass**. It requires at least 10% reductions in both errors on validation and test under both delays, plus positive lower 95% student-bootstrap bounds. See the notebook for the full bounds and subgroup diagnostics.

Under the 28-day assumption, **74.3%** of test records have usable history. Test MAE is **10.09** with history and **13.64** without it. These subsets contain different records.

The refreshed validation permutation importance ranks `weight`, `assessment_type`, `prior_mean` highest. Correlated inputs share importance; these are model sensitivities, not causal effects.

The previous fixed-recipe history model had test RMSE 15.0573 and MAE 11.0275. The tuned model has RMSE **14.9915** and MAE **10.9989**. This is a retrospective comparison on a previously inspected holdout; independent student/course data is needed to confirm performance. Grade-release timestamps are unavailable, so all history results depend on the stated availability assumptions.

## Run locally

Use Python 3.12 and run these commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.pipeline.train_pipeline
python app.py
```

The default training command refits the frozen tuned history recipe. To repeat
the bounded history search before refitting:

```powershell
python -m src.pipeline.train_pipeline --tune
```

This writes `artifacts/history_tuning_trials.json` and records the selected parameters
in model metadata. It does not rewrite the notebook, its comparison report, or the
frozen default recipe. To reproduce both feature-set searches and sensitivity/importance
analyses, restart and run all cells in `notebook/Model Training.ipynb`.

Open <http://127.0.0.1:5000>. The training step creates
`artifacts/history_model.joblib` (the complete fitted preprocessing + model pipeline)
and `artifacts/model_metadata.json` (schema, assessment schedule, categories, measured
metrics, versions, and model checksum). Artifacts are included when generated by this
project update; retraining is optional if the recorded dependency versions match.
Use only trusted local model files. Restart the app after replacing artifacts, since
the fitted pipeline is cached in memory. Training does not run during web requests.

For a WSGI server instead of Flask's development server:

```powershell
python -m waitress --listen=127.0.0.1:8080 app:application
```

This update prepares a runnable demo; it does not publish the app or provision hosting.
The Elastic Beanstalk entry point is `app:application`.

## Form inputs

Select the module, presentation, and assessment being predicted. The public OULAD
schedule supplies its assessment type, weight, and deadline. Enter gender, region,
highest education, deprivation band, age band, previous course attempts, studied
credits, and disability. Options retain the dataset's original category values.

Optionally add previous assessments from that same student's selected course offering,
with their score and submission day. Days are relative to the course start, not calendar
dates. The assessment selector supplies each previous deadline and ID, which are
necessary for consistent ordering and validation. Changing the course offering clears
previous entries to prevent accidentally mixing courses.

Leaving history empty uses `prior_count=0` and missing summary statistics, exactly the
no-history representation seen during training. The saved pipeline imputes those
statistics using training-fitted values. This is the history model operating without
history, not a switch to a separately trained contextual model.

## History rules and limitations

- Predict one day before the target deadline.
- A source grade is usable only when `max(deadline, submission_day) + 28 < prediction_day`.
- Do not enter the target assessment, duplicate assessments, or another student's results.
- Scores too recent for this rule are excluded and their count is shown in the result.
- Grade-release timestamps are absent from OULAD. The 28-day delay is an assumption,
  not a verified marking guarantee. In a real system, require actual grade-release times.
- Records with unknown deadlines cannot be selected or used as historical inputs.

The application uses the same tuned XGBoost recipe, student partitions,
imputation, encoding, and scaling as the executed notebook. Training fits the
frozen recipe on train + validation and reports held-out-student performance.
Earlier available grades belonging to those students may supply sequential inputs;
their prediction targets never fit the model.

Expected primary test results are R² **0.356**, RMSE **14.991**,
and MAE **10.999**. Runtime values come from the generated metadata.
These are overall assessment errors, not individual confidence intervals.

## Verify

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover deterministic tuning, train-only preprocessing, validation selection,
notebook/application feature parity, historical cutoff behavior, optional
history, malformed inputs, missing model artifacts, and real saved-pipeline predictions.
Train artifacts before running the integration tests. Notebooks and CSVs are not modified
by the application or training command.
