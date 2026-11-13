"""Check selection boundaries and the train-only preprocessing cache."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src import tuning


def test_search_is_reproducible_and_includes_originals():
    specs = tuning.search_specs()
    assert specs == tuning.search_specs()
    assert len(specs['Random Forest']) == len(specs['XGBoost']) == 7
    assert specs['Random Forest'][0]['min_samples_leaf'] == 10
    assert specs['K-Neighbors Regressor'][0]['n_neighbors'] == 25


def test_tuning_fits_preprocessing_only_on_training_and_selects_validation_minimum(monkeypatch):
    monkeypatch.setattr(tuning, 'search_specs', lambda seed: {
        'Decision Tree': [{'max_depth': 1, 'random_state': seed},
                          {'max_depth': 3, 'random_state': seed}]})
    X = pd.DataFrame({'x': [0., 1., 2., 3., 4., 5., np.nan, 1000., 1.5]})
    y = pd.Series([0., 0., 1., 1., 0., 0., 1., 9999., 0.])
    candidates, trials = tuning.tune_candidates(X, y, [0, 1, 2, 3, 4, 5], [6, 8],
        SimpleImputer(strategy='median'), seed=42)
    model = candidates['Decision Tree']
    assert model.named_steps['preprocessor'].statistics_[0] == 2.5
    best = trials.sort_values('Validation RMSE', kind='stable').iloc[0]
    assert model.named_steps['model'].max_depth == best.Parameters['max_depth']
    assert len(trials) == 2
    # The unused row's extreme target and feature cannot influence fitted parameters.
    X.loc[7, 'x'], y.loc[7] = -1e9, -1e9
    changed, changed_trials = tuning.tune_candidates(X, y, [0, 1, 2, 3, 4, 5], [6, 8],
        SimpleImputer(strategy='median'), seed=42)
    pd.testing.assert_frame_equal(trials, changed_trials)
    np.testing.assert_allclose(model.predict(X.loc[[6]]), changed['Decision Tree'].predict(X.loc[[6]]))
