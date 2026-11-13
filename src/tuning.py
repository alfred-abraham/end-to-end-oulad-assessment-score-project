"""Deterministic bounded validation search; test rows are never passed to search."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def make_estimator(name, params):
    classes = {'Linear Regression': LinearRegression,
               'K-Neighbors Regressor': KNeighborsRegressor,
               'Decision Tree': DecisionTreeRegressor,
               'Random Forest': RandomForestRegressor, 'XGBoost': XGBRegressor}
    return classes[name](**params)


def search_specs(seed=42):
    """Include each original recipe, plus fixed grid/random search candidates."""
    originals = {
        'Linear Regression': {},
        'K-Neighbors Regressor': {'n_neighbors': 25, 'n_jobs': 4},
        'Decision Tree': {'max_depth': 12, 'min_samples_leaf': 20, 'random_state': seed},
        'Random Forest': {'n_estimators': 150, 'max_depth': 16,
            'min_samples_leaf': 10, 'max_features': .8, 'random_state': seed, 'n_jobs': 4},
        'XGBoost': {'n_estimators': 300, 'max_depth': 4, 'learning_rate': .05,
            'subsample': .8, 'colsample_bytree': .8, 'objective': 'reg:squarederror',
            'tree_method': 'hist', 'random_state': seed, 'n_jobs': 4},
    }
    spaces = {
        'Linear Regression': {'fit_intercept': [True, False]},
        'K-Neighbors Regressor': {'n_neighbors': [15, 50], 'weights': ['uniform', 'distance']},
        'Decision Tree': {'max_depth': [8, 16, None], 'min_samples_leaf': [10, 40]},
        'Random Forest': {'n_estimators': [150, 250], 'max_depth': [12, 20, None],
            'min_samples_leaf': [3, 10, 20], 'max_features': [.6, 1.]},
        'XGBoost': {'n_estimators': [300, 600], 'max_depth': [3, 5, 7],
            'learning_rate': [.03, .07], 'min_child_weight': [1, 10],
            'reg_lambda': [1., 10.], 'subsample': [.8, 1.], 'colsample_bytree': [.8, 1.]},
    }
    output = {}
    for name, base in originals.items():
        sampled = (ParameterSampler(spaces[name], n_iter=6, random_state=seed)
                   if name in ['Random Forest', 'XGBoost'] else ParameterGrid(spaces[name]))
        recipes = [base] + [{**base, **p} for p in sampled]
        output[name] = list({json.dumps(p, sort_keys=True): p for p in recipes}.values())
    return output


def tune_candidates(X, y, train_idx, val_idx, preprocessor, label='', seed=42):
    """Cache preprocessing fitted on training students only, then rank by validation RMSE.

    Validation uses the established student-disjoint holdout, not row-wise CV.
    Each algorithm retains its lowest-RMSE recipe; insertion order breaks ties.
    """
    assert set(train_idx).isdisjoint(val_idx)
    fitted = clone(preprocessor)
    train = fitted.fit_transform(X.loc[train_idx], y.loc[train_idx])
    validation = fitted.transform(X.loc[val_idx])
    candidates, rows = {}, []
    for name, recipes in search_specs(seed).items():
        best_rmse = np.inf
        for trial, params in enumerate(recipes):
            estimator = make_estimator(name, params)
            estimator.fit(train, y.loc[train_idx])
            pred = estimator.predict(validation)
            rmse = float(np.sqrt(mean_squared_error(y.loc[val_idx], pred)))
            rows.append({'Feature set': label, 'Model': name, 'Trial': trial,
                         'Original recipe': trial == 0, 'Parameters': params,
                         'Validation RMSE': rmse,
                         'Validation MAE': float(mean_absolute_error(y.loc[val_idx], pred)),
                         'Validation R2': float(r2_score(y.loc[val_idx], pred))})
            print(f'{label}: {name} {trial + 1}/{len(recipes)} RMSE={rmse:.5f}', flush=True)
            if rmse < best_rmse:
                best_rmse = rmse
                candidates[name] = Pipeline([('preprocessor', fitted), ('model', estimator)])
    return candidates, pd.DataFrame(rows)


def selected_recipe():
    return json.loads(Path(__file__).with_name('model_selection.json').read_text(encoding='utf-8'))
