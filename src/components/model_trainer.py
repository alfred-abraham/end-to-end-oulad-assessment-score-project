"""Fit the notebook's validation-selected Random Forest without test-set tuning."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.components.data_transformation import DataTransformation
from src.config import SEED


def regression_metrics(actual, predicted):
    return {'R2': float(r2_score(actual, predicted)),
            'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
            'MAE': float(mean_absolute_error(actual, predicted))}


class ModelTrainer:
    def make_pipeline(self, features):
        return Pipeline([
            ('preprocessor', DataTransformation().get_data_transformer_object(features)),
            ('model', RandomForestRegressor(n_estimators=150, max_depth=16,
                min_samples_leaf=10, max_features=.8, random_state=SEED, n_jobs=-1)),
        ])

    def initiate_model_trainer(self, features, target, partitions):
        # Model and settings were selected on validation data in the notebook.
        # Refit exactly that recipe on train + validation, leaving test untouched.
        trainval = np.concatenate([partitions['train'], partitions['validation']])
        test = partitions['test']
        model = self.make_pipeline(features)
        model.fit(features.loc[trainval], target.loc[trainval])
        predictions = model.predict(features.loc[test])
        metrics = regression_metrics(target.loc[test], predictions)
        subsets = {}
        for name, mask in {
            'with_history': features.loc[test, 'prior_count'].gt(0).to_numpy(),
            'without_history': features.loc[test, 'prior_count'].eq(0).to_numpy(),
        }.items():
            if mask.sum() >= 2:
                subsets[name] = {'rows': int(mask.sum()),
                    **regression_metrics(target.loc[test].to_numpy()[mask], predictions[mask])}
        return model, metrics, subsets
