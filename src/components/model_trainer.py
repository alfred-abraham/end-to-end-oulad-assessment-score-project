"""Select on validation students and refit the frozen recipe on train + validation."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.components.data_transformation import DataTransformation
from src.config import SEED
from src.tuning import make_estimator, selected_recipe, tune_candidates


def regression_metrics(actual, predicted):
    return {'R2': float(r2_score(actual, predicted)),
            'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
            'MAE': float(mean_absolute_error(actual, predicted))}


class ModelTrainer:
    def __init__(self, tune=False):
        self.tune = tune
        self.selection = selected_recipe()['history']
        self.trials = None

    def make_pipeline(self, features):
        return Pipeline([
            ('preprocessor', DataTransformation().get_data_transformer_object(features)),
            ('model', make_estimator(self.selection['name'], self.selection['params'])),
        ])

    def initiate_model_trainer(self, features, target, partitions):
        if self.tune:
            _, self.trials = tune_candidates(features, target, partitions['train'],
                partitions['validation'],
                DataTransformation().get_data_transformer_object(features), 'History 28d', SEED)
            best = self.trials.sort_values('Validation RMSE', kind='stable').iloc[0]
            self.selection = {'name': best['Model'], 'params': best['Parameters'],
                              'validation_rmse': float(best['Validation RMSE'])}
        # Refit the validation-selected recipe; test metrics never choose parameters.
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
