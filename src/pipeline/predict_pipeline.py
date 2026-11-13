"""Validated OULAD inputs and inference through one saved sklearn pipeline."""
from functools import lru_cache
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, FEATURE_COLUMNS, METADATA_FILENAME, MODEL_FILENAME
from src.features import features_from_records
from src.utils import file_sha256, load_object

MAX_HISTORY_ROWS = 50


class InputError(ValueError):
    """An input that the user can correct in the form."""


class ModelUnavailable(RuntimeError):
    """Missing, stale, or incompatible local model artifacts."""


def number(value, label, minimum, maximum, integer=False):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise InputError(f'{label}: enter a number.') from None
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise InputError(f'{label}: enter a value from {minimum:g} to {maximum:g}.')
    if integer and not result.is_integer():
        raise InputError(f'{label}: enter a whole number.')
    return int(result) if integer else result


class CustomData:
    """Translate a web form into the exact contextual + history feature schema."""
    def __init__(self, form, metadata):
        self.form, self.metadata = form, metadata
        self.ignored_history_count = 0
        self.used_history_count = 0

    def get_data_as_data_frame(self):
        categories = self.metadata['categories']
        context = {}
        for field in ['code_module', 'code_presentation', 'gender', 'region',
                      'highest_education', 'imd_band', 'age_band', 'disability']:
            value = self.form.get(field, '')
            if value not in categories[field]:
                raise InputError(f'{field.replace("_", " ").capitalize()}: select a listed option.')
            context[field] = value
        catalog = {str(r['id_assessment']): r for r in self.metadata['assessment_catalog']}
        target = catalog.get(self.form.get('target_assessment_id', ''))
        if (target is None or target['code_module'] != context['code_module'] or
                target['code_presentation'] != context['code_presentation']):
            raise InputError('Select an assessment from the chosen module and presentation.')
        context.update(assessment_type=target['assessment_type'], weight=target['weight'])
        for field, label in [('num_of_prev_attempts', 'Previous course attempts'),
                             ('studied_credits', 'Studied credits')]:
            limits = self.metadata['numeric_limits'][field]
            context[field] = number(self.form.get(field), label, limits['min'], limits['max'], integer=True)

        lists = [self.form.getlist(key) for key in
                 ['prior_assessment_id', 'prior_score', 'prior_submitted']]
        if len({len(items) for items in lists}) != 1:
            raise InputError('Each previous assessment needs an assessment, score, and submission day.')
        if len(lists[0]) > MAX_HISTORY_ROWS:
            raise InputError(f'Enter at most {MAX_HISTORY_ROWS} previous assessments.')
        records, seen = [], set()
        for row, (assessment_id, score, submitted) in enumerate(zip(*lists), start=1):
            if not any(str(value).strip() for value in [assessment_id, score, submitted]):
                continue
            if not all(str(value).strip() for value in [assessment_id, score, submitted]):
                raise InputError(f'Previous assessment {row}: complete all three fields or remove the row.')
            assessment = catalog.get(assessment_id)
            if (assessment is None or assessment['code_module'] != context['code_module'] or
                    assessment['code_presentation'] != context['code_presentation']):
                raise InputError(f'Previous assessment {row}: choose one from the same course offering.')
            if assessment_id == str(target['id_assessment']):
                raise InputError('The assessment being predicted cannot be included in previous results.')
            if assessment_id in seen:
                raise InputError('Each previous assessment can be entered only once.')
            seen.add(assessment_id)
            records.append({'id_assessment': assessment['id_assessment'],
                'deadline': assessment['deadline'],
                'score': number(score, f'Previous assessment {row} score', 0, 100),
                'date_submitted': number(submitted, f'Previous assessment {row} submission day',
                                         -1000, 2000, integer=True)})
        history, ignored = features_from_records(records, target['deadline'] - 1,
                                                 self.metadata['release_lag_days'])
        self.ignored_history_count = ignored
        self.used_history_count = int(history['prior_count'])
        return pd.DataFrame([{**context, **history}], columns=FEATURE_COLUMNS)


class PredictPipeline:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        artifact_dir = Path(artifact_dir)
        model_path = artifact_dir / MODEL_FILENAME
        try:
            self.metadata = json.loads((artifact_dir / METADATA_FILENAME).read_text(encoding='utf-8'))
            if self.metadata['schema_version'] != 1 or self.metadata['feature_columns'] != FEATURE_COLUMNS:
                raise ValueError('Unsupported feature schema')
            if self.metadata['model_sha256'] != file_sha256(model_path):
                raise ValueError('Model and metadata do not match')
            for package in self.metadata['versions']:
                if importlib.metadata.version(package) != self.metadata['versions'][package]:
                    raise ValueError(f'{package} differs from the training environment')
            self.model = load_object(model_path)
            if list(self.model.feature_names_in_) != FEATURE_COLUMNS:
                raise ValueError('Saved model has the wrong inputs')
        except (OSError, ValueError, KeyError, ImportError, AttributeError) as exc:
            raise ModelUnavailable('The trained model is unavailable or incompatible. '
                'Install requirements.txt and run: python -m src.pipeline.train_pipeline') from exc

    def predict(self, features):
        if set(features.columns) != set(FEATURE_COLUMNS):
            raise InputError('Prediction inputs do not match the trained feature schema.')
        predictions = self.model.predict(features[FEATURE_COLUMNS])
        if not np.isfinite(predictions).all():
            raise RuntimeError('The model returned an invalid prediction.')
        return predictions

    def predict_form(self, form):
        data = CustomData(form, self.metadata)
        prediction = float(self.predict(data.get_data_as_data_frame())[0])
        return {'score': prediction, 'history_count': data.used_history_count,
                'ignored_history_count': data.ignored_history_count}


@lru_cache(maxsize=4)
def get_predictor(artifact_dir=str(ARTIFACT_DIR)):
    return PredictPipeline(artifact_dir)
