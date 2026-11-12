"""Run with: python -m src.pipeline.train_pipeline"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.config import (ARTIFACT_DIR, DATA_DIR, FEATURE_COLUMNS, MODEL_FILENAME,
                        METADATA_FILENAME, SCORE_RELEASE_LAG, SEED)
from src.utils import file_sha256, save_json, save_object


def make_catalog(data_dir):
    assessments = pd.read_csv(Path(data_dir) / 'assessments.csv')
    catalog = []
    for row in assessments.loc[assessments.date.notna()].itertuples():
        catalog.append({'id_assessment': int(row.id_assessment),
            'code_module': row.code_module, 'code_presentation': row.code_presentation,
            'assessment_type': row.assessment_type, 'deadline': int(row.date),
            'weight': float(row.weight)})
    return catalog


def train(data_dir=DATA_DIR, artifact_dir=ARTIFACT_DIR):
    print('Loading OULAD and allocating disjoint student groups...', flush=True)
    data, labels = DataIngestion(data_dir).initiate_data_ingestion()
    print('Building strictly earlier assessment histories...', flush=True)
    frame, X, y, partitions = DataTransformation().initiate_data_transformation(data, labels)
    print('Fitting the notebook-selected Random Forest pipeline...', flush=True)
    model, metrics, subsets = ModelTrainer().initiate_model_trainer(X, y, partitions)
    trainval = np.concatenate([partitions['train'], partitions['validation']])
    fitted_data = X.loc[trainval]
    categories = {column: sorted(fitted_data[column].dropna().unique().tolist())
                  for column in fitted_data.select_dtypes(exclude='number').columns}
    metadata = {
        'schema_version': 1, 'model': 'Random Forest with assessment history',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'release_lag_days': SCORE_RELEASE_LAG, 'prediction_offset_days': -1,
        'seed': SEED, 'feature_columns': FEATURE_COLUMNS,
        'categories': categories, 'assessment_catalog': make_catalog(data_dir),
        'numeric_limits': {column: {'min': float(fitted_data[column].min()),
                                   'max': float(fitted_data[column].max())}
                           for column in ['studied_credits', 'num_of_prev_attempts']},
        'test_metrics': metrics, 'test_subgroups': subsets,
        'split_counts': {name: {'rows': len(idx),
            'students': int(frame.loc[idx, 'id_student'].nunique())}
            for name, idx in partitions.items()},
        'excluded_unknown_deadline_rows': int(data.date.isna().sum()),
        'versions': {name: importlib.metadata.version(name)
                     for name in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
        'cleaned_data_sha256': file_sha256(Path(data_dir) / 'combinedfile_cleaned.csv'),
        'availability_note': 'Grade-release timestamps are absent. Assume availability '
            '28 days after the later of deadline and submission; this is not a guarantee.',
        'evaluation_note': 'Student-disjoint, retrospective evaluation with sequential '
            'history updates; not an independent blind or future-course test.',
    }
    artifact_dir = Path(artifact_dir)
    save_object(artifact_dir / MODEL_FILENAME, model)
    metadata['model_sha256'] = file_sha256(artifact_dir / MODEL_FILENAME)
    save_json(artifact_dir / METADATA_FILENAME, metadata)
    print(f'Saved pipeline and metadata to {artifact_dir}', flush=True)
    print(f'Test metrics: {metrics}', flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--artifact-dir', type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    train(args.data_dir, args.artifact_dir)


if __name__ == '__main__':
    main()
