"""Paths and feature schema shared by training and inference."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'notebook' / 'data'
ARTIFACT_DIR = PROJECT_ROOT / 'artifacts'
MODEL_FILENAME = 'history_model.joblib'
METADATA_FILENAME = 'model_metadata.json'
SEED = 42
SCORE_RELEASE_LAG = 28
CONTEXT_COLUMNS = [
    'code_module', 'code_presentation', 'assessment_type', 'weight', 'gender',
    'region', 'highest_education', 'imd_band', 'age_band', 'num_of_prev_attempts',
    'studied_credits', 'disability',
]
HISTORY_COLUMNS = [
    'prior_count', 'prior_mean', 'prior_std', 'prior_min', 'prior_max',
    'prior_latest_mean', 'prior_recent3_mean', 'prior_trend', 'days_since_latest_score',
]
FEATURE_COLUMNS = CONTEXT_COLUMNS + HISTORY_COLUMNS
GROUP_KEYS = ['id_student', 'code_module', 'code_presentation']
