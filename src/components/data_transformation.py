"""Feature construction and training-fitted preprocessing."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CONTEXT_COLUMNS, FEATURE_COLUMNS, SCORE_RELEASE_LAG
from src.features import build_history_features


class DataTransformation:
    def initiate_data_transformation(self, data, split_labels):
        frame = data.loc[data['date'].notna()].copy()
        history = build_history_features(frame, SCORE_RELEASE_LAG)
        features = pd.concat([frame[CONTEXT_COLUMNS], history], axis=1)[FEATURE_COLUMNS]
        partitions = {name: frame.index[split_labels.loc[frame.index].eq(name)]
                      for name in ['train', 'validation', 'test']}
        return frame, features, frame['score'], partitions

    def get_data_transformer_object(self, features):
        numeric = features.select_dtypes(include='number').columns.tolist()
        categorical = features.select_dtypes(exclude='number').columns.tolist()
        return ColumnTransformer([
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                ('scaler', StandardScaler()),
            ]), numeric),
            ('categorical', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), categorical),
        ])
