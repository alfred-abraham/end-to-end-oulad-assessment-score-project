"""Load OULAD and reproduce the notebook's student-separated partitions."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import DATA_DIR, SEED


class DataIngestion:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir

    def initiate_data_ingestion(self):
        from pathlib import Path
        data = pd.read_csv(Path(self.data_dir) / 'combinedfile_cleaned.csv')
        required = ['score', 'id_student', 'id_assessment', 'date', 'date_submitted',
                    'code_module', 'code_presentation']
        if not set(required).issubset(data.columns):
            raise ValueError('The cleaned OULAD dataset is missing required columns.')
        if not data['score'].between(0, 100).all():
            raise ValueError('Training scores must be present and between 0 and 100.')
        if data[['id_student', 'code_module', 'code_presentation', 'date_submitted']].isna().any().any():
            raise ValueError('Student groups and submission days must be present.')
        if data.duplicated(['id_student', 'id_assessment']).any():
            raise ValueError('Duplicate student/assessment pairs found.')
        outer = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
        trainval, test = next(outer.split(data, data.score, data.id_student))
        inner = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED + 1)
        train_rel, val_rel = next(inner.split(data.iloc[trainval],
            data.score.iloc[trainval], data.id_student.iloc[trainval]))
        labels = pd.Series(index=data.index, dtype='object')
        labels.iloc[trainval[train_rel]] = 'train'
        labels.iloc[trainval[val_rel]] = 'validation'
        labels.iloc[test] = 'test'
        assert labels.notna().all()
        assert pd.DataFrame({'student': data.id_student, 'split': labels}).groupby(
            'student')['split'].nunique().max() == 1
        return data, labels


if __name__ == '__main__':
    from src.pipeline.train_pipeline import main
    main()
