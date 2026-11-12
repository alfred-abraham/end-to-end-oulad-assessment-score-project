import ast
import json

import numpy as np
import pandas as pd
import pytest
from werkzeug.datastructures import MultiDict

from app import create_app
from src.config import ARTIFACT_DIR, DATA_DIR, FEATURE_COLUMNS, HISTORY_COLUMNS, PROJECT_ROOT
from src.features import build_history_features, features_from_records
from src.pipeline.predict_pipeline import CustomData, get_predictor


@pytest.fixture(scope='module')
def predictor():
    return get_predictor(str(ARTIFACT_DIR))


def valid_form(**changes):
    values = {'code_module': 'AAA', 'code_presentation': '2013J',
              'target_assessment_id': '1755', 'gender': 'M', 'region': 'East Anglian Region',
              'highest_education': 'HE Qualification', 'imd_band': '90-100%',
              'age_band': '55<=', 'num_of_prev_attempts': '0', 'studied_credits': '240',
              'disability': 'N'}
    values.update(changes)
    return MultiDict(values)


def add_history(form, assessment='1752', score='78', submitted='18'):
    for name, value in [('prior_assessment_id', assessment), ('prior_score', score),
                        ('prior_submitted', submitted)]:
        form.add(name, value)
    return form


def test_available_grade_strict_cutoff_and_future_invariance():
    records = [{'id_assessment': 1, 'deadline': 10, 'date_submitted': 9, 'score': 70},
               {'id_assessment': 2, 'deadline': 20, 'date_submitted': 40, 'score': 90}]
    assert features_from_records(records, 38)[0]['prior_count'] == 0
    result, excluded = features_from_records(records, 50)
    assert result['prior_count'] == 1 and result['prior_mean'] == 70 and excluded == 1
    records[1]['score'] = 0
    assert features_from_records(records, 50)[0] == result


def test_notebook_feature_parity_and_online_parity():
    data = pd.read_csv(DATA_DIR / 'combinedfile_cleaned.csv')
    data = data.loc[data.date.notna()].copy()
    selected = data.id_student.drop_duplicates().sample(60, random_state=42)
    data = data.loc[data.id_student.isin(selected)].copy()
    data['prediction_day'] = data['date'] - 1
    notebook = json.loads((PROJECT_ROOT / 'notebook' /
        'Model Training with Assessment History.ipynb').read_text(encoding='utf-8'))
    feature_cell = next(c for c in notebook['cells']
        if c['cell_type'] == 'code' and 'def history_features(' in ''.join(c['source']))
    syntax = ast.parse(''.join(feature_cell['source']))
    function = next(node for node in syntax.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'history_features')
    namespace = {'np': np, 'pd': pd, 'HISTORY_COLUMNS': HISTORY_COLUMNS,
                 'GROUP_KEYS': ['id_student', 'code_module', 'code_presentation']}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<notebook history>', 'exec'), namespace)
    expected, _ = namespace['history_features'](data, 28)
    actual = build_history_features(data, 28)
    pd.testing.assert_frame_equal(actual, expected)
    for _, group in data.groupby(['id_student', 'code_module', 'code_presentation']):
        for row in group.itertuples():
            records = [{'id_assessment': int(r.id_assessment), 'deadline': float(r.date),
                        'date_submitted': float(r.date_submitted), 'score': float(r.score)}
                       for r in group.itertuples() if r.id_assessment != row.id_assessment]
            online, _ = features_from_records(records, row.prediction_day)
            np.testing.assert_allclose(list(online.values()), actual.loc[row.Index].to_numpy(), equal_nan=True)


def test_no_history_and_blank_row_are_supported(predictor):
    form = valid_form()
    features = CustomData(form, predictor.metadata).get_data_as_data_frame()
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.prior_count.iloc[0] == 0
    assert features[HISTORY_COLUMNS[1:]].isna().all().all()
    blank = CustomData(add_history(valid_form(), '', '', ''), predictor.metadata).get_data_as_data_frame()
    pd.testing.assert_frame_equal(features, blank)
    result = predictor.predict_form(form)
    assert 0 <= result['score'] <= 100 and result['history_count'] == 0
    assert predictor is get_predictor(str(ARTIFACT_DIR))


def test_optional_history_uses_expected_features(predictor):
    form = add_history(valid_form())
    features = CustomData(form, predictor.metadata).get_data_as_data_frame()
    assert features.prior_count.iloc[0] == 1 and features.prior_mean.iloc[0] == 78
    assert features.days_since_latest_score.iloc[0] == (165 - 47)
    np.testing.assert_allclose(predictor.predict(features), predictor.model.predict(features))
    assert predictor.predict_form(form)['history_count'] == 1


@pytest.fixture
def client():
    return create_app({'TESTING': True}).test_client()


def test_routes_and_real_predictions(client):
    assert client.get('/').status_code == 200
    response = client.get('/predictdata')
    assert response.status_code == 200 and b'Previous assessment results' in response.data
    for form in [valid_form(), add_history(valid_form())]:
        response = client.post('/predictdata', data=form)
        assert response.status_code == 200
        assert b'id="predicted-score"' in response.data
        assert b'11.03 marks' in response.data
    assert b'No eligible previous scores were supplied' in client.post('/predictdata', data=valid_form()).data


def test_recent_history_is_excluded_and_reported(client):
    response = client.post('/predictdata', data=add_history(valid_form(), submitted='150'))
    assert response.status_code == 200 and b'Excluded 1 record' in response.data


@pytest.mark.parametrize('field,value', [
    ('gender', 'invalid'), ('num_of_prev_attempts', '0.5'), ('studied_credits', 'nan'),
    ('studied_credits', 'inf'), ('studied_credits', '-10'),
    ('target_assessment_id', '999999'), ('code_presentation', '2050J'),
])
def test_invalid_context_is_rejected(client, field, value):
    response = client.post('/predictdata', data=valid_form(**{field: value}))
    assert response.status_code == 400 and b'role="alert"' in response.data


@pytest.mark.parametrize('assessment,score,submitted', [
    ('1755', '78', '18'), ('1752', '101', '18'), ('1752', '-1', '18'),
    ('1752', 'nan', '18'), ('1752', 'inf', '18'), ('1752', '78', ''),
    ('1752', '78', '1.5'), ('24282', '78', '18'),
])
def test_invalid_history_is_rejected(client, assessment, score, submitted):
    response = client.post('/predictdata', data=add_history(valid_form(), assessment, score, submitted))
    assert response.status_code == 400


def test_duplicate_and_incomplete_history_arrays_rejected(client):
    form = add_history(add_history(valid_form()))
    assert client.post('/predictdata', data=form).status_code == 400
    form = valid_form()
    form.add('prior_assessment_id', '1752')
    assert client.post('/predictdata', data=form).status_code == 400


def test_missing_model_returns_helpful_503(tmp_path):
    client = create_app({'TESTING': True, 'ARTIFACT_DIR': str(tmp_path)}).test_client()
    response = client.get('/predictdata')
    assert response.status_code == 503 and b'python -m src.pipeline.train_pipeline' in response.data


def test_paths_do_not_depend_on_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = create_app({'TESTING': True}).test_client()
    assert client.post('/predictdata', data=valid_form()).status_code == 200


def test_saved_model_reproduces_notebook_metrics(predictor):
    assert predictor.metadata['test_metrics']['RMSE'] == pytest.approx(15.057307511156434, abs=1e-6)
    assert predictor.metadata['test_metrics']['MAE'] == pytest.approx(11.027464137848316, abs=1e-6)
    assert predictor.metadata['test_metrics']['R2'] == pytest.approx(.3506713319399456, abs=1e-6)
    counts = predictor.metadata['split_counts']
    assert counts['test']['rows'] == 33774
    assert sum(row['rows'] for row in counts.values()) == 168966
