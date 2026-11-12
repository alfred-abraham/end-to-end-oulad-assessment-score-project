"""Persist local, trusted model artifacts atomically."""
import hashlib
import json
from pathlib import Path
import joblib


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_object(file_path, obj):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    joblib.dump(obj, temp, compress=3)
    temp.replace(path)


def load_object(file_path):
    return joblib.load(file_path)


def save_json(file_path, value):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)
