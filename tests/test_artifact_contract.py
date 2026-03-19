import json
from pathlib import Path

import pytest

from app.model_loader import validate_artifact_dir


def _write_artifact(tmp_path: Path, meta: dict, model_file: str = "model.joblib") -> Path:
    d = tmp_path / meta["name"] / str(meta["version"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / model_file).write_bytes(b"dummy")
    return d


def test_validate_ok_classification_sklearn(tmp_path: Path):
    meta = {
        "name": "model",
        "version": "1",
        "task": "classification",
        "runtime": "sklearn",
        "model_file": "model.joblib",
        "n_features": 4,
        "classes": ["c0", "c1", "c2"],
        "contract": {
            "input_schema_version": "v1",
            "input": {"type": "float32", "shape": ["N", 4]},
            "output": {"type": "classification"},
        },
    }
    d = _write_artifact(tmp_path, meta, "model.joblib")
    out = validate_artifact_dir(str(d), deep=False)
    assert out["ok"] is True
    assert out["runtime"] == "sklearn"
    assert out["task"] == "classification"


def test_classification_requires_classes(tmp_path: Path):
    meta = {
        "name": "model",
        "version": "1",
        "task": "classification",
        "runtime": "sklearn",
        "model_file": "model.joblib",
        "n_features": 4,
        "classes": [],
    }
    d = _write_artifact(tmp_path, meta, "model.joblib")
    with pytest.raises(ValueError):
        validate_artifact_dir(str(d), deep=False)


def test_contract_shape_must_match_n_features(tmp_path: Path):
    meta = {
        "name": "model",
        "version": "1",
        "task": "classification",
        "runtime": "sklearn",
        "model_file": "model.joblib",
        "n_features": 4,
        "classes": ["c0", "c1"],
        "contract": {
            "input_schema_version": "v1",
            "input": {"type": "float32", "shape": ["N", 8]},
            "output": {"type": "classification"},
        },
    }
    d = _write_artifact(tmp_path, meta, "model.joblib")
    with pytest.raises(ValueError):
        validate_artifact_dir(str(d), deep=False)
