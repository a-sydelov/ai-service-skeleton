from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_SUBDIR = "dataset"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        values[key] = value

    return values


def _default_data_root() -> Path:
    env_value = os.getenv("DATA_ROOT")
    if env_value:
        return Path(env_value) / DATASET_SUBDIR

    env_file = _project_root() / ".env"
    file_values = _read_env_file(env_file)
    return Path(file_values.get("DATA_ROOT", "data")) / DATASET_SUBDIR


def _read_activity_map(data_root: Path) -> dict[int, str]:
    path = data_root / "activity_labels.txt"
    if not path.exists():
        raise FileNotFoundError(f"activity_labels.txt not found: {path}")

    mapping: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        key_str, value = line.split(maxsplit=1)
        mapping[int(key_str)] = value.strip()

    if not mapping:
        raise RuntimeError(f"No activity labels loaded from: {path}")
    return mapping


def _load_split(
    data_root: Path, split: str, activity_map: dict[int, str]
) -> tuple[np.ndarray, np.ndarray]:
    x_path = data_root / split / f"X_{split}.txt"
    y_path = data_root / split / f"y_{split}.txt"

    if not x_path.exists():
        raise FileNotFoundError(f"Feature file not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Label file not found: {y_path}")

    X = np.loadtxt(x_path, dtype=np.float32)
    y_numeric = np.loadtxt(y_path, dtype=np.int64)

    if X.ndim != 2:
        raise RuntimeError(f"Expected 2D features in {x_path}, got shape={X.shape}")
    if y_numeric.ndim != 1:
        raise RuntimeError(f"Expected 1D labels in {y_path}, got shape={y_numeric.shape}")
    if len(X) != len(y_numeric):
        raise RuntimeError(
            f"Feature/label row count mismatch in split={split}: {len(X)} != {len(y_numeric)}"
        )

    y = np.array([activity_map[int(v)] for v in y_numeric], dtype=object)
    return X, y


def _build_estimator(args: argparse.Namespace):
    if args.algo == "logreg":
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=args.C,
                        max_iter=args.max_iter,
                        n_jobs=1,
                        random_state=args.random_state,
                    ),
                ),
            ]
        )
        algo_meta = {
            "algo": "LogisticRegression",
            "C": args.C,
            "max_iter": args.max_iter,
            "scaler": "StandardScaler",
        }
        return clf, algo_meta

    if args.algo == "mlp":
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128,),
                        alpha=0.0001,
                        max_iter=args.max_iter,
                        early_stopping=False,
                        random_state=args.random_state,
                    ),
                ),
            ]
        )
        algo_meta = {
            "algo": "MLPClassifier",
            "hidden_layer_sizes": [128],
            "alpha": 0.0001,
            "max_iter": args.max_iter,
            "early_stopping": False,
            "scaler": "StandardScaler",
        }
        return clf, algo_meta

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=1,
    )
    algo_meta = {
        "algo": "RandomForestClassifier",
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
    }
    return clf, algo_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="model")
    parser.add_argument("--version", default="1")
    parser.add_argument("--out-root", default="/models")
    parser.add_argument("--data-root", default=str(_default_data_root()))
    parser.add_argument("--algo", choices=["logreg", "mlp", "rf"], default="logreg")
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)

    # Backward-compat placeholders to avoid breaking older make targets/flags.
    parser.add_argument("--n-features", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--n-classes", type=int, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_root) / args.name / str(args.version)
    out_dir.mkdir(parents=True, exist_ok=True)

    activity_map = _read_activity_map(data_root)
    classes = [activity_map[idx] for idx in sorted(activity_map)]

    X_train, y_train = _load_split(data_root, "train", activity_map)
    X_test, y_test = _load_split(data_root, "test", activity_map)

    clf, algo_meta = _build_estimator(args)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    model_file = "model.joblib"
    model_path = out_dir / model_file
    joblib.dump(clf, model_path)

    n_features = int(X_train.shape[1])

    meta = {
        "name": args.name,
        "version": str(args.version),
        "task": "classification",
        "runtime": "sklearn",
        "model_file": model_file,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_features": n_features,
        "classes": classes,
        "metrics": {"accuracy_test": acc},
        "training": {
            "dataset": "UCI HAR Dataset",
            "random_state": args.random_state,
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            **algo_meta,
        },
        "contract": {
            "input_schema_version": "v1",
            "input": {"type": "float32", "shape": ["N", n_features]},
            "output": {"type": "classification"},
        },
    }

    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {out_dir / 'metadata.json'}")
    print(f"Dataset root: {data_root}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Classes: {classes}")
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
