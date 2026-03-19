from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

SOURCE_ARCHIVE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
)

DATASET_SUBDIR = "dataset"
PAYLOADS_SUBDIR = "payloads"

REQUIRED_DATASET_FILES = [
    "activity_labels.txt",
    "train/X_train.txt",
    "train/y_train.txt",
    "test/X_test.txt",
    "test/y_test.txt",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
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


def get_data_root() -> Path:
    env_value = os.getenv("DATA_ROOT")
    if env_value:
        return Path(env_value)

    env_file = project_root() / ".env"
    file_values = _read_env_file(env_file)
    return Path(file_values.get("DATA_ROOT", "data"))


def dataset_root(data_root: Path) -> Path:
    return data_root / DATASET_SUBDIR


def payloads_root(data_root: Path) -> Path:
    return data_root / PAYLOADS_SUBDIR


def dataset_ready(data_root: Path) -> bool:
    root = dataset_root(data_root)
    return all((root / rel).exists() for rel in REQUIRED_DATASET_FILES)


def validate_dataset(data_root: Path) -> None:
    root = dataset_root(data_root)
    missing = [rel for rel in REQUIRED_DATASET_FILES if not (root / rel).exists()]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Dataset is incomplete in {root}. Missing: {joined}")


def cleanup_extraction_junk(target_root: Path) -> None:
    macosx_dir = target_root / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)

    for junk_name in ("._UCI HAR Dataset",):
        junk = target_root / junk_name
        if junk.exists():
            if junk.is_dir():
                shutil.rmtree(junk)
            else:
                junk.unlink()


def _load_label_mapping(dataset_path: Path) -> dict[int, str]:
    labels_file = dataset_path / "activity_labels.txt"
    if not labels_file.exists():
        raise FileNotFoundError(f"activity_labels.txt not found: {labels_file}")

    mapping: dict[int, str] = {}
    for line in labels_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        key_str, label = line.split(maxsplit=1)
        mapping[int(key_str)] = label.strip()

    if not mapping:
        raise RuntimeError(f"No labels loaded from: {labels_file}")
    return mapping


def _load_split(data_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    root = dataset_root(data_root)
    x_path = root / split / f"X_{split}.txt"
    y_path = root / split / f"y_{split}.txt"

    if not x_path.exists():
        raise FileNotFoundError(f"Feature file not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Label file not found: {y_path}")

    X = np.loadtxt(x_path, dtype=np.float32)
    y = np.loadtxt(y_path, dtype=np.int64)

    if X.ndim != 2:
        raise RuntimeError(f"Expected 2D features in {x_path}, got shape={X.shape}")
    if y.ndim != 1:
        raise RuntimeError(f"Expected 1D labels in {y_path}, got shape={y.shape}")
    if len(X) != len(y):
        raise RuntimeError(
            f"Feature/label row count mismatch in split={split}: {len(X)} != {len(y)}"
        )

    return X, y


def download_archive(target_file: Path) -> None:
    print(f"Downloading source archive to: {target_file}")
    with urllib.request.urlopen(SOURCE_ARCHIVE_URL) as response, target_file.open("wb") as out:
        shutil.copyfileobj(response, out)


def extract_archive(archive_file: Path, data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_file, "r") as zf:
        zf.extractall(data_root)
    cleanup_extraction_junk(data_root)

    extracted_root = data_root / "UCI HAR Dataset"
    final_root = dataset_root(data_root)

    if final_root.exists():
        shutil.rmtree(final_root)

    if not extracted_root.exists():
        raise RuntimeError(f"Expected extracted dataset directory not found: {extracted_root}")

    extracted_root.rename(final_root)


def row_to_payload(row: np.ndarray) -> dict:
    return {"instances": [[round(float(x), 6) for x in row.tolist()]]}


def save_payload_file(
    output_file: Path,
    payloads: list[dict],
    *,
    title: str,
    source_split: str,
    class_counts: dict[str, int],
    n_features: int,
    dataset_name: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "format": "predict-payload-pool/v1",
        "title": title,
        "dataset": dataset_name,
        "source_split": source_split,
        "n_payloads": len(payloads),
        "n_features": n_features,
        "class_counts": class_counts,
        "payloads": payloads,
    }
    output_file.write_text(json.dumps(content, indent=2), encoding="utf-8")
    print(f"Saved payload file: {output_file}")


def select_first_n_per_class(
    X: np.ndarray, y: np.ndarray, n_per_class: int
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []

    for class_id in sorted(np.unique(y).tolist()):
        idx = np.flatnonzero(y == class_id)
        take = idx[:n_per_class]
        for i in take:
            rows.append(X[i])
            labels.append(int(y[i]))

    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int64)


def select_boundary_n_per_class(
    X: np.ndarray, y: np.ndarray, n_per_class: int
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-8
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    Xs = (X - mean) / std

    class_ids = sorted(np.unique(y).tolist())
    centroids: dict[int, np.ndarray] = {}
    for class_id in class_ids:
        centroids[class_id] = Xs[y == class_id].mean(axis=0)

    rows: list[np.ndarray] = []
    labels: list[int] = []

    for class_id in class_ids:
        idx = np.flatnonzero(y == class_id)
        if len(idx) == 0:
            continue

        own = centroids[class_id]
        other_ids = [cid for cid in class_ids if cid != class_id]
        other_centroids = np.stack([centroids[cid] for cid in other_ids], axis=0)

        own_dist = np.sum((Xs[idx] - own) ** 2, axis=1)
        other_dist = np.sum((Xs[idx][:, None, :] - other_centroids[None, :, :]) ** 2, axis=2)
        nearest_other = np.min(other_dist, axis=1)

        gap = np.abs(nearest_other - own_dist)
        order = np.argsort(gap)
        take = idx[order[:n_per_class]]

        for i in take:
            rows.append(X[i])
            labels.append(int(y[i]))

    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int64)


def class_counts_from_ids(label_ids: np.ndarray, label_mapping: dict[int, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_id in sorted(np.unique(label_ids).tolist()):
        counts[label_mapping[int(class_id)]] = int(np.sum(label_ids == class_id))
    return counts


def build_payload_files(
    data_root: Path,
    *,
    split: str,
    smoke_per_class: int,
    load_per_class: int,
    boundary_per_class: int,
) -> None:
    validate_dataset(data_root)

    root = dataset_root(data_root)
    out_dir = payloads_root(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_mapping = _load_label_mapping(root)
    X, y = _load_split(data_root, split)

    smoke_rows, smoke_ids = select_first_n_per_class(X, y, smoke_per_class)
    load_rows, load_ids = select_first_n_per_class(X, y, load_per_class)
    boundary_rows, boundary_ids = select_boundary_n_per_class(X, y, boundary_per_class)

    save_payload_file(
        out_dir / "smoke.json",
        [row_to_payload(row) for row in smoke_rows],
        title="Smoke payloads",
        source_split=split,
        class_counts=class_counts_from_ids(smoke_ids, label_mapping),
        n_features=int(X.shape[1]),
        dataset_name="UCI HAR Dataset",
    )
    save_payload_file(
        out_dir / "load.json",
        [row_to_payload(row) for row in load_rows],
        title="Load payloads",
        source_split=split,
        class_counts=class_counts_from_ids(load_ids, label_mapping),
        n_features=int(X.shape[1]),
        dataset_name="UCI HAR Dataset",
    )
    save_payload_file(
        out_dir / "boundary.json",
        [row_to_payload(row) for row in boundary_rows],
        title="Boundary payloads",
        source_split=split,
        class_counts=class_counts_from_ids(boundary_ids, label_mapping),
        n_features=int(X.shape[1]),
        dataset_name="UCI HAR Dataset",
    )

    print(f"Payload files are ready in: {out_dir}")


def cmd_load(
    data_root: Path,
    *,
    force: bool,
    split: str,
    smoke_per_class: int,
    load_per_class: int,
    boundary_per_class: int,
) -> int:
    if dataset_ready(data_root) and not force:
        print(f"Dataset is already ready: {dataset_root(data_root)}")
        build_payload_files(
            data_root,
            split=split,
            smoke_per_class=smoke_per_class,
            load_per_class=load_per_class,
            boundary_per_class=boundary_per_class,
        )
        return 0

    dataset_path = dataset_root(data_root)
    if force and dataset_path.exists():
        print(f"Removing existing dataset directory: {dataset_path}")
        shutil.rmtree(dataset_path)

    data_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dataset_assets_") as tmp_dir:
        archive_file = Path(tmp_dir) / "dataset_source.zip"
        download_archive(archive_file)
        extract_archive(archive_file, data_root)

    validate_dataset(data_root)

    build_payload_files(
        data_root,
        split=split,
        smoke_per_class=smoke_per_class,
        load_per_class=load_per_class,
        boundary_per_class=boundary_per_class,
    )

    print(f"Dataset is ready: {dataset_root(data_root)}")
    return 0


def cmd_build_payloads(
    data_root: Path,
    *,
    split: str,
    smoke_per_class: int,
    load_per_class: int,
    boundary_per_class: int,
) -> int:
    build_payload_files(
        data_root,
        split=split,
        smoke_per_class=smoke_per_class,
        load_per_class=load_per_class,
        boundary_per_class=boundary_per_class,
    )
    return 0


def cmd_clear(data_root: Path) -> int:
    if not data_root.exists():
        print(f"Nothing to remove: {data_root}")
        return 0

    dataset_path = dataset_root(data_root)
    payloads_path = payloads_root(data_root)

    if dataset_path.exists():
        print(f"Removing dataset directory: {dataset_path}")
        shutil.rmtree(dataset_path)

    if payloads_path.exists():
        print(f"Removing payloads directory: {payloads_path}")
        shutil.rmtree(payloads_path)

    leftovers = [p for p in data_root.iterdir()] if data_root.exists() else []
    if data_root.exists() and not leftovers:
        print(f"Keeping empty data root: {data_root}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download dataset assets, generate payload files, or clear generated content."
    )
    parser.add_argument(
        "action",
        choices=["load", "build-payloads", "clear"],
        help="load: download dataset and build payloads; build-payloads: regenerate payload files; clear: remove generated content",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Optional override for DATA_ROOT from environment/.env",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="For 'load': remove existing dataset directory and re-download source files",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Dataset split used for payload generation (default: test)",
    )
    parser.add_argument(
        "--smoke-per-class",
        type=int,
        default=1,
        help="Number of smoke payloads per class (default: 1)",
    )
    parser.add_argument(
        "--load-per-class",
        type=int,
        default=12,
        help="Number of load payloads per class (default: 12)",
    )
    parser.add_argument(
        "--boundary-per-class",
        type=int,
        default=12,
        help="Number of boundary payloads per class (default: 12)",
    )

    args = parser.parse_args()
    data_root = Path(args.root) if args.root else get_data_root()

    try:
        if args.action == "load":
            return cmd_load(
                data_root,
                force=args.force,
                split=args.split,
                smoke_per_class=args.smoke_per_class,
                load_per_class=args.load_per_class,
                boundary_per_class=args.boundary_per_class,
            )
        if args.action == "build-payloads":
            return cmd_build_payloads(
                data_root,
                split=args.split,
                smoke_per_class=args.smoke_per_class,
                load_per_class=args.load_per_class,
                boundary_per_class=args.boundary_per_class,
            )
        return cmd_clear(data_root)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
