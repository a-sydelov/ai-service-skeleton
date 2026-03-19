"""Route-level check for a concrete model version behind the gateway.

Examples (inside the compose network):
  python -m scripts.test_model_version \
    --url http://gateway:8080/v1/predict \
    --model-dir /models/model/1

  python -m scripts.test_model_version \
    --url http://gateway:8080/v2/predict \
    --model-dir /models/model/2

Exit codes:
- 0 OK
- 2 request/response mismatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx


class SmokeError(RuntimeError):
    """Raised when the route check fails."""


def _fail(msg: str) -> int:
    print(f"TEST_MODEL_VERSION_FAIL: {msg}", file=sys.stderr)
    return 2


def _load_metadata(model_dir: Path) -> dict[str, Any]:
    metadata_file = model_dir / "metadata.json"
    if not metadata_file.exists():
        raise SmokeError(f"metadata.json not found: {metadata_file}")

    try:
        raw = json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SmokeError(f"invalid JSON in {metadata_file}: {exc}") from exc

    version = raw.get("version")
    task = raw.get("task")
    n_features = raw.get("n_features")

    if not version:
        raise SmokeError(f"missing 'version' in {metadata_file}")
    if task not in {"classification", "regression", "embedding"}:
        raise SmokeError(f"unsupported or missing 'task' in {metadata_file}: {task!r}")
    if not isinstance(n_features, int) or n_features <= 0:
        raise SmokeError(f"invalid 'n_features' in {metadata_file}: {n_features!r}")

    return {
        "version": str(version),
        "task": str(task),
        "n_features": n_features,
    }


def _make_row(n_features: int, offset: int) -> list[float]:
    row: list[float] = []
    for i in range(n_features):
        base = ((i + offset) % 17) - 8
        row.append(round(base / 10.0, 6))
    return row


def _make_payload(n_features: int, batch_size: int) -> dict[str, Any]:
    return {
        "instances": [_make_row(n_features, offset=i) for i in range(batch_size)],
    }


def _validate_prediction(pred: Any, task: str, index: int) -> None:
    if not isinstance(pred, dict):
        raise SmokeError(f"prediction[{index}] must be an object, got {type(pred).__name__}")

    if task == "classification":
        label = pred.get("label")
        confidence = pred.get("confidence")
        if not isinstance(label, str) or not label:
            raise SmokeError(f"prediction[{index}] missing non-empty 'label'")
        if not isinstance(confidence, (int, float)):
            raise SmokeError(f"prediction[{index}] missing numeric 'confidence'")
        confidence_value = float(confidence)
        if confidence_value < 0.0 or confidence_value > 1.0:
            raise SmokeError(f"prediction[{index}] has out-of-range confidence={confidence_value}")
        return

    if task == "regression":
        if not isinstance(pred.get("value"), (int, float)):
            raise SmokeError(f"prediction[{index}] missing numeric 'value'")
        return

    if task == "embedding":
        embedding = pred.get("embedding")
        if embedding is None:
            return
        if not isinstance(embedding, list) or not all(
            isinstance(x, (int, float)) for x in embedding
        ):
            raise SmokeError(f"prediction[{index}] has invalid 'embedding'")
        return

    raise SmokeError(f"unsupported task={task!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="e.g. http://gateway:8080/v1/predict")
    parser.add_argument("--model-dir", required=True, help="e.g. /models/model/1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    if args.batch_size <= 0:
        return _fail(f"--batch-size must be > 0, got {args.batch_size}")

    try:
        metadata = _load_metadata(Path(args.model_dir))
        payload = _make_payload(metadata["n_features"], args.batch_size)
    except SmokeError as exc:
        return _fail(str(exc))

    url = args.url.rstrip("/")
    request_id = f"test-model-version-{metadata['version']}"

    try:
        with httpx.Client(timeout=args.timeout) as client:
            response = client.post(url, json=payload, headers={"X-Request-ID": request_id})
    except httpx.HTTPError as exc:
        return _fail(f"POST {url} failed: {exc}")

    if response.status_code != 200:
        return _fail(f"POST {url} status={response.status_code} body={response.text[:300]}")

    try:
        body = response.json()
    except ValueError:
        return _fail(f"POST {url} returned non-JSON body: {response.text[:300]}")

    if body.get("model_version") != metadata["version"]:
        return _fail(
            f"expected model_version={metadata['version']!r}, got {body.get('model_version')!r}"
        )

    if body.get("task") != metadata["task"]:
        return _fail(f"expected task={metadata['task']!r}, got {body.get('task')!r}")

    predictions = body.get("predictions")
    if not isinstance(predictions, list):
        return _fail("response missing list 'predictions'")

    if len(predictions) != args.batch_size:
        return _fail(f"expected {args.batch_size} predictions, got {len(predictions)}")

    try:
        for index, pred in enumerate(predictions):
            _validate_prediction(pred, metadata["task"], index)
    except SmokeError as exc:
        return _fail(str(exc))

    print(
        "TEST_MODEL_VERSION_OK "
        f"version={metadata['version']} task={metadata['task']} "
        f"n_features={metadata['n_features']} batch={args.batch_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
