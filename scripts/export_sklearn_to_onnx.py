"""Sklearn -> ONNX export.

Converts a joblib model to ONNX and updates metadata.json:
- runtime=onnx
- model_file=model.onnx
- onnx: input_name/output_name (+ output_is_logits if requested)

This keeps the inference runtime decoupled from training libraries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def load_meta(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("metadata.json must be a JSON object")
    return raw


def infer_task(meta: Dict[str, Any], model: Any) -> str:
    if "task" in meta:
        return str(meta["task"]).strip().lower()
    if hasattr(model, "predict_proba"):
        return "classification"
    return "regression"


def infer_n_features(meta: Dict[str, Any], model: Any) -> int:
    if "n_features" in meta:
        return int(meta["n_features"])
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    raise ValueError("Cannot infer n_features. Provide it in metadata.json or pass --n-features.")


def infer_classes(meta: Dict[str, Any], model: Any) -> Optional[list]:
    if "classes" in meta and isinstance(meta["classes"], list) and meta["classes"]:
        return list(meta["classes"])
    if hasattr(model, "classes_"):
        return [str(x) for x in list(model.classes_)]
    return None


def pick_onnx_io(session: ort.InferenceSession, task: str) -> Tuple[str, str]:
    in_name = session.get_inputs()[0].name
    outs = session.get_outputs()

    def is_float(o) -> bool:
        return "float" in (o.type or "").lower()

    def rank(o) -> int:
        shp = o.shape
        return len(shp) if isinstance(shp, (list, tuple)) else 0

    # For label tasks (task=classification) we prefer the 2D float output (probabilities or logits).
    if task == "classification":
        for o in outs:
            if is_float(o) and rank(o) == 2:
                return in_name, o.name
        return in_name, outs[0].name

    for o in outs:
        if is_float(o) and rank(o) in (1, 2):
            return in_name, o.name
    return in_name, outs[0].name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--joblib-file", default="model.joblib")
    p.add_argument("--onnx-file", default="model.onnx")
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--n-features", type=int, default=0)
    p.add_argument("--task", default="")
    p.add_argument("--output-is-logits", action="store_true")
    args = p.parse_args()

    d = Path(args.dir)
    meta_path = d / "metadata.json"
    meta = load_meta(meta_path)

    model_path = d / args.joblib_file
    if not model_path.exists():
        raise SystemExit(f"joblib model not found: {model_path}")

    model = joblib.load(model_path)
    task = args.task.strip().lower() if args.task else infer_task(meta, model)
    if task not in ("classification", "regression", "embedding"):
        raise SystemExit(f"Unsupported task={task}. Use classification|regression|embedding")

    n_features = args.n_features if args.n_features > 0 else infer_n_features(meta, model)

    initial_types = [("input", FloatTensorType([None, n_features]))]
    options = None
    if task == "classification":
        options = {id(model): {"zipmap": False}}

    onx = convert_sklearn(
        model, initial_types=initial_types, options=options, target_opset=args.opset
    )
    onnx_path = d / args.onnx_file
    onnx_path.write_bytes(onx.SerializeToString())

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name, output_name = pick_onnx_io(sess, task)

    if meta_path.exists():
        (d / "metadata.json.bak").write_text(
            meta_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    new_meta = dict(meta)
    new_meta["runtime"] = "onnx"
    new_meta["model_file"] = args.onnx_file
    new_meta["task"] = task
    new_meta["n_features"] = int(n_features)
    new_meta.setdefault("name", d.parent.name)
    new_meta.setdefault("version", d.name)

    if task == "classification":
        classes = infer_classes(meta, model)
        if not classes:
            raise SystemExit(
                "classification requires classes. Provide in metadata.json or ensure model has classes_"
            )
        new_meta["classes"] = list(classes)

    new_meta["onnx"] = dict(new_meta.get("onnx") or {})
    new_meta["onnx"]["input_name"] = input_name
    new_meta["onnx"]["output_name"] = output_name
    if task == "classification" and args.output_is_logits:
        new_meta["onnx"]["output_is_logits"] = True

    # Keep contract consistent if present.
    contract = new_meta.get("contract")
    if isinstance(contract, dict):
        out = contract.get("output")
        if isinstance(out, dict):
            out["type"] = task

    meta_path.write_text(json.dumps(new_meta, indent=2), encoding="utf-8")
    print(f"ONNX written: {onnx_path}")
    print(f"metadata updated: {meta_path}")
    print(f"input={input_name} output={output_name} task={task} n_features={n_features}")


if __name__ == "__main__":
    main()
