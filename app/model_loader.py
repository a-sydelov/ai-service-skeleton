"""Model artifact loader + validator.

This module enforces a minimal artifact contract stored in <MODEL_DIR>/metadata.json.

Required fields:
- name, version, task, runtime, model_file, n_features

Optional fields:
- classes (required for classification)
- contract: input/output schema (recommended for portability)
- onnx: input/output names (recommended for stable ONNX serving)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from app.model import OnnxRuntimeAdapter, SklearnAdapter

SUPPORTED_RUNTIMES = {"sklearn", "onnx"}
SUPPORTED_TASKS = {"classification", "regression", "embedding"}
SUPPORTED_INPUT_TYPES = {"float32"}


def _require(meta: Dict[str, Any], key: str) -> Any:
    if key not in meta:
        raise ValueError(f"metadata.json missing required field: '{key}'")
    return meta[key]


def _validate_contract(meta: Dict[str, Any]) -> None:
    contract = meta.get("contract")
    if contract is None:
        return
    if not isinstance(contract, dict):
        raise ValueError("metadata.json field 'contract' must be an object")

    n_features = int(meta["n_features"])
    task = str(meta["task"]).strip().lower()

    inp = contract.get("input")
    if inp is not None:
        if not isinstance(inp, dict):
            raise ValueError("metadata.json contract.input must be an object")

        itype = inp.get("type")
        if itype is None:
            raise ValueError(
                "metadata.json contract.input.type is required when contract.input is provided"
            )
        itype = str(itype).strip().lower()
        if itype not in SUPPORTED_INPUT_TYPES:
            raise ValueError(
                f"metadata.json contract.input.type must be one of {sorted(SUPPORTED_INPUT_TYPES)}"
            )

        shape = inp.get("shape")
        if shape is None:
            raise ValueError(
                "metadata.json contract.input.shape is required when contract.input is provided"
            )
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(
                "metadata.json contract.input.shape must be a 2-element list, e.g. ['N', n_features]"
            )

        dim1 = shape[1]
        try:
            dim1_i = int(dim1)
        except Exception as e:
            raise ValueError(
                "metadata.json contract.input.shape[1] must be an int equal to n_features"
            ) from e
        if dim1_i != n_features:
            raise ValueError(
                f"metadata.json contract.input.shape[1]={dim1_i} must match n_features={n_features}"
            )

    out = contract.get("output")
    if out is not None:
        if not isinstance(out, dict):
            raise ValueError("metadata.json contract.output must be an object")
        otype = out.get("type")
        if otype is None:
            raise ValueError(
                "metadata.json contract.output.type is required when contract.output is provided"
            )
        otype = str(otype).strip().lower()
        if otype not in SUPPORTED_TASKS:
            raise ValueError(
                f"metadata.json contract.output.type must be one of {sorted(SUPPORTED_TASKS)}"
            )
        if otype != task:
            raise ValueError(
                f"metadata.json contract.output.type='{otype}' must match task='{task}'"
            )


def _validate_metadata(meta: Dict[str, Any]) -> Tuple[str, str, str, str, int]:
    name = str(_require(meta, "name"))
    # Ensure the contract includes an explicit model version.
    _require(meta, "version")
    runtime = str(_require(meta, "runtime")).strip().lower()
    task = str(_require(meta, "task")).strip().lower()
    model_file = str(_require(meta, "model_file"))
    n_features = int(_require(meta, "n_features"))

    if runtime not in SUPPORTED_RUNTIMES:
        raise ValueError(
            f"Unsupported runtime='{runtime}'. Supported: {sorted(SUPPORTED_RUNTIMES)}"
        )
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task='{task}'. Supported: {sorted(SUPPORTED_TASKS)}")
    if n_features <= 0:
        raise ValueError("metadata.json field 'n_features' must be > 0")

    if task == "classification":
        classes = meta.get("classes")
        if not isinstance(classes, list) or len(classes) == 0:
            raise ValueError(
                "classification task requires non-empty list field 'classes' in metadata.json"
            )

    _validate_contract(meta)

    if runtime == "onnx":
        onnx_cfg = meta.get("onnx")
        if onnx_cfg is not None and not isinstance(onnx_cfg, dict):
            raise ValueError("metadata.json field 'onnx' must be an object when provided")
        if isinstance(onnx_cfg, dict):
            if "input_name" in onnx_cfg and not isinstance(onnx_cfg["input_name"], str):
                raise ValueError("metadata.json onnx.input_name must be string when provided")
            if "output_name" in onnx_cfg and not isinstance(onnx_cfg["output_name"], str):
                raise ValueError("metadata.json onnx.output_name must be string when provided")
            if "output_is_logits" in onnx_cfg and not isinstance(
                onnx_cfg["output_is_logits"], bool
            ):
                raise ValueError(
                    "metadata.json onnx.output_is_logits must be boolean when provided"
                )

    return runtime, task, name, model_file, n_features


def validate_artifact_dir(model_dir: str, deep: bool = False) -> Dict[str, Any]:
    """Validate an artifact directory without starting the server."""
    d = Path(model_dir)
    meta_path = d / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in MODEL_DIR: {d}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    runtime, task, name, model_file, n_features = _validate_metadata(meta)

    model_path = d / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    summary: Dict[str, Any] = {
        "ok": True,
        "model_dir": str(d),
        "name": name,
        "version": str(meta.get("version")),
        "runtime": runtime,
        "task": task,
        "n_features": n_features,
        "model_file": model_file,
    }

    # Deep ONNX validation checks the configured I/O names actually exist in the model.
    if runtime == "onnx" and deep:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = [i.name for i in sess.get_inputs()]
        outputs = [o.name for o in sess.get_outputs()]
        onnx_cfg = meta.get("onnx") or {}
        in_name = onnx_cfg.get("input_name")
        out_name = onnx_cfg.get("output_name")
        if in_name is not None and in_name not in inputs:
            raise ValueError(f"onnx.input_name='{in_name}' not found in model inputs={inputs}")
        if out_name is not None and out_name not in outputs:
            raise ValueError(f"onnx.output_name='{out_name}' not found in model outputs={outputs}")
        summary["onnx_io"] = {
            "inputs": inputs,
            "outputs": outputs,
            "providers": sess.get_providers(),
        }

    return summary


def load_model_from_dir(model_dir: str):
    """Load model adapter from MODEL_DIR."""
    d = Path(model_dir)
    meta_path = d / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in MODEL_DIR: {d}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    runtime, _task, _name, model_file, _n_features = _validate_metadata(meta)

    model_path = d / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if runtime == "sklearn":
        sk_adapter = SklearnAdapter(str(model_path), str(meta_path))
        sk_adapter.load()
        return sk_adapter
    if runtime == "onnx":
        onnx_adapter = OnnxRuntimeAdapter(str(model_path), str(meta_path))
        onnx_adapter.load()
        return onnx_adapter

    raise ValueError(f"Unsupported runtime='{runtime}'")


__all__ = ["load_model_from_dir", "validate_artifact_dir"]
