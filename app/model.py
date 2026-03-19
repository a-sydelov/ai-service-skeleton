"""Model adapters.

This skeleton supports two runtimes:
- sklearn (joblib) as a simple reference
- onnxruntime as a portable production-friendly runtime

Adapters expose:
- info: model metadata (name/version/task/n_features)
- runtime + runtime_details: required for /info troubleshooting
- predict(instances): normalized output format across tasks
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    task: str  # classification|regression|embedding
    n_features: int
    classes: Optional[List[str]] = None
    embedding_dim: Optional[int] = None


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    out = e / np.sum(e, axis=1, keepdims=True)
    # mypy + numpy typing: ensure the return type is treated as ndarray
    return np.asarray(out, dtype=np.float32)


class SklearnAdapter:
    """Sklearn adapter (joblib)."""

    def __init__(self, model_path: str, metadata_path: str):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model: Any = None
        self.info: Optional[ModelInfo] = None
        self._meta: Dict[str, Any] = {}
        self.runtime: str = "sklearn"
        self.runtime_details: Dict[str, Any] = {}

    def load(self) -> None:
        self.model = joblib.load(self.model_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)

        self.runtime = str(self._meta.get("runtime", "sklearn")).strip().lower()
        task = str(self._meta["task"]).strip().lower()

        classes = self._meta.get("classes")
        if classes is not None:
            classes = list(classes)

        self.info = ModelInfo(
            name=str(self._meta["name"]),
            version=str(self._meta["version"]),
            task=task,
            n_features=int(self._meta["n_features"]),
            classes=classes,
            embedding_dim=self._meta.get("embedding_dim"),
        )

        self.runtime_details = {"format": "joblib", "path": self.model_path}

    def _to_matrix(self, instances: List[List[float]]) -> np.ndarray:
        if self.info is None:
            raise RuntimeError("Model not loaded")
        X = np.asarray(instances, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.info.n_features:
            raise ValueError(f"Expected shape [N, {self.info.n_features}] but got {tuple(X.shape)}")
        return X

    def _predict_classification(self, X: np.ndarray, *, include_proba: bool) -> Dict[str, Any]:
        if self.model is None or self.info is None:
            raise RuntimeError("Model not loaded")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("classification task requires sklearn model with predict_proba()")
        if not self.info.classes:
            raise ValueError("classification task requires non-empty 'classes' in metadata.json")

        proba = np.asarray(self.model.predict_proba(X), dtype=np.float32)
        preds: List[Dict[str, Any]] = []
        for row in proba:
            idx = int(np.argmax(row))
            pred: Dict[str, Any] = {
                "label": str(self.info.classes[idx]),
                "confidence": float(row[idx]),
            }
            if include_proba:
                pred["proba"] = row.tolist()
            preds.append(pred)
        return {"task": "classification", "predictions": preds}

    def _predict_regression(self, X: np.ndarray) -> Dict[str, Any]:
        if self.model is None or self.info is None:
            raise RuntimeError("Model not loaded")
        if not hasattr(self.model, "predict"):
            raise ValueError("regression task requires sklearn model with predict()")
        y = np.asarray(self.model.predict(X), dtype=np.float32).reshape(-1)
        preds = [{"label": "regression", "confidence": 1.0, "value": float(v)} for v in y]
        return {"task": "regression", "predictions": preds}

    def _predict_embedding(self, X: np.ndarray, *, include_embedding: bool) -> Dict[str, Any]:
        if self.model is None or self.info is None:
            raise RuntimeError("Model not loaded")
        if hasattr(self.model, "transform"):
            E = self.model.transform(X)
        elif hasattr(self.model, "predict"):
            E = self.model.predict(X)
        else:
            raise ValueError(
                "embedding task requires sklearn model with transform() or predict() returning vectors"
            )

        E = np.asarray(E, dtype=np.float32)
        if E.ndim != 2:
            raise ValueError(f"embedding output must be 2D [N,D], got shape {tuple(E.shape)}")

        emb_dim = int(E.shape[1])
        self.info = ModelInfo(
            self.info.name,
            self.info.version,
            self.info.task,
            self.info.n_features,
            self.info.classes,
            emb_dim,
        )
        preds: List[Dict[str, Any]] = []
        for row in E:
            pred: Dict[str, Any] = {"label": "embedding", "confidence": 1.0}
            if include_embedding:
                pred["embedding"] = row.tolist()
            preds.append(pred)
        return {"task": "embedding", "predictions": preds}

    def predict(
        self,
        instances: List[List[float]],
        *,
        include_proba: bool = False,
        include_embedding: bool = True,
    ) -> Dict[str, Any]:
        if self.info is None:
            raise RuntimeError("Model not loaded")
        X = self._to_matrix(instances)
        if self.info.task == "classification":
            return self._predict_classification(X, include_proba=include_proba)
        if self.info.task == "regression":
            return self._predict_regression(X)
        if self.info.task == "embedding":
            return self._predict_embedding(X, include_embedding=include_embedding)
        raise ValueError(
            f"Unsupported task='{self.info.task}'. Supported: classification|regression|embedding"
        )


class OnnxRuntimeAdapter:
    """ONNXRuntime adapter."""

    def __init__(self, model_path: str, metadata_path: str):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.info: Optional[ModelInfo] = None
        self._meta: Dict[str, Any] = {}
        self._sess: Any = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._output_is_logits: bool = False
        self.runtime: str = "onnx"
        self.runtime_details: Dict[str, Any] = {}

    def load(self) -> None:
        import onnxruntime as ort

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)

        self.runtime = str(self._meta.get("runtime", "onnx")).strip().lower()
        task = str(self._meta["task"]).strip().lower()

        classes = self._meta.get("classes")
        if classes is not None:
            classes = list(classes)

        self.info = ModelInfo(
            name=str(self._meta["name"]),
            version=str(self._meta["version"]),
            task=task,
            n_features=int(self._meta["n_features"]),
            classes=classes,
            embedding_dim=self._meta.get("embedding_dim"),
        )

        onnx_cfg = self._meta.get("onnx") or {}
        self._output_is_logits = bool(onnx_cfg.get("output_is_logits", False))

        self._sess = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self._input_name = str(onnx_cfg.get("input_name") or self._sess.get_inputs()[0].name)
        self._output_name = str(onnx_cfg.get("output_name") or self._sess.get_outputs()[0].name)

        def _io_obj(x: Any) -> Dict[str, Any]:
            shape = x.shape
            if isinstance(shape, (list, tuple)):
                shape = [None if s is None else s for s in shape]
            return {"name": x.name, "shape": shape, "type": x.type}

        self.runtime_details = {
            "format": "onnx",
            "path": self.model_path,
            "input_name": self._input_name,
            "output_name": self._output_name,
            "output_is_logits": self._output_is_logits,
            "inputs": [_io_obj(i) for i in self._sess.get_inputs()],
            "outputs": [_io_obj(o) for o in self._sess.get_outputs()],
            "providers": list(self._sess.get_providers()),
        }

    def _to_matrix(self, instances: List[List[float]]) -> np.ndarray:
        if self.info is None:
            raise RuntimeError("Model not loaded")
        X = np.asarray(instances, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.info.n_features:
            raise ValueError(f"Expected shape [N, {self.info.n_features}] but got {tuple(X.shape)}")
        return X

    def _run(self, X: np.ndarray) -> np.ndarray:
        if self._sess is None:
            raise RuntimeError("Model not loaded")
        y = self._sess.run([self._output_name], {self._input_name: X})[0]
        return np.asarray(y)

    def _predict_classification(self, X: np.ndarray, *, include_proba: bool) -> Dict[str, Any]:
        if self.info is None:
            raise RuntimeError("Model not loaded")
        if not self.info.classes:
            raise ValueError("classification task requires non-empty 'classes' in metadata.json")

        out = self._run(X).astype(np.float32)
        if out.ndim != 2:
            raise ValueError(
                f"classification output must be 2D [N,C], got shape {tuple(out.shape)}"
            )
        if self._output_is_logits:
            out = _softmax(out)

        preds: List[Dict[str, Any]] = []
        for row in out:
            idx = int(np.argmax(row))
            pred: Dict[str, Any] = {
                "label": str(self.info.classes[idx]),
                "confidence": float(row[idx]),
            }
            if include_proba:
                pred["proba"] = row.tolist()
            preds.append(pred)
        return {"task": "classification", "predictions": preds}

    def _predict_regression(self, X: np.ndarray) -> Dict[str, Any]:
        out = self._run(X).astype(np.float32).reshape(-1)
        preds = [{"label": "regression", "confidence": 1.0, "value": float(v)} for v in out]
        return {"task": "regression", "predictions": preds}

    def _predict_embedding(self, X: np.ndarray, *, include_embedding: bool) -> Dict[str, Any]:
        if self.info is None:
            raise RuntimeError("Model not loaded")
        out = self._run(X).astype(np.float32)
        if out.ndim != 2:
            raise ValueError(f"embedding output must be 2D [N,D], got shape {tuple(out.shape)}")

        emb_dim = int(out.shape[1])
        self.info = ModelInfo(
            self.info.name,
            self.info.version,
            self.info.task,
            self.info.n_features,
            self.info.classes,
            emb_dim,
        )
        preds: List[Dict[str, Any]] = []
        for row in out:
            pred: Dict[str, Any] = {"label": "embedding", "confidence": 1.0}
            if include_embedding:
                pred["embedding"] = row.tolist()
            preds.append(pred)
        return {"task": "embedding", "predictions": preds}

    def predict(
        self,
        instances: List[List[float]],
        *,
        include_proba: bool = False,
        include_embedding: bool = True,
    ) -> Dict[str, Any]:
        if self.info is None:
            raise RuntimeError("Model not loaded")
        X = self._to_matrix(instances)
        if self.info.task == "classification":
            return self._predict_classification(X, include_proba=include_proba)
        if self.info.task == "regression":
            return self._predict_regression(X)
        if self.info.task == "embedding":
            return self._predict_embedding(X, include_embedding=include_embedding)
        raise ValueError(
            f"Unsupported task='{self.info.task}'. Supported: classification|regression|embedding"
        )
