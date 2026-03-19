"""Inference service.

Endpoints:
- /predict: online inference
- /info: runtime + model metadata (for ops/debug)
- /metrics: Prometheus endpoint
- /admin/reload: manual hot-reload (token-protected)

Design notes:
- Enforces request size and batch limits to protect against memory abuse.
- On reload failure, keeps MODEL_UP intact (old model may still serve) and increments
  reload_failed counters.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    generate_latest,
    multiprocess,
)

from app.compare_client import ComparePublisher
from app.config import Settings, get_settings
from app.logging_conf import setup_logging
from app.model_loader import load_model_from_dir
from app.telemetry import (
    HTTP_INFLIGHT,
    HTTP_REQ_SIZE,
    HTTP_RESP_SIZE,
    INFERENCE_INSTANCE_INFO,
    MODEL_ACTIVE_INFO,
    MODEL_BATCH,
    MODEL_CONF,
    MODEL_EMBED_NORM,
    MODEL_ERRORS_TOTAL,
    MODEL_INFERENCE_LATENCY,
    MODEL_OUTPUT_VALUE,
    MODEL_PRED_COUNT,
    MODEL_PRED_LABEL,
    MODEL_RELOAD_FAILED_TOTAL,
    MODEL_RELOAD_SUCCESS_TOTAL,
    MODEL_RELOAD_TOTAL,
    MODEL_UP,
    REQ_COUNT,
    REQ_LATENCY,
)

logger = logging.getLogger("app")

_model_lock = threading.RLock()
_model = None
_container_id: str = ""
_active_model_info_labels: tuple[str, str, str, str, str] | None = None
_compare_publisher: ComparePublisher | None = None


def _detect_container_id() -> str:
    """Best-effort container id detection.

    We prefer a full 64-hex Docker/containerd id so we can join with cAdvisor.
    """

    hostname = (os.getenv("HOSTNAME") or "").strip()

    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as f:
            for line in f:
                m = re.search(r"([0-9a-f]{64})", line)
                if m:
                    return m.group(1)
    except Exception:
        pass

    return hostname or "unknown"


class BadRequestError(ValueError):
    pass


def _set_active_model(m) -> None:
    global _model
    with _model_lock:
        _model = m


def _get_active_model():
    return _model


def _publish_active_model_metrics(m, *, model_dir: str) -> None:
    """Publish the active model identity.

    In Prometheus multiprocess mode, `.clear()` does not reliably remove old labelsets.
    We keep the metric bounded by explicitly setting the previous labelset to 0.
    """

    global _active_model_info_labels
    if _active_model_info_labels is not None:
        try:
            MODEL_ACTIVE_INFO.labels(*_active_model_info_labels).set(0)
        except Exception:
            pass
        _active_model_info_labels = None

    if m and m.info:
        runtime = getattr(m, "runtime", "unknown")
        labels = (m.info.name, m.info.version, model_dir, m.info.task, runtime)
        MODEL_ACTIVE_INFO.labels(*labels).set(1)
        _active_model_info_labels = labels


async def _enforce_body_limit(request: Request, max_bytes: int) -> Optional[Response]:
    """Reject oversized bodies early.

    If Content-Length is missing, stream and cap.
    """
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > max_bytes:
                return JSONResponse(
                    {"error": "payload_too_large", "max_bytes": max_bytes},
                    status_code=413,
                )
        except Exception:
            pass

    if not cl:
        total = 0
        chunks: list[bytes] = []
        async for chunk in request.stream():
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                return JSONResponse(
                    {"error": "payload_too_large", "max_bytes": max_bytes},
                    status_code=413,
                )
        request._body = b"".join(chunks)  # type: ignore[attr-defined]

    return None


def _is_full_metrics(settings: Settings) -> bool:
    return settings.metrics_profile == "full"


def _parse_bool_override(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise BadRequestError(f"Invalid boolean query parameter value: {raw!r}")


def _parse_predict_payload(payload: Any) -> list[list[float]]:
    if not isinstance(payload, dict):
        raise BadRequestError("JSON body must be an object")

    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise BadRequestError("Field 'instances' must be a list of feature vectors")

    for row in instances:
        if not isinstance(row, list):
            raise BadRequestError("Each item in 'instances' must be a list")

    return instances


def _extract_compare_event(
    request: Request,
    m,
    preds: list[dict[str, Any]],
    *,
    duration_seconds: float,
) -> Optional[dict[str, Any]]:
    role = (request.headers.get("X-Compare-Role") or "").strip().lower()
    route_variant = (request.headers.get("X-Compare-Variant") or "").strip().lower()

    if role not in {"primary", "shadow"} or route_variant not in {"v1", "v2"}:
        return None
    if not m or not m.info or m.info.task != "classification":
        return None

    compact_preds: list[dict[str, Any]] = []
    for pred in preds:
        label = pred.get("label")
        confidence = pred.get("confidence")
        if label is None or confidence is None:
            return None
        compact_preds.append({"label": str(label), "confidence": float(confidence)})

    if not compact_preds:
        return None

    return {
        "request_id": request.state.request_id,
        "role": role,
        "route_variant": route_variant,
        "task": m.info.task,
        "model_name": m.info.name,
        "model_version": m.info.version,
        "duration_seconds": float(duration_seconds),
        "predictions": compact_preds,
    }


def _record_predict_metrics(
    settings: Settings, m, preds: list[dict[str, Any]], batch_size: int
) -> None:
    task = m.info.task
    MODEL_PRED_COUNT.labels(m.info.name, m.info.version, task).inc(len(preds))
    MODEL_BATCH.labels(m.info.name, m.info.version, task).observe(batch_size)

    if not _is_full_metrics(settings):
        return

    for p in preds:
        label = str(p.get("label", "unknown"))
        conf = float(p.get("confidence", 1.0))

        MODEL_PRED_LABEL.labels(m.info.name, m.info.version, task, label).inc()
        MODEL_CONF.labels(m.info.name, m.info.version, task).observe(conf)

        if task == "regression" and p.get("value") is not None:
            try:
                MODEL_OUTPUT_VALUE.labels(m.info.name, m.info.version).observe(float(p["value"]))
            except Exception:
                pass

        if task == "embedding" and p.get("embedding") is not None:
            try:
                vec = p["embedding"]
                norm = float((sum(float(x) * float(x) for x in vec)) ** 0.5)
                MODEL_EMBED_NORM.labels(m.info.name, m.info.version).observe(norm)
            except Exception:
                pass


def create_app() -> FastAPI:
    settings: Settings = get_settings()
    app = FastAPI(title="Inference Service", version=settings.service_version)
    app.state.settings = settings

    @app.on_event("startup")
    def startup() -> None:
        setup_logging(settings.log_level)
        global _container_id, _compare_publisher
        _container_id = _detect_container_id()
        INFERENCE_INSTANCE_INFO.labels(_container_id).set(1)
        _compare_publisher = ComparePublisher(
            enabled=settings.compare_enabled,
            url=settings.compare_url,
            timeout_s=settings.compare_timeout_s,
            queue_size=settings.compare_queue_size,
        )
        _compare_publisher.start()
        m = load_model_from_dir(settings.model_dir)
        _set_active_model(m)
        MODEL_UP.set(1)
        _publish_active_model_metrics(m, model_dir=settings.model_dir)

    @app.on_event("shutdown")
    def shutdown() -> None:
        global _compare_publisher
        if _compare_publisher is not None:
            _compare_publisher.stop()
            _compare_publisher = None

    @app.middleware("http")
    async def mw(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid

        method = request.method
        path = request.url.path

        if method == "POST" and path == "/predict":
            early_resp = await _enforce_body_limit(request, settings.max_request_bytes)
            if early_resp is not None:
                early_resp.headers["X-Request-ID"] = rid
                return early_resp

        HTTP_INFLIGHT.labels(method, path).inc()

        if _is_full_metrics(settings):
            cl = request.headers.get("content-length")
            if cl:
                try:
                    HTTP_REQ_SIZE.labels(method, path).observe(int(cl))
                except Exception:
                    pass

        t0 = time.perf_counter()
        status = "500"
        response: Response
        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            logger.exception(
                "unhandled_exception",
                extra={"request_id": rid, "method": method, "path": path},
            )
            response = JSONResponse(
                {"error": "internal_error", "request_id": rid},
                status_code=500,
            )
            status = "500"
        finally:
            dt = time.perf_counter() - t0
            REQ_LATENCY.labels(method, path).observe(dt)
            REQ_COUNT.labels(method, path, status).inc()

            response.headers["X-Request-ID"] = rid
            if _is_full_metrics(settings):
                rcl = response.headers.get("content-length")
                if rcl:
                    try:
                        HTTP_RESP_SIZE.labels(method, path, status).observe(int(rcl))
                    except Exception:
                        pass

            HTTP_INFLIGHT.labels(method, path).dec()

        return response

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        m = _get_active_model()
        if not m or not m.info:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return {
            "status": "ready",
            "model": {
                "name": m.info.name,
                "version": m.info.version,
                "task": m.info.task,
            },
        }

    @app.get("/info")
    def info():
        m = _get_active_model()
        if not m or not m.info:
            return JSONResponse({"error": "model_not_loaded"}, status_code=503)

        runtime = getattr(m, "runtime", "unknown")
        runtime_details = getattr(m, "runtime_details", {}) or {}

        return {
            "service": {
                "name": settings.service_name,
                "version": settings.service_version,
            },
            "model": {
                "name": m.info.name,
                "version": m.info.version,
                "task": m.info.task,
                "runtime": runtime,
                "n_features": m.info.n_features,
                "classes": m.info.classes,
                "embedding_dim": m.info.embedding_dim,
            },
            "runtime_details": runtime_details,
            "model_dir": settings.model_dir,
            "reload_enabled": bool(settings.reload_token),
            "limits": {
                "max_request_bytes": settings.max_request_bytes,
                "max_batch_size": settings.max_batch_size,
            },
        }

    @app.post("/predict")
    async def predict(request: Request):
        rid = request.state.request_id
        t_predict0 = time.perf_counter()

        try:
            payload = await request.json()
        except Exception:
            MODEL_ERRORS_TOTAL.labels("predict", "invalid_json").inc()
            return JSONResponse(
                {"error": "bad_request", "details": "Invalid JSON body", "request_id": rid},
                status_code=400,
            )

        try:
            instances = _parse_predict_payload(payload)
            if len(instances) > settings.max_batch_size:
                return JSONResponse(
                    {
                        "error": "batch_too_large",
                        "max_batch_size": settings.max_batch_size,
                        "request_id": rid,
                    },
                    status_code=413,
                )

            include_proba = _parse_bool_override(
                request.query_params.get("include_proba"),
                settings.classification_include_proba_default,
            )
            include_embedding = _parse_bool_override(
                request.query_params.get("include_embedding"),
                settings.embedding_include_vector_default,
            )
        except BadRequestError as e:
            MODEL_ERRORS_TOTAL.labels("predict", "bad_request").inc()
            return JSONResponse(
                {"error": "bad_request", "details": str(e), "request_id": rid},
                status_code=400,
            )

        m = _get_active_model()
        if not m or not m.info:
            return JSONResponse(
                {"error": "model_not_loaded", "request_id": rid},
                status_code=503,
            )

        runtime = getattr(m, "runtime", "unknown")

        t_inf0 = time.perf_counter()
        try:
            out = await run_in_threadpool(
                m.predict,
                instances,
                include_proba=include_proba,
                include_embedding=include_embedding,
            )
            dt_inf = time.perf_counter() - t_inf0
            MODEL_INFERENCE_LATENCY.labels(
                m.info.name,
                m.info.version,
                m.info.task,
                runtime,
            ).observe(dt_inf)
        except ValueError as e:
            MODEL_ERRORS_TOTAL.labels("predict", "value_error").inc()
            return JSONResponse(
                {"error": "bad_request", "details": str(e), "request_id": rid},
                status_code=400,
            )
        except Exception:
            MODEL_ERRORS_TOTAL.labels("predict", "exception").inc()
            raise

        preds = out.get("predictions") or []
        _record_predict_metrics(settings, m, preds, len(instances))

        compare_event = _extract_compare_event(
            request,
            m,
            preds,
            duration_seconds=time.perf_counter() - t_predict0,
        )
        if compare_event is not None and _compare_publisher is not None:
            _compare_publisher.publish(compare_event)

        return JSONResponse(
            {
                "model_name": m.info.name,
                "model_version": m.info.version,
                "task": m.info.task,
                "predictions": preds,
                "request_id": rid,
            }
        )

    @app.post("/admin/reload")
    def reload_model(request: Request):
        rid = request.state.request_id

        if not settings.reload_token:
            MODEL_RELOAD_TOTAL.labels("disabled").inc()
            return JSONResponse(
                {"error": "reload_disabled", "request_id": rid},
                status_code=404,
            )

        token = request.headers.get("X-Reload-Token", "")
        if token != settings.reload_token:
            MODEL_RELOAD_TOTAL.labels("unauthorized").inc()
            return JSONResponse(
                {"error": "unauthorized", "request_id": rid},
                status_code=403,
            )

        try:
            m = load_model_from_dir(settings.model_dir)
            _set_active_model(m)
            _publish_active_model_metrics(m, model_dir=settings.model_dir)

            if _container_id:
                INFERENCE_INSTANCE_INFO.labels(_container_id).set(1)

            MODEL_UP.set(1)
            MODEL_RELOAD_TOTAL.labels("success").inc()
            MODEL_RELOAD_SUCCESS_TOTAL.inc()

            return {
                "status": "reloaded",
                "model": {
                    "name": m.info.name,
                    "version": m.info.version,
                    "task": m.info.task,
                },
                "request_id": rid,
            }
        except Exception as e:
            MODEL_RELOAD_TOTAL.labels("failed").inc()
            MODEL_RELOAD_FAILED_TOTAL.inc()
            logger.exception(
                "model_reload_failed",
                extra={"request_id": rid, "model_dir": settings.model_dir},
            )
            return JSONResponse(
                {"error": "reload_failed", "details": str(e), "request_id": rid},
                status_code=500,
            )

    @app.get("/metrics")
    def metrics():
        d = (os.getenv("PROMETHEUS_MULTIPROC_DIR") or "").strip()
        if d:
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            payload = generate_latest(registry)
        else:
            payload = generate_latest()

        return PlainTextResponse(payload.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
