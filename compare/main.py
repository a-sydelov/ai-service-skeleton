from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from compare.telemetry import (
    CONF_DELTA,
    LABEL_PAIR,
    MISMATCH_BY_PRIMARY_LABEL,
    MISMATCH_TOTAL,
    PRIMARY_LATENCY,
    PRIMARY_REQUESTS,
    SHADOW_LATENCY,
    SHADOW_REQUESTS,
)

logger = logging.getLogger("compare")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


class ComparePrediction(BaseModel):
    label: str
    confidence: float


class CompareEvent(BaseModel):
    request_id: str = Field(min_length=1)
    role: str
    route_variant: str
    task: str
    duration_seconds: float = Field(ge=0.0)
    predictions: list[ComparePrediction]
    model_name: str | None = None
    model_version: str | None = None


@dataclass
class _PendingPair:
    created_at: float
    primary: CompareEvent | None = None
    shadow: CompareEvent | None = None


@dataclass
class _PairStore:
    ttl_seconds: float
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _items: dict[str, _PendingPair] = field(default_factory=dict)

    def add(self, event: CompareEvent) -> CompareEvent | None:
        now = time.time()
        with self._lock:
            self._evict_expired_locked(now)
            item = self._items.get(event.request_id)
            if item is None:
                item = _PendingPair(created_at=now)
                self._items[event.request_id] = item

            if event.role == "primary":
                item.primary = event
            else:
                item.shadow = event

            if item.primary is not None and item.shadow is not None:
                pair = item
                del self._items[event.request_id]
                return pair.primary, pair.shadow

        return None

    def _evict_expired_locked(self, now: float) -> None:
        expired = [
            rid for rid, item in self._items.items() if now - item.created_at > self.ttl_seconds
        ]
        for rid in expired:
            self._items.pop(rid, None)


class _PairProcessor:
    def __init__(self, ttl_seconds: float):
        self.store = _PairStore(ttl_seconds=ttl_seconds)

    def record(self, event: CompareEvent) -> None:
        self._observe_flow_metrics(event)
        pair = self.store.add(event)
        if pair is None:
            return
        primary, shadow = pair
        self._compare_classification(primary, shadow)

    @staticmethod
    def _observe_flow_metrics(event: CompareEvent) -> None:
        if event.role == "primary":
            PRIMARY_REQUESTS.labels(event.route_variant, "success").inc()
            PRIMARY_LATENCY.labels(event.route_variant).observe(event.duration_seconds)
        else:
            SHADOW_REQUESTS.labels(event.route_variant, "success").inc()
            SHADOW_LATENCY.labels(event.route_variant).observe(event.duration_seconds)

    @staticmethod
    def _compare_classification(primary: CompareEvent, shadow: CompareEvent) -> None:
        primary_preds = primary.predictions
        shadow_preds = shadow.predictions
        primary_variant = primary.route_variant
        shadow_variant = shadow.route_variant

        if len(primary_preds) != len(shadow_preds):
            MISMATCH_TOTAL.labels(primary_variant, shadow_variant, "prediction_count").inc()

        for pp, sp in zip(primary_preds, shadow_preds, strict=False):
            pair_group = "match" if pp.label == sp.label else "mismatch"
            LABEL_PAIR.labels(
                primary_variant,
                shadow_variant,
                pair_group,
                pp.label,
                sp.label,
            ).inc()
            CONF_DELTA.labels(primary_variant, shadow_variant).observe(
                abs(float(pp.confidence) - float(sp.confidence))
            )
            if pp.label != sp.label:
                MISMATCH_BY_PRIMARY_LABEL.labels(primary_variant, shadow_variant, pp.label).inc()
                MISMATCH_TOTAL.labels(primary_variant, shadow_variant, "top1_label").inc()


def _ttl_seconds() -> float:
    raw = (os.getenv("COMPARE_PAIR_TTL_SECONDS") or "60").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid COMPARE_PAIR_TTL_SECONDS={raw!r}") from exc
    if value <= 0:
        raise RuntimeError("COMPARE_PAIR_TTL_SECONDS must be > 0")
    return value


def create_app() -> FastAPI:
    processor = _PairProcessor(ttl_seconds=_ttl_seconds())
    app = FastAPI(title="Compare Collector", version="0.1.0")
    app.state.processor = processor

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/event", status_code=202)
    def event(payload: CompareEvent) -> dict[str, str]:
        if payload.role not in {"primary", "shadow"}:
            raise HTTPException(status_code=400, detail="role must be primary|shadow")
        if payload.route_variant not in {"v1", "v2"}:
            raise HTTPException(status_code=400, detail="route_variant must be v1|v2")
        if payload.task != "classification":
            return {"status": "ignored", "reason": "task_not_supported"}

        processor.record(payload)
        return {"status": "accepted"}

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        payload = generate_latest()
        return PlainTextResponse(payload.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
