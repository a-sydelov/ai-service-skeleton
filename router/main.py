from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess
from pydantic import BaseModel

from app.logging_conf import setup_logging
from router.telemetry import (
    ROUTER_ACTIVE_VARIANT,
    ROUTER_CUTOVER,
    ROUTER_ROLLBACK,
    ROUTER_ROLLOUT_EVENTS_TOTAL,
    ROUTER_ROLLOUT_LAST_EVENT_TIMESTAMP_SECONDS,
    ROUTER_SERVE_SHADOW_RATE,
    ROUTER_SHADOW_SAMPLE,
    ROUTER_SHADOW_SAMPLE_RATE,
    ROUTER_SPLIT,
    ROUTER_SPLIT_OFF,
    ROUTER_UP,
)

logger = logging.getLogger("router")


@dataclass(frozen=True)
class RouterState:
    primary_variant: str
    shadow_variant: str
    shadow_sample_rate: float
    serve_shadow_rate: float
    version: int
    updated_at: str


class AdminConfigPatch(BaseModel):
    primary_variant: Optional[str] = None
    shadow_variant: Optional[str] = None
    shadow_sample_rate: Optional[float] = None
    serve_shadow_rate: Optional[float] = None


LOG_LEVEL = ""
V1_URL = ""
V2_URL = ""
ADMIN_TOKEN = ""
STATE_FILE = ""


def _env_required(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _env_optional(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _norm_variant(value: str) -> str:
    return (value or "").strip().lower()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _weights_v1_v2(primary: str, shadow: str, serve_shadow_rate: float) -> str:
    p = _norm_variant(primary)
    s = _norm_variant(shadow)
    r = max(0.0, min(1.0, float(serve_shadow_rate)))

    w1 = 0.0
    w2 = 0.0

    if p == "v1":
        if s == "v2" and r > 0.0:
            w1 = 1.0 - r
            w2 = r
        else:
            w1 = 1.0
    elif p == "v2":
        if s == "v1" and r > 0.0:
            w2 = 1.0 - r
            w1 = r
        else:
            w2 = 1.0

    return f"v1={w1:.2f},v2={w2:.2f}"


def _state_to_public_dict(state: RouterState) -> Dict[str, Any]:
    return {
        "v1_url": V1_URL,
        "v2_url": V2_URL,
        "primary_variant": state.primary_variant,
        "shadow_variant": state.shadow_variant,
        "shadow_sample_rate": state.shadow_sample_rate,
        "serve_shadow_rate": state.serve_shadow_rate,
        "admin_enabled": bool(ADMIN_TOKEN),
        "limits": {
            "max_request_bytes": int(_env_required("MAX_REQUEST_BYTES")),
            "max_batch_size": int(_env_required("MAX_BATCH_SIZE")),
        },
        "control_plane_only": True,
        "state_version": state.version,
        "updated_at": state.updated_at,
    }


def _state_to_json_dict(state: RouterState) -> Dict[str, Any]:
    return asdict(state)


def _float01(name: str, value: float) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be 0..1")
    return number


def _state_from_dict(data: Dict[str, Any], fallback: RouterState) -> RouterState:
    primary_variant = _norm_variant(str(data.get("primary_variant", fallback.primary_variant)))
    shadow_variant = _norm_variant(str(data.get("shadow_variant", fallback.shadow_variant)))
    shadow_sample_rate = _float01(
        "shadow_sample_rate", data.get("shadow_sample_rate", fallback.shadow_sample_rate)
    )
    serve_shadow_rate = _float01(
        "serve_shadow_rate", data.get("serve_shadow_rate", fallback.serve_shadow_rate)
    )

    if primary_variant not in {"v1", "v2"}:
        raise ValueError("primary_variant must be v1|v2")
    if shadow_variant not in {"v1", "v2", "off"}:
        raise ValueError("shadow_variant must be v1|v2|off")
    if shadow_variant != "off" and shadow_variant == primary_variant:
        raise ValueError("shadow_variant must differ from primary_variant or be off")
    if shadow_variant == "off" and (shadow_sample_rate > 0.0 or serve_shadow_rate > 0.0):
        raise ValueError("shadow_variant=off requires shadow_sample_rate=0 and serve_shadow_rate=0")

    return RouterState(
        primary_variant=primary_variant,
        shadow_variant=shadow_variant,
        shadow_sample_rate=shadow_sample_rate,
        serve_shadow_rate=serve_shadow_rate,
        version=int(data.get("version", fallback.version)),
        updated_at=str(data.get("updated_at", fallback.updated_at)),
    )


def _write_state(state: RouterState) -> None:
    directory = os.path.dirname(STATE_FILE)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{STATE_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(_state_to_json_dict(state), fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp_path, STATE_FILE)


def _load_state(default_state: RouterState) -> RouterState:
    if not os.path.exists(STATE_FILE):
        _write_state(default_state)
        return default_state

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError("state payload must be an object")
        return _state_from_dict(payload, default_state)
    except Exception:
        logger.exception("state_load_failed", extra={"state_file": STATE_FILE})
        _write_state(default_state)
        return default_state


def _maybe_reload_state() -> RouterState:
    current: RouterState = app.state.state
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            return current
        new_state = _state_from_dict(payload, current)
        app.state.state = new_state
        _publish_config_metrics(new_state)
        return new_state
    except FileNotFoundError:
        return current
    except Exception:
        logger.exception("state_reload_failed", extra={"state_file": STATE_FILE})
        return current


def _publish_config_metrics(state: RouterState) -> None:
    known_variants = ("v1", "v2", "off")

    for variant in known_variants:
        ROUTER_ACTIVE_VARIANT.labels("primary", variant).set(
            1 if variant == state.primary_variant else 0
        )
        ROUTER_ACTIVE_VARIANT.labels("shadow", variant).set(
            1 if variant == state.shadow_variant else 0
        )

    ROUTER_SERVE_SHADOW_RATE.set(state.serve_shadow_rate)
    ROUTER_SHADOW_SAMPLE_RATE.set(state.shadow_sample_rate)


def _admin_enabled() -> bool:
    return bool(ADMIN_TOKEN)


def _check_admin(request: Request) -> bool:
    return request.headers.get("X-Admin-Token", "").strip() == ADMIN_TOKEN


@asynccontextmanager
async def lifespan(app: FastAPI):
    global LOG_LEVEL, V1_URL, V2_URL, ADMIN_TOKEN, STATE_FILE

    LOG_LEVEL = _env_required("LOG_LEVEL")
    V1_URL = _env_required("V1_URL").rstrip("/")
    V2_URL = _env_required("V2_URL").rstrip("/")
    ADMIN_TOKEN = _env_optional("ADMIN_TOKEN")
    STATE_FILE = _env_required("ROUTER_STATE_FILE")

    setup_logging(LOG_LEVEL)
    app.state.client = httpx.AsyncClient()

    default_state = _state_from_dict(
        {
            "primary_variant": _env_required("PRIMARY_VARIANT"),
            "shadow_variant": _env_required("SHADOW_VARIANT"),
            "shadow_sample_rate": float(_env_required("SHADOW_SAMPLE_RATE")),
            "serve_shadow_rate": float(_env_required("SERVE_SHADOW_RATE")),
            "version": 1,
            "updated_at": _utc_now(),
        },
        RouterState(
            primary_variant="v1",
            shadow_variant="off",
            shadow_sample_rate=0.0,
            serve_shadow_rate=0.0,
            version=1,
            updated_at=_utc_now(),
        ),
    )
    app.state.state = _load_state(default_state)
    _publish_config_metrics(app.state.state)
    ROUTER_UP.set(1)

    try:
        yield
    finally:
        ROUTER_UP.set(0)
        await app.state.client.aclose()


app = FastAPI(title="Shadow Router Control Plane", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info() -> Dict[str, Any]:
    state = _maybe_reload_state()
    return _state_to_public_dict(state)


@app.get("/variants/info")
async def variants_info() -> Dict[str, Any]:
    client: httpx.AsyncClient = app.state.client

    async def _get(url: str) -> Dict[str, Any]:
        try:
            response = await client.get(url, timeout=2.5)
            if response.status_code != 200:
                return {"ok": False, "status": response.status_code}
            return {"ok": True, "data": response.json()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    v1 = await _get(f"{V1_URL}/info")
    v2 = await _get(f"{V2_URL}/info")
    return {"v1": v1, "v2": v2}


@app.get("/admin/config")
def admin_get_config(request: Request):
    if not _admin_enabled():
        return JSONResponse({"error": "admin_disabled"}, status_code=404)
    if not _check_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=403)

    state = _maybe_reload_state()
    return {
        "primary_variant": state.primary_variant,
        "shadow_variant": state.shadow_variant,
        "shadow_sample_rate": state.shadow_sample_rate,
        "serve_shadow_rate": state.serve_shadow_rate,
        "version": state.version,
        "updated_at": state.updated_at,
    }


@app.post("/admin/config")
def admin_set_config(patch: AdminConfigPatch, request: Request):
    if not _admin_enabled():
        return JSONResponse({"error": "admin_disabled"}, status_code=404)
    if not _check_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=403)

    old = _maybe_reload_state()

    try:
        new_state = _state_from_dict(
            {
                **_state_to_json_dict(old),
                **patch.model_dump(exclude_none=True),
                "version": old.version + 1,
                "updated_at": _utc_now(),
            },
            old,
        )
    except ValueError as exc:
        return JSONResponse({"error": "bad_request", "details": str(exc)}, status_code=400)

    app.state.state = new_state
    _write_state(new_state)
    _publish_config_metrics(new_state)

    at = _utc_now()
    ts = int(time.time())

    primary_changed = old.primary_variant != new_state.primary_variant
    serve_changed = old.serve_shadow_rate != new_state.serve_shadow_rate
    shadow_sample_changed = old.shadow_sample_rate != new_state.shadow_sample_rate

    if patch.primary_variant is not None and primary_changed:
        if new_state.primary_variant == "v2":
            ROUTER_CUTOVER.labels(
                "control",
                "success",
                "none",
                at,
                old.primary_variant,
                new_state.primary_variant,
                old.shadow_variant,
                new_state.shadow_variant,
            ).set(ts)
        else:
            ROUTER_ROLLBACK.labels(
                "control",
                "success",
                "none",
                at,
                old.primary_variant,
                new_state.primary_variant,
                old.shadow_variant,
                new_state.shadow_variant,
            ).set(ts)

    if patch.serve_shadow_rate is not None and serve_changed:
        w_before = _weights_v1_v2(old.primary_variant, old.shadow_variant, old.serve_shadow_rate)
        w_after = _weights_v1_v2(
            new_state.primary_variant,
            new_state.shadow_variant,
            new_state.serve_shadow_rate,
        )
        if new_state.serve_shadow_rate == 0.0 and old.serve_shadow_rate > 0.0:
            ROUTER_SPLIT_OFF.labels("control", "success", "none", at, w_before, w_after).set(ts)
        else:
            ROUTER_SPLIT.labels("control", "success", "none", at, w_before, w_after).set(ts)

    if patch.shadow_sample_rate is not None and shadow_sample_changed:
        ROUTER_SHADOW_SAMPLE.labels(
            "control",
            "success",
            "none",
            at,
            new_state.shadow_variant,
            f"{old.shadow_sample_rate:.2f}",
            f"{new_state.shadow_sample_rate:.2f}",
        ).set(ts)

    if primary_changed:
        event = "cutover" if new_state.primary_variant == "v2" else "rollback"
        ROUTER_ROLLOUT_EVENTS_TOTAL.labels(
            event, old.primary_variant, new_state.primary_variant
        ).inc()
        ROUTER_ROLLOUT_LAST_EVENT_TIMESTAMP_SECONDS.labels(event).set(time.time())

    return {"status": "updated", "version": new_state.version, "updated_at": new_state.updated_at}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
