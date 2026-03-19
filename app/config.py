"""Environment-backed settings.

Core runtime values are sourced from `.env` via docker-compose.
New performance flags keep backward-compatible fallbacks so older `.env`
files do not break immediately when only code files are replaced.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel


def _env_required(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _env_optional(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_optional(name)
    if not raw:
        return default
    v = raw.lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean env var {name}={raw!r}")


class Settings(BaseModel):
    model_dir: str
    reload_token: str
    log_level: str
    service_name: str
    service_version: str

    # Safety limits
    max_request_bytes: int
    max_batch_size: int

    # Performance/runtime behavior
    metrics_profile: str
    classification_include_proba_default: bool
    embedding_include_vector_default: bool

    # Async compare publisher (shadow pairing off-path)
    compare_enabled: bool
    compare_url: str
    compare_timeout_s: float
    compare_queue_size: int


@lru_cache
def get_settings() -> Settings:
    metrics_profile = (_env_optional("METRICS_PROFILE") or "lean").lower()
    if metrics_profile not in {"lean", "full"}:
        raise RuntimeError("METRICS_PROFILE must be one of: lean, full")

    return Settings(
        model_dir=_env_required("MODEL_DIR"),
        reload_token=_env_optional("RELOAD_TOKEN"),
        log_level=_env_required("LOG_LEVEL"),
        service_name=_env_required("SERVICE_NAME"),
        service_version=_env_required("SERVICE_VERSION"),
        max_request_bytes=int(_env_required("MAX_REQUEST_BYTES")),
        max_batch_size=int(_env_required("MAX_BATCH_SIZE")),
        metrics_profile=metrics_profile,
        classification_include_proba_default=_env_bool("CLASSIFICATION_INCLUDE_PROBA", False),
        embedding_include_vector_default=_env_bool("EMBEDDING_INCLUDE_VECTOR", True),
        compare_enabled=_env_bool("COMPARE_ENABLED", False),
        compare_url=_env_optional("COMPARE_URL"),
        compare_timeout_s=float(_env_optional("COMPARE_TIMEOUT_S") or "0.5"),
        compare_queue_size=int(_env_optional("COMPARE_QUEUE_SIZE") or "2048"),
    )
