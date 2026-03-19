"""Prometheus metrics.

Principles:
- HTTP metrics should be generic and reusable across model tasks.
- Model metrics should be stable identifiers (name/version/task/runtime), not ad-hoc strings.
"""

from prometheus_client import Counter, Gauge, Histogram

REQ_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQ_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "path"])
# In multiprocess mode, inflight should sum across workers.
HTTP_INFLIGHT = Gauge(
    "http_in_flight",
    "In-flight HTTP requests",
    ["method", "path"],
    multiprocess_mode="sum",
)

HTTP_REQ_SIZE = Histogram(
    "http_request_size_bytes",
    "HTTP request size in bytes (from Content-Length)",
    ["method", "path"],
    buckets=(0, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, float("inf")),
)

HTTP_RESP_SIZE = Histogram(
    "http_response_size_bytes",
    "HTTP response size in bytes (from Content-Length)",
    ["method", "path", "status"],
    buckets=(0, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, float("inf")),
)

MODEL_UP = Gauge(
    "model_loaded",
    "1 if model is loaded, else 0",
    multiprocess_mode="max",
)

# This is intentionally a "gauge-as-info": it carries labels describing the loaded artifact.
MODEL_ACTIVE_INFO = Gauge(
    "model_active_info",
    "Active model info (value=1)",
    ["model_name", "model_version", "model_dir", "task", "runtime"],
    multiprocess_mode="max",
)

# Per-container/process identity (value=1). This is a low-cardinality join key
# for infra/container metrics (cAdvisor) and inference metrics.
INFERENCE_INSTANCE_INFO = Gauge(
    "inference_instance_info",
    "Inference instance identity (value=1)",
    ["container_id"],
    multiprocess_mode="max",
)

MODEL_RELOAD_TOTAL = Counter("model_reload_total", "Total model reload attempts", ["status"])

MODEL_RELOAD_SUCCESS_TOTAL = Counter("model_reload_success_total", "Total successful model reloads")
MODEL_RELOAD_FAILED_TOTAL = Counter("model_reload_failed_total", "Total failed model reloads")

MODEL_PRED_COUNT = Counter(
    "model_predictions_total", "Total model predictions", ["model_name", "model_version", "task"]
)
MODEL_PRED_LABEL = Counter(
    "model_prediction_label_total",
    "Predicted labels total",
    ["model_name", "model_version", "task", "label"],
)

MODEL_CONF = Histogram(
    "model_prediction_confidence",
    "Prediction confidence histogram",
    ["model_name", "model_version", "task"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

MODEL_BATCH = Histogram(
    "model_batch_size",
    "Batch size per request",
    ["model_name", "model_version", "task"],
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, float("inf")),
)

# Pure inference latency (excluding HTTP middleware and serialization).
MODEL_INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "Model inference duration (predict call only)",
    ["model_name", "model_version", "task", "runtime"],
    buckets=(
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        float("inf"),
    ),
)

# Stable error counters for model stages (low-cardinality enums only).
MODEL_ERRORS_TOTAL = Counter(
    "model_errors_total",
    "Total model errors by stage and reason",
    ["stage", "reason"],
)

MODEL_OUTPUT_VALUE = Histogram(
    "model_output_value",
    "Regression output value histogram (task=regression)",
    ["model_name", "model_version"],
    buckets=(-1000, -100, -10, -1, -0.1, 0, 0.1, 1, 10, 100, 1000, float("inf")),
)

MODEL_EMBED_NORM = Histogram(
    "model_embedding_norm",
    "Embedding vector L2 norm histogram (task=embedding)",
    ["model_name", "model_version"],
    buckets=(0, 0.5, 1, 2, 5, 10, 20, 50, 100, float("inf")),
)
