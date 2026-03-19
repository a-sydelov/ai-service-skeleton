"""Comparison metrics exported by the compare collector."""

from prometheus_client import Counter, Histogram

PRIMARY_REQUESTS = Counter(
    "shadow_primary_requests_total", "Primary (served) requests total", ["variant", "status"]
)
SHADOW_REQUESTS = Counter("shadow_requests_total", "Shadow requests total", ["variant", "status"])

PRIMARY_LATENCY = Histogram(
    "shadow_primary_request_duration_seconds",
    "Primary request duration seconds",
    ["variant"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float("inf")),
)
SHADOW_LATENCY = Histogram(
    "shadow_request_duration_seconds",
    "Shadow request duration seconds",
    ["variant"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float("inf")),
)

MISMATCH_TOTAL = Counter(
    "shadow_mismatch_total",
    "Shadow mismatches total",
    ["primary_variant", "shadow_variant", "reason"],
)

MISMATCH_BY_PRIMARY_LABEL = Counter(
    "shadow_mismatch_by_primary_label_total",
    "Shadow mismatches total grouped by primary predicted label",
    ["primary_variant", "shadow_variant", "primary_label"],
)

LABEL_PAIR = Counter(
    "shadow_label_pair_total",
    "Primary vs shadow label pairs",
    ["primary_variant", "shadow_variant", "pair_group", "primary_label", "shadow_label"],
)

CONF_DELTA = Histogram(
    "shadow_confidence_delta",
    "Absolute delta between primary and shadow top-1 confidence",
    ["primary_variant", "shadow_variant"],
    buckets=(0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0),
)
