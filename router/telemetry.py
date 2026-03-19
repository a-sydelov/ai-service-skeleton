from prometheus_client import Counter, Gauge

ROUTER_UP = Gauge(
    "router_up",
    "1 if router is running, else 0",
    multiprocess_mode="max",
)

# Active variants (value=1) — displayed in Grafana as "Primary" / "Shadow".
ROUTER_ACTIVE_VARIANT = Gauge(
    "router_active_variant",
    "Active variants by role (value=1)",
    ["role", "variant"],
    multiprocess_mode="max",
)

# Rollout knobs as numeric gauges (0..1). Easier to graph than label-only snapshots.
ROUTER_SERVE_SHADOW_RATE = Gauge(
    "router_serve_shadow_rate",
    "Fraction of served requests routed to the shadow variant (0..1)",
    multiprocess_mode="max",
)

ROUTER_SHADOW_SAMPLE_RATE = Gauge(
    "router_shadow_sample_rate",
    "Fraction of served requests mirrored to shadow for comparison (0..1)",
    multiprocess_mode="max",
)

# Rollout events (used for Grafana annotations / audit tables).
ROUTER_ROLLOUT_EVENTS_TOTAL = Counter(
    "router_rollout_events_total",
    "Rollout events total",
    ["event", "from_primary", "to_primary"],
)

ROUTER_ROLLOUT_LAST_EVENT_TIMESTAMP_SECONDS = Gauge(
    "router_rollout_last_event_timestamp_seconds",
    "Last rollout event unix timestamp seconds",
    ["event"],
    multiprocess_mode="max",
)

# Manual control events (one series per action occurrence).
# Value: unix timestamp seconds of the action.
ROUTER_CUTOVER = Gauge(
    "router_cutover",
    "Cutover action (value=unix timestamp seconds)",
    [
        "group",
        "result",
        "error",
        "at",
        "primary_before",
        "primary_after",
        "shadow_before",
        "shadow_after",
    ],
    multiprocess_mode="max",
)

ROUTER_ROLLBACK = Gauge(
    "router_rollback",
    "Rollback action (value=unix timestamp seconds)",
    [
        "group",
        "result",
        "error",
        "at",
        "primary_before",
        "primary_after",
        "shadow_before",
        "shadow_after",
    ],
    multiprocess_mode="max",
)

ROUTER_SHADOW_SAMPLE = Gauge(
    "router_shadow_sample",
    "Shadow-sample action (value=unix timestamp seconds)",
    [
        "group",
        "result",
        "error",
        "at",
        "shadow",
        "rate_before",
        "rate_after",
    ],
    multiprocess_mode="max",
)

ROUTER_SPLIT = Gauge(
    "router_split",
    "Split action (value=unix timestamp seconds)",
    [
        "group",
        "result",
        "error",
        "at",
        "weights_before",
        "weights_after",
    ],
    multiprocess_mode="max",
)

ROUTER_SPLIT_OFF = Gauge(
    "router_split_off",
    "Split-off action (value=unix timestamp seconds)",
    [
        "group",
        "result",
        "error",
        "at",
        "weights_before",
        "weights_after",
    ],
    multiprocess_mode="max",
)
