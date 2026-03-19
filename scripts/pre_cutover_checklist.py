"""Pre-cutover checklist.

Queries router + Prometheus and prints a concise recommendation.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Tuple

import httpx


def prom_query(client: httpx.Client, prom_url: str, expr: str) -> Dict[str, Any]:
    r = client.get(f"{prom_url.rstrip('/')}/api/v1/query", params={"query": expr}, timeout=5.0)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected Prometheus response type: {type(data)}")
    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {data}")
    return data


def parse_vector(result: Dict[str, Any]) -> List[Tuple[Dict[str, str], float]]:
    out: List[Tuple[Dict[str, str], float]] = []
    res = (result.get("data", {}) or {}).get("result", []) or []
    for item in res:
        metric = item.get("metric", {}) or {}
        value = item.get("value", [None, "nan"])
        try:
            v = float(value[1])
        except Exception:
            v = float("nan")
        out.append((metric, v))
    return out


def _first_value(vec: List[Tuple[Dict[str, str], float]]) -> float:
    return vec[0][1] if vec else 0.0


def fmt_kv(title: str, kv: Dict[str, Any]) -> str:
    s = [f"{title}:"]
    for k, v in kv.items():
        s.append(f"  - {k}: {v}")
    return "\n".join(s)


def fmt_pct01(x: float) -> str:
    """Format a 0..1 rate as a human-friendly percent."""
    pct = float(x) * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{pct:.0f}%"
    return f"{pct:.1f}%"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--router-url", default="http://router:8080")
    p.add_argument("--prom-url", default="http://prometheus:9090")
    p.add_argument("--window", default="5m")
    p.add_argument("--primary-error-th", type=float, default=0.02)
    p.add_argument("--primary-p95-th", type=float, default=0.50)
    p.add_argument("--shadow-error-th", type=float, default=0.05)
    p.add_argument("--mismatch-rate-th", type=float, default=0.20)
    args = p.parse_args()

    router_url = args.router_url.rstrip("/")
    prom_url = args.prom_url.rstrip("/")
    w = args.window

    issues: List[str] = []
    warnings: List[str] = []

    with httpx.Client() as client:
        try:
            rinfo = client.get(f"{router_url}/info", timeout=3.0).json()
        except Exception as e:
            print(f"FAILED: router /info: {e}", file=sys.stderr)
            return 2

        try:
            vinfo = client.get(f"{router_url}/variants/info", timeout=5.0).json()
        except Exception as e:
            print(f"FAILED: router /variants/info: {e}", file=sys.stderr)
            return 2

        primary = str(rinfo.get("primary_variant", "v1"))
        shadow = str(rinfo.get("shadow_variant", "off"))
        sample = float(rinfo.get("shadow_sample_rate", 0.0) or 0.0)
        serve_shadow_rate = float(rinfo.get("serve_shadow_rate", 0.0) or 0.0)

        served_shadow_pct = serve_shadow_rate if shadow not in ("", "off") else 0.0
        served_primary_pct = 1.0 - served_shadow_pct

        router_view: Dict[str, Any] = {
            "v1_url": rinfo.get("v1_url"),
            "v2_url": rinfo.get("v2_url"),
            "primary_variant": primary,
            "shadow_variant": shadow,
            "served_split_primary": fmt_pct01(served_primary_pct),
            "served_split_shadow": fmt_pct01(served_shadow_pct),
            "shadow_sample_rate": fmt_pct01(sample),
            "admin_enabled": rinfo.get("admin_enabled"),
            "limits": rinfo.get("limits"),
        }

        print("=== PRE-CUTOVER CHECKLIST (read-only) ===")
        print(fmt_kv("Router", router_view))
        print()

        def _cond(v: Dict[str, Any]) -> str:
            if not v.get("ok"):
                return f"NOT_OK ({v})"
            data = v.get("data", {})
            model = data.get("model", {})
            return (
                f"ok runtime={model.get('runtime')} "
                f"name={model.get('name')} "
                f"ver={model.get('version')} "
                f"task={model.get('task')}"
            )

        print("Variants:")
        print(f"  - v1: {_cond(vinfo.get('v1', {}))}")
        print(f"  - v2: {_cond(vinfo.get('v2', {}))}")
        print()

        if not vinfo.get("v1", {}).get("ok"):
            issues.append("v1 /info not ok")
        if not vinfo.get("v2", {}).get("ok"):
            issues.append("v2 /info not ok")

        up_vec = parse_vector(
            prom_query(client, prom_url, 'sum by (job) (up{job=~"inference_v[12]"})')
        )
        print("Up replicas by job:")
        for m, v in up_vec:
            job = m.get("job", "?")
            print(f"  - {job}: {v:.0f}")
            if v <= 0:
                issues.append(f"{job} has 0 up targets")
        print()

        act_expr = (
            "sum by (job, model_name, model_version, task, model_dir, runtime) "
            '(model_active_info{job=~"inference_v[12]",runtime=~".+"})'
        )
        act_vec = parse_vector(prom_query(client, prom_url, act_expr))
        print("Active model summary (grouped):")
        if not act_vec:
            warnings.append("model_active_info is empty (wait for scrapes or check inference)")
            print("  (empty)")
        else:
            for m, v in act_vec:
                print(
                    "  - "
                    f"job={m.get('job', '?')} "
                    f"name={m.get('model_name', '?')} "
                    f"ver={m.get('model_version', '?')} "
                    f"task={m.get('task', '?')} "
                    f"runtime={m.get('runtime', '?')} "
                    f"dir={m.get('model_dir', '?')} "
                    f"count={v:.0f}"
                )
        print()

        primary_err = (
            f'sum(rate(http_requests_total{{variant="{primary}",path="/predict",status=~"5.."}}[{w}]))'
            f' / clamp_min(sum(rate(http_requests_total{{variant="{primary}",path="/predict"}}[{w}])), 0.001)'
        )
        primary_p95 = (
            f"histogram_quantile(0.95, "
            f'sum by (le) (rate(shadow_primary_request_duration_seconds_bucket{{variant="{primary}"}}[{w}])))'
        )
        primary_rps = (
            f'sum(rate(shadow_primary_requests_total{{variant="{primary}",status="success"}}[{w}]))'
        )

        perr_v = _first_value(parse_vector(prom_query(client, prom_url, primary_err)))
        pp95_v = _first_value(parse_vector(prom_query(client, prom_url, primary_p95)))
        prps_v = _first_value(parse_vector(prom_query(client, prom_url, primary_rps)))

        print(f"Primary metrics (variant={primary}, window={w}):")
        print(f"  - rps_success: {prps_v:.4f}")
        print(f"  - error_ratio: {perr_v:.4f}")
        print(f"  - p95_latency_s: {pp95_v:.4f}")
        print()

        if perr_v > args.primary_error_th:
            issues.append(f"Primary error_ratio {perr_v:.4f} > {args.primary_error_th}")
        if pp95_v > args.primary_p95_th:
            warnings.append(f"Primary p95 {pp95_v:.4f}s > {args.primary_p95_th}s")

        if shadow.lower() == "off" or sample <= 0.0:
            warnings.append("Shadow is OFF (no shadow gating signals)")
        else:
            shadow_err = (
                f'sum(rate(http_requests_total{{variant="{shadow}",path="/predict",status=~"5.."}}[{w}]))'
                f' / clamp_min(sum(rate(http_requests_total{{variant="{shadow}",path="/predict"}}[{w}])), 0.001)'
            )
            shadow_p95 = (
                f"histogram_quantile(0.95, "
                f'sum by (le) (rate(shadow_request_duration_seconds_bucket{{variant="{shadow}"}}[{w}])))'
            )
            mismatch_rate = f"sum(rate(shadow_mismatch_total[{w}]))"
            conf_p95 = (
                f"histogram_quantile(0.95, sum by (le) (rate(shadow_confidence_delta_bucket[{w}])))"
            )

            serr_v = _first_value(parse_vector(prom_query(client, prom_url, shadow_err)))
            sp95_v = _first_value(parse_vector(prom_query(client, prom_url, shadow_p95)))
            mm_v = _first_value(parse_vector(prom_query(client, prom_url, mismatch_rate)))
            cd_v = _first_value(parse_vector(prom_query(client, prom_url, conf_p95)))

            print(f"Shadow metrics (variant={shadow}, window={w}):")
            print(f"  - error_ratio: {serr_v:.4f}")
            print(f"  - p95_latency_s: {sp95_v:.4f}")
            print(f"  - mismatch_rate_per_s: {mm_v:.4f}")
            print(f"  - confidence_delta_p95: {cd_v:.4f}")
            print()

            if serr_v > args.shadow_error_th:
                warnings.append(f"Shadow error_ratio {serr_v:.4f} > {args.shadow_error_th}")
            if mm_v > args.mismatch_rate_th:
                warnings.append(f"Shadow mismatch_rate {mm_v:.4f}/s > {args.mismatch_rate_th}/s")

        print("Recommendation:")
        if issues:
            print("  - STATUS: RED (do NOT cutover)")
        elif warnings:
            print("  - STATUS: YELLOW (cutover only if you accept risk)")
        else:
            print("  - STATUS: GREEN (safe to cutover)")

        if issues:
            print("Issues:")
            for x in issues:
                print(f"  - {x}")
        if warnings:
            print("Warnings:")
            for x in warnings:
                print(f"  - {x}")

        print()
        print("Manual actions:")
        print("  - show variants info:  curl -s http://localhost:8080/variants/info | head -n 80")
        print("  - cutover:             make cutover")
        print("  - rollback:            make rollback")
        print("  - serve split:         make split PCT=10        # serve 10% by shadow (canary)")
        print("  - shadow sample:       make shadow-sample PCT=10 # mirror 10% to shadow (compare)")

        return 2 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
