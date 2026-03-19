from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import httpx

DEFAULT_ROUTER_URL = "http://router:8080"
DEFAULT_GATEWAY_URL = "http://gateway:8080"
DEFAULT_STATE_FILE = "/state/rollout_state.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _env_float(name: str) -> float:
    return float(_env(name))


def _pct_to_rate(pct: float) -> float:
    if pct < 0.0 or pct > 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    return pct / 100.0


def _float01(value: Any, field_name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {number}")
    return number


def _normalize_variant(value: str) -> str:
    return (value or "").strip().lower()


def _build_state(
    *,
    primary_variant: str,
    shadow_variant: str,
    shadow_sample_rate: Any,
    serve_shadow_rate: Any,
    version: int,
    updated_at: str,
) -> Dict[str, Any]:
    primary_variant = _normalize_variant(primary_variant)
    shadow_variant = _normalize_variant(shadow_variant)
    shadow_sample_rate = _float01(shadow_sample_rate, "shadow_sample_rate")
    serve_shadow_rate = _float01(serve_shadow_rate, "serve_shadow_rate")

    if primary_variant not in {"v1", "v2"}:
        raise ValueError("primary_variant must be v1|v2")
    if shadow_variant not in {"v1", "v2", "off"}:
        raise ValueError("shadow_variant must be v1|v2|off")
    if shadow_variant != "off" and shadow_variant == primary_variant:
        raise ValueError("shadow_variant must differ from primary_variant or be off")
    if shadow_variant == "off" and (shadow_sample_rate > 0.0 or serve_shadow_rate > 0.0):
        raise ValueError("shadow_variant=off requires shadow_sample_rate=0 and serve_shadow_rate=0")

    return {
        "primary_variant": primary_variant,
        "shadow_variant": shadow_variant,
        "shadow_sample_rate": shadow_sample_rate,
        "serve_shadow_rate": serve_shadow_rate,
        "version": version,
        "updated_at": updated_at,
    }


def _admin_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Admin-Token": _env("ADMIN_TOKEN"),
    }


def _post_config(payload: Dict[str, Any], router_url: str, timeout: float) -> int:
    url = f"{router_url.rstrip('/')}/admin/config"
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=_admin_headers(), json=payload)
        print(response.text)
        response.raise_for_status()
    return 0


def cmd_cutover(args: argparse.Namespace) -> int:
    payload = {
        "primary_variant": "v2",
        "shadow_variant": "v1",
    }
    return _post_config(payload, args.router_url, args.timeout)


def cmd_rollback(args: argparse.Namespace) -> int:
    payload = {
        "primary_variant": "v1",
        "shadow_variant": "v2",
    }
    return _post_config(payload, args.router_url, args.timeout)


def cmd_shadow_sample(args: argparse.Namespace) -> int:
    payload = {"shadow_sample_rate": _pct_to_rate(args.pct)}
    return _post_config(payload, args.router_url, args.timeout)


def cmd_split(args: argparse.Namespace) -> int:
    payload = {"serve_shadow_rate": _pct_to_rate(args.pct)}
    return _post_config(payload, args.router_url, args.timeout)


def cmd_split_off(args: argparse.Namespace) -> int:
    payload = {"serve_shadow_rate": 0.0}
    return _post_config(payload, args.router_url, args.timeout)


def _default_state() -> Dict[str, Any]:
    return _build_state(
        primary_variant=_env("PRIMARY_VARIANT"),
        shadow_variant=_env("SHADOW_VARIANT"),
        shadow_sample_rate=_env_float("SHADOW_SAMPLE_RATE"),
        serve_shadow_rate=_env_float("SERVE_SHADOW_RATE"),
        version=1,
        updated_at=_utc_now(),
    )


def cmd_reset_state(args: argparse.Namespace) -> int:
    path = Path(args.state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _default_state()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{path} reset")
    return 0


def cmd_require_up(args: argparse.Namespace) -> int:
    url = f"{args.gateway_url.rstrip('/')}/healthz"
    try:
        with httpx.Client(timeout=args.timeout) as client:
            response = client.get(url)
            response.raise_for_status()
    except Exception:
        print("ERROR: gateway is not up. Run: make up", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control-plane admin operations executed inside the router container."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler in (("cutover", cmd_cutover), ("rollback", cmd_rollback)):
        sub = subparsers.add_parser(name)
        sub.add_argument("--router-url", default=DEFAULT_ROUTER_URL)
        sub.add_argument("--timeout", type=float, default=10.0)
        sub.set_defaults(func=handler)

    sub = subparsers.add_parser("shadow-sample")
    sub.add_argument("--pct", type=float, required=True)
    sub.add_argument("--router-url", default=DEFAULT_ROUTER_URL)
    sub.add_argument("--timeout", type=float, default=10.0)
    sub.set_defaults(func=cmd_shadow_sample)

    sub = subparsers.add_parser("split")
    sub.add_argument("--pct", type=float, required=True)
    sub.add_argument("--router-url", default=DEFAULT_ROUTER_URL)
    sub.add_argument("--timeout", type=float, default=10.0)
    sub.set_defaults(func=cmd_split)

    sub = subparsers.add_parser("split-off")
    sub.add_argument("--router-url", default=DEFAULT_ROUTER_URL)
    sub.add_argument("--timeout", type=float, default=10.0)
    sub.set_defaults(func=cmd_split_off)

    sub = subparsers.add_parser("reset-state")
    sub.add_argument(
        "--state-file",
        default=os.environ.get("ROUTER_STATE_FILE", DEFAULT_STATE_FILE),
    )
    sub.set_defaults(func=cmd_reset_state)

    sub = subparsers.add_parser("require-up")
    sub.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    sub.add_argument("--timeout", type=float, default=5.0)
    sub.set_defaults(func=cmd_require_up)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
