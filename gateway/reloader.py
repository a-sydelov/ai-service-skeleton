from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("gateway_reloader")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@dataclass(frozen=True)
class RolloutState:
    primary_variant: str
    shadow_variant: str
    shadow_sample_rate: float
    serve_shadow_rate: float
    version: int
    updated_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_variant(value: str) -> str:
    return (value or "").strip().lower()


def float01(value: Any) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError("rate must be 0..1")
    return number


def build_state(
    *,
    primary_variant: str,
    shadow_variant: str,
    shadow_sample_rate: Any,
    serve_shadow_rate: Any,
    version: int,
    updated_at: str,
) -> RolloutState:
    primary_variant = normalize_variant(primary_variant)
    shadow_variant = normalize_variant(shadow_variant)
    shadow_sample_rate = float01(shadow_sample_rate)
    serve_shadow_rate = float01(serve_shadow_rate)

    if primary_variant not in {"v1", "v2"}:
        raise ValueError("primary_variant must be v1|v2")
    if shadow_variant not in {"v1", "v2", "off"}:
        raise ValueError("shadow_variant must be v1|v2|off")
    if shadow_variant != "off" and shadow_variant == primary_variant:
        raise ValueError("shadow_variant must differ from primary_variant or be off")
    if shadow_variant == "off" and (shadow_sample_rate > 0.0 or serve_shadow_rate > 0.0):
        raise ValueError("shadow_variant=off requires shadow_sample_rate=0 and serve_shadow_rate=0")

    return RolloutState(
        primary_variant=primary_variant,
        shadow_variant=shadow_variant,
        shadow_sample_rate=shadow_sample_rate,
        serve_shadow_rate=serve_shadow_rate,
        version=version,
        updated_at=updated_at,
    )


def default_state() -> RolloutState:
    return build_state(
        primary_variant=(normalize_variant(os.getenv("PRIMARY_VARIANT", "v1")) or "v1"),
        shadow_variant=(normalize_variant(os.getenv("SHADOW_VARIANT", "v2")) or "v2"),
        shadow_sample_rate=os.getenv("SHADOW_SAMPLE_RATE", "0.1"),
        serve_shadow_rate=os.getenv("SERVE_SHADOW_RATE", "0.0"),
        version=1,
        updated_at=utc_now(),
    )


def state_from_payload(payload: dict[str, Any], fallback: RolloutState) -> RolloutState:
    return build_state(
        primary_variant=str(payload.get("primary_variant", fallback.primary_variant)),
        shadow_variant=str(payload.get("shadow_variant", fallback.shadow_variant)),
        shadow_sample_rate=payload.get("shadow_sample_rate", fallback.shadow_sample_rate),
        serve_shadow_rate=payload.get("serve_shadow_rate", fallback.serve_shadow_rate),
        version=int(payload.get("version", fallback.version)),
        updated_at=str(payload.get("updated_at", fallback.updated_at)),
    )


def load_state(path: Path) -> RolloutState:
    base = default_state()
    if not path.exists():
        write_state(path, base)
        return base

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("rollout state must be a JSON object")
        return state_from_payload(payload, base)
    except Exception:
        LOG.exception("state_load_failed", extra={"state_file": str(path)})
        write_state(path, base)
        return base


def write_state(path: Path, state: RolloutState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state.__dict__, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def compact_percent(rate: float) -> str:
    value = rate * 100.0
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return f"{text}%"


def render_http_conf(state: RolloutState) -> str:
    primary = state.primary_variant
    shadow = state.shadow_variant

    served_variant_map = [
        "map $served_bucket $served_variant {",
        f"    default {primary};",
        f"    primary {primary};",
        f"    shadow {shadow if shadow != 'off' else primary};",
        "}",
    ]

    if shadow == "off" or state.serve_shadow_rate <= 0.0:
        serve_bucket = [
            "map $request_id_for_rollout $served_bucket {",
            "    default primary;",
            "}",
        ]
    elif state.serve_shadow_rate >= 1.0:
        serve_bucket = [
            "map $request_id_for_rollout $served_bucket {",
            "    default shadow;",
            "}",
        ]
    else:
        serve_bucket = [
            'split_clients "${request_id_for_rollout}:serve" $served_bucket {',
            f"    {compact_percent(state.serve_shadow_rate)} shadow;",
            "    * primary;",
            "}",
        ]

    if shadow == "off" or state.shadow_sample_rate <= 0.0:
        shadow_sample = [
            "map $request_id_for_rollout $shadow_sample_bucket {",
            "    default off;",
            "}",
        ]
    elif state.shadow_sample_rate >= 1.0:
        shadow_sample = [
            "map $request_id_for_rollout $shadow_sample_bucket {",
            "    default on;",
            "}",
        ]
    else:
        shadow_sample = [
            'split_clients "${request_id_for_rollout}:shadow" $shadow_sample_bucket {',
            f"    {compact_percent(state.shadow_sample_rate)} on;",
            "    * off;",
            "}",
        ]

    if shadow == "off":
        shadow_map = [
            'map "$served_variant:$shadow_sample_bucket" $shadow_internal_uri {',
            "    default /__shadow_off;",
            "}",
        ]
    elif shadow == "v1":
        shadow_map = [
            'map "$served_variant:$shadow_sample_bucket" $shadow_internal_uri {',
            "    default /__shadow_off;",
            '    "v2:on" /__shadow_v1;',
            "}",
        ]
    else:
        shadow_map = [
            'map "$served_variant:$shadow_sample_bucket" $shadow_internal_uri {',
            "    default /__shadow_off;",
            '    "v1:on" /__shadow_v2;',
            "}",
        ]

    predict_uri_map = [
        "map $served_variant $predict_internal_uri {",
        "    default /__predict_v1;",
        "    v1 /__predict_v1;",
        "    v2 /__predict_v2;",
        "}",
    ]

    lines = [
        "# generated by gateway/reloader.py — do not edit",
        "map $http_x_request_id $request_id_for_rollout {",
        "    default $http_x_request_id;",
        '    "" $request_id;',
        "}",
        *serve_bucket,
        *served_variant_map,
        *predict_uri_map,
        *shadow_sample,
        *shadow_map,
    ]
    return "\n".join(lines) + "\n"


def render_predict_conf(state: RolloutState) -> str:
    if state.shadow_variant == "off" or state.shadow_sample_rate <= 0.0:
        return "# generated by gateway/reloader.py — do not edit\nmirror off;\n"
    return (
        "# generated by gateway/reloader.py — do not edit\n"
        "mirror /__shadow_dispatch;\n"
        "mirror_request_body on;\n"
    )


def file_hash(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.exists():
            digest.update(b"<missing>")
            continue
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_check(nginx_conf: str) -> None:
    subprocess.run(["nginx", "-t", "-c", nginx_conf], check=True, capture_output=True, text=True)


def run_reload() -> None:
    subprocess.run(["nginx", "-s", "reload"], check=True, capture_output=True, text=True)


def run_check_with_candidates(
    nginx_conf: str,
    http_out: Path,
    http_candidate: Path,
    predict_out: Path,
    predict_candidate: Path,
) -> None:
    conf_text = Path(nginx_conf).read_text(encoding="utf-8")
    conf_text = conf_text.replace(str(http_out), str(http_candidate))
    conf_text = conf_text.replace(str(predict_out), str(predict_candidate))
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as fh:
        fh.write(conf_text)
        temp_conf = fh.name
    try:
        run_check(temp_conf)
    finally:
        try:
            os.unlink(temp_conf)
        except FileNotFoundError:
            pass


def render_once(
    state_file: Path, http_out: Path, predict_out: Path, nginx_conf: str, do_reload: bool
) -> None:
    state = load_state(state_file)
    http_conf = render_http_conf(state)
    predict_conf = render_predict_conf(state)

    http_out.parent.mkdir(parents=True, exist_ok=True)
    predict_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(http_out.parent), encoding="utf-8"
    ) as http_fh:
        http_fh.write(http_conf)
        http_candidate = Path(http_fh.name)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(predict_out.parent), encoding="utf-8"
    ) as predict_fh:
        predict_fh.write(predict_conf)
        predict_candidate = Path(predict_fh.name)

    try:
        run_check_with_candidates(
            nginx_conf, http_out, http_candidate, predict_out, predict_candidate
        )
        http_candidate.replace(http_out)
        predict_candidate.replace(predict_out)
        if do_reload:
            run_reload()
    finally:
        for candidate in (http_candidate, predict_candidate):
            if candidate.exists():
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass

    LOG.info(
        "rollout_applied primary=%s shadow=%s serve_shadow_rate=%.4f shadow_sample_rate=%.4f version=%s",
        state.primary_variant,
        state.shadow_variant,
        state.serve_shadow_rate,
        state.shadow_sample_rate,
        state.version,
    )


def watch_loop(
    state_file: Path, http_out: Path, predict_out: Path, nginx_conf: str, interval: float
) -> None:
    last_seen = file_hash(state_file)
    while True:
        time.sleep(interval)
        current = file_hash(state_file)
        if current == last_seen:
            continue
        try:
            render_once(state_file, http_out, predict_out, nginx_conf, do_reload=True)
            last_seen = current
        except Exception:
            LOG.exception("rollout_apply_failed", extra={"state_file": str(state_file)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--http-out", required=True)
    parser.add_argument("--predict-out", required=True)
    parser.add_argument("--nginx-conf", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_file = Path(args.state_file)
    http_out = Path(args.http_out)
    predict_out = Path(args.predict_out)

    if args.once:
        render_once(state_file, http_out, predict_out, args.nginx_conf, do_reload=False)
        return 0

    if args.watch:
        watch_loop(state_file, http_out, predict_out, args.nginx_conf, args.interval)
        return 0

    raise SystemExit("use --once or --watch")


if __name__ == "__main__":
    raise SystemExit(main())
