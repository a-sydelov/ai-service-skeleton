"""Load generator.

Profiles:
- smoke
- load
- boundary

Payload file is resolved automatically from:
  DATA_ROOT / "payloads" / f"{profile}.json"

Examples:
- smoke    -> data/payloads/smoke.json
- load     -> data/payloads/load.json
- boundary -> data/payloads/boundary.json

Supported payload file formats:
1) A single request body:
   {"instances": [[...]]}

2) A payload pool file:
   {
     "format": "predict-payload-pool/v1",
     "payloads": [
       {"instances": [[...]]},
       {"instances": [[...]]}
     ]
   }

For pool files, requests are sent in round-robin order across payloads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class Phase:
    name: str
    duration_s: float
    rps: float
    concurrency: int


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        values[key] = value

    return values


def _data_root() -> Path:
    env_value = os.getenv("DATA_ROOT")
    if env_value:
        return Path(env_value)

    env_file = _project_root() / ".env"
    file_values = _read_env_file(env_file)
    return Path(file_values.get("DATA_ROOT", "data"))


def _payload_file_for_profile(profile: str) -> Path:
    return _data_root() / "payloads" / f"{profile}.json"


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs2 = sorted(xs)
    k = int((len(xs2) - 1) * q)
    return xs2[k]


def _load_payloads_from_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Payload file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "payloads" in data:
        payloads = data.get("payloads")
        if not isinstance(payloads, list) or not payloads:
            raise RuntimeError(f"Payload file has empty or invalid 'payloads' list: {path}")
        for idx, payload in enumerate(payloads):
            if not isinstance(payload, dict):
                raise RuntimeError(f"Payload at index {idx} must be a JSON object: {path}")
        return payloads

    if isinstance(data, dict):
        return [data]

    raise RuntimeError(
        f"Payload file must contain either a JSON object or a 'payloads' list: {path}"
    )


async def _one(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    sem: asyncio.Semaphore,
) -> tuple[bool, float]:
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json=payload, timeout=5.0)
            ok = 200 <= r.status_code < 300
        except Exception:
            ok = False
        dt = time.perf_counter() - t0
        return ok, dt


async def run_phase(url: str, payloads: list[dict[str, Any]], phase: Phase) -> dict[str, Any]:
    if not payloads:
        raise RuntimeError("Payload pool is empty")

    sem = asyncio.Semaphore(phase.concurrency)
    limits = httpx.Limits(
        max_keepalive_connections=phase.concurrency,
        max_connections=phase.concurrency * 2,
    )

    ok_count = 0
    err_count = 0
    latencies: list[float] = []

    async with httpx.AsyncClient(limits=limits) as client:
        start = time.perf_counter()
        next_t = start
        interval = 1.0 / phase.rps if phase.rps > 0 else 0.0

        tasks: list[asyncio.Task] = []
        request_no = 0

        async def _collect(t: asyncio.Task) -> None:
            nonlocal ok_count, err_count
            ok, dt = await t
            latencies.append(dt)
            if ok:
                ok_count += 1
            else:
                err_count += 1

        while True:
            now = time.perf_counter()
            if now - start >= phase.duration_s:
                break

            if phase.rps > 0:
                if now < next_t:
                    await asyncio.sleep(next_t - now)
                next_t += interval

            payload = payloads[request_no % len(payloads)]
            request_no += 1

            t = asyncio.create_task(_one(client, url, payload, sem))
            tasks.append(asyncio.create_task(_collect(t)))

        if tasks:
            await asyncio.gather(*tasks)

    total = ok_count + err_count
    return {
        "phase": phase.name,
        "duration_s": phase.duration_s,
        "rps_target": phase.rps,
        "concurrency": phase.concurrency,
        "payload_pool_size": len(payloads),
        "requests_total": total,
        "ok": ok_count,
        "errors": err_count,
        "p50_s": _percentile(latencies, 0.50),
        "p95_s": _percentile(latencies, 0.95),
        "max_s": max(latencies) if latencies else 0.0,
    }


def profile_by_name(name: str) -> list[Phase]:
    name = name.strip().lower()

    if name == "smoke":
        return [Phase("smoke", duration_s=10, rps=5, concurrency=10)]

    if name == "load":
        return [
            Phase("warmup", duration_s=15, rps=5, concurrency=20),
            Phase("steady", duration_s=45, rps=20, concurrency=50),
            Phase("spike", duration_s=15, rps=40, concurrency=80),
        ]

    if name == "boundary":
        return [
            Phase("warmup", duration_s=10, rps=5, concurrency=20),
            Phase("steady", duration_s=30, rps=15, concurrency=40),
            Phase("spike", duration_s=10, rps=25, concurrency=60),
        ]

    raise ValueError("unknown profile. Use: smoke|load|boundary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=os.getenv("PROFILE", "smoke"))
    parser.add_argument("--url", default=os.getenv("TARGET_URL", "http://gateway:8080/predict"))
    args = parser.parse_args()

    profile = args.profile.strip().lower()
    payload_file = _payload_file_for_profile(profile)
    payloads = _load_payloads_from_file(payload_file)
    phases = profile_by_name(profile)

    print(f"Load profile: {profile}")
    print(f"Target: {args.url}")
    print(f"Data root: {_data_root()}")
    print(f"Payload file: {payload_file}")
    print(f"Payload pool size: {len(payloads)}")
    print()

    out = []
    for ph in phases:
        res = asyncio.run(run_phase(args.url, payloads, ph))
        out.append(res)
        print(res)

    total = sum(x["requests_total"] for x in out)
    ok = sum(x["ok"] for x in out)
    err = sum(x["errors"] for x in out)
    print()
    print(f"TOTAL: {total} ok={ok} errors={err} err_ratio={(err / max(total, 1)):.4f}")


if __name__ == "__main__":
    main()
