import asyncio
import collections
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, TypedDict

import httpx

URL = os.getenv("TARGET_URL", "http://localhost:8080/predict")
TOTAL = int(os.getenv("TOTAL", "64000"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "6"))
PAYLOAD_SET = os.getenv("PAYLOAD_SET", "load")  # smoke | load | boundary


class Stats(TypedDict):
    ok: int
    bad: int
    err: int
    latencies: list[float]
    codes: collections.Counter[int]
    errors: collections.Counter[str]
    sample_bad_body: str | None
    sample_exception: str | None


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


def _payload_file_for_set(payload_set: str) -> Path:
    return _data_root() / "payloads" / f"{payload_set}.json"


PAYLOAD_FILE = str(_payload_file_for_set(PAYLOAD_SET))


def load_payloads(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str)
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


PAYLOAD_POOL = load_payloads(PAYLOAD_FILE)


def get_payload(request_no: int) -> dict[str, Any]:
    return PAYLOAD_POOL[request_no % len(PAYLOAD_POOL)]


async def worker(
    request_no: int,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    stats: Stats,
) -> None:
    async with sem:
        payload = get_payload(request_no)
        started = time.perf_counter()
        try:
            r = await client.post(
                URL,
                json=payload,
                headers={"X-Request-ID": f"py-{uuid.uuid4()}"},
            )
            elapsed = time.perf_counter() - started
            stats["latencies"].append(elapsed)
            stats["codes"][r.status_code] += 1

            if 200 <= r.status_code < 300:
                stats["ok"] += 1
            else:
                stats["bad"] += 1
                if stats["sample_bad_body"] is None:
                    stats["sample_bad_body"] = r.text[:1000]

        except Exception as e:
            elapsed = time.perf_counter() - started
            stats["latencies"].append(elapsed)
            stats["err"] += 1
            stats["errors"][type(e).__name__] += 1
            if stats["sample_exception"] is None:
                stats["sample_exception"] = repr(e)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(int(len(values) * p), len(values) - 1)
    return values[idx]


async def main() -> None:
    stats: Stats = {
        "ok": 0,
        "bad": 0,
        "err": 0,
        "latencies": [],
        "codes": collections.Counter[int](),
        "errors": collections.Counter[str](),
        "sample_bad_body": None,
        "sample_exception": None,
    }

    sem = asyncio.Semaphore(CONCURRENCY)
    timeout = httpx.Timeout(10.0, connect=5.0)
    limits = httpx.Limits(
        max_connections=CONCURRENCY,
        max_keepalive_connections=CONCURRENCY,
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        started = time.perf_counter()
        tasks = [worker(i, client, sem, stats) for i in range(TOTAL)]
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - started

    p50 = percentile(stats["latencies"], 0.50)
    p95 = percentile(stats["latencies"], 0.95)
    p99 = percentile(stats["latencies"], 0.99)
    rps = stats["ok"] / total_time if total_time > 0 else 0.0

    print(f"URL={URL}")
    print(f"PAYLOAD_SET={PAYLOAD_SET}")
    print(f"TOTAL={TOTAL}")
    print(f"CONCURRENCY={CONCURRENCY}")
    print(f"DATA_ROOT={_data_root()}")
    print(f"PAYLOAD_FILE={PAYLOAD_FILE}")
    print(f"PAYLOAD_POOL_SIZE={len(PAYLOAD_POOL)}")
    print(f"OK={stats['ok']} BAD={stats['bad']} ERR={stats['err']}")
    print(f"TIME={total_time:.2f}s")
    print(f"RPS={rps:.2f}")
    print(f"P50={p50 * 1000:.1f}ms P95={p95 * 1000:.1f}ms P99={p99 * 1000:.1f}ms")
    print(f"STATUS CODES={dict(stats['codes'])}")

    if stats["errors"]:
        print(f"ERROR TYPES={dict(stats['errors'])}")

    if stats["sample_bad_body"] is not None:
        print("\nSAMPLE BAD BODY:")
        print(stats["sample_bad_body"])

    if stats["sample_exception"] is not None:
        print("\nSAMPLE EXCEPTION:")
        print(stats["sample_exception"])


if __name__ == "__main__":
    asyncio.run(main())
