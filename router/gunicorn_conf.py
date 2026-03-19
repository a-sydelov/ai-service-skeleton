"""Gunicorn config for Prometheus multiprocess mode.

This file keeps the runtime behavior stable with multiple Gunicorn workers:
- Cleans PROMETHEUS_MULTIPROC_DIR on startup to avoid stale metrics after restarts.
- Marks dead worker processes so Prometheus multiprocess collector doesn't accumulate ghosts.

The router is launched by docker-compose with `-c /app/router/gunicorn_conf.py`.
"""

from __future__ import annotations

import glob
import os

from prometheus_client import multiprocess


def on_starting(server) -> None:  # pragma: no cover
    d = (os.getenv("PROMETHEUS_MULTIPROC_DIR") or "").strip()
    if not d:
        return

    os.makedirs(d, exist_ok=True)
    # Only delete Prometheus multiprocess files; keep any other shared state.
    for p in glob.glob(os.path.join(d, "*.db")):
        try:
            os.remove(p)
        except Exception:
            pass


def child_exit(server, worker) -> None:  # pragma: no cover
    try:
        multiprocess.mark_process_dead(worker.pid)
    except Exception:
        # Best-effort cleanup; never block worker lifecycle.
        pass
