"""Gunicorn config for Prometheus multiprocess mode (inference service).

We run the inference API with multiple Gunicorn workers. In Prometheus multiprocess
mode, we must:
- Clean PROMETHEUS_MULTIPROC_DIR on startup to avoid stale rows after restarts.
- Mark dead workers so the multiprocess collector doesn't keep "ghost" processes.

The inference container is launched with:
  gunicorn ... -c /app/app/gunicorn_conf.py
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
    for p in glob.glob(os.path.join(d, "*.db")):
        try:
            os.remove(p)
        except Exception:
            pass


def child_exit(server, worker) -> None:  # pragma: no cover
    try:
        multiprocess.mark_process_dead(worker.pid)
    except Exception:
        pass
