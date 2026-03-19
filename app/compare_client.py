from __future__ import annotations

import logging
import queue
import threading
from typing import Any

import httpx

logger = logging.getLogger("app.compare")


class ComparePublisher:
    def __init__(
        self,
        *,
        enabled: bool,
        url: str,
        timeout_s: float,
        queue_size: int,
    ):
        self.enabled = enabled and bool(url.strip())
        self.url = url.strip()
        self.timeout_s = timeout_s
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max(queue_size, 1))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="compare-publisher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)
        self._thread = None

    def publish(self, payload: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            logger.warning("compare_publish_queue_full")
            return False

    def _run(self) -> None:
        with httpx.Client(timeout=self.timeout_s) as client:
            while not self._stop.is_set():
                try:
                    payload = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if payload is None:
                    self._queue.task_done()
                    continue
                try:
                    r = client.post(self.url, json=payload)
                    if r.status_code >= 400:
                        logger.warning(
                            "compare_publish_failed",
                            extra={"status_code": r.status_code, "body": r.text[:200]},
                        )
                except Exception:
                    logger.exception("compare_publish_exception")
                finally:
                    self._queue.task_done()
