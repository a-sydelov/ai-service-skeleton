from __future__ import annotations

import logging

from pythonjsonlogger.json import JsonFormatter


def setup_logging(log_level: str) -> None:
    root = logging.getLogger()
    root.setLevel(log_level)

    handler = logging.StreamHandler()
    formatter = JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        json_ensure_ascii=False,
    )
    handler.setFormatter(formatter)
    root.handlers = [handler]
