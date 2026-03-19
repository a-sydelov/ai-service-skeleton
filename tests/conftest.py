"""Pytest configuration.

This repository is intentionally a lightweight skeleton and may be executed either:
- inside the Docker image (WORKDIR=/app)
- directly on the host during development

When running on the host, ensure the repo root is on sys.path so imports like
`from app...` work without requiring an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
