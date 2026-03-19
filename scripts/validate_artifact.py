"""Artifact validation CLI."""

from __future__ import annotations

import argparse
import json
import sys

from app.model_loader import validate_artifact_dir


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Artifact directory, e.g. /models/model/1")
    p.add_argument(
        "--deep", action="store_true", help="Deep validation (for ONNX: check io names exist)"
    )
    args = p.parse_args()

    try:
        summary = validate_artifact_dir(args.dir, deep=args.deep)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as e:
        print(f"VALIDATION_FAILED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
