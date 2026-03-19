# Contributing

This repository is a production-oriented skeleton. The goal is to keep it **small**, **explicit**, and **easy to reuse**.
Contributions should preserve that: avoid turning this into a platform.

## Development setup

### Requirements
- Python 3.11+
- Docker + Docker Compose v2
- `make`

### Install dependencies (local)
```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt -r requirements-dev.txt
```

### Pre-commit (recommended)
Pre-commit keeps formatting/lint consistent across contributors.

```bash
pip install pre-commit
pre-commit install

pre-commit run --all-files
```

## Quality gates

### Format + lint
```bash
ruff format .
ruff check .
```

### Type checking
```bash
mypy app router scripts
```

### Tests
```bash
pytest
```

## End-to-end sanity (Docker Compose)
```bash
docker compose down --remove-orphans
docker compose build --no-cache

python -m scripts.train_reference_sklearn --name model --version 1 --out-root ./models --algo logreg
python -m scripts.train_reference_sklearn --name model --version 2 --out-root ./models --algo mlp

docker compose up -d --build --scale inference-v1=1 --scale inference-v2=1
curl -fsS http://localhost:8080/healthz
curl -fsS -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  --data-binary @.github/fixtures/ci_smoke_request.json
docker compose down --remove-orphans
```

## Contribution guidelines
- Keep changes **minimal** and **explicit**.
- Prefer small, composable changes over “framework” abstractions.
- Avoid adding dependencies unless clearly justified.
- If you change runtime behavior, update `README.md` and/or `docs/commands.md`.

## Pull requests
- Explain motivation and trade-offs.
- Include tests for non-trivial logic.
- Ensure CI passes (ruff/mypy/pytest + compose smoke).
