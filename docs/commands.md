# Commands

## Why this file exists

This file is a compact operator reference for the repository. It documents the commands used to start the stack, inspect serving state, generate traffic, validate artifacts, exercise manual rollout, and run code-quality or per-variant checks.

## Quick help

```bash
make help
```

## Setup

```bash
cp .env.example .env
```

Model artifacts are already included in the repository, so a local demo does not require retraining before startup.

## Start the stack

```bash
make up V1_REPLICAS=2 V2_REPLICAS=1
```

## Inspect the running system

```bash
curl http://localhost:8080/info
curl http://localhost:8080/variants/info
```

## UIs and endpoints

- Gateway:        `http://localhost:8080`
- Rollout info:   `http://localhost:8080/info`
- Variants info:  `http://localhost:8080/variants/info`
- Prometheus:     `http://localhost:9090`
- Grafana:        `http://localhost:3000` (`admin/admin`)
- Alertmanager:   `http://localhost:9093`

## Command execution model

Most operational commands run through Docker Compose. A smaller set of helper commands runs on the host machine.

- **Dockerized commands:** `make up`, `make down`, `make ps`, `make logs`, `make scale`, `make smoke`, `make load`, `make boundary`, `make validate-v1`, `make validate-v2`, `make checklist`, `make shadow-sample`, `make split`, `make split-off`, `make cutover`, `make rollback`, `make reset-state`, `make test-v1`, `make test-v2`, `make fmt`, `make lint`, `make typecheck`, `make unit`, `make test`, `make train-v1`, `make train-v2`, `make export-onnx-v1`, `make export-onnx-v2`
- **Host-side commands:** `make load-dataset`, `make clear-dataset`, `make traffic`, and direct `curl` requests

**Host requirements:** Python `3.11+`, `numpy`, `httpx`, and `curl`.

## Scaling

```bash
make scale V1_REPLICAS=3 V2_REPLICAS=2
```

This only changes the inference replica counts. It does not rebuild the stack and is independent from the traffic-generation profiles.

## Generate traffic

### Host-side quick traffic

```bash
make traffic N=250 P=20
```

### Prepare payloads for built-in traffic profiles

The built-in containerized profiles read payload files from `data/payloads/`. On a clean checkout, prepare them first:

```bash
make load-dataset
```

### Containerized traffic profiles

```bash
make smoke
make load
make boundary
```

### Clear generated payloads

```bash
make clear-dataset
```

## Manual rollout controls

### Shadow mirroring for comparison only

```bash
make shadow-sample PCT=10
make shadow-sample PCT=0
```

### Canary-style live traffic exposure for the shadow version

```bash
make split PCT=10
make split PCT=50
make split-off
```

### Read-only pre-cutover checklist

```bash
make checklist
```

### Promote or revert primary/shadow assignment

```bash
make cutover
make rollback
```

### Restore rollout state from `.env`

```bash
make reset-state
```

## Artifact validation

```bash
make validate-v1
make validate-v2
```

## Per-variant checks

These are the main verification targets when only one model version was changed.

```bash
make test-v1
make test-v2
```

Each target includes:

- unit tests
- artifact validation for the selected version
- direct route check for that version through the gateway
- short system smoke
- read-only checklist

## Code quality

```bash
make fmt
make lint
make typecheck
make unit
make test
```

## Refresh sklearn artifacts (optional)

Use these only when you want to replace the bundled model artifacts.

```bash
make train-v1
make train-v2
```

## Export sklearn artifacts to ONNX (optional)

```bash
make export-onnx-v1 TASK=<task> N_FEATURES=<n>
make export-onnx-v2 TASK=<task> N_FEATURES=<n>
```

Examples of `TASK`: `classification`, `regression`, `embedding`.

After updating ONNX artifacts, recreate the inference services:

```bash
docker compose up -d --build --force-recreate inference-v1 inference-v2
```

## Stop the stack

```bash
make down
```