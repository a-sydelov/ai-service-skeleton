SHELL := /bin/bash

ifneq (,$(wildcard .env))
include .env
export
endif




OPSET ?= 12
JOBLIB_FILE ?= model.joblib
ONNX_FILE ?= model.onnx
MODEL_TASK ?= classification
TASK ?= $(MODEL_TASK)
N_FEATURES ?= 561
OUTPUT_IS_LOGITS ?= 0

TEST_IMAGE ?= ai-service-skeleton:test
TEST_IMAGE_STAMP ?= .cache/test-image.stamp
DOCKER_RUN_TEST = docker run --rm --user "$$(id -u):$$(id -g)" -v "$$PWD":/src -w /src $(TEST_IMAGE)

# ---------------- Public targets (minimal UX) ----------------
.PHONY: help up down ps logs scale \
        train-v1 train-v2 export-onnx-v1 export-onnx-v2 \
        traffic load boundary load-dataset clear-dataset \
        smoke \
        test-v1 test-v2 \
        checklist cutover rollback \
        validate-v1 validate-v2 \
        fmt lint typecheck unit test \
        shadow-sample split split-off reset-state clean-workspace

# ---------------- Internal targets (not advertised) ----------------
.PHONY: _require-up _code _test-model-version-v1 _test-model-version-v2 _build_test_image

# help — show a concise command overview.
help:
	@echo ""
	@echo "AI Service Skeleton — commands"
	@echo ""
	@echo "Lifecycle:"
	@echo "  make up [V1_REPLICAS=2 V2_REPLICAS=1]"
	@echo "  make down | make ps | make logs"
	@echo "  make scale V1_REPLICAS=3 V2_REPLICAS=2"
	@echo ""
	@echo "Artifacts:"
	@echo "  make train-v1 | make train-v2"
	@echo "  make export-onnx-v1 TASK=<task> N_FEATURES=<n>  # task: classification|regression|embedding"
	@echo "  make export-onnx-v2 TASK=<task> N_FEATURES=<n>  # task: classification|regression|embedding"
	@echo ""
	@echo "Traffic tools (kept separate by design):"
	@echo "  make traffic N=250 P=20      # quick fill dashboards"
	@echo "  make load                    # baseline load profile via gateway"
	@echo ""
	@echo "System smoke (entrypoint check):"
	@echo "  make smoke                   # short profile via gateway"
	@echo ""
	@echo "Per-variant checks (run independently):"
	@echo "  make test-v1                 # code tests + v1 artifact + v1 smoke + system smoke + checklist"
	@echo "  make test-v2                 # code tests + v2 artifact + v2 smoke + system smoke + checklist"
	@echo ""
	@echo "Manual rollout:"
	@echo "  make checklist               # read-only pre-cutover gate"
	@echo "  make cutover | make rollback # manual (no auto)"
	@echo ""
	@echo "  make clean-workspace         # remove local generated files and reset workspace"


# up — start the full stack (router + gateway + prometheus + grafana + inference variants).
# Example:
#   make up V1_REPLICAS=2 V2_REPLICAS=3
up:
	docker compose up --build --scale inference-v1=$(V1_REPLICAS) --scale inference-v2=$(V2_REPLICAS)


# down — stop and remove containers (keeps local files).
down:
	docker compose down --remove-orphans


# ps — show running containers in the compose project.
ps:
	docker compose ps


# logs — follow service logs.
logs:
	docker compose logs -f --tail=200


# scale — change inference replica counts without rebuilding.
# Example:
#   make scale V1_REPLICAS=2 V2_REPLICAS=3
scale:
	docker compose up -d --no-build --scale inference-v1=$(V1_REPLICAS) --scale inference-v2=$(V2_REPLICAS) --no-deps inference-v1 inference-v2


# train-v1 — train and write artifact for v1 under /models.
train-v1:
	docker compose --profile train run --rm trainer --name $(MODEL_NAME) --version 1 --out-root /models --algo logreg


# train-v2 — train and write artifact for v2 under /models.
train-v2:
	docker compose --profile train run --rm trainer --name $(MODEL_NAME) --version 2 --out-root /models --algo mlp


# export-onnx-v1 — export v1 sklearn artifact to ONNX.
# Example:
#   make export-onnx-v1 TASK=<task> N_FEATURES=<n>  # task: classification|regression|embedding
export-onnx-v1:
	@FLAGS=""; \
	if [ "$(OUTPUT_IS_LOGITS)" = "1" ]; then FLAGS="--output-is-logits"; fi; \
	docker compose run --rm --entrypoint python trainer -m scripts.export_sklearn_to_onnx \
	  --dir /models/$(MODEL_NAME)/1 \
	  --joblib-file $(JOBLIB_FILE) \
	  --onnx-file $(ONNX_FILE) \
	  --opset $(OPSET) \
	  --task $(TASK) \
	  --n-features $(N_FEATURES) \
	  $$FLAGS


# export-onnx-v2 — export v2 sklearn artifact to ONNX.
# Example:
#   make export-onnx-v2 TASK=<task> N_FEATURES=<n>  # task: classification|regression|embedding
export-onnx-v2:
	@FLAGS=""; \
	if [ "$(OUTPUT_IS_LOGITS)" = "1" ]; then FLAGS="--output-is-logits"; fi; \
	docker compose run --rm --entrypoint python trainer -m scripts.export_sklearn_to_onnx \
	  --dir /models/$(MODEL_NAME)/2 \
	  --joblib-file $(JOBLIB_FILE) \
	  --onnx-file $(ONNX_FILE) \
	  --opset $(OPSET) \
	  --task $(TASK) \
	  --n-features $(N_FEATURES) \
	  $$FLAGS


# traffic — generate N requests to populate metrics (best-effort).
# Example:
#   make traffic N=200 P=6
traffic:
	@TOTAL=$${N:-200} CONCURRENCY=$${P:-20} PAYLOAD_SET=$${PAYLOAD_SET:-load} python -m scripts.load_test

# smoke — short profile against the gateway (sanity check).
smoke:
	docker compose --profile load run --rm --no-deps loadgen --profile smoke

# load — regular load profile against the gateway.
load:
	docker compose --profile load run --rm --no-deps loadgen --profile load

# boundary — boundary-focused profile for provoking class mismatches.
boundary:
	docker compose --profile load run --rm --no-deps loadgen --profile boundary


# validate-v1 — validate v1 artifact contract (deep checks).
validate-v1:
	docker compose exec -T inference-v1 python -m scripts.validate_artifact --dir $(MODEL_DIR_V1) --deep


# validate-v2 — validate v2 artifact contract (deep checks).
validate-v2:
	docker compose exec -T inference-v2 python -m scripts.validate_artifact --dir $(MODEL_DIR_V2) --deep


# checklist — read-only pre-cutover checklist (router config + variants + metrics).
checklist:
	docker compose exec -T router python -m scripts.pre_cutover_checklist --router-url http://router:8080 --prom-url http://prometheus:9090


# cutover — make v2 primary (manual switch; no automation).
# Example:
#   make cutover
cutover:
	@docker compose exec -T router python -m scripts.admin_ops cutover


# rollback — make v1 primary (manual switch; no automation).
# Example:
#   make rollback
rollback:
	@docker compose exec -T router python -m scripts.admin_ops rollback

# ---------------- Per-variant tests (what you asked for) ----------------
# These targets are designed for the common workflow:
# - you replace ONLY v1 artifact -> run `make test-v1`
# - you replace ONLY v2 artifact -> run `make test-v2`
#
# test-vX includes:
#   1) code tests (unit; optionally lint/typecheck via `make test`)
#   2) artifact validation for that variant
#   3) direct route check to that variant behind gateway (/v1/predict or /v2/predict)
#   4) system smoke via gateway (short) to populate Prometheus for checklist
#   5) read-only checklist
test-v1: _code validate-v1 _require-up _test-model-version-v1 smoke checklist

test-v2: _code validate-v2 _require-up _test-model-version-v2 smoke checklist

_require-up:
	@docker compose exec -T router python -m scripts.admin_ops require-up

_test-model-version-v1:
	docker compose exec -T router python -m scripts.test_model_version \
	  --url http://gateway:8080/v1/predict \
	  --model-dir $(MODEL_DIR_V1)

_test-model-version-v2:
	docker compose exec -T router python -m scripts.test_model_version \
	  --url http://gateway:8080/v2/predict \
	  --model-dir $(MODEL_DIR_V2)

# ---------------- Code quality (dockerized) ----------------
# Kept as separate targets; `test-v1/test-v2` call `_code` (unit) by default.

# fmt — format code (ruff format).
fmt: _build_test_image
	@$(DOCKER_RUN_TEST) bash -lc "ruff format ."


# lint — formatting + lint checks (fails CI if changes needed).
lint: _build_test_image
	@$(DOCKER_RUN_TEST) bash -lc "ruff format --check . && ruff check ."


# typecheck — static type checks (mypy).
typecheck: _build_test_image
	@$(DOCKER_RUN_TEST) bash -lc "mypy app router scripts"


# unit — run unit tests (pytest).
unit: _build_test_image
	# Use `python -m pytest` (instead of the `pytest` console script) so the project root
	# is reliably on sys.path across environments.
	@$(DOCKER_RUN_TEST) bash -lc "python -m pytest"


# test — full code quality gate (lint + typecheck + unit).
test: lint typecheck unit

_code: unit

# shadow-sample — mirror a sampled subset of requests to the shadow variant for *comparison only*.
# - Purpose: evaluate quality/latency/error of the candidate model without affecting user responses.
# - Effect: updates router config field `shadow_sample_rate` (0.0..1.0). Responses are still served by `primary_variant`.
# - Example:
#     make shadow-sample PCT=10
#     make shadow-sample PCT=0   # disable mirroring
#
# PCT is 0..100 (integer or float).
shadow-sample:
	@docker compose exec -T router python -m scripts.admin_ops shadow-sample --pct "$${PCT:-10}"

# split — canary serving: serve PCT% of requests by the *shadow* variant (manual rollout control).
# - Purpose: gradually expose real users to the candidate model (served traffic is split between primary and shadow).
# - Effect: updates router config field `serve_shadow_rate` (0.0..1.0).
# - Example:
#     make split PCT=10
#     make split PCT=50
#
# PCT is 0..100 (integer or float).
split:
	@docker compose exec -T router python -m scripts.admin_ops split --pct "$${PCT:-0}"

# split-off — disable canary serving (100% served by primary).
# - Purpose: return to a stable state after experiments.
# - Effect: sets router config field `serve_shadow_rate` to 0.0.
# - Example:
#     make split-off
split-off:
	@docker compose exec -T router python -m scripts.admin_ops split-off

_build_test_image:
	@:

$(TEST_IMAGE_STAMP): Dockerfile.dev requirements.txt requirements-dev.txt
	@mkdir -p .cache
	@docker build -f Dockerfile.dev -t $(TEST_IMAGE) . 1>/dev/null
	@touch $(TEST_IMAGE_STAMP)

_build_test_image: $(TEST_IMAGE_STAMP)

# reset-state — restore rollout state file to the values from .env.
reset-state:
	@docker compose exec -T router python -m scripts.admin_ops reset-state

load-dataset:
	python scripts/load_dataset.py load

clear-dataset:
	python scripts/load_dataset.py clear

clean-workspace:
	@bash scripts/clean_workspace.sh