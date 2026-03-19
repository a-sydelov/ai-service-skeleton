# Practical Guide: Plugging Your Model into the Skeleton

This repository expects **model artifacts on disk** and treats them as an external serving contract. You can replace model versions without changing service code, as long as the artifact layout and metadata remain valid.

## 1. Artifact contract

Artifacts live on the host under `./models/` and are mounted read-only into the inference containers.

Directory layout:

```text
models/
  <MODEL_NAME>/
    1/
      metadata.json
      model.joblib   (sklearn runtime)  OR  model.onnx (onnx runtime)
    2/
      metadata.json
      model.joblib   OR  model.onnx
```

Minimal `metadata.json` example:

```json
{
  "name": "model",
  "version": "2",
  "task": "classification",
  "runtime": "onnx",
  "model_file": "model.onnx",
  "n_features": 4,
  "classes": ["cat", "dog", "horse"],
  "onnx": {
    "input_name": "input",
    "output_name": "probabilities",
    "output_is_logits": false
  }
}
```

Supported values:

- `task`: `classification` | `regression` | `embedding`
- `runtime`: `sklearn` (`joblib`) | `onnx` (`onnxruntime`)

## 2. Place a new model version into `v2`

A new candidate version is normally placed into `./models/model/2/`:

```text
./models/model/2/metadata.json
./models/model/2/model.onnx   (or model.joblib)
```

If needed, rebuild only the `inference-v2` service:

```bash
docker compose up -d --no-deps --build inference-v2
```

Then validate the artifact contract and smoke-check `v2`:

```bash
make test-v2
```

## 3. ONNX workflow

If your base artifact is a scikit-learn model stored as `joblib`, you can export it to ONNX with the included script.

Export `v2`:

```bash
make export-onnx-v2 TASK=<task> N_FEATURES=<n>
```

Then refresh `inference-v2` and run the version check:

```bash
docker compose up -d --no-deps --build inference-v2
make test-v2
```

## 4. Shadow evaluation and comparison

The repository uses **primary/shadow model versions** for controlled rollout and paired comparison. The router manages rollout state and administrative operations, while prediction traffic is handled by the gateway and the inference pools.

To mirror a portion of requests to the shadow version for comparison only:

```bash
make shadow-sample PCT=10
```

To disable shadow sampling:

```bash
make shadow-sample PCT=0
```

This path is useful when you want to evaluate a candidate version under live request flow without changing the response returned to the client.

## 5. Controlled live traffic exposure

This skeleton also supports **manual canary-style serving**. A defined percentage of live requests can be served by the current shadow version while the remaining traffic continues to be handled by the primary version.

Serve 5% of live traffic through the shadow version:

```bash
make split PCT=5
```

Serve 50%:

```bash
make split PCT=50
```

Stop live traffic exposure and return it to 0%:

```bash
make split-off
```

Notes:

- `split` affects **served live traffic**.
- `shadow-sample` affects **mirrored comparison traffic**.
- These controls can be used independently or together, depending on how you want to evaluate the candidate version.

## 6. Promotion and recovery

Promotion is handled as an explicit rollout operation.

Promote the current shadow version to primary:

```bash
make cutover
```

Restore the previous primary/shadow assignment:

```bash
make rollback
```

Return rollout state to the defaults defined in `.env`:

```bash
make reset-state
```

## 7. What to watch in Grafana

Open Grafana:

- http://localhost:3000

Dashboards:

- **Serving and Rollout Dashboard** — active primary/shadow assignment, routing topology, release operations, traffic split, and request flow across serving instances
- **Primary/Shadow Comparison Dashboard** — shadow coverage, mismatch rate, class-level comparison signals, and candidate-versus-primary behavior
- **Infrastructure Dashboard** — container- and host-level CPU, memory, and runtime health for the local stack