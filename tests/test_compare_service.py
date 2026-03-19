from fastapi.testclient import TestClient

from compare.main import create_app


def test_compare_service_pairs_primary_and_shadow_events() -> None:
    client = TestClient(create_app())

    primary = {
        "request_id": "req-1",
        "role": "primary",
        "route_variant": "v1",
        "task": "classification",
        "duration_seconds": 0.012,
        "predictions": [
            {"label": "WALKING", "confidence": 0.91},
            {"label": "SITTING", "confidence": 0.77},
        ],
        "model_name": "model",
        "model_version": "1",
    }
    shadow = {
        "request_id": "req-1",
        "role": "shadow",
        "route_variant": "v2",
        "task": "classification",
        "duration_seconds": 0.018,
        "predictions": [
            {"label": "STANDING", "confidence": 0.81},
            {"label": "SITTING", "confidence": 0.70},
        ],
        "model_name": "model",
        "model_version": "2",
    }

    assert client.post("/event", json=primary).status_code == 202
    assert client.post("/event", json=shadow).status_code == 202

    metrics = client.get("/metrics").text
    assert 'shadow_primary_requests_total{status="success",variant="v1"} 1.0' in metrics
    assert 'shadow_requests_total{status="success",variant="v2"} 1.0' in metrics
    assert (
        'shadow_mismatch_total{primary_variant="v1",reason="top1_label",shadow_variant="v2"} 1.0'
        in metrics
    )
    assert (
        'shadow_mismatch_by_primary_label_total{primary_label="WALKING",primary_variant="v1",shadow_variant="v2"} 1.0'
        in metrics
    )
    assert (
        'shadow_label_pair_total{pair_group="mismatch",primary_label="WALKING",primary_variant="v1",shadow_label="STANDING",shadow_variant="v2"} 1.0'
        in metrics
    )
    assert (
        'shadow_label_pair_total{pair_group="match",primary_label="SITTING",primary_variant="v1",shadow_label="SITTING",shadow_variant="v2"} 1.0'
        in metrics
    )
    assert 'shadow_confidence_delta_count{primary_variant="v1",shadow_variant="v2"} 2.0' in metrics


def test_compare_service_ignores_non_classification_events() -> None:
    client = TestClient(create_app())
    payload = {
        "request_id": "req-embedding",
        "role": "primary",
        "route_variant": "v1",
        "task": "embedding",
        "duration_seconds": 0.01,
        "predictions": [{"label": "embedding", "confidence": 1.0}],
    }

    r = client.post("/event", json=payload)
    assert r.status_code == 202
    assert r.json() == {"status": "ignored", "reason": "task_not_supported"}
