from __future__ import annotations

import pytest

from router.main import RouterState, _state_from_dict


def _fallback(**overrides: object) -> RouterState:
    base = RouterState(
        primary_variant="v1",
        shadow_variant="v2",
        shadow_sample_rate=0.10,
        serve_shadow_rate=0.00,
        version=1,
        updated_at="2026-03-19T00:00:00Z",
    )
    return RouterState(**{**base.__dict__, **overrides})


def test_state_from_dict_accepts_valid_state() -> None:
    state = _state_from_dict(
        {
            "primary_variant": "V2",
            "shadow_variant": "v1",
            "shadow_sample_rate": 0.25,
            "serve_shadow_rate": 0.40,
            "version": 7,
            "updated_at": "2026-03-19T10:20:30Z",
        },
        _fallback(),
    )

    assert state == RouterState(
        primary_variant="v2",
        shadow_variant="v1",
        shadow_sample_rate=0.25,
        serve_shadow_rate=0.40,
        version=7,
        updated_at="2026-03-19T10:20:30Z",
    )


def test_state_from_dict_uses_fallback_for_missing_fields() -> None:
    fallback = _fallback(
        primary_variant="v2",
        shadow_variant="v1",
        shadow_sample_rate=0.15,
        serve_shadow_rate=0.35,
        version=9,
        updated_at="2026-03-18T11:22:33Z",
    )

    state = _state_from_dict({}, fallback)

    assert state == fallback


def test_state_from_dict_allows_shadow_off_with_zero_rates() -> None:
    state = _state_from_dict(
        {
            "primary_variant": "v1",
            "shadow_variant": "off",
            "shadow_sample_rate": 0.0,
            "serve_shadow_rate": 0.0,
        },
        _fallback(),
    )

    assert state.primary_variant == "v1"
    assert state.shadow_variant == "off"
    assert state.shadow_sample_rate == 0.0
    assert state.serve_shadow_rate == 0.0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "primary_variant": "v3",
                "shadow_variant": "v2",
                "shadow_sample_rate": 0.1,
                "serve_shadow_rate": 0.0,
            },
            "primary_variant must be v1|v2",
        ),
        (
            {
                "primary_variant": "v1",
                "shadow_variant": "blue",
                "shadow_sample_rate": 0.1,
                "serve_shadow_rate": 0.0,
            },
            "shadow_variant must be v1|v2|off",
        ),
        (
            {
                "primary_variant": "v1",
                "shadow_variant": "v1",
                "shadow_sample_rate": 0.1,
                "serve_shadow_rate": 0.0,
            },
            "shadow_variant must differ from primary_variant or be off",
        ),
        (
            {
                "primary_variant": "v1",
                "shadow_variant": "off",
                "shadow_sample_rate": 0.1,
                "serve_shadow_rate": 0.0,
            },
            "shadow_variant=off requires shadow_sample_rate=0 and serve_shadow_rate=0",
        ),
        (
            {
                "primary_variant": "v1",
                "shadow_variant": "off",
                "shadow_sample_rate": 0.0,
                "serve_shadow_rate": 0.2,
            },
            "shadow_variant=off requires shadow_sample_rate=0 and serve_shadow_rate=0",
        ),
        (
            {
                "primary_variant": "v1",
                "shadow_variant": "v2",
                "shadow_sample_rate": -0.01,
                "serve_shadow_rate": 0.0,
            },
            "shadow_sample_rate must be 0..1",
        ),
        (
            {
                "primary_variant": "v1",
                "shadow_variant": "v2",
                "shadow_sample_rate": 0.1,
                "serve_shadow_rate": 1.01,
            },
            "serve_shadow_rate must be 0..1",
        ),
    ],
)
def test_state_from_dict_rejects_invalid_values(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _state_from_dict(payload, _fallback())
