from gateway.reloader import RolloutState, compact_percent, render_http_conf, render_predict_conf


def _state(**overrides):
    base = RolloutState(
        primary_variant="v1",
        shadow_variant="v2",
        shadow_sample_rate=0.1,
        serve_shadow_rate=0.0,
        version=1,
        updated_at="2026-03-09T00:00:00Z",
    )
    return RolloutState(**{**base.__dict__, **overrides})


def test_compact_percent_strips_trailing_zeroes() -> None:
    assert compact_percent(0.1) == "10%"
    assert compact_percent(0.125) == "12.5%"
    assert compact_percent(0.333333) == "33.3333%"


def test_render_http_conf_contains_predict_routing_map() -> None:
    conf = render_http_conf(_state())
    assert "map $served_variant $predict_internal_uri" in conf
    assert "v1 /__predict_v1;" in conf
    assert '"v1:on" /__shadow_v2;' in conf


def test_render_predict_conf_turns_mirror_off_when_shadow_disabled() -> None:
    conf = render_predict_conf(_state(shadow_variant="off", shadow_sample_rate=0.0))
    assert conf.strip().endswith("mirror off;")
