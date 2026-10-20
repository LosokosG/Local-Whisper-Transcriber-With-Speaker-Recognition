from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta, timezone

import pytest
from jsonschema import Draft202012Validator, FormatChecker


def signal(
    signal_id: str,
    candidate: str,
    *,
    kind: str = "visible_ui",
    strength: float = 0.9,
    group: str | None = None,
    authoritative: bool = False,
) -> dict[str, object]:
    return {
        "id": signal_id,
        "kind": kind,
        "candidate_start_utc": candidate,
        "source": f"fixture:{signal_id}",
        "source_ref": f"frame-{signal_id}",
        "detail": f"evidence for {signal_id}",
        "strength": strength,
        "independent_group": group or signal_id,
        "authoritative": authoritative,
    }


def test_event_evidence_schema_is_current_locatable_and_strict() -> None:
    from transcript_pipeline.event_time import (
        EVENT_EVIDENCE_SCHEMA_PATH,
        load_event_evidence_schema,
    )

    assert EVENT_EVIDENCE_SCHEMA_PATH.is_file()
    on_disk = json.loads(EVENT_EVIDENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    schema = load_event_evidence_schema()
    assert schema == on_disk
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(schema)

    valid = {
        "schema_version": 1,
        "recording_duration_seconds": 5273.385,
        "signals": [signal("media", "2026-07-17T13:00:36Z")],
    }
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    assert not list(validator.iter_errors(valid))

    extra_root = {**valid, "unexpected": True}
    extra_signal = {
        **valid,
        "signals": [{**valid["signals"][0], "unexpected": True}],
    }
    mtime = {
        **valid,
        "signals": [
            {
                **valid["signals"][0],
                "kind": "filesystem_mtime",
            }
        ],
    }
    illegal_authority = {
        **valid,
        "signals": [
            {
                **valid["signals"][0],
                "authoritative": True,
                "kind": "visible_ui",
            }
        ],
    }
    unauthoritative_override = {
        **valid,
        "signals": [
            {
                **valid["signals"][0],
                "authoritative": False,
                "kind": "user_override",
            }
        ],
    }
    for invalid in (
        extra_root,
        extra_signal,
        mtime,
        illegal_authority,
        unauthoritative_override,
    ):
        assert list(validator.iter_errors(invalid))


def test_typed_evidence_validation_normalizes_utc_and_round_trips() -> None:
    from transcript_pipeline.event_time import (
        EventEvidence,
        SignalKind,
        validate_event_evidence,
    )

    evidence = validate_event_evidence(
        {
            "schema_version": 1,
            "recording_duration_seconds": 1.25,
            "signals": [
                signal(
                    "creation-time",
                    "2026-07-17T13:00:36+00:00",
                    kind="media_creation_time",
                )
            ],
        }
    )

    assert isinstance(evidence, EventEvidence)
    assert evidence.signals[0].kind is SignalKind.MEDIA_CREATION_TIME
    assert evidence.signals[0].candidate_start_utc == datetime(
        2026, 7, 17, 13, 0, 36, tzinfo=UTC
    )
    assert evidence.to_dict()["signals"][0]["candidate_start_utc"] == (
        "2026-07-17T13:00:36Z"
    )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda data: data.update(recording_duration_seconds=0), "duration"),
        (lambda data: data.update(recording_duration_seconds=math.inf), "finite"),
        (
            lambda data: data["signals"].append(dict(data["signals"][0])),
            "unique",
        ),
        (
            lambda data: data["signals"][0].update(
                candidate_start_utc="2026-07-17T13:00:36"
            ),
            "timezone",
        ),
        (
            lambda data: data["signals"][0].update(strength=math.nan),
            "finite",
        ),
        (
            lambda data: data["signals"][0].update(
                kind="spoken_date", authoritative=True
            ),
            "authoritative",
        ),
        (
            lambda data: data["signals"][0].update(
                kind="user_override", authoritative=False
            ),
            "authoritative",
        ),
    ],
)
def test_evidence_semantics_reject_invalid_values(mutator: object, match: str) -> None:
    from transcript_pipeline.event_time import validate_event_evidence

    data = {
        "schema_version": 1,
        "recording_duration_seconds": 60.0,
        "signals": [signal("one", "2026-07-17T13:00:36Z")],
    }
    mutator(data)  # type: ignore[operator]

    with pytest.raises(ValueError, match=match):
        validate_event_evidence(data)


def test_resolver_matches_corroborated_sample_and_renders_local_time() -> None:
    from transcript_pipeline.event_time import resolve_event_time

    signals = [
        signal(
            "media",
            "2026-07-17T13:00:36Z",
            kind="media_creation_time",
            strength=0.98,
            group="embedded-metadata",
        ),
        signal(
            "ui",
            "2026-07-17T13:01:05Z",
            kind="visible_ui",
            strength=0.94,
            group="screen-capture",
        ),
        signal(
            "spoken",
            "2026-07-17T13:00:12Z",
            kind="spoken_date",
            strength=0.82,
            group="transcript",
        ),
        signal(
            "slide",
            "2026-07-17T13:01:00Z",
            kind="visible_ui",
            strength=0.75,
            group="slide-clock",
        ),
    ]

    result = resolve_event_time(signals, 5273.385, timezone_name="Europe/Berlin")

    assert result["status"] == "resolved"
    assert result["resolution_basis"] == "corroborated_evidence"
    assert result["start_utc"] == "2026-07-17T13:00:36Z"
    assert result["start_unix"] == 1_784_293_236
    assert result["end_utc"] == "2026-07-17T14:28:29.385000Z"
    assert result["duration_seconds"] == 5273.385
    assert result["timezone"] == "Europe/Berlin"
    assert result["start_local"] == "2026-07-17T15:00:36+02:00"
    assert result["end_local"] == "2026-07-17T16:28:29.385000+02:00"
    assert result["local_date"] == "2026-07-17"
    assert result["ranked_candidates"][0]["signal_ids"] == [
        "media",
        "slide",
        "spoken",
        "ui",
    ]
    assert result["ranked_candidates"][0]["strong_independent_groups"] == 4


def test_resolver_blocks_equal_plausible_candidates_instead_of_guessing() -> None:
    from transcript_pipeline.event_time import resolve_event_time

    result = resolve_event_time(
        [
            signal("a1", "2026-07-17T13:00:00Z", group="a-ui"),
            signal("a2", "2026-07-17T13:00:20Z", group="a-transcript"),
            signal("b1", "2026-07-18T13:00:00Z", group="b-ui"),
            signal("b2", "2026-07-18T13:00:20Z", group="b-transcript"),
        ],
        60,
    )

    assert result["status"] == "blocked"
    assert result["blocker"] == "ambiguous_event_time"
    assert "start_utc" not in result
    assert len(result["ranked_candidates"]) == 2


def test_resolver_blocks_one_signal_repeated_source_and_weak_evidence() -> None:
    from transcript_pipeline.event_time import resolve_event_time

    one = resolve_event_time(
        [signal("one", "2026-07-17T13:00:00Z")],
        60,
    )
    repeated = resolve_event_time(
        [
            signal(
                f"same-{index}",
                f"2026-07-17T13:00:{index:02d}Z",
                group="same-source",
            )
            for index in range(5)
        ],
        60,
    )
    weak = resolve_event_time(
        [
            signal("weak-1", "2026-07-17T13:00:00Z", strength=0.69),
            signal("weak-2", "2026-07-17T13:00:10Z", strength=0.4),
        ],
        60,
    )

    for result in (one, repeated, weak):
        assert result["status"] == "blocked"
        assert result["blocker"] == "ambiguous_event_time"
    assert repeated["ranked_candidates"][0]["strong_independent_groups"] == 1
    assert weak["ranked_candidates"][0]["strong_independent_groups"] == 0


def test_authoritative_user_override_resolves_and_is_recorded() -> None:
    from transcript_pipeline.event_time import resolve_event_time

    result = resolve_event_time(
        [
            signal(
                "operator",
                "2026-07-19T09:30:00Z",
                kind="user_override",
                strength=0.1,
                group="operator-decision",
                authoritative=True,
            ),
            signal("conflict", "2026-07-17T13:00:00Z"),
        ],
        90,
    )

    assert result["status"] == "resolved"
    assert result["resolution_basis"] == "authoritative_user_override"
    assert result["start_utc"] == "2026-07-19T09:30:00Z"
    assert result["authoritative_signal_ids"] == ["operator"]


def test_conflicting_authoritative_overrides_are_ambiguous() -> None:
    from transcript_pipeline.event_time import resolve_event_time

    result = resolve_event_time(
        [
            signal(
                "override-a",
                "2026-07-17T13:00:00Z",
                kind="user_override",
                authoritative=True,
            ),
            signal(
                "override-b",
                "2026-07-18T13:00:00Z",
                kind="user_override",
                authoritative=True,
            ),
        ],
        90,
    )

    assert result["status"] == "blocked"
    assert result["blocker"] == "ambiguous_event_time"


def test_authoritative_overrides_within_normal_cluster_must_match_millisecond() -> None:
    from transcript_pipeline.event_time import resolve_event_time

    result = resolve_event_time(
        [
            signal(
                "override-a",
                "2026-07-17T13:00:00Z",
                kind="user_override",
                authoritative=True,
            ),
            signal(
                "override-b",
                "2026-07-17T13:00:30Z",
                kind="user_override",
                authoritative=True,
            ),
        ],
        90,
    )

    assert result["status"] == "blocked"
    assert result["blocker"] == "ambiguous_event_time"
    assert result["authoritative_override_tolerance_millis"] == 0
    assert [
        candidate["candidate_start_utc"]
        for candidate in result["ranked_candidates"]
    ] == ["2026-07-17T13:00:00Z", "2026-07-17T13:00:30Z"]
    assert [
        candidate["signal_ids"] for candidate in result["ranked_candidates"]
    ] == [["override-a"], ["override-b"]]


def test_matching_authoritative_override_references_resolve_and_are_all_recorded(
) -> None:
    from transcript_pipeline.event_time import resolve_event_time

    result = resolve_event_time(
        [
            signal(
                "override-a",
                "2026-07-17T13:00:00.123Z",
                kind="user_override",
                authoritative=True,
                group="operator-a",
            ),
            signal(
                "override-b",
                "2026-07-17T13:00:00.123456Z",
                kind="user_override",
                authoritative=True,
                group="operator-b",
            ),
            signal("conflict", "2026-07-18T13:00:00Z"),
        ],
        90,
    )

    assert result["status"] == "resolved"
    assert result["resolution_basis"] == "authoritative_user_override"
    assert result["start_utc"] == "2026-07-17T13:00:00.123000Z"
    assert result["authoritative_signal_ids"] == ["override-a", "override-b"]
    assert result["ranked_candidates"][0]["signal_ids"] == [
        "override-a",
        "override-b",
    ]


def test_more_independent_groups_provide_a_clear_lead_over_conflict() -> None:
    from transcript_pipeline.event_time import resolve_event_time

    result = resolve_event_time(
        [
            signal("leader-1", "2026-07-17T13:00:00Z", group="leader-1"),
            signal("leader-2", "2026-07-17T13:00:10Z", group="leader-2"),
            signal("leader-3", "2026-07-17T13:00:20Z", group="leader-3"),
            signal("runner-1", "2026-07-18T13:00:00Z", group="runner-1"),
            signal("runner-2", "2026-07-18T13:00:10Z", group="runner-2"),
        ],
        60,
    )

    assert result["status"] == "resolved"
    assert result["start_utc"] == "2026-07-17T13:00:00Z"


def test_close_scores_with_same_corroboration_remain_ambiguous() -> None:
    from transcript_pipeline.event_time import resolve_event_time

    result = resolve_event_time(
        [
            signal("a1", "2026-07-17T13:00:00Z", strength=0.9),
            signal("a2", "2026-07-17T13:00:10Z", strength=0.9),
            signal("b1", "2026-07-18T13:00:00Z", strength=0.85),
            signal("b2", "2026-07-18T13:00:10Z", strength=0.85),
        ],
        60,
    )

    assert result["status"] == "blocked"
    assert result["blocker"] == "ambiguous_event_time"


def test_exact_score_tie_at_strength_threshold_is_ambiguous_despite_more_groups(
) -> None:
    from transcript_pipeline.event_time import resolve_event_time

    result = resolve_event_time(
        [
            signal(
                f"three-{index}",
                f"2026-07-17T13:00:{index * 10:02d}Z",
                kind="media_creation_time",
                strength=0.5,
            )
            for index in range(3)
        ]
        + [
            signal("two-1", "2026-07-18T13:00:00Z", kind="media_creation_time", strength=0.75),
            signal("two-2", "2026-07-18T13:00:10Z", kind="media_creation_time", strength=0.75),
        ],
        60,
        strong_threshold=0.5,
        score_lead=0.000_001,
    )

    assert result["ranked_candidates"][0]["score"] == 1.5
    assert result["ranked_candidates"][0]["strong_independent_groups"] == 3
    assert result["ranked_candidates"][1]["score"] == 1.5
    assert result["status"] == "blocked"
    assert result["blocker"] == "ambiguous_event_time"


def test_exact_score_tie_with_any_other_eligible_candidate_is_ambiguous() -> None:
    from transcript_pipeline.event_time import resolve_event_time

    result = resolve_event_time(
        [
            signal(
                f"leader-{index}",
                f"2026-07-17T13:00:{index:02d}Z",
                kind="media_creation_time",
                strength=0.5,
            )
            for index in range(3)
        ]
        + [
            signal("runner-1", "2026-07-18T13:00:00Z", kind="media_creation_time", strength=0.9),
            signal("runner-2", "2026-07-18T13:00:10Z", kind="media_creation_time", strength=0.9),
            signal("tied-1", "2026-07-19T13:00:00Z", kind="media_creation_time", strength=0.75),
            signal("tied-2", "2026-07-19T13:00:10Z", kind="media_creation_time", strength=0.75),
        ],
        60,
        strong_threshold=0.5,
    )

    assert [
        candidate["score"] for candidate in result["ranked_candidates"]
    ] == [1.5, 1.8, 1.5]
    assert result["status"] == "blocked"
    assert result["blocker"] == "ambiguous_event_time"


def test_resolver_accepts_typed_signals_and_non_utc_aware_input_is_rejected() -> None:
    from transcript_pipeline.event_time import EventTimeSignal, SignalKind, resolve_event_time

    first = EventTimeSignal(
        id="typed-a",
        kind=SignalKind.SPOKEN_DATE,
        candidate_start_utc=datetime(2026, 7, 17, 13, tzinfo=UTC),
        source="transcript:15",
        detail="speaker states the date and time",
        strength=0.8,
        independent_group="audio-transcript",
    )
    second = EventTimeSignal(
        id="typed-b",
        kind=SignalKind.VISIBLE_UI,
        candidate_start_utc=datetime(2026, 7, 17, 13, 1, tzinfo=UTC),
        source="frame:25",
        detail="clock visible in the UI",
        strength=0.9,
        independent_group="video-frame",
    )

    assert resolve_event_time([first, second], 30)["status"] == "resolved"

    with pytest.raises(ValueError, match="UTC"):
        EventTimeSignal(
            id="offset",
            kind=SignalKind.VISIBLE_UI,
            candidate_start_utc=datetime(
                2026, 7, 17, 15, tzinfo=timezone(timedelta(hours=2))
            ),
            source="frame:25",
            detail="local clock",
            strength=0.9,
            independent_group="video-frame",
        )


@pytest.mark.parametrize(
    "invalid_signal",
    [
        {
            **signal("extra", "2026-07-17T13:00:00Z"),
            "unexpected": "not part of the evidence contract",
        },
        signal("lowercase-z", "2026-07-17T13:00:00z"),
        signal("unknown-kind", "2026-07-17T13:00:00Z", kind="mtime"),
    ],
)
def test_resolver_raw_mappings_enforce_signal_json_schema(
    invalid_signal: dict[str, object],
) -> None:
    from transcript_pipeline.event_time import resolve_event_time

    with pytest.raises(ValueError, match="schema violation"):
        resolve_event_time([invalid_signal], 60)


def test_resolver_rejects_invalid_duration_and_options() -> None:
    from transcript_pipeline.event_time import resolve_event_time

    signals = [signal("one", "2026-07-17T13:00:00Z")]
    for duration in (0, -1, math.inf, math.nan, True):
        with pytest.raises((TypeError, ValueError)):
            resolve_event_time(signals, duration)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timezone"):
        resolve_event_time(signals, 60, timezone_name="Not/A_Zone")
    for score_lead in (0, -1, math.inf, math.nan, True):
        with pytest.raises((TypeError, ValueError), match="score lead"):
            resolve_event_time(signals, 60, score_lead=score_lead)
