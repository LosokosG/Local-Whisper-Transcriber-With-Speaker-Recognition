"""Infer a recording's real-world start time from independent, auditable evidence."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jsonschema import Draft202012Validator, FormatChecker

EVENT_EVIDENCE_SCHEMA_PATH = (
    Path(__file__).parent / "schemas" / "event-evidence.schema.json"
)
EVENT_EVIDENCE_SCHEMA_VERSION = 1

DEFAULT_CLUSTER_TOLERANCE_SECONDS = 120.0
DEFAULT_STRONG_SIGNAL_THRESHOLD = 0.7
DEFAULT_MINIMUM_INDEPENDENT_GROUPS = 2
DEFAULT_SCORE_LEAD = 0.25
DEFAULT_TIMEZONE_NAME = "UTC"
AUTHORITATIVE_OVERRIDE_TOLERANCE_MILLIS = 0

_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_SIGNAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class SignalKind(StrEnum):
    """Allowed evidence origins; filesystem timestamps are intentionally absent."""

    MEDIA_CREATION_TIME = "media_creation_time"
    VISIBLE_UI = "visible_ui"
    SPOKEN_DATE = "spoken_date"
    USER_OVERRIDE = "user_override"


# Strength is the primary weight. These fixed reliability multipliers make score
# ordering explicit without allowing multiple observations from one source group
# to inflate corroboration.
KIND_RELIABILITY_WEIGHT: Mapping[SignalKind, float] = {
    SignalKind.MEDIA_CREATION_TIME: 1.0,
    SignalKind.VISIBLE_UI: 0.95,
    SignalKind.SPOKEN_DATE: 0.9,
    SignalKind.USER_OVERRIDE: 1.0,
}

_KIND_TIE_BREAK_PRIORITY: Mapping[SignalKind, int] = {
    SignalKind.USER_OVERRIDE: 0,
    SignalKind.MEDIA_CREATION_TIME: 1,
    SignalKind.VISIBLE_UI: 2,
    SignalKind.SPOKEN_DATE: 3,
}


def _require_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _require_finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_aware(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value


def _require_utc(value: object, field: str) -> datetime:
    timestamp = _require_aware(value, field)
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must use UTC")
    return timestamp.astimezone(UTC)


def _parse_utc_timestamp(value: object, field: str) -> datetime:
    text = _require_nonempty_string(value, field)
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must use UTC")
    return parsed.astimezone(UTC)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def datetime_to_unix_millis(value: datetime) -> int:
    """Return the containing Unix millisecond for an aware datetime."""

    timestamp = _require_aware(value, "datetime").astimezone(UTC)
    delta = timestamp - _UNIX_EPOCH
    return (
        delta.days * 86_400_000
        + delta.seconds * 1_000
        + delta.microseconds // 1_000
    )


def unix_millis_to_datetime(unix_millis: int) -> datetime:
    """Convert an integer Unix millisecond to an aware UTC datetime."""

    millis = _require_integer(unix_millis, "Unix milliseconds")
    if millis < 0:
        raise ValueError("Unix milliseconds must be nonnegative")
    try:
        return _UNIX_EPOCH + timedelta(milliseconds=millis)
    except OverflowError as exc:
        raise ValueError("Unix milliseconds are outside datetime range") from exc


@dataclass(frozen=True, slots=True)
class EventTimeSignal:
    """One auditable claim about the recording's UTC start time."""

    id: str
    kind: SignalKind
    candidate_start_utc: datetime
    source: str
    detail: str
    strength: float
    independent_group: str
    source_ref: str | None = None
    authoritative: bool = False

    def __post_init__(self) -> None:
        identifier = _require_nonempty_string(self.id, "signal id")
        if not _SIGNAL_ID_RE.fullmatch(identifier):
            raise ValueError(
                "signal id is not stable or contains unsupported characters"
            )
        try:
            kind = (
                self.kind
                if isinstance(self.kind, SignalKind)
                else SignalKind(self.kind)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported signal kind: {self.kind!r}") from exc
        timestamp = _require_utc(self.candidate_start_utc, "candidate_start_utc")
        source = _require_nonempty_string(self.source, "signal source")
        detail = _require_nonempty_string(self.detail, "signal detail")
        group = _require_nonempty_string(
            self.independent_group, "signal independent_group"
        )
        if len(group) > 128:
            raise ValueError("signal independent_group must not exceed 128 characters")
        source_ref = self.source_ref
        if source_ref is not None:
            source_ref = _require_nonempty_string(source_ref, "signal source_ref")
        strength = _require_finite_number(self.strength, "signal strength")
        if not 0 <= strength <= 1:
            raise ValueError("signal strength must be between 0 and 1")
        if not isinstance(self.authoritative, bool):
            raise TypeError("signal authoritative must be a boolean")
        if (kind is SignalKind.USER_OVERRIDE) != self.authoritative:
            raise ValueError(
                "user_override kind and authoritative=true must occur together"
            )

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "candidate_start_utc", timestamp)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "detail", detail)
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "independent_group", group)
        object.__setattr__(self, "source_ref", source_ref)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EventTimeSignal:
        if not isinstance(value, Mapping):
            raise TypeError("signal must be a mapping")
        try:
            timestamp = _parse_utc_timestamp(
                value.get("candidate_start_utc"), "candidate_start_utc"
            )
            return cls(
                id=value.get("id"),  # type: ignore[arg-type]
                kind=value.get("kind"),  # type: ignore[arg-type]
                candidate_start_utc=timestamp,
                source=value.get("source"),  # type: ignore[arg-type]
                source_ref=value.get("source_ref"),  # type: ignore[arg-type]
                detail=value.get("detail"),  # type: ignore[arg-type]
                strength=value.get("strength"),  # type: ignore[arg-type]
                independent_group=value.get(
                    "independent_group"
                ),  # type: ignore[arg-type]
                authoritative=value.get("authoritative", False),
            )
        except (TypeError, ValueError) as exc:
            signal_id = value.get("id", "<unknown>")
            raise type(exc)(f"invalid signal {signal_id!r}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind.value,
            "candidate_start_utc": _utc_iso(self.candidate_start_utc),
            "source": self.source,
            "detail": self.detail,
            "strength": self.strength,
            "independent_group": self.independent_group,
            "authoritative": self.authoritative,
        }
        if self.source_ref is not None:
            result["source_ref"] = self.source_ref
        return result


@dataclass(frozen=True, slots=True)
class EventEvidence:
    """Versioned evidence bundle plus the recording duration it describes."""

    schema_version: int
    recording_duration_seconds: float
    signals: tuple[EventTimeSignal, ...]

    def __post_init__(self) -> None:
        version = _require_integer(self.schema_version, "schema_version")
        if version != EVENT_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {EVENT_EVIDENCE_SCHEMA_VERSION}"
            )
        duration = _require_finite_number(
            self.recording_duration_seconds, "recording duration"
        )
        if duration <= 0:
            raise ValueError("recording duration must be positive")
        signals = tuple(self.signals)
        if not signals:
            raise ValueError("event evidence must contain at least one signal")
        if not all(isinstance(item, EventTimeSignal) for item in signals):
            raise TypeError("event evidence signals must be EventTimeSignal values")
        identifiers = [item.id for item in signals]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("signal ids must be unique")

        object.__setattr__(self, "recording_duration_seconds", duration)
        object.__setattr__(self, "signals", signals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recording_duration_seconds": self.recording_duration_seconds,
            "signals": [item.to_dict() for item in self.signals],
        }


def load_event_evidence_schema() -> dict[str, Any]:
    """Load the packaged event evidence JSON Schema."""

    try:
        schema = json.loads(EVENT_EVIDENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("event evidence schema cannot be loaded") from exc
    if not isinstance(schema, dict):
        raise RuntimeError("event evidence schema root must be an object")
    return schema


def _validate_against_schema(document: Mapping[str, Any]) -> None:
    validator = Draft202012Validator(
        load_event_evidence_schema(), format_checker=FormatChecker()
    )
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "root"
        raise ValueError(
            f"event evidence schema violation at {location}: {error.message}"
        )


def validate_event_evidence(document: Mapping[str, Any]) -> EventEvidence:
    """Validate a JSON-like evidence document with schema and Python semantics."""

    if not isinstance(document, Mapping):
        raise TypeError("event evidence must be a mapping")
    version = _require_integer(document.get("schema_version"), "schema_version")
    duration = _require_finite_number(
        document.get("recording_duration_seconds"), "recording duration"
    )
    raw_signals = document.get("signals")
    if not isinstance(raw_signals, Sequence) or isinstance(
        raw_signals, (str, bytes, bytearray)
    ):
        raise TypeError("event evidence signals must be an array")
    signals = tuple(EventTimeSignal.from_mapping(value) for value in raw_signals)
    evidence = EventEvidence(version, duration, signals)
    _validate_against_schema(document)
    return evidence


@dataclass(slots=True)
class _CandidateCluster:
    anchor: datetime
    signals: list[EventTimeSignal]

    @property
    def authoritative_signals(self) -> tuple[EventTimeSignal, ...]:
        return tuple(signal for signal in self.signals if signal.authoritative)

    def best_by_group(self) -> dict[str, EventTimeSignal]:
        selected: dict[str, EventTimeSignal] = {}
        for signal in self.signals:
            current = selected.get(signal.independent_group)
            if current is None or _signal_rank_key(signal) < _signal_rank_key(current):
                selected[signal.independent_group] = signal
        return selected


def _weighted_strength(signal: EventTimeSignal) -> float:
    return signal.strength * KIND_RELIABILITY_WEIGHT[signal.kind]


def _signal_rank_key(signal: EventTimeSignal) -> tuple[float, int, str]:
    return (
        -_weighted_strength(signal),
        _KIND_TIE_BREAK_PRIORITY[signal.kind],
        signal.id,
    )


def _representative(cluster: _CandidateCluster) -> EventTimeSignal:
    authority = cluster.authoritative_signals
    choices = authority if authority else tuple(cluster.signals)
    return min(choices, key=_signal_rank_key)


def _cluster_signals(
    signals: Sequence[EventTimeSignal], tolerance_seconds: float
) -> list[_CandidateCluster]:
    tolerance = timedelta(seconds=tolerance_seconds)
    ordered = sorted(signals, key=lambda item: (item.candidate_start_utc, item.id))
    clusters: list[_CandidateCluster] = []
    for signal in ordered:
        if not clusters or signal.candidate_start_utc - clusters[-1].anchor > tolerance:
            clusters.append(
                _CandidateCluster(signal.candidate_start_utc, [signal])
            )
        else:
            clusters[-1].signals.append(signal)
    return clusters


def _cluster_authoritative_signals(
    signals: Sequence[EventTimeSignal],
) -> list[_CandidateCluster]:
    """Group overrides only when they name the identical Unix millisecond."""

    by_millisecond: dict[int, list[EventTimeSignal]] = {}
    for signal in signals:
        millis = datetime_to_unix_millis(signal.candidate_start_utc)
        by_millisecond.setdefault(millis, []).append(signal)
    return [
        _CandidateCluster(unix_millis_to_datetime(millis), grouped)
        for millis, grouped in sorted(by_millisecond.items())
    ]


def _cluster_metrics(
    cluster: _CandidateCluster, strong_threshold: float
) -> tuple[int, float, tuple[str, ...]]:
    best = cluster.best_by_group()
    strong_groups = tuple(
        sorted(
            group
            for group, signal in best.items()
            if signal.strength >= strong_threshold
        )
    )
    score = math.fsum(_weighted_strength(signal) for signal in best.values())
    return len(strong_groups), score, strong_groups


def _cluster_rank_key(
    cluster: _CandidateCluster, strong_threshold: float
) -> tuple[int, int, float, datetime, str]:
    strong_count, score, _ = _cluster_metrics(cluster, strong_threshold)
    representative = _representative(cluster)
    return (
        -int(bool(cluster.authoritative_signals)),
        -strong_count,
        -score,
        representative.candidate_start_utc,
        representative.id,
    )


def _ranked_candidate(
    cluster: _CandidateCluster, strong_threshold: float, rank: int
) -> dict[str, Any]:
    representative = _representative(cluster)
    best_by_group = cluster.best_by_group()
    strong_count, score, strong_groups = _cluster_metrics(
        cluster, strong_threshold
    )
    return {
        "rank": rank,
        "candidate_start_utc": _utc_iso(
            cluster.anchor
            if cluster.authoritative_signals
            else representative.candidate_start_utc
        ),
        "score": round(score, 6),
        "independent_group_count": len(best_by_group),
        "independent_groups": sorted(best_by_group),
        "strong_independent_groups": strong_count,
        "strong_group_ids": list(strong_groups),
        "authoritative": bool(cluster.authoritative_signals),
        "signal_ids": sorted(signal.id for signal in cluster.signals),
        "evidence": [
            signal.to_dict()
            for signal in sorted(cluster.signals, key=lambda item: item.id)
        ],
    }


def _coerce_signals(
    signals: Sequence[EventTimeSignal | Mapping[str, Any]],
) -> tuple[EventTimeSignal, ...]:
    if not isinstance(signals, Sequence) or isinstance(
        signals, (str, bytes, bytearray)
    ):
        raise TypeError("signals must be a sequence")
    serialized: list[dict[str, Any]] = []
    for signal in signals:
        if isinstance(signal, EventTimeSignal):
            serialized.append(signal.to_dict())
        elif isinstance(signal, Mapping):
            serialized.append(dict(signal))
        else:
            raise TypeError("signal must be a mapping or EventTimeSignal")
    if not serialized:
        raise ValueError("at least one event-time signal is required")
    _validate_against_schema(
        {
            "schema_version": EVENT_EVIDENCE_SCHEMA_VERSION,
            "recording_duration_seconds": 1,
            "signals": serialized,
        }
    )
    normalized = tuple(
        signal
        if isinstance(signal, EventTimeSignal)
        else EventTimeSignal.from_mapping(signal)
        for signal in signals
    )
    identifiers = [signal.id for signal in normalized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("signal ids must be unique")
    return normalized


def _validate_resolver_options(
    duration_seconds: object,
    tolerance_seconds: object,
    strong_threshold: object,
    minimum_independent_groups: object,
    score_lead: object,
) -> tuple[float, float, float, int, float]:
    duration = _require_finite_number(duration_seconds, "recording duration")
    if duration <= 0:
        raise ValueError("recording duration must be positive")
    tolerance = _require_finite_number(tolerance_seconds, "cluster tolerance")
    if tolerance <= 0:
        raise ValueError("cluster tolerance must be positive")
    threshold = _require_finite_number(strong_threshold, "strong threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("strong threshold must be between 0 and 1")
    minimum_groups = _require_integer(
        minimum_independent_groups, "minimum independent groups"
    )
    if minimum_groups < 2:
        raise ValueError("minimum independent groups must be at least 2")
    lead = _require_finite_number(score_lead, "score lead")
    if lead <= 0:
        raise ValueError("score lead must be positive")
    return duration, tolerance, threshold, minimum_groups, lead


def resolve_event_time(
    signals: Sequence[EventTimeSignal | Mapping[str, Any]],
    duration_seconds: float,
    *,
    tolerance_seconds: float = DEFAULT_CLUSTER_TOLERANCE_SECONDS,
    strong_threshold: float = DEFAULT_STRONG_SIGNAL_THRESHOLD,
    minimum_independent_groups: int = DEFAULT_MINIMUM_INDEPENDENT_GROUPS,
    score_lead: float = DEFAULT_SCORE_LEAD,
    timezone_name: str = DEFAULT_TIMEZONE_NAME,
) -> dict[str, Any]:
    """Resolve a recording start or return an explicit ambiguity blocker.

    Non-authoritative candidates require at least two independent groups whose
    raw strengths are at least 0.7 by default. Signals within one independent
    group contribute only their strongest weighted observation. If two eligible
    candidates have the same corroboration count, the leader must exceed the
    runner-up score by 0.25. Otherwise no date is selected.

    Authoritative user overrides bypass corroboration only when every override
    names the identical Unix millisecond (zero-millisecond tolerance).
    """

    duration, tolerance, threshold, minimum_groups, required_lead = (
        _validate_resolver_options(
            duration_seconds,
            tolerance_seconds,
            strong_threshold,
            minimum_independent_groups,
            score_lead,
        )
    )
    normalized = _coerce_signals(signals)
    timezone_text = _require_nonempty_string(timezone_name, "timezone name")
    try:
        local_zone = ZoneInfo(timezone_text)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown timezone: {timezone_text}") from exc

    authoritative_signals = tuple(
        signal for signal in normalized if signal.authoritative
    )
    if authoritative_signals:
        ordinary_signals = tuple(
            signal for signal in normalized if not signal.authoritative
        )
        clusters = _cluster_authoritative_signals(authoritative_signals)
        if ordinary_signals:
            clusters.extend(_cluster_signals(ordinary_signals, tolerance))
    else:
        clusters = _cluster_signals(normalized, tolerance)
    clusters.sort(key=lambda cluster: _cluster_rank_key(cluster, threshold))
    ranked = [
        _ranked_candidate(cluster, threshold, index)
        for index, cluster in enumerate(clusters, start=1)
    ]
    authoritative_clusters = [
        cluster for cluster in clusters if cluster.authoritative_signals
    ]

    selected: _CandidateCluster | None = None
    resolution_basis: str | None = None
    if len(authoritative_clusters) == 1:
        selected = authoritative_clusters[0]
        resolution_basis = "authoritative_user_override"
    elif not authoritative_clusters:
        eligible = [
            cluster
            for cluster in clusters
            if _cluster_metrics(cluster, threshold)[0] >= minimum_groups
        ]
        if eligible:
            leader = eligible[0]
            leader_groups, leader_score, _ = _cluster_metrics(leader, threshold)
            clear_lead = len(eligible) == 1
            if len(eligible) > 1:
                runner_groups, runner_score, _ = _cluster_metrics(
                    eligible[1], threshold
                )
                has_score_tie = any(
                    _cluster_metrics(candidate, threshold)[1] == leader_score
                    for candidate in eligible[1:]
                )
                clear_lead = not has_score_tie and (
                    leader_groups > runner_groups
                    or (
                        leader_groups == runner_groups
                        and leader_score - runner_score >= required_lead
                    )
                )
            if clear_lead:
                selected = leader
                resolution_basis = "corroborated_evidence"

    common: dict[str, Any] = {
        "schema_version": EVENT_EVIDENCE_SCHEMA_VERSION,
        "duration_seconds": duration,
        "tolerance_seconds": tolerance,
        "strong_threshold": threshold,
        "minimum_independent_groups": minimum_groups,
        "score_lead_required": required_lead,
        "authoritative_override_tolerance_millis": (
            AUTHORITATIVE_OVERRIDE_TOLERANCE_MILLIS
        ),
        "ranked_candidates": ranked,
    }
    if selected is None:
        return {
            **common,
            "status": "blocked",
            "blocker": "ambiguous_event_time",
            "blocker_detail": (
                "event time lacks independent corroboration or a clear lead"
            ),
        }

    representative = _representative(selected)
    start_utc = (
        selected.anchor
        if resolution_basis == "authoritative_user_override"
        else representative.candidate_start_utc
    )
    end_utc = start_utc + timedelta(seconds=duration)
    start_local = start_utc.astimezone(local_zone)
    end_local = end_utc.astimezone(local_zone)
    authority_ids = sorted(signal.id for signal in selected.authoritative_signals)
    return {
        **common,
        "status": "resolved",
        "resolution_basis": resolution_basis,
        "start_utc": _utc_iso(start_utc),
        "end_utc": _utc_iso(end_utc),
        "start_unix": datetime_to_unix_millis(start_utc) // 1_000,
        "timezone": timezone_text,
        "start_local": start_local.isoformat(),
        "end_local": end_local.isoformat(),
        "local_date": start_local.date().isoformat(),
        "authoritative_signal_ids": authority_ids,
    }


__all__ = [
    "AUTHORITATIVE_OVERRIDE_TOLERANCE_MILLIS",
    "DEFAULT_CLUSTER_TOLERANCE_SECONDS",
    "DEFAULT_MINIMUM_INDEPENDENT_GROUPS",
    "DEFAULT_SCORE_LEAD",
    "DEFAULT_STRONG_SIGNAL_THRESHOLD",
    "DEFAULT_TIMEZONE_NAME",
    "EVENT_EVIDENCE_SCHEMA_PATH",
    "EVENT_EVIDENCE_SCHEMA_VERSION",
    "EventEvidence",
    "EventTimeSignal",
    "KIND_RELIABILITY_WEIGHT",
    "SignalKind",
    "datetime_to_unix_millis",
    "load_event_evidence_schema",
    "resolve_event_time",
    "unix_millis_to_datetime",
    "validate_event_evidence",
]
