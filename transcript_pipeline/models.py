"""Typed state records and UTC serialization helpers used by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Mapping


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_utc_iso(value: datetime) -> str:
    """Serialize a timezone-aware datetime as an unambiguous UTC timestamp."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("UTC serialization requires a timezone-aware datetime")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def from_utc_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("UTC timestamp must be a non-empty ISO-8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("UTC timestamp must include a timezone offset")
    return parsed.astimezone(UTC)


def utc_now_iso() -> str:
    return to_utc_iso(utc_now())


# Clear aliases for callers that prefer the longer spelling.
to_iso8601_utc = to_utc_iso
parse_iso8601_utc = from_utc_iso


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    type: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type, "message": self.message}
        if self.details:
            result["details"] = dict(self.details)
        return result


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    path: str
    sha256: str | None = None
    validation: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"path": self.path}
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        if self.validation:
            result["validation"] = dict(self.validation)
        return result


@dataclass(frozen=True, slots=True)
class BlockerRecord:
    id: str
    stage: str
    code: str
    message: str
    created_at: str
    details: Mapping[str, Any] = field(default_factory=dict)
    resolved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "created_at": self.created_at,
        }
        if self.details:
            result["details"] = dict(self.details)
        if self.resolved_at is not None:
            result["resolved_at"] = self.resolved_at
        return result


@dataclass(frozen=True, slots=True)
class StageRecord:
    status: StageStatus = StageStatus.PENDING
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    artifacts: Mapping[str, str] = field(default_factory=dict)
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)
    error: ErrorSummary | None = None
    blocker_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status.value,
            "attempts": self.attempts,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "artifacts": dict(self.artifacts),
            "artifact_hashes": dict(self.artifact_hashes),
            "validation": dict(self.validation),
            "blocker_ids": list(self.blocker_ids),
        }
        if self.error is not None:
            result["error"] = self.error.to_dict()
        return result
